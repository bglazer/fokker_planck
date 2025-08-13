import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad

# -----------------------------
# p(x,t): same as yours, small tweak in dx_dt
# -----------------------------

class Pxt(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim + 1, hidden_dim, bias=True))
        layers.append(nn.SiLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1, bias=True))
        self.layers = nn.ModuleList(layers)
        self.model  = nn.Sequential(*layers)

        # Optional time scaling (kept from your code, but off by default)
        self.tscale = nn.Parameter(torch.ones(input_dim + 1), requires_grad=False)

    def xts(self, x, ts):
        # x: (B, D), ts: (T,)
        xs  = x.repeat((ts.shape[0], 1, 1))           # (T, B, D)
        ts_ = ts.repeat((x.shape[0], 1)).T.unsqueeze(2)  # (T, B, 1)
        return torch.cat((xs, ts_), dim=2)            # (T, B, D+1)

    def forward(self, xts):
        # xts: (T, B, D+1)
        s = xts.shape
        x = xts.reshape(-1, xts.shape[-1])            # (T*B, D+1)
        # first layer with optional scaling on inputs
        layer0 = self.layers[0]
        x = (x * self.tscale) @ layer0.weight.T + layer0.bias
        for layer in self.layers[1:]:
            x = layer(x)
        x = x.reshape(s[:-1] + (-1,))                 # (T, B, 1)
        return x

    def log_pxt(self, x, ts):
        return self.forward(self.xts(x, ts))

    def pxt(self, x, ts):
        return torch.exp(self.log_pxt(x, ts))

    def set_tscale(self, tscale):
        self.tscale.data[-1] = tscale

    def log_px(self, x, ts):
        return torch.logsumexp(self.log_pxt(x, ts), dim=0)

    def dx_dt(self, x, ts):
        """
        Returns:
          dq_dx: (T, B, D)
          dq_dt: (T, B, 1)  # physical time derivative
        """
        xts = self.xts(x, ts)
        xts.requires_grad_(True)
        q = self.forward(xts)                          # (T, B, 1)
        g = grad(q, xts, grad_outputs=torch.ones_like(q),
                 create_graph=True, retain_graph=True)[0]    # (T, B, D+1)
        dq_dx = g[..., :-1]
        dq_dt_internal = g[..., -1:]                       # derivative w.r.t. raw input t
        # Chain-rule correction if time was scaled inside forward by a gain s_t:
        # if forward used t_scaled = s_t * t, then autograd gives dq/dt = (dq/dt_scaled) * s_t.
        # To recover dq/dt_scaled (or to unscale an exaggerated magnitude), divide by s_t.
        s_t = None
        # last entry is the time channel scale
        s_t = self.tscale[-1]
        dq_dt = dq_dt_internal / s_t
        return dq_dx, dq_dt

# -----------------------------
# Potential flow: u = ∇_x φ(x,t), div u = Δ_x φ
# -----------------------------

class Phi(nn.Module):
    """
    Scalar potential φ(x). Drift u = ∇_x φ, divergence = Δ_x φ.
    """
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        layers = []

        layers += [nn.Linear(input_dim, hidden_dim, bias=True)]
        layers += [nn.SiLU()]

        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=True)]
            layers += [nn.SiLU()]

        layers += [nn.Linear(hidden_dim, 1, bias=True)]   # scalar φ
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def u(self, x):
        """Compute the gradient ∇_x φ(x)"""
        x.requires_grad_(True)                            # make x a variable
        y = self.forward(x)                                  # shape (N,1)
        grad_outputs = torch.ones_like(y)                    # make gradient scalar
        g = grad(y, x, grad_outputs=grad_outputs,
                 create_graph=True, retain_graph=True)[0]
        # x.requires_grad_(False)
        return g
    
    
    def grad_and_laplacian(self, x):
        """
        Compute u = ∇_x φ(x) and Δ_x φ.
        Args:
          x: (B, D)
          estimator: "hutchinson" or "exact"
          n_probe: probes for Hutchinson
        Returns:
          u:   (B, D)
          lap: (B, 1)
        """
        u = self.u(x)

        # Δφ = sum_i ∂^2 φ / ∂x_i^2
        lap = 0.0
        for i in range(u.shape[-1]):
            gi = u[..., i]                            
            dgi_dxts = grad(gi, x, grad_outputs=torch.ones_like(gi),
                            create_graph=True, retain_graph=True)[0]  
            lap_i = dgi_dxts[..., i:i+1]             # pick derivative wrt x_i
            lap = lap + lap_i

        return u, lap

class CellDelta(nn.Module):
    """
    CellDelta is a learned model of cell differentiation in single-cell RNA-seq single timepoint data.
    It models the developmental trajectory of cells as a driven stochastic process
    # TODO expand this docstring to describe the rationale of the model
    """
    def __init__(self, input_dim, 
                 ux_hidden_dim, ux_layers,
                 pxt_hidden_dim, pxt_layers,
                 device='cpu') -> None:
        """
        Initialize the CellDelta model with the given hyperparameters.

        Args:
            input_dim (int): The dimensionality of the input data.
            ux_hidden_dim (int): The number of hidden units in each layer for the UX model.
            ux_layers (int): The number of layers in the UX model.
            pxt_hidden_dim (int): The number of hidden units in each layer for the PXT model.
            pxt_layers (int): The number of layers in the PXT model.
            loss_type (str, optional): The type of loss to use for training. Defaults to 'nce'. Must be one of ('nce', 'ence', 'self').
            device (torch.device): The device to use for the model.

        Returns:
            None
        """
        super().__init__()
        # self.phi = Ux(input_dim, ux_hidden_dim, ux_layers, ux_batch_norm).to(device)
        self.phi = Phi(input_dim, ux_hidden_dim, ux_layers).to(device)
        self.pxt = Pxt(input_dim, pxt_hidden_dim, pxt_layers).to(device)
        self.device = device
        # Add the component models (ux, pxt, nce) to a module list
        self.models = torch.nn.ModuleDict({'phi':self.phi, 'pxt':self.pxt})
    
    def nce_loss(self, x, noise, ts, scale):
        """
        Compute the Noise-Contrastive Estimation (NCE) loss for the given data.

        Args:
            x (torch.tensor): The input data of shape (n_cells, n_genes).
            ts (torch.tensor): The time points at which to evaluate the model of shape (n_timesteps,).
        
        Returns:
            torch.tensor: The NCE loss value.
            torch.tensor: The sample vs noise classification accuracy.
        """
        y = noise.sample((x.shape[0],))

        log_scale = torch.log(torch.tensor(scale))
        logp_x = self.pxt.log_px(x, ts) + log_scale # logp(x)
        logq_x = noise.log_prob(x).unsqueeze(1) # logq(x)
        logp_y = self.pxt.log_px(y, ts) + log_scale # logp(y)
        logq_y = noise.log_prob(y).unsqueeze(1) # logq(y)

        value_x = logp_x - torch.logaddexp(logp_x, logq_x)  # logp(x)/(logp(x) + logq(x))
        value_y = logq_y - torch.logaddexp(logp_y, logq_y)  # logq(y)/(logp(y) + logq(y))

        v = value_x.mean() + value_y.mean()

        # Classification of noise vs target
        r_x = torch.sigmoid(logp_x - logq_x)
        r_y = torch.sigmoid(logq_y - logp_y)

        # Compute the classification accuracy
        acc = ((r_x > 1/2).sum() + (r_y > 1/2).sum()).cpu().numpy() / (len(x) + len(y))
        
        return -v, acc
        
    def fokker_planck_loss(self, x, ts):
        """
        Residual: r = dq/dt + u·∇q + div u,   with u = ∇φ,  div u = Δφ
        Returns mean squared residual.
        x:  (B, D)
        ts: (T,)
        """
        # score and time-derivative of q = log p
        dq_dx, dq_dt = self.pxt.dx_dt(x, ts)  # (T,B,D), (T,B,1)

        # potential drift and Laplacian
        u, lap = self.phi.grad_and_laplacian(x)  # (B,D), (B,1)

        # FP residual
        residual = dq_dt + (u * dq_dx).sum(dim=-1, keepdim=True) + lap   # (T,B,1)

        return (residual ** 2).mean()

    def consistency_loss(self, X, ts):
        """
        Ensure that each timepoint has a similar mean probability to t=0
        """
        log_pxt = self.pxt.log_pxt(X, ts)
        l_cons = (log_pxt[1:].mean(1) - log_pxt[0].mean())**2
        return l_cons.mean()

    def time_responsibilities(self, X, ts):
        # log_pxt: (T,B,1) -> (T,B); softmax over T gives p(t|x) up to proportionality
        logp = self.pxt.log_pxt(X, ts).squeeze(-1)          # (T,B)
        w = torch.softmax(logp, dim=0)                      # sum_T w = 1 for each b
        return w

    def time_prior_kl(self, X, ts, prior="uniform"):
        w = self.time_responsibilities(X, ts)               # (T,B), sum over T = 1
        T = w.size(0)
        if prior == "uniform":
            logu = -np.log(T)
            kl_b = (w * (torch.log(w + 1e-8) - logu)).sum(dim=0)  # (B,)
        else:
            raise NotImplementedError
        return kl_b.mean()
    
    def divergence_penalty(self, x, lam=1e-2):
        _, lap = self.phi.grad_and_laplacian(x)  # (B,1)
        return lam * (lap**2).mean()

    def optimize(self, X, X0, ts, px_noise, p0_noise, p0_alpha=1,
                 pxt_lr=5e-4, ux_lr=1e-3, fokker_planck_alpha=1, 
                 consistency_alpha=None,
                 time_prior_kl_alpha=None,
                 div_penalty_alpha=None,
                 p_alpha=None,
                 n_epochs=100, n_samples=1000, verbose=False):
        """
        Optimize the cell delta model parameters using the provided training data.

        Args:
            X (torch.tensor): The input training data of shape (n_cells, n_genes).
            X0 (torch.tensor): The initial state of the cells of shape (n_cells, n_genes).
            ts (torch.tensor): The time points at which to evaluate the model of shape (n_timesteps,).
            px_noise (torch.distributions): The noise distribution to use for the NCE loss for the overall distribution.
            p0_noise (torch.distributions): The noise distribution to use for the NCE loss for the initial conditions.
            p0_alpha (float, optional): The weight to apply to the initial conditions loss. Defaults to 1.
            pxt_lr (float, optional): The learning rate for the PXT model. Defaults to 5e-4.
            ux_lr (float, optional): The learning rate for the UX model. Defaults to 1e-3.
            fokker_planck_alpha (bool, optional): The weight to apply to the Fokker-Planck loss term. Defaults to 1.            n_epochs (int, optional): The number of epochs to train for. Defaults to 100.
            n_samples (int, optional): The number of data samples to use in training for each epoch. Defaults to 1000.
            verbose (bool, optional): Whether to print the optimization progress. Defaults to False.

        Returns:
            dict: A dictionary containing the loss values for each epoch.
        """
        self.pxt_optimizer = torch.optim.Adam(self.pxt.parameters(), lr=pxt_lr, weight_decay=1e-3)
        self.phi_optimizer = torch.optim.Adam(self.phi.parameters(), lr=ux_lr, weight_decay=1e-3)

        # Convenience variable for the time t=0
        zero = torch.zeros(1).to(self.device)
        
        l_nce_pxs = np.zeros(n_epochs)
        l_nce_p0s = np.zeros(n_epochs)
        l_fps = np.zeros(n_epochs)
        
        n_samples = min(n_samples, len(X))

        for epoch in range(n_epochs):
            # Sample from the data distribution
            rand_idxs = torch.randperm(len(X))[:n_samples]
            # TODO do I need to clone here?
            x = X[rand_idxs].detach()
            x0 = X0.detach()

            self.pxt_optimizer.zero_grad()
            self.phi_optimizer.zero_grad()

            # Calculate the Noise-Constrastive Loss of the distribution
            # of p(x,t) marginalized over t: p(x) = \int p(x,t) dt
            if p_alpha is not None:
                l_nce_px, acc_px = self.nce_loss(x, px_noise, ts=ts, scale=1/ts.shape[0])
                l_nce_px.backward()
            else:
                l_nce_px = zero
                acc_px = zero
            
            # Calculate the Noise-Constrastive Loss of the initial distribution
            if p0_alpha is not None:
                l_nce_p0, acc_p0 = self.nce_loss(x0, p0_noise, ts=zero, scale=1)
                l_nce_p0 = l_nce_p0 * p0_alpha
                l_nce_p0.backward()
            else:
                l_nce_p0 = zero
                acc_p0 = zero
                       
            if fokker_planck_alpha is not None:
                # Calculate the Fokker-Planck loss
                l_fp = self.fokker_planck_loss(x, ts)*fokker_planck_alpha
                l_fp.backward()
            else:
                l_fp = zero

            if consistency_alpha is not None:
                l_cons = self.consistency_loss(x, ts)*consistency_alpha
                l_cons.backward()
            else:
                l_cons = zero

            if time_prior_kl_alpha is not None:
                l_tpk = self.time_prior_kl(X, ts)*time_prior_kl_alpha
                l_tpk.backward()
            else:
                l_tpk = zero

            if div_penalty_alpha is not None:
                l_div = self.divergence_penalty(x)*div_penalty_alpha
                l_div.backward()
            else:
                l_div = zero

            self.pxt_optimizer.step()
            self.phi_optimizer.step()

            # Record the losses
            l_nce_pxs[epoch] = float(l_nce_px.mean())
            l_fps[epoch] = float(l_fp.mean())
            
            if verbose:
                print(f'{epoch:6d} l_nce_px={float(l_nce_px):.5f}, acc_px={float(acc_px):.5f}, '
                    f'l_nce_p0={float(l_nce_p0): .5f}, '
                    f'acc_p0={float(acc_p0): .5f}, '
                    f'l_fp={float(l_fp):.5f}, '
                    f'l_cons={float(l_cons):.5f}, '
                    f'l_tpk={float(l_tpk):.5f}, '
                    f'l_div={float(l_div):.5f}'
                    )
                if (epoch+1) % 50 == 0:
                    # FP term stats
                    fp = self.fp_terms(x, ts, lap_estimator="exact")
                    print(f"[FP] {epoch} stats:", {k: round(v,5) for k,v in fp["stats"].items()})

                    # Grad conflict stats (cheap if done occasionally)
                    gc = self.grad_conflict(x, x0, ts, px_noise, p0_noise)
                    print(f"[GRADS] {epoch}:", {k: round(v,5) for k,v in gc.items()})

                    fpT = self.fp_per_time(x, ts)
                    print("[FP/T]", {k: np.round(v, 4).tolist() for k,v in fpT.items()})

                    vfit = self.eval_const_u_fit(x, ts)
                    print("[CONST-U]", {k: round(v, 5) for k,v in vfit.items()})

                    par = self.parallel_fraction(x, ts)
                    print("[U‖]", {k: round(v, 5) for k,v in par.items()})

                    # inside the verbose block, after the parallel_fraction print:
                    tv_stats = self.time_variation_logp(x, ts)
                    fd_stats = self.fp_density_residual(x, ts)
                    div_stats = self.divergence_stats(x)
                    print("[TV]", {k: round(v, 5) for k, v in tv_stats.items()})
                    print("[FD]", {k: round(v, 5) for k, v in fd_stats.items()})
                    print("[DIV]", {k: round(v, 5) for k, v in div_stats.items()})


        return {'l_nce_px': l_nce_pxs, 'l_nce_p0': l_nce_p0s, 'l_fp': l_fps, 'l_tpk': l_tpk}
    
    def optimize_fokker_planck(self, X, ts,
                               ux_lr=1e-3, fokker_planck_alpha=1,
                               noise=None,
                               n_epochs=100, n_samples=1000, verbose=False):
        """
        Optimize the Fokker-Planck component of the loss independently of the NCE component.
        """
        self.phi_optimizer = torch.optim.Adam(self.phi.parameters(), lr=ux_lr)

        l_fps = np.zeros(n_epochs)
        l_fp0s = np.zeros(n_epochs)
        
        n_samples = min(n_samples, len(X))

        if noise is not None:
            # Create a Gaussian distribution
            normal = torch.distributions.Normal(loc=torch.zeros(n_samples,device=self.device), 
                                                scale=torch.ones(n_samples,device=self.device)*noise)

        for epoch in range(n_epochs):
            # Sample from the data distribution
            rand_idxs = torch.randperm(len(X))[:n_samples]
            x = X[rand_idxs].clone().detach()
            if noise is not None:
                # Add Gaussian noise to the data
                x = x + normal.sample().unsqueeze(1)

            self.pxt_optimizer.zero_grad()
            self.phi_optimizer.zero_grad()

            # Calculate the Fokker-Planck loss
            l_fp = self.fokker_planck_loss(x, ts)*fokker_planck_alpha
            l_fp.backward()

            # Fokker-Planck loss at t=0
            # l_fp0 = self.fokker_planck_loss(x, ts[:1])*fokker_planck_alpha
            # l_fp0.backward()
            l_fp0 = torch.zeros(1).to(self.device)

            self.phi_optimizer.step()

            # Record the losses
            l_fps[epoch] = float(l_fp.mean())
            l_fp0s[epoch] = float(l_fp0.mean())
            
            if verbose:
                print(f'{epoch} l_fp={float(l_fp):.5f}, l_fp0={float(l_fp0):.5f}')
                
        return {'l_fp': l_fps, 'l_fp0': l_fp0s}
    
    def optimize_initial_conditions(self, X0, ts, p0_noise, pxt_lr=1e-3,
                                    n_epochs=100, verbose=False, scale=1):
        """
        Optimize the initial conditions of the model.
        """
        nce_loss = self.nce_loss

        self.pxt_optimizer = torch.optim.Adam(self.pxt.parameters(), lr=pxt_lr)

        l_nce_p0s = np.zeros(n_epochs)
        
        for epoch in range(n_epochs):
            ti = torch.randint(0, len(ts), (1,)).item()
            t = ts[ti,None].detach()

            self.pxt_optimizer.zero_grad()
            
            # Fit every time to the initial distribution
            l_nce_p0, acc_p0 = nce_loss(X0, p0_noise, ts=t, scale=scale)
            l_nce_p0.backward()

            self.pxt_optimizer.step()

            # Record the losses
            l_nce_p0s[epoch] = float(l_nce_p0.mean())
            
            if verbose:
                print(f'{epoch} l_nce_p0={float(l_nce_p0):.5f}, acc_p0={float(acc_p0):.5f}')
                
        return {'l_nce_p0': l_nce_p0s}

    def simulate(self, X0, tsim, sigma=1, zero_boundary=True):
        """
        Simulate the stochastic differential equation using the Euler-Maruyama method
        with the learned drift term u(x)
        """
        x = X0.clone().detach()
        xts = torch.zeros((len(tsim), x.shape[0], x.shape[1]), device='cpu')
        ht = tsim[1] - tsim[0]
        zero_boundary = zero_boundary
        # sigma = torch.ones_like(x)*sigma
        
        for i in range(len(tsim)):
            # Compute the drift term
            u = self.phi.u(x)
            # Compute the diffusion term
            # Generate a set of random numbers
            dW = torch.randn_like(x) * torch.sqrt(ht)
            # Compute the change in x
            dx = u * ht + sigma * dW
            # dx = dx.squeeze(0)
            # Update x
            x = x + dx
            if zero_boundary:
                x[x < 0] = 0.0
            xts[i,:,:] = x.cpu().detach()
        return xts
    
    def fp_terms(self, x, ts, lap_estimator="exact", n_probe=8):
        """
        Returns detached tensors of each FP term and summary stats.
        """
        with torch.enable_grad():
            dq_dx, dq_dt = self.pxt.dx_dt(x, ts)               # (T,B,D), (T,B,1)
            u, lap_exact = self.phi.grad_and_laplacian(x)      # (B,D), (B,1)

            adv = (u * dq_dx).sum(dim=-1, keepdim=True)        # (T,B,1)
            if lap_estimator == "exact":
                divu = lap_exact.expand(dq_dt.shape)           # broadcast over T
            elif lap_estimator == "zero":
                # helpful to see what happens if Δφ effectively vanishes
                divu = torch.zeros_like(dq_dt)
            else:
                raise ValueError("lap_estimator must be 'exact' or 'zero'")

            residual = dq_dt + adv + divu                      # (T,B,1)

        # Summaries across (T,B)
        stats = {}
        stats.update(_tensor_stats("dq_dt", dq_dt))
        stats.update(_tensor_stats("adv",   adv))
        stats.update(_tensor_stats("divu",  divu))
        stats.update(_tensor_stats("res",   residual))

        # How aligned is adv with -dq_dt? (For a translating Gaussian, this should be ~1)
        with torch.no_grad():
            a = adv.flatten()
            b = (-dq_dt).flatten()
            dot = float((a*b).mean())
            corr = float(torch.corrcoef(torch.stack([a, b]))[0,1]) if a.numel() == b.numel() and a.numel() > 1 else float('nan')
            stats.update({"adv_vs_neg_dqdt_dot": dot, "adv_vs_neg_dqdt_corr": corr})

        return {"dq_dt": dq_dt.detach(), "adv": adv.detach(), "divu": divu.detach(),
                "residual": residual.detach(), "stats": stats}
    
    def grad_conflict(self, x, x0, ts, px_noise, p0_noise):
        # Compute NCE grads
        self.pxt_optimizer.zero_grad()
        self.phi_optimizer.zero_grad()
        l_nce_px, _ = self.nce_loss(x,  px_noise, ts=ts,  scale=1/ts.shape[0])
        l_nce_p0, _ = self.nce_loss(x0, p0_noise, ts=torch.zeros(1,device=x.device), scale=1)
        lnce = l_nce_px + l_nce_p0
        lnce.backward(retain_graph=True)
        g_pxt_nce = _flat_grads(self.pxt)
        g_phi_nce = _flat_grads(self.phi)

        # Compute FP grads
        self.pxt_optimizer.zero_grad()
        self.phi_optimizer.zero_grad()
        l_fp = self.fokker_planck_loss(x, ts)
        l_fp.backward(retain_graph=True)
        g_pxt_fp = _flat_grads(self.pxt)
        g_phi_fp = _flat_grads(self.phi)

        return {
            "cos_pxt": _cosine(g_pxt_nce, g_pxt_fp),
            "cos_phi": _cosine(g_phi_nce, g_phi_fp),
            "||g_pxt_nce||": float(g_pxt_nce.norm()),
            "||g_pxt_fp||":  float(g_pxt_fp .norm()),
            "||g_phi_nce||": float(g_phi_nce.norm()),
            "||g_phi_fp||":  float(g_phi_fp .norm()),
        }
    
    def fp_per_time(self, x, ts):
        dq_dx, dq_dt = self.pxt.dx_dt(x, ts)         # (T,B,D), (T,B,1)
        u, lap = self.phi.grad_and_laplacian(x)      # (B,D), (B,1)
        adv = (u * dq_dx).sum(-1, keepdim=True)      # (T,B,1)
        divu = lap.expand_as(dq_dt)
        res = dq_dt + adv + divu                     # (T,B,1)
        with torch.no_grad():
            return {
                "res_mean_per_T": res.mean(1).flatten().cpu().numpy()[::10],
                "res_abs_mean_per_T": res.abs().mean(1).flatten().cpu().numpy()[::10],
                "dq_dt_abs_mean_per_T": dq_dt.abs().mean(1).flatten().cpu().numpy()[::10],
                "adv_abs_mean_per_T": adv.abs().mean(1).flatten().cpu().numpy()[::10],
                "divu_abs_mean_per_T": divu.abs().mean(1).flatten().cpu().numpy()[::10],
            }
        
    def parallel_fraction(self, x, ts):
        dq_dx, _ = self.pxt.dx_dt(x, ts)       # (T,B,D)
        u, _ = self.phi.grad_and_laplacian(x)  # (B,D)
        with torch.no_grad():
            # project u onto dq_dx (per T,B), report mean fraction
            T,B,D = dq_dx.shape
            u_expanded = u.unsqueeze(0).expand(T,-1,-1)              # (T,B,D)
            num = (u_expanded * dq_dx).sum(-1)                       # (T,B)
            den = (u_expanded.norm(dim=-1) * dq_dx.norm(dim=-1) + 1e-8)
            cos = (num / den).abs()                                  # |cos|
        return {"u_parallel_frac_mean": float(cos.mean()),
                "u_parallel_frac_p25": float(cos.quantile(0.25)),
                "u_parallel_frac_p75": float(cos.quantile(0.75))}
    
    def eval_const_u_fit(self, x, ts):
        # need autograd inside dx_dt to compute derivatives
        dq_dx, dq_dt = self.pxt.dx_dt(x, ts)              # (T,B,D), (T,B,1)
        # compute v_hat on detached copies (no graph needed for lstsq)
        v_hat = torch.linalg.lstsq((-dq_dx.detach()).reshape(-1, dq_dx.shape[-1]),
                                dq_dt.detach().reshape(-1, 1)).solution.squeeze(1)  # (D,)

        # u and comparisons don't need to build higher-order grads here; detach for safety
        u_pred, _ = self.phi.grad_and_laplacian(x)        # uses enable_grad internally
        with torch.no_grad():
            cos = torch.nn.functional.cosine_similarity(u_pred, v_hat, dim=-1).mean()
            mse = ((u_pred - v_hat)**2).mean()
            norm = v_hat.norm()
        return {"cos_u_vhat": float(cos), "mse_u_vhat": float(mse), "v_hat_norm": float(norm)}

    def time_variation_logp(self, X, ts):
        logp = self.pxt.log_pxt(X, ts).squeeze(-1)  # (T,B)
        var_t = logp.var(dim=0)                     # (B,)
        return {"var_t_logp_mean": float(var_t.mean()),
                "var_t_logp_p25":  float(var_t.quantile(0.25)),
                "var_t_logp_p75":  float(var_t.quantile(0.75))}
    
    def fp_density_residual(self, X, ts):
        dq_dx, dq_dt = self.pxt.dx_dt(X, ts)                 # (T,B,D),(T,B,1)
        u, lap = self.phi.grad_and_laplacian(X)              # (B,D),(B,1)
        adv = (u * dq_dx).sum(-1, keepdim=True)              # (T,B,1)
        rlog = dq_dt + adv + lap.expand_as(dq_dt)            # (T,B,1)
        p = torch.exp(self.pxt.log_pxt(X, ts))               # (T,B,1)
        rden = p * rlog
        return {"rlog_abs_mean": float(rlog.abs().mean()),
                "rden_abs_mean": float(rden.abs().mean())}

    def divergence_stats(self, X):
        _, lap = self.phi.grad_and_laplacian(X)      # (B,1)
        return {"div_abs_mean": float(lap.abs().mean()),
                "div_p95": float(lap.abs().quantile(0.95))}

# --- utilities you can add somewhere in your file ---

def estimate_v_const(dq_dx, dq_dt):
    # dq_dx: (T,B,D), dq_dt: (T,B,1)
    T,B,D = dq_dx.shape
    A = (-dq_dx).reshape(T*B, D)
    b = dq_dt.reshape(T*B, 1)
    # lstsq handles rank deficiency; returns (D,1)
    v_hat = torch.linalg.lstsq(A, b).solution.squeeze(1)   # (D,)
    return v_hat

def _tensor_stats(name, t):
    # returns compact stats for logging
    with torch.no_grad():
        return {
            f'{name}_mean': float(t.mean()),
            f'{name}_std':  float(t.std()),
            f'{name}_abs_mean': float(t.abs().mean()),
            f'{name}_p95': float(t.abs().quantile(0.95)),
        }

def _grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return (total ** 0.5)

def _flat_grads(module):
    """Concatenate grads for all params; use zeros for missing grads so length is stable."""
    vecs = []
    for p in module.parameters():
        if p.grad is None:
            vecs.append(torch.zeros_like(p).reshape(-1))
        else:
            vecs.append(p.grad.detach().reshape(-1))
    if not vecs:  # module with no params
        return torch.zeros(1)
    return torch.cat(vecs)

def _cosine(a, b, eps=1e-12):
    """Cosine similarity that’s tolerant of tiny norms."""
    # (Lengths should now match; keep a guard anyway.)
    if a.numel() != b.numel():
        m = min(a.numel(), b.numel())
        a, b = a[:m], b[:m]
    an, bn = a.norm(), b.norm()
    if an < eps or bn < eps:
        return 0.0
    return float((a @ b) / (an * bn + eps))
