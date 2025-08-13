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
        layers.append(nn.LeakyReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(nn.LeakyReLU())
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
    def __init__(self, input_dim, hidden_dim, n_layers, batch_norm=False):
        super().__init__()
        layers = []

        layers += [nn.Linear(input_dim, hidden_dim, bias=True)]
        layers += [nn.LeakyReLU()]

        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=True)]
            layers += [nn.LeakyReLU()]

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
        self.phi = Phi(input_dim, ux_hidden_dim, ux_layers, batch_norm=False).to(device)
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

    def optimize(self, X, X0, ts, px_noise, p0_noise, p0_alpha=1,
                 pxt_lr=5e-4, ux_lr=1e-3, fokker_planck_alpha=1, 
                 l_consistency_alpha=None,
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

            if l_consistency_alpha is not None:
                l_cons = self.consistency_loss(x, ts)*l_consistency_alpha
                l_cons.backward()
            else:
                l_cons = zero

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
                    )
                
        return {'l_nce_px': l_nce_pxs, 'l_nce_p0': l_nce_p0s, 'l_fp': l_fps}
    
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