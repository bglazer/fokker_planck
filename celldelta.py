import torch
from torch.nn import Linear, LeakyReLU
import torch.nn as nn
import numpy as np
from torch.autograd import grad

# ---------------------------------
# Utils
# ---------------------------------

def rademacher_like(t):
    return (torch.randint_like(t, low=0, high=2) * 2 - 1).to(dtype=t.dtype)

# ---------------------------------
# p(x,t): log-density network
# ---------------------------------

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
        xs  = x.repeat((ts.shape[0], 1, 1))                 # (T, B, D)
        ts_ = ts.repeat((x.shape[0], 1)).T.unsqueeze(2)     # (T, B, 1)
        return torch.cat((xs, ts_), dim=2)                  # (T, B, D+1)

    def forward(self, xts):
        # xts: (T, B, D+1)
        s = xts.shape
        x = xts.reshape(-1, xts.shape[-1])                  # (T*B, D+1)
        # first layer with optional scaling on inputs
        layer0 = self.layers[0]
        x = (x * self.tscale) @ layer0.weight.T + layer0.bias
        for layer in self.layers[1:]:
            x = layer(x)
        x = x.reshape(s[:-1] + (-1,))                       # (T, B, 1)
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
          dq_dt: (T, B, 1)
        """
        xts = self.xts(x, ts)
        xts.requires_grad_(True)
        q = self.forward(xts)                                # (T, B, 1)
        g = grad(q, xts, grad_outputs=torch.ones_like(q),
                 create_graph=True, retain_graph=True)[0]     # (T, B, D+1)
        return g[..., :-1], g[..., -1:]                      # (dq/dx, dq/dt)

    def score_time_and_laplacian(self, x, ts, estimator: str = "hutchinson", n_probe: int = 4):
        """
        Compute score ∇_x q, time derivative ∂_t q, and Laplacian Δ_ x q.
        Returns:
          dq_dx: (T, B, D)
          dq_dt: (T, B, 1)
          lap_q: (T, B, 1)
        """
        xts = self.xts(x, ts)
        xts.requires_grad_(True)
        q = self.forward(xts)                                # (T, B, 1)
        g_xts = grad(q, xts, grad_outputs=torch.ones_like(q),
                     create_graph=True, retain_graph=True)[0] # (T, B, D+1)
        dq_dx = g_xts[..., :-1]
        dq_dt = g_xts[..., -1:]

        if estimator == "exact":
            lap = 0.0
            for i in range(dq_dx.shape[-1]):
                gi = dq_dx[..., i]                           # (T, B)
                dgi_dxts = grad(gi, xts, grad_outputs=torch.ones_like(gi),
                                 create_graph=True, retain_graph=True)[0]
                lap_i = dgi_dxts[..., i:i+1]                 # pick derivative wrt x_i
                lap = lap + lap_i
            lap_q = lap
        else:
            # Hutchinson estimator for trace of Hessian wrt x
            lap_q = 0.0
            D = dq_dx.shape[-1]
            for _ in range(n_probe):
                z = rademacher_like(dq_dx)                   # (T, B, D)
                z_ext = torch.cat([z, torch.zeros_like(dq_dt)], dim=-1)  # (T,B,D+1)
                Hz_ext = grad(g_xts, xts, grad_outputs=z_ext,
                              create_graph=True, retain_graph=True)[0]    # (T,B,D+1)
                Hz_x = Hz_ext[..., :-1]                      # (T, B, D)
                lap_q = lap_q + (Hz_x * z).sum(dim=-1, keepdim=True)      # (T, B, 1)
            lap_q = lap_q / float(n_probe)
        return dq_dx, dq_dt, lap_q

# ---------------------------------
# Score-aligned drift: u = α(x) * ∇_x q(x,t)
# ---------------------------------

class Alpha(nn.Module):
    """
    Scalar field α(x). Drift u = α(x) ∇_x q(x,t). Time-invariant for interpretability.
    """
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim, bias=True), nn.LeakyReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=True), nn.LeakyReLU()]
        layers += [nn.Linear(hidden_dim, 1, bias=True)]  # scalar α(x)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, D)
        return self.net(x)  # (B, 1)

    def alpha_and_gradx(self, x, ts):
        """Return α(x) and ∇_x α(x), broadcast across time axis to (T,B,*)."""
        x = x.requires_grad_(True)
        a = self.forward(x)                                # (B, 1)
        gradx = grad(a.sum(), x, create_graph=True)[0]     # (B, D)
        T = ts.shape[0]
        aT = a.unsqueeze(0).expand(T, -1, -1)              # (T, B, 1)
        gradxT = gradx.unsqueeze(0).expand(T, -1, -1)      # (T, B, D)
        return aT, gradxT

# ---------------------------------
# End-to-end system with score-aligned drift
# ---------------------------------

class CellDelta(nn.Module):

    """
    CellDelta with score-aligned drift u = α(x,t) ∇_x q(x,t).
    """
    def __init__(self, input_dim,
                 ux_hidden_dim, ux_layers,
                 pxt_hidden_dim, pxt_layers,
                 device='cpu') -> None:
        super().__init__()
        self.alpha = Alpha(input_dim, ux_hidden_dim, ux_layers).to(device)
        self.pxt = Pxt(input_dim, pxt_hidden_dim, pxt_layers).to(device)
        self.device = device
        self.models = torch.nn.ModuleDict({'alpha': self.alpha, 'pxt': self.pxt})

    # -----------------------------
    # Losses
    # -----------------------------

    def nce_loss(self, x, noise, ts, scale):
        """Noise-Contrastive Estimation for the t-marginal log p(x)."""
        y = noise.sample((x.shape[0],)).to(device=x.device, dtype=x.dtype)
        log_scale = torch.log(torch.as_tensor(scale, device=x.device, dtype=x.dtype))
        logp_x = self.pxt.log_px(x, ts) + log_scale
        logq_x = noise.log_prob(x).unsqueeze(1)
        logp_y = self.pxt.log_px(y, ts) + log_scale
        logq_y = noise.log_prob(y).unsqueeze(1)

        value_x = logp_x - torch.logaddexp(logp_x, logq_x)
        value_y = logq_y - torch.logaddexp(logp_y, logq_y)
        v = value_x.mean() + value_y.mean()

        # classification accuracy
        r_x = torch.sigmoid(logp_x - logq_x)
        r_y = torch.sigmoid(logq_y - logp_y)
        acc = ((r_x > 0.5).sum() + (r_y > 0.5).sum()).float().cpu().numpy() / (len(x) + len(y))
        return -v, acc

    def fokker_planck_loss(self, x, ts, lap_estimator: str = 'hutchinson', n_probe: int = 4):
        """
        Residual r = ∂_t q + u·∇q + ∇·u,  with u = α ∇q and ∇·u = ∇α·∇q + α Δq.
        Returns mean squared residual.
        x:  (B, D)
        ts: (T,)
        """
        dq_dx, dq_dt, lap_q = self.pxt.score_time_and_laplacian(x, ts, estimator=lap_estimator, n_probe=n_probe)
        alpha, grad_alpha = self.alpha.alpha_and_gradx(x, ts)  # (T,B,1), (T,B,D)

        score_sq = (dq_dx * dq_dx).sum(dim=-1, keepdim=True)   # ||∇q||^2, (T,B,1)
        u_dot_gradq = alpha * score_sq                         # α ||∇q||^2
        div_u = (grad_alpha * dq_dx).sum(dim=-1, keepdim=True) + alpha * lap_q

        residual = dq_dt + u_dot_gradq + div_u                 # (T,B,1)
        return (residual ** 2).mean()

    def consistency_loss(self, X, ts):
        log_pxt = self.pxt.log_pxt(X, ts)
        l_cons = (log_pxt[1:].mean(1) - log_pxt[0].mean())**2
        return l_cons.mean()

    # -----------------------------
    # Optimizers / training loops
    # -----------------------------

    def optimize(self, X, X0, ts, px_noise, p0_noise, p0_alpha=1.0,
                 pxt_lr=5e-4, ux_lr=1e-3, fokker_planck_alpha=1.0,
                 l_consistency_alpha=None, p_alpha=None,
                 n_epochs=100, n_samples=1000, verbose=False,
                 lap_estimator: str = 'hutchinson', n_probe: int = 4):
        self.pxt_optimizer = torch.optim.Adam(self.pxt.parameters(), lr=pxt_lr, weight_decay=1e-3)
        self.alpha_optimizer = torch.optim.Adam(self.alpha.parameters(), lr=ux_lr, weight_decay=1e-3)

        zero = torch.zeros(1, device=self.device)
        l_nce_pxs = np.zeros(n_epochs)
        l_nce_p0s = np.zeros(n_epochs)
        l_fps     = np.zeros(n_epochs)
        n_samples = min(n_samples, len(X))

        for epoch in range(n_epochs):
            rand_idxs = torch.randperm(len(X), device=X.device)[:n_samples]
            x = X[rand_idxs].detach()
            x0 = X0.detach()

            self.pxt_optimizer.zero_grad()
            self.alpha_optimizer.zero_grad()

            # NCE for p(x)
            if p_alpha is not None:
                l_nce_px, acc_px = self.nce_loss(x, px_noise, ts=ts, scale=1/ts.shape[0])
                (l_nce_px * float(p_alpha)).backward()
            else:
                l_nce_px, acc_px = zero, zero

            # NCE for p(x, t=0)
            if p0_alpha is not None and p0_alpha != 0:
                l_nce_p0, acc_p0 = self.nce_loss(x0, p0_noise, ts=torch.zeros(1, device=x0.device), scale=1.0)
                (l_nce_p0 * float(p0_alpha)).backward()
            else:
                l_nce_p0, acc_p0 = zero, zero

            # FP residual
            if fokker_planck_alpha is not None and fokker_planck_alpha != 0:
                l_fp = self.fokker_planck_loss(x, ts, lap_estimator=lap_estimator, n_probe=n_probe) * float(fokker_planck_alpha)
                l_fp.backward()
            else:
                l_fp = zero

            # Optional consistency
            if l_consistency_alpha is not None and l_consistency_alpha != 0:
                l_cons = self.consistency_loss(x, ts) * float(l_consistency_alpha)
                l_cons.backward()
            else:
                l_cons = zero

            self.pxt_optimizer.step()
            self.alpha_optimizer.step()

            l_nce_pxs[epoch] = float(l_nce_px)
            l_nce_p0s[epoch] = float(l_nce_p0)
            l_fps[epoch]     = float(l_fp)

            if verbose:
                print(f"{epoch:6d} l_nce_px={float(l_nce_px):.5f}, acc_px={float(acc_px):.5f}, "
                      f"l_nce_p0={float(l_nce_p0): .5f}, acc_p0={float(acc_p0): .5f}, "
                      f"l_fp={float(l_fp):.5f}, l_cons={float(l_cons):.5f}")

        return {'l_nce_px': l_nce_pxs, 'l_nce_p0': l_nce_p0s, 'l_fp': l_fps}

    def optimize_fokker_planck(self, X, ts,
                               ux_lr=1e-3, fokker_planck_alpha=1.0,
                               noise=None, n_epochs=100, n_samples=1000, verbose=False,
                               lap_estimator: str = 'hutchinson', n_probe: int = 4):
        self.alpha_optimizer = torch.optim.Adam(self.alpha.parameters(), lr=ux_lr)
        l_fps = np.zeros(n_epochs)
        n_samples = min(n_samples, len(X))

        normal = None
        if noise is not None:
            normal = torch.distributions.Normal(loc=torch.zeros(n_samples, device=self.device),
                                                scale=torch.ones(n_samples, device=self.device) * noise)

        for epoch in range(n_epochs):
            rand_idxs = torch.randperm(len(X), device=X.device)[:n_samples]
            x = X[rand_idxs].clone().detach()
            if normal is not None:
                x = x + normal.sample().unsqueeze(1)

            self.alpha_optimizer.zero_grad()
            l_fp = self.fokker_planck_loss(x, ts, lap_estimator=lap_estimator, n_probe=n_probe) * float(fokker_planck_alpha)
            l_fp.backward()
            self.alpha_optimizer.step()

            l_fps[epoch] = float(l_fp)
            if verbose:
                print(f"{epoch} l_fp={float(l_fp):.5f}")
        return {'l_fp': l_fps}

    def optimize_initial_conditions(self, X0, ts, p0_noise, pxt_lr=1e-3,
                                    n_epochs=100, verbose=False, scale=1.0):
        self.pxt_optimizer = torch.optim.Adam(self.pxt.parameters(), lr=pxt_lr)
        l_nce_p0s = np.zeros(n_epochs)
        for epoch in range(n_epochs):
            ti = torch.randint(0, len(ts), (1,), device=X0.device).item()
            t = ts[ti:ti+1].detach()
            self.pxt_optimizer.zero_grad()
            l_nce_p0, acc_p0 = self.nce_loss(X0, p0_noise, ts=t, scale=scale)
            l_nce_p0.backward()
            self.pxt_optimizer.step()
            l_nce_p0s[epoch] = float(l_nce_p0)
            if verbose:
                print(f"{epoch} l_nce_p0={float(l_nce_p0):.5f}, acc_p0={float(acc_p0):.5f}")
        return {'l_nce_p0': l_nce_p0s}

    # -----------------------------
    # Simulation (Euler–Maruyama)
    # -----------------------------
    # -----------------------------
    # Velocity field
    # -----------------------------

    def u(self, x, t):
        """
        Compute velocity u(x,t) = α(x) * ∇_x q(x,t).
        Args:
            x: Tensor (D,) or (B, D)
            t: scalar (float/int or 0-dim tensor) or 1-D tensor of shape (T,)
        Returns:
            If t is scalar: (B, D) or (D,) if x was (D,)
            If t has length T: (T, B, D)
        Notes:
            - Uses pxt.dx_dt (no Hessian) for efficiency.
            - Keeps autograd graph so you can backprop through u if desired.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, D)
            squeeze_b = True
        else:
            squeeze_b = False
        # Normalize/prepare time tensor
        if isinstance(t, (float, int)):
            ts = torch.tensor([t], device=x.device, dtype=x.dtype)
        else:
            ts = t.to(device=x.device, dtype=x.dtype)
            if ts.dim() == 0:
                ts = ts.unsqueeze(0)
        # Score ∇_x q and scalar α(x)
        dq_dx, _ = self.pxt.dx_dt(x, ts)        # (T, B, D)
        alpha = self.alpha(x)                   # (B, 1)
        alphaT = alpha.unsqueeze(0).expand(ts.shape[0], -1, -1)  # (T, B, 1)
        u = alphaT * dq_dx                      # (T, B, D)
        if ts.shape[0] == 1:
            u = u[0]                            # (B, D)
            if squeeze_b:
                u = u.squeeze(0)                # (D,)
        return u



    def simulate(self, X0, tsim, sigma=1.0, zero_boundary=True):
        """Simulate with u = α(x,t) ∇_x q(x,t)."""
        x = X0.clone().detach()
        xts = torch.zeros((len(tsim), x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
        xts[0] = x
        for i in range(len(tsim) - 1):
            t_i = tsim[i:i+1]
            ht  = tsim[i+1] - tsim[i]
            dq_dx, _, _ = self.pxt.score_time_and_laplacian(x, t_i, estimator='hutchinson', n_probe=2)
            dq_dx_i = dq_dx[0]                               # (B, D)
            alpha_i, _ = self.alpha.alpha_and_gradx(x, t_i)
            alpha_i = alpha_i[0]                             # (B, 1)
            u = alpha_i * dq_dx_i                            # (B, D)
            dW = torch.randn_like(x) * torch.sqrt(ht)
            x = x + u * ht + sigma * dW
            if zero_boundary:
                x = torch.clamp(x, min=0.0)
            xts[i+1] = x
        return xts.detach().cpu()
