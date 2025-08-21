#%%
import torch
import numpy as np
from weak_flow import (
    PotentialNet,
    WeakFlowTrainer,
)
from typing import Optional

#%%
# Standardization
class ZScoreStandardizer:
    def __init__(self, eps: float = 1e-8):
        self.mu = None
        self.sigma = None
        self.eps = eps

    def fit(self, X0: np.ndarray, X: np.ndarray, use_union: bool = True):
        if use_union:
            U = np.vstack([X0, X])
        else:
            U = X0
        self.mu = U.mean(0)
        self.sigma = U.std(0, ddof=0)
        self.sigma = np.where(self.sigma < self.eps, 1.0, self.sigma)
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        assert self.mu is not None, "Call fit(...) first"
        return (A - self.mu) / self.sigma

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        assert self.mu is not None, "Call fit(...) first"
        return Z * self.sigma + self.mu

    def to_torch(self, device="cpu"):
        mu_t = torch.from_numpy(self.mu.astype(np.float32)).to(device)
        sigma_t = torch.from_numpy(self.sigma.astype(np.float32)).to(device)
        return mu_t, sigma_t

# Auto tuner (unchanged API, but uses trainer's EMA metrics)
class AutoTuner:
    def __init__(self,
                 u2_band=(5.0, 25.0),
                 g2_band=(1.0, 10.0),
                 w_u_bounds=(1e-5, 1e-1),
                 w_s_bounds=(1e-3, 10.0),
                 gap_target=2.0):
        self.u2_lo, self.u2_hi = u2_band
        self.g2_lo, self.g2_hi = g2_band
        self.wu_lo, self.wu_hi = w_u_bounds
        self.ws_lo, self.ws_hi = w_s_bounds
        self.gap_target = gap_target
    def step(self, trainer: WeakFlowTrainer, gap_d_ma: float, U2_ma: float, G2_ma: float):
        cfg = trainer.cfg; changed = {}
        if U2_ma > self.u2_hi:
            cfg.drift_l2_weight = min(cfg.drift_l2_weight * 2.0, self.wu_hi)
            changed['drift_l2_weight'] = cfg.drift_l2_weight
        elif U2_ma < self.u2_lo and gap_d_ma > self.gap_target:
            cfg.drift_l2_weight = max(cfg.drift_l2_weight / 1.5, self.wu_lo)
            changed['drift_l2_weight'] = cfg.drift_l2_weight
        if G2_ma > self.g2_hi:
            cfg.sobolev_weight = min(cfg.sobolev_weight * 1.5, self.ws_hi)
            changed['sobolev_weight'] = cfg.sobolev_weight
            # cfg.diffusion_D = max(getattr(cfg, 'diffusion_D', 0.0), 1e-3)
            # changed['diffusion_D'] = cfg.diffusion_D
        elif G2_ma < self.g2_lo and gap_d_ma > self.gap_target:
            cfg.sobolev_weight = max(cfg.sobolev_weight / 1.5, self.ws_lo)
            changed['sobolev_weight'] = cfg.sobolev_weight
        if gap_d_ma < self.gap_target and (self.u2_lo <= U2_ma <= self.u2_hi) and (self.g2_lo <= G2_ma <= self.g2_hi):
            cfg.T = min(cfg.T * 1.25, 4.0); changed['T'] = cfg.T
        return changed
    
# PCA helpers

def pca_fit(X: np.ndarray, n: int = 2):
    Xc = X - X.mean(0, keepdims=True)
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    comps = VT[:n, :]
    mean = X.mean(0)
    return mean, comps


def pca_project(X: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    return (X - mean) @ comps.T


def pca_unproject(Z: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    return Z @ comps + mean


def drift_batch(
    potential: PotentialNet,
    X: np.ndarray,
    device: Optional[torch.device] = None,
    bs: int = 4096,
) -> np.ndarray:
    device = device or next(iter(potential.parameters())).device
    was_training = potential.training
    potential.eval()
    out = []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i : i + bs]).float().to(device)
        xb.requires_grad_(True)
        phi = potential(xb)
        u = torch.autograd.grad(phi.sum(), xb)[0]
        out.append(u.detach().cpu().numpy())
    if was_training:
        potential.train()
    return np.concatenate(out, axis=0)


# ODE-based utilities (forward push + reverse-time log p_t)

def _drift_eval(potential: PotentialNet, x: torch.Tensor) -> torch.Tensor:
    with torch.set_grad_enabled(True):
        x_req = x.detach().requires_grad_(True)
        phi = potential(x_req)
        u = torch.autograd.grad(
            phi.sum(), x_req, create_graph=False, retain_graph=False
        )[0]
    return u.detach()


def _hutch_divergence(
    potential: PotentialNet, x: torch.Tensor, n_probe: int = 1
) -> torch.Tensor:
    with torch.set_grad_enabled(True):
        x_req = x.detach().requires_grad_(True)
        phi = potential(x_req)
        u = torch.autograd.grad(phi.sum(), x_req, create_graph=True, retain_graph=True)[
            0
        ]
        out = 0.0
        for k in range(n_probe):
            v = torch.randn_like(x_req)
            v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
            retain = k < n_probe - 1
            vjp = torch.autograd.grad(
                u, x_req, v, retain_graph=retain, create_graph=False
            )[0]
            out = out + (v * vjp).sum(dim=1, keepdim=True)
    return out.detach() / float(n_probe)


def rk4_forward_step(
    potential: PotentialNet, x: torch.Tensor, dt: float
) -> torch.Tensor:
    k1 = _drift_eval(potential, x)
    k2 = _drift_eval(potential, x + 0.5 * dt * k1)
    k3 = _drift_eval(potential, x + 0.5 * dt * k2)
    k4 = _drift_eval(potential, x + dt * k3)
    return (x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).detach()


def rk4_reverse_step(
    potential: PotentialNet, x: torch.Tensor, ds: float
) -> torch.Tensor:
    k1 = -_drift_eval(potential, x)
    k2 = -_drift_eval(potential, x + 0.5 * ds * k1)
    k3 = -_drift_eval(potential, x + 0.5 * ds * k2)
    k4 = -_drift_eval(potential, x + ds * k3)
    return (x + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).detach()


def pushforward(
    potential: PotentialNet, X0: torch.Tensor, t: float, steps: int = 80
) -> torch.Tensor:
    dt = float(t) / max(1, int(steps))
    x = X0.detach()
    for _ in range(max(1, int(steps))):
        x = rk4_forward_step(potential, x, dt)
    return x


def pushforward_snapshots(
    potential: PotentialNet, X0: torch.Tensor, T: float, K: int = 32
):
    """Uniform-grid forward snapshots at times {0, dt, 2dt, ..., T} with dt=T/K."""
    K = int(K)
    dt = float(T) / max(1, K)
    x = X0.detach()
    snaps = [x.detach().clone()]
    for _ in range(1, K + 1):
        x = rk4_forward_step(potential, x, dt)
        snaps.append(x.detach().clone())
    return snaps


def logpt_reverse(
    potential: PotentialNet,
    logp0_fn,
    X: torch.Tensor,
    t: float,
    steps: int = 80,
    n_probe: int = 2,
) -> torch.Tensor:
    if t <= 0:
        lp = logp0_fn(X)
        return lp if lp.ndim == 2 else lp.view(-1, 1)
    ds = float(t) / max(1, int(steps))
    x = X.detach()
    div_int = torch.zeros(X.size(0), 1, device=X.device)
    for _ in range(max(1, int(steps))):
        y1 = x
        d1 = _hutch_divergence(potential, y1, n_probe)
        k1 = -_drift_eval(potential, y1)
        y2 = x + 0.5 * ds * k1
        d2 = _hutch_divergence(potential, y2, n_probe)
        k2 = -_drift_eval(potential, y2)
        y3 = x + 0.5 * ds * k2
        d3 = _hutch_divergence(potential, y3, n_probe)
        k3 = -_drift_eval(potential, y3)
        y4 = x + ds * k3
        d4 = _hutch_divergence(potential, y4, n_probe)
        k4 = -_drift_eval(potential, y4)
        x = (x + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).detach()
        div_int = div_int + (ds / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)
    lp0 = logp0_fn(x)
    if lp0.ndim == 1:
        lp0 = lp0.view(-1, 1)
    return (lp0 - div_int).detach()

# ---- generic integrator + generator for arbitrary drift fields ----

def rk4_step_u(u_fn, x: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = u_fn(x)
    k2 = u_fn(x + 0.5 * dt * k1)
    k3 = u_fn(x + 0.5 * dt * k2)
    k4 = u_fn(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_field(u_fn, x0: torch.Tensor, K: int, dt: float) -> torch.Tensor:
    x = x0.detach()
    for _ in range(int(K)):
        x = rk4_step_u(u_fn, x, dt)
    return x


def generate_mixture_from_field(
    u_fn, d: int, N0: int, N: int, T: float, K: int, sigma: float = 1.0, seed: int = 0
):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(0.0, sigma, size=(N0, d)).astype(np.float32)
    Xstart = rng.normal(0.0, sigma, size=(N, d)).astype(np.float32)
    k_idx = rng.integers(low=0, high=int(K) + 1, size=N, dtype=np.int64)
    dt = float(T) / float(K)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        Xstart_t = torch.from_numpy(Xstart).float().to(dev)
        X_t = integrate_field(u_fn, Xstart_t, K=int(K), dt=dt)
        X = X_t.cpu().numpy()
    ts = k_idx.astype(np.float32) * dt
    return X0, X, ts
