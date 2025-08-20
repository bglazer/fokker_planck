# weakflow.py
# Neural weak-form learning of a drift field u(x)=∇phi(x) from (p0, p) with uniform time-mixture on [0,T].
# No ODE solves; uses only local derivatives (directional derivatives) of a neural critic.

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# -----------------------
# Utilities
# -----------------------

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

def make_mlp(in_dim: int, out_dim: int, width: int, depth: int,
             spectral_norm: bool = False, last_linear_init_scale: float = 1.0) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(depth):
        lin = nn.Linear(d, width)
        if spectral_norm:
            lin = nn.utils.spectral_norm(lin)
        layers += [lin, nn.SiLU()]
        d = width
    last = nn.Linear(d, out_dim)
    # small init on last layer to stabilize early training
    nn.init.uniform_(last.weight, -last_linear_init_scale / math.sqrt(d), last_linear_init_scale / math.sqrt(d))
    nn.init.zeros_(last.bias)
    if spectral_norm:
        last = nn.utils.spectral_norm(last)
    layers += [last]
    return nn.Sequential(*layers)

def grad_scalar_output(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    y: (B, 1) scalar per sample, requires_grad=True
    x: (B, d) input with requires_grad=True
    returns: dy/dx shape (B, d)
    """
    grad = torch.autograd.grad(y.sum(), x, create_graph=True, retain_graph=True)[0]
    return grad

# -----------------------
# Models
# -----------------------

class PotentialNet(nn.Module):
    """
    phi_theta(x): scalar potential; drift is u(x)=∇phi_theta(x)
    """
    def __init__(self, d: int, width: int = 256, depth: int = 4):
        super().__init__()
        self.net = make_mlp(d, 1, width, depth, spectral_norm=False, last_linear_init_scale=1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B,1)

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        """
        u(x) = ∇_x phi_theta(x)  -> shape (B, d)
        """
        x = x.requires_grad_(True)
        phi = self.forward(x)      # (B,1)
        u = grad_scalar_output(phi, x)  # (B,d)
        return u

class CriticNet(nn.Module):
    """
    f_psi(x): scalar critic / test function
    """
    def __init__(self, d: int, width: int = 256, depth: int = 4, spectral_norm: bool = True):
        super().__init__()
        self.net = make_mlp(d, 1, width, depth, spectral_norm=spectral_norm, last_linear_init_scale=1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B,1)

# -----------------------
# Weak-operator building blocks
# -----------------------

def L_f(critic: CriticNet, potential: PotentialNet, x: torch.Tensor) -> torch.Tensor:
    """
    L_theta f(x) = u(x) · ∇ f(x), with gradients enabled w.r.t. x.
    Note: parameters of critic/potential can be frozen outside (set_requires_grad),
    but we must NOT wrap this in torch.no_grad(), or ∇_x won't be recorded.
    """
    x_req = x.requires_grad_(True)
    # u(x) = ∇_x phi(x); parameters may be frozen by the caller; keep grad wrt x enabled
    u = potential.drift(x_req)        # (B,d)
    f = critic(x_req)                 # (B,1)
    grad_f = grad_scalar_output(f, x_req)  # (B,d)
    return (u * grad_f).sum(dim=1, keepdim=True)  # (B,1)

def L2_f(critic: CriticNet, potential: PotentialNet, x: torch.Tensor) -> torch.Tensor:
    """
    L^2 f = (u · ∇)(u · ∇ f). Requires one extra backward (HVP-style).
    """
    x_req = x.requires_grad_(True)
    L1 = L_f(critic, potential, x_req)       # (B,1)
    grad_L1 = grad_scalar_output(L1, x_req)  # (B,d)
    u = potential.drift(x_req)               # (B,d)
    return (u * grad_L1).sum(dim=1, keepdim=True)  # (B,1)

def RT_apply(critic: CriticNet, potential: PotentialNet, x0: torch.Tensor, T: float, order: int = 1) -> torch.Tensor:
    """
    R_T f ≈ f + (T/2) L f + (T^2/6) L^2 f
    """
    f = critic(x0)  # (B,1)
    if order >= 1:
        f = f + 0.5 * T * L_f(critic, potential, x0)
    if order >= 2:
        f = f + (T**2 / 6.0) * L2_f(critic, potential, x0)
    return f

# -----------------------
# Penalties / regularizers
# -----------------------

def sobolev_penalty(critic: CriticNet, x: torch.Tensor, weight: float) -> torch.Tensor:
    """
    Sobolev/L2 gradient penalty: E[||∇ f||^2] on provided points.
    Encourages bounded Lipschitz behavior without expensive WGAN-GP interpolation.
    """
    if weight <= 0.0:
        return torch.zeros((), device=x.device)
    x_req = x.requires_grad_(True)
    f = critic(x_req)  # (B,1)
    gradf = grad_scalar_output(f, x_req)  # (B,d)
    return weight * (gradf.pow(2).sum(dim=1).mean())

def drift_l2_penalty(potential: PotentialNet, x: torch.Tensor, weight: float) -> torch.Tensor:
    """
    Simple smoothness surrogate: E[||u(x)||^2].
    For stronger smoothing one can estimate ||∇u||_F^2 via Hutchinson, but this is cheap & effective.
    """
    if weight <= 0.0:
        return torch.zeros((), device=x.device)
    u = potential.drift(x)  # (B,d)
    return weight * (u.pow(2).sum(dim=1).mean())

# -----------------------
# Training
# -----------------------

@dataclass
class Config:
    d: int                           # data dimension
    T: float = 1.0                   # mixing window [0,T]
    order: int = 1                   # 1 or 2 for R_T expansion order
    batch_size: int = 512
    critic_width: int = 256
    critic_depth: int = 4
    potential_width: int = 256
    potential_depth: int = 4
    lr_critic: float = 2e-4
    lr_potential: float = 2e-4
    steps: int = 20000
    n_critic: int = 1                # critic updates per potential update
    sobolev_weight: float = 5e-2     # gradient penalty weight on critic
    drift_l2_weight: float = 1e-4    # L2(u) smoothing
    spectral_norm_critic: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    log_every: int = 50

class WeakFlowTrainer:
    def __init__(self, cfg: Config, X0: np.ndarray, X: np.ndarray, seed: int = 123):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # data
        x0 = torch.from_numpy(X0).float()
        x  = torch.from_numpy(X).float()
        assert x0.shape[1] == cfg.d and x.shape[1] == cfg.d, "Input dim mismatch"

        ds0 = TensorDataset(x0)
        ds  = TensorDataset(x)
        self.loader0 = DataLoader(ds0, batch_size=cfg.batch_size, shuffle=True, drop_last=True, pin_memory=True)
        self.loader  = DataLoader(ds,  batch_size=cfg.batch_size, shuffle=True, drop_last=True, pin_memory=True)
        self.it0 = iter(self.loader0)
        self.it  = iter(self.loader)

        # models
        self.critic = CriticNet(cfg.d, cfg.critic_width, cfg.critic_depth, spectral_norm=cfg.spectral_norm_critic).to(self.device)
        self.potential = PotentialNet(cfg.d, cfg.potential_width, cfg.potential_depth).to(self.device)

        # opt
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic, betas=(0.5, 0.9))
        self.opt_p = torch.optim.Adam(self.potential.parameters(), lr=cfg.lr_potential, betas=(0.5, 0.9))

        self.scaler = torch.amp.GradScaler(enabled=cfg.mixed_precision)

    def _next_batch(self, it, loader):
        try:
            (x,) = next(it)
        except StopIteration:
            it = iter(loader)
            (x,) = next(it)
        return x.to(self.device, non_blocking=True), it

    def critic_step(self) -> Tuple[float, float]:
        cfg = self.cfg
        # freeze potential; unfreeze critic
        set_requires_grad(self.potential, False)
        set_requires_grad(self.critic, True)

        x, self.it = self._next_batch(self.it, self.loader)     # from p
        x0, self.it0 = self._next_batch(self.it0, self.loader0) # from p0

        with torch.amp.autocast(enabled=cfg.mixed_precision, device_type=self.device.type):
            # gap: E_p[f] - E_p0[R_T f]; do not backprop into potential in critic step
            f_p = self.critic(x).mean()
            rt  = RT_apply(self.critic, self.potential, x0, cfg.T, order=cfg.order).mean()
            gap = f_p - rt

            # Sobolev penalty on both supports (stabilizes critic)
            sp = 0.5 * sobolev_penalty(self.critic, x, cfg.sobolev_weight) \
               + 0.5 * sobolev_penalty(self.critic, x0, cfg.sobolev_weight)

            loss = -(gap) + sp  # ascent on critic <=> minimize -gap

        self.opt_c.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt_c)
        self.scaler.update()

        return gap.item(), sp.item()

    def drift_step(self) -> Tuple[float, float, float]:
        cfg = self.cfg
        # freeze critic; unfreeze potential
        set_requires_grad(self.critic, False)
        set_requires_grad(self.potential, True)

        x, self.it = self._next_batch(self.it, self.loader)     # from p
        x0, self.it0 = self._next_batch(self.it0, self.loader0) # from p0

        with torch.amp.autocast(enabled=cfg.mixed_precision, device_type=self.device.type):
            # gap: E_p[f] - E_p0[R_T f]; do not backprop into critic in drift step
            # drift_step()
            f_p = self.critic(x).mean().detach()
            rt  = RT_apply(self.critic, self.potential, x0, cfg.T, order=cfg.order).mean()
            gap = f_p - rt

            # smoothness penalty on u(x) evaluated on both supports
            sm = 0.5 * drift_l2_penalty(self.potential, x, cfg.drift_l2_weight) \
               + 0.5 * drift_l2_penalty(self.potential, x0, cfg.drift_l2_weight)

            loss = (gap + sm)  # minimize wrt theta

        self.opt_p.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt_p)
        self.scaler.update()

        return gap.item(), sm.item(), loss.item()

    def train(self):
        cfg = self.cfg
        for step in range(1, cfg.steps + 1):
            # n_critic steps
            for _ in range(cfg.n_critic):
                gap_c, sp = self.critic_step()
            gap_d, sm, loss_d = self.drift_step()

            if step % cfg.log_every == 0:
                print(f"[{step:06d}] gap_c={gap_c:+.4e}  gap_d={gap_d:+.4e}  sob={sp:.3e}  smooth={sm:.3e}  loss_d={loss_d:+.4e}")

        print("Training done.")

# -----------------------
# Convenience API
# -----------------------

def train(X0: np.ndarray,
          X: np.ndarray,
          T: float,
          order: int = 1,
          steps: int = 20000,
          batch_size: int = 512,
          device: Optional[str] = None) -> Tuple[PotentialNet, CriticNet]:
    """
    Train u(x)=∇phi_theta(x) and f_psi(x) from samples X0 ~ p0, X ~ p (time-mixture).
    Returns (potential_net, critic_net).
    """
    d = X0.shape[1]
    cfg = Config(
        d=d, T=T, order=order,
        steps=steps, batch_size=batch_size,
        device=(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    trainer = WeakFlowTrainer(cfg, X0, X)
    trainer.train()
    return trainer.potential.eval(), trainer.critic.eval()
import contextlib

# ---- helpers (no training grads) ----

def _drift_eval(potential: PotentialNet, x: torch.Tensor) -> torch.Tensor:
    """Evaluate u(x)=∇phi(x); returns detached tensor (B,d).
    Uses first-order autograd only (no create_graph)."""
    with torch.set_grad_enabled(True):
        x_req = x.detach().requires_grad_(True)
        phi = potential(x_req)                       # (B,1)
        u = torch.autograd.grad(phi.sum(), x_req, create_graph=False, retain_graph=False)[0]
    return u.detach()


def _hutch_divergence(potential: PotentialNet, x: torch.Tensor, n_probe: int = 1) -> torch.Tensor:
    """Estimate div u(x) with Hutchinson; returns (B,1), detached.
    We need second-order grads (through u wrt x), but this is inference-only."""
    with torch.set_grad_enabled(True):
        x_req = x.detach().requires_grad_(True)
        # need create_graph=True so we can differentiate u wrt x
        phi = potential(x_req)
        u = torch.autograd.grad(phi.sum(), x_req, create_graph=True, retain_graph=True)[0]
        out = 0.0
        for k in range(n_probe):
            v = torch.randn_like(x_req)
            v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
            retain = (k < n_probe - 1)
            vjp = torch.autograd.grad(u, x_req, v, retain_graph=retain, create_graph=False)[0]  # J^T v
            out = out + (v * vjp).sum(dim=1, keepdim=True)  # v^T J^T v; E[...] = tr(J) = div u
    return out.detach() / float(n_probe)


# ---- RK4 integrators ----

def rk4_forward_step(potential: PotentialNet, x: torch.Tensor, dt: float) -> torch.Tensor:
    """One RK4 step for x' = u(x)."""
    k1 = _drift_eval(potential, x)
    k2 = _drift_eval(potential, x + 0.5 * dt * k1)
    k3 = _drift_eval(potential, x + 0.5 * dt * k2)
    k4 = _drift_eval(potential, x + dt * k3)
    return (x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).detach()


def rk4_reverse_step(potential: PotentialNet, x: torch.Tensor, ds: float) -> torch.Tensor:
    """One RK4 step for y' = -u(y) (reverse time)."""
    k1 = -_drift_eval(potential, x)
    k2 = -_drift_eval(potential, x + 0.5 * ds * k1)
    k3 = -_drift_eval(potential, x + 0.5 * ds * k2)
    k4 = -_drift_eval(potential, x + ds * k3)
    return (x + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).detach()


# ---- Public APIs ----

def pushforward(potential: PotentialNet, X0: torch.Tensor, t: float, steps: int = 80) -> torch.Tensor:
    """Push a batch X0 ~ p0 forward under x' = u(x) for time t using RK4.
    No training grads are recorded. Returns positions at time t on same device."""
    dt = float(t) / max(1, int(steps))
    x = X0.detach()
    for _ in range(max(1, int(steps))):
        x = rk4_forward_step(potential, x, dt)
    return x


def pushforward_snapshots(potential: PotentialNet, X0: torch.Tensor, T: float, K: int = 32, steps_per_unit: int = 40):
    """Return K+1 snapshots Y_k at times t_k=k*T/K using RK4 forward integration."""
    steps_total = max(1, int(steps_per_unit * float(T)))
    dt = float(T) / steps_total
    between = max(1, steps_total // K)
    x = X0.detach()
    snaps = [x.detach().clone()]
    cut = between
    for s in range(1, steps_total + 1):
        x = rk4_forward_step(potential, x, dt)
        if s == cut:
            snaps.append(x.detach().clone())
            cut += between
            if len(snaps) == K + 1:
                break
    while len(snaps) < K + 1:
        snaps.append(x.detach().clone())
    return snaps  # list of tensors length K+1


def logpt_reverse(potential: PotentialNet,
                  logp0_fn,
                  X: torch.Tensor,
                  t: float,
                  steps: int = 80,
                  n_probe: int = 2) -> torch.Tensor:
    """Compute log p_t(X) by reverse-time solve and divergence accumulation.

    Args:
        potential: trained PotentialNet (drift u=∇phi).
        logp0_fn: callable taking a tensor (B,d) and returning log p0(x) with shape (B,1) or (B,).
        X: (B,d) tensor of query points at time t.
        t: scalar time ≥ 0.
        steps: RK4 steps along [0,t].
        n_probe: Hutchinson probe count per step (1–4 is typical).

    Returns:
        log p_t(X) as a tensor (B,1).
    """
    if t <= 0:
        lp = logp0_fn(X)
        return lp if lp.ndim == 2 else lp.view(-1, 1)

    ds = float(t) / max(1, int(steps))
    x = X.detach()
    div_int = torch.zeros(X.size(0), 1, device=X.device)

    for _ in range(max(1, int(steps))):
        # RK4–consistent divergence quadrature at stage points
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

        # advance state and integral
        x = (x + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).detach()
        div_int = div_int + (ds / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)

    # evaluate initial log-density at the recovered preimage
    lp0 = logp0_fn(x)
    if lp0.ndim == 1:
        lp0 = lp0.view(-1, 1)
    return (lp0 - div_int).detach()


# ---- Optional: round-trip certification ----

def roundtrip_error(potential: PotentialNet, X: torch.Tensor, t: float, steps: int = 80) -> torch.Tensor:
    """Compute ||Φ_t(Φ_{-t}(X)) - X||_2 per sample with same RK4 discretization."""
    x0_hat = X.detach()
    for _ in range(max(1, int(steps))):
        x0_hat = rk4_reverse_step(potential, x0_hat, float(t) / max(1, int(steps)))
    x_fwd = x0_hat.detach()
    for _ in range(max(1, int(steps))):
        x_fwd = rk4_forward_step(potential, x_fwd, float(t) / max(1, int(steps)))
    return ((x_fwd - X.detach())**2).sum(dim=1).sqrt()