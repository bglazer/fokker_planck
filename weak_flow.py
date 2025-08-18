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

# -----------------------
# Example (commented)
# -----------------------
# if __name__ == "__main__":
#     # Dummy synthetic 2D data to sanity-check the loop
#     np.random.seed(0)
#     N0, N = 20000, 20000
#     d = 2
#     # p0: Gaussian at (-2,0)
#     X0 = np.random.randn(N0, d).astype(np.float32) + np.array([-2.0, 0.0], np.float32)
#     # p: 50/50 mixture of two Gaussians (time-mixture stand-in)
#     X1 = np.random.randn(N//2, d).astype(np.float32) + np.array([+2.0, 0.0], np.float32)
#     X2 = np.random.randn(N//2, d).astype(np.float32) + np.array([-2.0, 2.5], np.float32)
#     X  = np.concatenate([X1, X2], axis=0)
#     # Train (small T, order=1)
#     potential, critic = train(X0, X, T=0.5, order=1, steps=5000, batch_size=1024)
#     # Use potential.drift(torch.tensor([...])) to evaluate u(x)
