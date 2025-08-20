# weakflow.py
# Neural weak-form learning of a drift field u(x)=∇phi(x) from (p0, p) with uniform time-mixture on [0,T].
# No ODE solves; uses only local derivatives (directional derivatives) of a neural critic.

from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# =====================
# Utilities
# =====================


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def make_mlp(
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    spectral_norm: bool = False,
    last_linear_init_scale: float = 1e-2,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        lin = nn.Linear(d, width)
        if spectral_norm:
            lin = nn.utils.spectral_norm(lin)
        layers += [lin, nn.SiLU()]
        d = width
    last = nn.Linear(d, out_dim)
    # small init on last layer to stabilize early training
    nn.init.uniform_(
        last.weight,
        -last_linear_init_scale / math.sqrt(max(d, 1)),
        last_linear_init_scale / math.sqrt(max(d, 1)),
    )
    nn.init.zeros_(last.bias)
    if spectral_norm:
        last = nn.utils.spectral_norm(last)
    layers += [last]
    return nn.Sequential(*layers)


def grad_scalar_output(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y: (B,1) scalar output; x: (B,d). Return ∂y/∂x with graph for higher-order ops."""
    return torch.autograd.grad(y.sum(), x, create_graph=True, retain_graph=True)[0]


# =====================
# Models
# =====================
class PotentialNet(nn.Module):
    """φ_θ(x): scalar potential; drift is u(x)=∇φ_θ(x)."""

    def __init__(self, d: int, width: int = 256, depth: int = 4) -> None:
        super().__init__()
        self.net = make_mlp(
            d, 1, width, depth, spectral_norm=False, last_linear_init_scale=1e-2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        phi = self.forward(x)
        return grad_scalar_output(phi, x)


class CriticNet(nn.Module):
    """f_ψ(x): scalar critic / test function."""

    def __init__(
        self, d: int, width: int = 256, depth: int = 4, spectral_norm: bool = True
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            d, 1, width, depth, spectral_norm=spectral_norm, last_linear_init_scale=1e-2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================
# Weak-operator building blocks
# =====================


def laplacian_hutch(critic: CriticNet, x: torch.Tensor, m: int = 2) -> torch.Tensor:
    """Hutchinson estimator of Δf(x)=tr(∇²f). Uses Hessian–vector products only.
    Returns (B,1)."""
    x_req = x.requires_grad_(True)
    f = critic(x_req)
    g = grad_scalar_output(f, x_req)  # ∇f
    lap = 0.0
    for k in range(m):
        v = torch.randn_like(x_req)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        Hv = torch.autograd.grad((g * v).sum(), x_req, retain_graph=(k < m - 1))[0]
        lap = lap + (Hv * v).sum(dim=1, keepdim=True)  # vᵀH v
    return lap / float(m)


def L_f(critic: CriticNet, potential: PotentialNet, x: torch.Tensor) -> torch.Tensor:
    """(L f)(x) = u·∇f with respect to x; parameters may be frozen by caller."""
    x_req = x.requires_grad_(True)
    u = potential.drift(x_req)
    f = critic(x_req)
    grad_f = grad_scalar_output(f, x_req)
    return (u * grad_f).sum(dim=1, keepdim=True)


def L_f_with_diffusion(
    critic: CriticNet,
    potential: PotentialNet,
    x: torch.Tensor,
    D_scalar: float,
    lap_probes: int = 2,
) -> torch.Tensor:
    """(L f)(x) with optional isotropic diffusion: u·∇f + D Δf."""
    drift_term = L_f(critic, potential, x)
    if D_scalar is None or D_scalar <= 0.0:
        return drift_term
    diff_term = D_scalar * laplacian_hutch(critic, x, m=max(1, int(lap_probes)))
    return drift_term + diff_term


def L2_f(critic: CriticNet, potential: PotentialNet, x: torch.Tensor) -> torch.Tensor:
    """(L^2 f)(x) = (u·∇)(u·∇f)."""
    x_req = x.requires_grad_(True)
    L1 = L_f(critic, potential, x_req)
    grad_L1 = grad_scalar_output(L1, x_req)
    u = potential.drift(x_req)
    return (u * grad_L1).sum(dim=1, keepdim=True)


def RT_apply(
    critic: CriticNet,
    potential: PotentialNet,
    x0: torch.Tensor,
    T: float,
    order: int = 1,
) -> torch.Tensor:
    """R_T f ≈ f + (T/2)Lf + (T^2/6)L^2 f."""
    f = critic(x0)
    if order >= 1:
        f = f + 0.5 * T * L_f(critic, potential, x0)
    if order >= 2:
        f = f + (T**2 / 6.0) * L2_f(critic, potential, x0)
    return f


def RT_apply_with_diffusion(
    critic: CriticNet,
    potential: PotentialNet,
    x0: torch.Tensor,
    T: float,
    order: int = 1,
    D_scalar: float = 0.0,
    lap_probes: int = 2,
) -> torch.Tensor:
    """Same as RT_apply but includes isotropic diffusion term D Δf in 𝓛.
    For stability we apply order-1 when D>0; when D==0 we honor the requested order."""
    f = critic(x0)
    if order >= 1:
        L1 = L_f_with_diffusion(
            critic, potential, x0, D_scalar=D_scalar, lap_probes=lap_probes
        )
        f = f + 0.5 * T * L1
    if (order >= 2) and (D_scalar <= 0.0):
        f = f + (T**2 / 6.0) * L2_f(critic, potential, x0)
    return f


# =====================
# Penalties
# =====================


def sobolev_penalty(critic: CriticNet, x: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0.0:
        return torch.zeros((), device=x.device)
    x_req = x.requires_grad_(True)
    f = critic(x_req)
    gradf = grad_scalar_output(f, x_req)
    return weight * (gradf.pow(2).sum(dim=1).mean())


def drift_l2_penalty(
    potential: PotentialNet, x: torch.Tensor, weight: float
) -> torch.Tensor:
    if weight <= 0.0:
        return torch.zeros((), device=x.device)
    u = potential.drift(x)
    return weight * (u.pow(2).sum(dim=1).mean())


# =====================
# Training
# =====================
@dataclass
class Config:
    d: int
    T: float = 1.0
    order: int = 1
    batch_size: int = 512
    critic_width: int = 256
    critic_depth: int = 4
    potential_width: int = 256
    potential_depth: int = 4
    lr_critic: float = 2e-4
    lr_potential: float = 2e-4
    steps: int = 20000
    n_critic: int = 1
    sobolev_weight: float = 5e-2
    drift_l2_weight: float = 1e-4
    spectral_norm_critic: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    log_every: int = 50
    # new: optional diffusion
    diffusion_D: float = 0.0
    lap_probes: int = 2


class WeakFlowTrainer:
    def __init__(
        self, cfg: Config, X0: np.ndarray, X: np.ndarray, seed: int = 123
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # data
        x0 = torch.from_numpy(X0).float()
        x = torch.from_numpy(X).float()
        assert x0.shape[1] == cfg.d and x.shape[1] == cfg.d, "Input dim mismatch"
        ds0 = TensorDataset(x0)
        ds = TensorDataset(x)
        self.loader0 = DataLoader(
            ds0,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        self.loader = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, pin_memory=True
        )
        self.it0 = iter(self.loader0)
        self.it = iter(self.loader)
        # models
        self.critic = CriticNet(
            cfg.d,
            cfg.critic_width,
            cfg.critic_depth,
            spectral_norm=cfg.spectral_norm_critic,
        ).to(self.device)
        self.potential = PotentialNet(
            cfg.d, cfg.potential_width, cfg.potential_depth
        ).to(self.device)
        # opt
        self.opt_c = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.lr_critic, betas=(0.5, 0.9)
        )
        self.opt_p = torch.optim.Adam(
            self.potential.parameters(), lr=cfg.lr_potential, betas=(0.5, 0.9)
        )
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
        set_requires_grad(self.potential, False)
        set_requires_grad(self.critic, True)
        x, self.it = self._next_batch(self.it, self.loader)  # from p
        x0, self.it0 = self._next_batch(self.it0, self.loader0)  # from p0
        with torch.amp.autocast(
            enabled=cfg.mixed_precision, device_type=self.device.type
        ):
            f_p = self.critic(x).mean()
            rt = RT_apply_with_diffusion(
                self.critic,
                self.potential,
                x0,
                cfg.T,
                order=cfg.order,
                D_scalar=cfg.diffusion_D,
                lap_probes=cfg.lap_probes,
            ).mean()
            gap = f_p - rt
            sp = 0.5 * sobolev_penalty(
                self.critic, x, cfg.sobolev_weight
            ) + 0.5 * sobolev_penalty(self.critic, x0, cfg.sobolev_weight)
            loss = -(gap) + sp
        self.opt_c.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt_c)
        self.scaler.update()
        return gap.item(), sp.item()

    def drift_step(self) -> Tuple[float, float, float]:
        cfg = self.cfg
        set_requires_grad(self.critic, False)
        set_requires_grad(self.potential, True)
        x, self.it = self._next_batch(self.it, self.loader)
        x0, self.it0 = self._next_batch(self.it0, self.loader0)
        with torch.amp.autocast(
            enabled=cfg.mixed_precision, device_type=self.device.type
        ):
            f_p = self.critic(x).mean().detach()
            rt = RT_apply_with_diffusion(
                self.critic,
                self.potential,
                x0,
                cfg.T,
                order=cfg.order,
                D_scalar=cfg.diffusion_D,
                lap_probes=cfg.lap_probes,
            ).mean()
            gap = f_p - rt
            sm = 0.5 * drift_l2_penalty(
                self.potential, x, cfg.drift_l2_weight
            ) + 0.5 * drift_l2_penalty(self.potential, x0, cfg.drift_l2_weight)
            loss = gap + sm
        self.opt_p.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt_p)
        self.scaler.update()
        return gap.item(), sm.item(), loss.item()

    def train(self) -> None:
        cfg = self.cfg
        for step in range(1, cfg.steps + 1):
            for _ in range(cfg.n_critic):
                gap_c, sp = self.critic_step()
            gap_d, sm, loss_d = self.drift_step()
            if step % cfg.log_every == 0:
                print(
                    f"[{step:06d}] gap_c={gap_c:+.4e} gap_d={gap_d:+.4e} sob={sp:.3e} smooth={sm:.3e} loss_d={loss_d:+.4e}"
                )
        print("Training done.")
