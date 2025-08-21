# weakflow.py
# Neural weak-form learning of a drift field u(x)=∇phi(x) from (p0, p) with uniform time-mixture on [0,T].
# Optimized + consolidated: avoids redundant forward/grad evals, caches metrics during steps.

from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import numpy as np
import torch
import torch.nn as nn
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
# Weak-operator building blocks (optimized)
# =====================


def _precompute_local(critic: CriticNet, potential: PotentialNet, x: torch.Tensor):
    """Compute f, ∇f, and u once from x (keeps graph for higher-order ops)."""
    x_req = x.requires_grad_(True)
    f = critic(x_req)  # (B,1)
    grad_f = grad_scalar_output(f, x_req)  # (B,d)
    u = potential.drift(x_req)  # (B,d)
    return x_req, f, grad_f, u


def _laplacian_hutch_from_grad(
    grad_f: torch.Tensor, x: torch.Tensor, m: int = 2
) -> torch.Tensor:
    """Hutchinson estimator of Δf using precomputed ∇f. Returns (B,1)."""
    lap = 0.0
    for k in range(m):
        v = torch.randn_like(x)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        Hv = torch.autograd.grad(
            (grad_f * v).sum(), x, retain_graph=(k < m - 1), create_graph=True
        )[0]
        lap = lap + (Hv * v).sum(dim=1, keepdim=True)
    return lap / float(m)


def _L1_from_parts(grad_f: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return (u * grad_f).sum(dim=1, keepdim=True)


def RT_apply(
    critic: CriticNet,
    potential: PotentialNet,
    x0: torch.Tensor,
    T: float,
    order: int = 1,
) -> torch.Tensor:
    """R_T f ≈ f + (T/2)Lf + (T^2/6)L^2 f (no diffusion). Optimized: single critic forward."""
    x_req, f0, grad_f, u = _precompute_local(critic, potential, x0)
    out = f0
    if order >= 1:
        L1 = _L1_from_parts(grad_f, u)
        out = out + 0.5 * T * L1
    if order >= 2:
        grad_L1 = grad_scalar_output(L1, x_req)
        L2 = _L1_from_parts(grad_L1, u)
        out = out + (T**2 / 6.0) * L2
    return out


def RT_apply_with_diffusion(
    critic: CriticNet,
    potential: PotentialNet,
    x0: torch.Tensor,
    T: float,
    order: int = 1,
    D_scalar: float = 0.0,
    lap_probes: int = 2,
) -> torch.Tensor:
    """R_T with optional isotropic diffusion term D Δf in 𝓛.
    For stability we keep order-1 when D>0."""
    x_req, f0, grad_f, u = _precompute_local(critic, potential, x0)
    L1 = _L1_from_parts(grad_f, u)
    if D_scalar is not None and D_scalar > 0.0:
        lap = _laplacian_hutch_from_grad(grad_f, x_req, m=max(1, int(lap_probes)))
        L1 = L1 + D_scalar * lap
        # order forced to 1 when diffusion present (as in original code)
        return f0 + 0.5 * T * L1
    # D==0: honor requested order
    out = f0 + 0.5 * T * L1
    if order >= 2:
        grad_L1 = grad_scalar_output(L1, x_req)
        L2 = _L1_from_parts(grad_L1, u)
        out = out + (T**2 / 6.0) * L2
    return out


# =====================
# Penalties (return metrics for free)
# =====================


def sobolev_penalty_and_g2(
    critic: CriticNet, x: torch.Tensor, weight: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    if weight <= 0.0:
        z = torch.zeros((), device=x.device)
        return z, z
    x_req = x.requires_grad_(True)
    f = critic(x_req)
    gradf = grad_scalar_output(f, x_req)
    g2 = gradf.pow(2).sum(dim=1).mean()
    return weight * g2, g2.detach()


def drift_l2_penalty_and_u2(
    potential: PotentialNet, x: torch.Tensor, weight: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    if weight <= 0.0:
        z = torch.zeros((), device=x.device)
        return z, z
    u = potential.drift(x)
    u2 = u.pow(2).sum(dim=1).mean()
    return weight * u2, u2.detach()


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
            pin_memory=(self.device.type == "cuda"),
        )
        self.loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(self.device.type == "cuda"),
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
        # metrics ema
        self.u2_ma: Optional[float] = None
        self.g2_ma: Optional[float] = None
        self.gap_d_ma: Optional[float] = None
        self.ema_decay = 0.9

    def _ema_update(self, old: Optional[float], val: float) -> float:
        return val if old is None else self.ema_decay * old + (1 - self.ema_decay) * val

    def _next_batch(self, it, loader):
        try:
            (x,) = next(it)
        except StopIteration:
            it = iter(loader)
            (x,) = next(it)
        return x.to(self.device, non_blocking=True), it

    def critic_step(self) -> Tuple[float, float, float]:
        """Returns: gap_c, sobolev_penalty, G2 (mean ||∇f||^2)."""
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
            sp_x, g2_x = sobolev_penalty_and_g2(self.critic, x, cfg.sobolev_weight)
            sp_x0, g2_x0 = sobolev_penalty_and_g2(self.critic, x0, cfg.sobolev_weight)
            sp = 0.5 * sp_x + 0.5 * sp_x0
            g2 = 0.5 * g2_x + 0.5 * g2_x0
            loss = -(gap) + sp
        self.opt_c.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt_c)
        self.scaler.update()
        # update ema metric
        self.g2_ma = float(self._ema_update(self.g2_ma, g2.item()))
        return gap.item(), sp.item(), self.g2_ma

    def drift_step(self) -> Tuple[float, float, float, float]:
        """Returns: gap_d, smooth_penalty, loss_d, U2 (ema)."""
        cfg = self.cfg
        set_requires_grad(self.critic, False)
        set_requires_grad(self.potential, True)
        x, self.it = self._next_batch(self.it, self.loader)
        x0, self.it0 = self._next_batch(self.it0, self.loader0)
        with torch.amp.autocast(
            enabled=cfg.mixed_precision, device_type=self.device.type
        ):
            f_p = self.critic(x).mean().detach()  # do not backprop through critic
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
            sm_x, u2_x = drift_l2_penalty_and_u2(self.potential, x, cfg.drift_l2_weight)
            sm_x0, u2_x0 = drift_l2_penalty_and_u2(
                self.potential, x0, cfg.drift_l2_weight
            )
            sm = 0.5 * sm_x + 0.5 * sm_x0
            u2 = 0.5 * u2_x + 0.5 * u2_x0
            loss = gap + sm
        self.opt_p.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt_p)
        self.scaler.update()
        # ema updates
        self.u2_ma = float(self._ema_update(self.u2_ma, u2.item()))
        self.gap_d_ma = float(self._ema_update(self.gap_d_ma, gap.item()))
        return gap.item(), sm.item(), loss.item(), self.u2_ma

    def train(self) -> None:
        cfg = self.cfg
        for step in range(1, cfg.steps + 1):
            for _ in range(cfg.n_critic):
                gap_c, sp, g2 = self.critic_step()
            gap_d, sm, loss_d, u2 = self.drift_step()
            if step % cfg.log_every == 0:
                print(
                    f"[{step:06d}] gap_c={gap_c:+.4e} gap_d={gap_d:+.4e} sob={sp:.3e} smooth={sm:.3e} "
                    f"u2_ma={self.u2_ma:.3e} g2_ma={self.g2_ma:.3e} loss_d={loss_d:+.4e}"
                )
        print("Training done.")
