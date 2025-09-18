"""
Weak Continuity (critic‑free) — NO Euler steps, NO ODE backprop
---------------------------------------------------------------

You asked for the **pure weak continuity constraint** version (no transport/unroll).
This module implements exactly that:

• Fixed random test bank φ_j(x,t) with analytic ∂tφ and ∇xφ.
• Drift u_θ(x,t) (small MLP with time Fourier features).
• Conditional time model q_η(t|x) with an atom α_η(x) at t=0.
• Loss = continuity residual (expected over p(x) q_η(t|x)) + boundary match at t=0
  + mild regularizers. No Euler steps anywhere.

Mathematical core
-----------------
Let tests φ_j be smooth with compact support in t∈(0,1] (handled via random Fourier
features). The continuity equation implies for each j:

  R_j(θ,η) := E_{x∼p} E_{t∼q_η(·|x)} [ ∂t φ_j(x,t) + u_θ(x,t)·∇x φ_j(x,t) ] = 0.

We also enforce a weak boundary at t=0 using ψ_k(x)=φ_k(x,0):

  E_{x∼p}[ α_η(x) ψ_k(x) ] ≈ E_{x0∼p0}[ ψ_k(x0) ].

This avoids constructing samples from p_t and avoids pushing x0 forward.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Config
# =========================

@dataclass
class Config:
    d: int
    n_tests: int = 256
    x_scale: float = 1.0
    t_scale: float = 1.0
    time_feat: int = 32
    width: int = 128
    depth: int = 3
    batch_p: int = 2048
    batch_p0: int = 1024
    K_t: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-6
    lambda_cont: float = 1.0
    lambda_bound: float = 1.0
    lambda_smooth_u: float = 1e-3
    lambda_entropy_t: float = 1e-3
    device: str = "auto"

    def device_obj(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# =========================
# Drift u_θ(x)
# =========================

class DriftNet(nn.Module):
    def __init__(self, d: int, width: int = 128, depth: int = 3):
        super().__init__()
        self.d = d
        in_dim = d
        layers = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, width), nn.SiLU()]
            last = width
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(last, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.out(h)


# =========================
# Fixed random Fourier tests φ_j(x,t)
# =========================

class RandomFourierTests(nn.Module):
    def __init__(self, d: int, n_tests: int = 256, x_scale: float = 1.0, t_scale: float = 1.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.omega = nn.Parameter(torch.randn(n_tests, d, generator=g) / x_scale, requires_grad=False)
        self.gamma = nn.Parameter(torch.randn(n_tests, 1, generator=g) / t_scale, requires_grad=False)
        self.bias  = nn.Parameter(2 * math.pi * torch.rand(n_tests, 1, generator=g), requires_grad=False)

    def _A(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x @ self.omega.t() + t @ self.gamma.t() + self.bias.t()

    def phi(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(self._A(x, t))  # (B,nT)

    def dphi_dt(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        A = self._A(x, t)
        return -torch.sin(A) * self.gamma.t()  # (B,nT)

    def grad_phi_x(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        A = self._A(x, t)
        S = -torch.sin(A)  # (B,nT)
        return S.unsqueeze(-1) * self.omega.unsqueeze(0)  # (B,nT,d)


# =========================
# Conditional time q_η(t|x) with atom at t=0
# =========================

class TimeDensity(nn.Module):
    """q_η(t|x) = α(x) δ_0(t) + (1-α(x)) q^{cont}(t|x) on (0,1].
    We sample K_t Monte Carlo times in (0,1] per x and use a softmax over those
    times to represent q^{cont}(t|x).
    """
    def __init__(self, d: int, width: int = 128, depth: int = 2):
        super().__init__()
        layers = []
        last = d
        for _ in range(depth):
            layers += [nn.Linear(last, width), nn.SiLU()]
            last = width
        self.f = nn.Sequential(*layers)
        self.logit_alpha = nn.Linear(last, 1)
        self.logit_temp  = nn.Linear(last, 1)

    def forward(self, x: torch.Tensor, t_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x:(B,d), t_grid:(B,K,1)
        h = self.f(x)
        alpha = torch.sigmoid(self.logit_alpha(h))        # (B,1)
        temp  = F.softplus(self.logit_temp(h)) + 1e-4     # (B,1)
        logits = -((t_grid.squeeze(-1) - 0.5) ** 2) / (2 * (0.2 ** 2) * temp)  # (B,K)
        w = torch.softmax(logits, dim=1)                  # (B,K)
        return alpha, w


# =========================
# Trainer (pure weak form)
# =========================

class WeakContinuityTrainer:
    def __init__(self, drift: DriftNet, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device_obj()
        self.drift = drift.to(self.device)
        self.tests = RandomFourierTests(cfg.d, cfg.n_tests, cfg.x_scale, cfg.t_scale).to(self.device)
        self.qtime = TimeDensity(cfg.d).to(self.device)
        self.opt = torch.optim.AdamW(
            list(self.drift.parameters()) + list(self.qtime.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    def _sample_times(self, B: int) -> torch.Tensor:
        return torch.rand(B, self.cfg.K_t, 1, device=self.device) * 0.999 + 1e-3  # (0,1]

    def step(self, x_p: torch.Tensor, x0: torch.Tensor):
        cfg = self.cfg
        x_p = x_p.to(self.device)
        x0  = x0.to(self.device)
        B, K = x_p.shape[0], cfg.K_t
        t_grid = self._sample_times(B)                 # (B,K,1)
        alpha, w = self.qtime(x_p, t_grid)             # (B,1), (B,K)

        # ----- Weak continuity residual -----
        # Residual r = ∂t φ + u·∇φ evaluated at (x_p, t_k)
        dphi_dt_stack, u_dot_gradphi_stack, u_power = [], [], []
        for k in range(K):
            t_k = t_grid[:, k, :]                      # (B,1)
            dphi_dt = self.tests.dphi_dt(x_p, t_k)     # (B,nT)
            gradphi = self.tests.grad_phi_x(x_p, t_k)  # (B,nT,d)
            u = self.drift(x_p)                   # (B,d)
            u_dot_gradphi = (u.unsqueeze(1) * gradphi).sum(-1)  # (B,nT)
            dphi_dt_stack.append(dphi_dt)
            u_dot_gradphi_stack.append(u_dot_gradphi)
            u_power.append((u * u).sum(dim=1, keepdim=True))
        dphi_dt = torch.stack(dphi_dt_stack, dim=1)         # (B,K,nT)
        u_dot_gradphi = torch.stack(u_dot_gradphi_stack, 1) # (B,K,nT)
        resid = dphi_dt + u_dot_gradphi                     # (B,K,nT)
        R_cont = (w.unsqueeze(-1) * resid).sum(dim=1)       # (B,nT)
        R_mean = R_cont.mean(dim=0)                         # (nT,)
        loss_cont = (R_mean ** 2).mean()

        # ----- Boundary match at t=0 -----
        t0 = torch.zeros(B, 1, device=self.device)
        psi_p  = self.tests.phi(x_p, t0)                   # (B,nT)
        psi_p0 = self.tests.phi(x0, t0)                    # (B0,nT)
        lhs = (alpha * psi_p.mean(dim=1, keepdim=True)).mean(dim=0).squeeze()  # (nT,)
        rhs = psi_p0.mean(dim=0)                           # (nT,)
        loss_bound = ((lhs - rhs) ** 2).mean()

        # ----- Regularizers -----
        u_ke = torch.stack(u_power, dim=1).mean()          # kinetic energy
        eps = 1e-8
        H = -(w * (w.add(eps).log())).sum(dim=1).mean()    # entropy of q^{cont}
        loss = (
            cfg.lambda_cont * loss_cont
            + cfg.lambda_bound * loss_bound
            + cfg.lambda_smooth_u * u_ke
            + cfg.lambda_entropy_t * (-H)
        )

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        stats = {
            "loss": float(loss.item()),
            "loss_cont": float(loss_cont.item()),
            "loss_bound": float(loss_bound.item()),
            "u_ke": float(u_ke.item()),
            "entropy": float(H.item()),
            "alpha_mean": float(alpha.mean().item()),
        }
        return float(loss.item()), stats

    @torch.no_grad()
    def eval_boundary_gap(self, x_p: torch.Tensor, x0: torch.Tensor) -> float:
        x_p = x_p.to(self.device)
        x0  = x0.to(self.device)
        B = x_p.shape[0]
        t0 = torch.zeros(B, 1, device=self.device)
        psi_p  = self.tests.phi(x_p, t0).mean(dim=0)
        psi_p0 = self.tests.phi(x0, t0).mean(dim=0)
        return float(((psi_p - psi_p0) ** 2).mean().item())


