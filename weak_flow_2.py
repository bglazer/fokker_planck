# Complete PyTorch implementation for learning a drift field u_θ(x)
# Two modes:
#   (A) GENERAL  E_{bar p}[ (div u + u · ∇bar_q) * ψ ]  =  E_{p0}[ ψ ]
#   (B) POTENTIAL u=∇φ and use  E_{bar p}[ ∇φ · ∇ψ ]  =  E_{p0}[ ψ ]
#
# Notes
# - GENERAL uses precomputed pooled score ∇bar_q(x) and Hutchinson for div u.
# - POTENTIAL avoids divergence & scores entirely (ratio-free weak form).
# - ψ is a fixed random-feature family (no adversarial critic).
#
# You provide:
#   X_pool, d) pooled samples across time (uniform mixture)
#   X0,    d) samples at t=0
#   score_bar_model (only for GENERAL)(x)->(B,d) ≈ ∇ log bar p(x)
#%%
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from score_matching import ScoreMatching
#%%
# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed = 123):
    torch.manual_seed(seed)
    np.random.seed(seed)


def rademacher(shape, device=None):
    """±1 entries with equal probability."""
    return (torch.randint(0, 2, shape, device=device, dtype=torch.int8).float() * 2.0 - 1.0)


def grad_wrt_x(scalar_per_sample: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return ∇_x of scalar_per_sample (B,) w.r.t. x (B,d)."""
    return torch.autograd.grad(
        scalar_per_sample.sum(), x, create_graph=True, retain_graph=True
    )[0]


# ---------------------------
# Models
# ---------------------------

def make_mlp(in_dim, out_dim, width = 256, depth = 3, act=nn.SiLU) -> nn.Sequential:
    layers = []
    last = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(last, width), act()]
        last = width
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class PotentialNet(nn.Module):
    """POTENTIAL mode(x)^d -> R (scalar potential)."""
    def __init__(self, d, width = 256, depth = 3):
        super().__init__()
        self.net = make_mlp(d, 1, width, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)


class PotentialDrift(nn.Module):
    """Wrapper that returns u=∇φ for convenience."""
    def __init__(self, phi):
        super().__init__()
        self.phi = phi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        phi_x = self.phi(x)                  # (B,)
        grad_phi = grad_wrt_x(phi_x, x)      # (B,d)
        return grad_phi

class DriftNet(nn.Module):
    """Learn drift field u(x) directly."""
    def __init__(self, d, width=256, depth=3):
        super().__init__()
        self.net = make_mlp(d, d, width, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RandomFourierPsi(nn.Module):
    """Fixed random-feature test functions ψ(x) ∈ R^M.
    ψ_j(x) = sqrt(2/M) * cos( x W[:,j] + b_j ), with W ~ N(0, σ^2 I).

    Provides both ψ(x) and its analytic gradient ∇ψ(x) for POTENTIAL mode.
    """
    def __init__(self, d, M = 64, sigma = 1.0, seed = 123):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(d, M, generator=g) * sigma
        b = 2 * math.pi * torch.rand(M, generator=g)
        self.register_buffer("W", W)  # (d,M)
        self.register_buffer("b", b)  # (M,)
        self.scale = (2.0 / M) ** 0.5

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x,d) -> (B,M)
        z = x @ self.W + self.b
        return self.scale * torch.cos(z)

    @torch.no_grad()
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """Return ∇ψ(x) (B,d,M)."""
        z = x @ self.W + self.b                      # (B,M)
        s = torch.sin(z)                              # (B,M)
        # ∇_x cos(z_j) = -sin(z_j) * W[:,j]
        return -self.scale * s.unsqueeze(1) * self.W.unsqueeze(0)  # (B,1,M)*(1,d,M)->(B,d,M)


# ---------------------------
# Hutchinson divergence (GENERAL mode)
# ---------------------------

def hutchinson_divergence(u_model: nn.Module, x: torch.Tensor, K = 2) -> torch.Tensor:
    """Estimate div u(x) = tr(∇u) using K Hutchinson probes. Returns (B,)."""
    B, d = x.shape
    x = x.requires_grad_(True)
    div = 0.0
    for _ in range(K):
        eps = rademacher((B, d), device=x.device)
        u = u_model(x)                       # (B,d)
        s = (u * eps).sum(dim=1)             # (B,)
        g = grad_wrt_x(s, x)                 # (B,d)
        div = div + (g * eps).sum(dim=1)     # (B,)
    return div / K


# ---------------------------
# Config / Datasets
# ---------------------------

@dataclass
class Config:
    d: int = 2
    batch_size: int = 1024
    lr: float = 1e-3
    steps: int = 50_000
    K_hutch: int = 2
    width: int = 256
    depth: int = 3
    n_features: int = 64
    psi_sigma: float = 1.0
    weight_decay: float = 0.0
    clip_grad: float = 5.0
    reg_l2: float = 1e-4
    log_every: int = 200
    device: str = "cuda"


def make_loader(X: np.ndarray, batch_size, device) -> DataLoader:
    t = torch.as_tensor(X, dtype=torch.float32, device=device)
    ds = TensorDataset(t)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


#%%
# ---------------------------
# Training
# ---------------------------

"""Train drift field 

Objective per feature j:
    E_{bar p}[ (div u + u·score_bar) * ψ_j ] = E_{p0}[ ψ_j ]
    Loss = mean_j (lhs_j - rhs_j)^2
"""
cfg = Config(
    d=2,
    batch_size = 1024,
    lr = 1e-3,
    steps = 50_000,
    K_hutch = 2,
    width = 256,
    depth = 3,
    n_features = 64,
    psi_sigma = 1.0,
    weight_decay = 0.0,
    clip_grad = 5.0,
    reg_l2 = 1e-4,
    log_every = 200,
    device = "cuda",
)

#%%
seed = 42
set_seed(seed)

N0 = 5_000
N  = 5_000
T = 5
rng = np.random.default_rng(seed)
v = np.zeros(cfg.d, dtype=np.float32)
v[0] = float(1)
X0 = rng.normal(0.0, 1.0, size=(N0, cfg.d)).astype(np.float32)
ts = rng.integers(low=0, high=T + 1, size=N).astype(np.float32)
eps = rng.normal(0.0, 1.0, size=(N, cfg.d)).astype(np.float32)
X = ts[:, None] * v[None, :] + eps
EX0, EX = X0.mean(0), X.mean(0)
print(
    f"[sanity] mean(X0)[0]={EX0[0]:+.3f}, mean(X)[0]={EX[0]:+.3f}"
)
#%%
# Make a two dimensional X that is a mixture of two Gaussians
# X_mog = np.concatenate([
#     rng.normal(loc=-1.0, scale=0.5, size=(N0 // 2, cfg.d)),
#     rng.normal(loc=1.0, scale=0.5, size=(N0 // 2, cfg.d))
# ])
# plt.scatter(X_mog[:, 0], X_mog[:, 1], s=5, alpha=0.5)
#%%
snet = make_mlp(2, 2, width=256, depth=3)  # Example score network

#%%
# Train Score Matching model
D = 2   # input dimension
M = 512  # the number of neurons in scale (s) and translation (t) nets
alpha = 0.1
sigma = 0.1
eta = 0.05
T = 10
lr = 1e-4 # learning rate
num_epochs = 1000 # max. number of epochs

score_model = ScoreMatching(snet, alpha, sigma, eta, D, T, device=cfg.device)
score_model = score_model.to(cfg.device)
#%%
optimizer = torch.optim.Adamax([p for p in score_model.parameters() if p.requires_grad == True], lr=lr)

#%%
Xt = torch.tensor(X, device=cfg.device, dtype=torch.float32)
X0t = torch.tensor(X0, device=cfg.device, dtype=torch.float32)
#%%
for i in range(500):
    # Forward pass
    loss = score_model(Xt)
    if i%100 == 0:
        print(f"Epoch {i+1}, Loss: {loss.item()}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#%%
score_model0 = ScoreMatching(snet, alpha, sigma, eta, D, T, device=cfg.device)
score_model0 = score_model.to(cfg.device)
#%%
optimizer = torch.optim.Adamax([p for p in score_model0.parameters() if p.requires_grad == True], lr=lr)

#%%
for i in range(500):
    # Forward pass
    loss = score_model0(X0t)
    if i%100 == 0:
        print(f"Epoch {i+1}, Loss: {loss.item()}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#%%
# Draw samples from the score model and plot them over the data distribution as a check
import matplotlib.pyplot as plt

with torch.no_grad():
    score_model.T = 200
    samples = score_model.sample(1000, sigma=1)
samples = samples.cpu().numpy()
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=5, label="Data", alpha=0.5)
plt.scatter(samples[:, 0], samples[:, 1], s=5, label="Samples", alpha=0.5)
plt.legend()
plt.title("Data Distribution vs. Samples from Score Model")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

#%%
# Draw samples from the score model and plot them over the data distribution as a check
import matplotlib.pyplot as plt

with torch.no_grad():
    score_model0.T = 200
    samples = score_model0.sample(1000, sigma=1)
samples = samples.cpu().numpy()
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=5, label="Data", alpha=0.5)
plt.scatter(samples[:, 0], samples[:, 1], s=5, label="Samples", alpha=0.5)
plt.legend()
plt.title("Data Distribution vs. Samples from Score Model")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


#%%
device = cfg.device

# Data
loader_pool = make_loader(X, cfg.batch_size, device)
loader_p0   = make_loader(X0, cfg.batch_size, device)
iter_pool = iter(loader_pool)
iter_p0   = iter(loader_p0)

# Test features
psi = RandomFourierPsi(cfg.d, cfg.n_features, cfg.psi_sigma).to(device)

stats: dict[str, list[float]] = {"loss": [], "lhs_norm": [], "rhs_norm": [], "res_norm": []}

u_theta = DriftNet(cfg.d, cfg.width, cfg.depth).to(device)
score_bar_model = score_model.to(device)
for p in score_bar_model.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(u_theta.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

for step in range(1, cfg.steps + 1):
    try:
        (x_bar,) = next(iter_pool)
    except StopIteration:
        iter_pool = iter(loader_pool)
        (x_bar,) = next(iter_pool)
    try:
        (x0,) = next(iter_p0)
    except StopIteration:
        iter_p0 = iter(loader_p0)
        (x0,) = next(iter_p0)

    # LHS under bar p
    u = u_theta(x_bar)                                 # (B,d)
    div_u = hutchinson_divergence(u_theta, x_bar, cfg.K_hutch)   # (B,)
    s_bar = score_bar_model(x_bar)                     # (B,d)
    term = div_u + (u * s_bar).sum(dim=1)              # (B,)

    with torch.no_grad():
        psi_bar = psi(x_bar)                           # (B,M)
        psi_0   = psi(x0)                              # (B,M)
    lhs_vec = (term[:, None] * psi_bar).mean(dim=0)    # (M,)
    rhs_vec = psi_0.mean(dim=0)                        # (M,)

    resid = lhs_vec - rhs_vec                          # (M,)
    loss_main = (resid ** 2).mean()

    # Regularize u magnitude under bar p
    reg = cfg.reg_l2 * (u.pow(2).sum(dim=1)).mean()
    loss = loss_main + reg

    opt.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.clip_grad is not None:
        nn.utils.clip_grad_norm_(u_theta.parameters(), cfg.clip_grad)
    opt.step()

    if step % cfg.log_every == 0:
        stats["loss"].append(float(loss_main.detach().cpu()))
        stats["lhs_norm"].append(float(lhs_vec.norm().detach().cpu()))
        stats["rhs_norm"].append(float(rhs_vec.norm().detach().cpu()))
        stats["res_norm"].append(float(resid.norm().detach().cpu()))
        print(f"[Step {step:6d}] loss={loss_main.item():.4e} | res_norm={resid.norm().item():.4e} | u_l2={u.pow(2).mean().item():.4e}")

#%%
# Plot a scatter plot of the data, then plot a quiver of arrows showing the direction of u
import matplotlib.pyplot as plt

with torch.no_grad():
    # Data scatter (optional)
    x_bar = X

    # Build a regular grid
    x_min, y_min = X.min(0) 
    x_max, y_max = X.max(0)
    n_per_axis = 25
    gx = np.linspace(x_min, x_max, n_per_axis)
    gy = np.linspace(y_min, y_max, n_per_axis)
    GX, GY = np.meshgrid(gx, gy)
    grid_points = np.stack([GX.ravel(), GY.ravel()], axis=1)  # (n_per_axis^2, 2)

    # Evaluate drift on grid
    gp_t = torch.tensor(grid_points, device=device, dtype=torch.float32)
    U = u_theta(gp_t).cpu().numpy()  # (N,2)
    Ux = U[:, 0].reshape(GX.shape)
    Uy = U[:, 1].reshape(GY.shape)
    Umag = np.sqrt(Ux**2 + Uy**2)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_bar[:, 0], x_bar[:, 1], s=5, alpha=0.25, label="Data")
    plt.quiver(GX, GY, Ux, Uy, Umag, cmap="viridis", angles='xy', scale_units='xy', scale=1.0, width=0.003)
    cbar = plt.colorbar()
    cbar.set_label("|u(x)|")
    plt.title("Drift field u_theta over grid")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
