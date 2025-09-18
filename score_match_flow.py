#%%
"""
Single-model training of a time-independent flow u(x) from p0 and \bar p using precomputed scores and a precomputed
potential g with \nabla g = s0 - sbar.

- No adversary, no test functions, no trajectory unrolling.
- Two targets supported:
    * "linearized": y(x) = g_c(x)  (first-order linearization of r0 - 1)
    * "exact":      y(x) = r0(x) - 1 with r0(x) = exp(g(x) - logZ) (stable log-sum-exp)

Loss (uniform mixing):
    L(θ, τ) = E_{x~\bar p} [ ( div u_θ(x) + u_θ(x)·sbar(x) - τ·y(x) )^2 ]
              + λ_u E||u_θ(x)||^2 + λ_J E||∇u_θ(x)||_F^2

Assumes you have:
  - X_train, X_val: samples from \bar p
  - sbar_train, sbar_val: score of \bar p at those points (exact or frozen score net outputs)
  - g_train, g_val: scalar potential values with ∇g ≈ s0 - sbar at those points

Optionally, if using target="exact", r0_train/val can be precomputed and passed; otherwise we compute from g via log-sum-exp.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from score_matching import ScoreMatching
from matplotlib.colors import Normalize
import numpy as np

#%%
# =========================
# Utilities
# =========================

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# TODO implement with torch native logsumexp - log(N)
def logmeanexp(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Stable log(mean(exp(t)))."""
    # m = t.max(dim=dim, keepdim=True).values
    # return (t - m).exp().mean(dim=dim).log() + m.squeeze(dim)
    return torch.logsumexp(t, dim=dim) - math.log(t.shape[dim])

# TODO figure out the issues with grad detachment
def hutchinson_divergence(u_fn, x, n_probes=1):
    outs = []
    for _ in range(n_probes):
        v = torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        x_probe = x.detach().clone().requires_grad_(True)  # fresh leaf
        u = u_fn(x_probe)                                  # Forward must use x_probe directly
        dot = (u * v).sum(dim=1)
        grad_x = torch.autograd.grad(dot.sum(), x_probe, create_graph=True)[0]
        outs.append((grad_x * v).sum(dim=1, keepdim=True))
    return torch.mean(torch.stack(outs, 0), 0)

# TODO figure out issues with grad detachment
def frobenius_jacobian_norm(u_fn, x: torch.Tensor, n_probes: int = 1) -> torch.Tensor:
    B, D = x.shape
    acc = torch.zeros(B, 1, device=x.device)
    for _ in range(n_probes):
        v = torch.empty_like(x).normal_()
        x_probe = x.detach().clone().requires_grad_(True)   # fresh leaf per probe
        u = u_fn(x_probe)
        dot = (u * v).sum(dim=1)
        grad_x = torch.autograd.grad(dot.sum(), x_probe, create_graph=True)[0]
        acc += (grad_x**2).sum(dim=1, keepdim=True)
    return acc / float(n_probes)



# =========================
# Models
# =========================

def make_mlp(in_dim: int, out_dim: int, width: int, depth: int, activation=nn.SiLU) -> nn.Sequential:
    layers = []
    d_in = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(d_in, width), activation()]
        d_in = width
    layers += [nn.Linear(d_in, out_dim)]
    return nn.Sequential(*layers)


class VectorField(nn.Module):
    def __init__(self, d: int, width: int = 128, depth: int = 4):
        super().__init__()
        self.net = make_mlp(d, d, width, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PotentialFlow(nn.Module):
    """u = ∇φ(x). Caller must ensure x.requires_grad_(True)."""
    def __init__(self, d: int, width: int = 128, depth: int = 4):
        super().__init__()
        self.phi = make_mlp(d, 1, width, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # no detach/clone here
        phi = self.phi(x)                                  # (B,1)
        grad = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        return grad                                        # (B,D)


# =========================
# Data
# =========================
class BarPDataset(Dataset):
    def __init__(self, X: np.ndarray, sbar: np.ndarray, y: np.ndarray):
        assert X.shape[0] == sbar.shape[0] == y.shape[0]
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.sbar = torch.as_tensor(sbar, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.sbar[idx], self.y[idx]
#%%

# =========================
# Config
# =========================
@dataclass
class TrainConfig:
    d: int
    potential_flow: bool = False            # use u = ∇φ by default
    width: int = 128
    depth: int = 4
    batch_size: int = 512
    epochs: int = 200
    lr: float = 1e-3
    tau_init: float = 10.0                  # ≈ 1/T if known
    learn_tau: bool = False
    lambda_u: float = 1e-3
    lambda_J: float = 1e-3
    div_probes: int = 2
    jac_probes: int = 1
    target: str = "linearized"            # "linearized" or "exact"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: Optional[float] = 1.0
    seed: int = 0

#%%
# =========================
# Training
# =========================
class SingleModelTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        if cfg.potential_flow:
            self.model = PotentialFlow(cfg.d, cfg.width, cfg.depth).to(self.device)
        else:
            self.model = VectorField(cfg.d, cfg.width, cfg.depth).to(self.device)
        # τ parameter (positive via softplus)
        self.tau_param = nn.Parameter(torch.tensor(math.log(math.exp(cfg.tau_init) - 1.0), dtype=torch.float32)) if cfg.learn_tau else None
        params = list(self.model.parameters()) + ([self.tau_param] if self.tau_param is not None else [])
        self.opt = torch.optim.Adam(params, lr=cfg.lr)

    def tau(self) -> torch.Tensor:
        if self.tau_param is None:
            return torch.tensor(self.cfg.tau_init, device=self.device)
        return F.softplus(self.tau_param) + 1e-6

    def residual(self, x: torch.Tensor, sbar: torch.Tensor):
        x_req = x.detach().clone().requires_grad_(True)        # leaf with grad
        u = self.model(x_req)                                  # PotentialFlow uses x_req
        div_u = hutchinson_divergence(lambda z: self.model(z), x, n_probes=self.cfg.div_probes)
        dot = (u * sbar).sum(dim=1, keepdim=True)
        return div_u + dot, u

    def reg_terms(self, x: torch.Tensor):
        x_req = x.detach().clone().requires_grad_(True)        # leaf with grad
        u = self.model(x_req)
        u2 = (u**2).sum(dim=1, keepdim=True)
        J2 = frobenius_jacobian_norm(lambda z: self.model(z), x, n_probes=self.cfg.jac_probes)
        return u2, J2

    def train(self,
              train_ds: BarPDataset,
              val_ds: Optional[BarPDataset] = None,
              verbose: bool = True) -> dict:
        cfg = self.cfg
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False) if val_ds is not None else None

        best_val = float('inf')
        best_state = None

        for epoch in range(cfg.epochs):
            self.model.train()
            tot_loss = 0.0
            for xb, sbarb, yb in train_loader:
                xb = xb.to(self.device)
                sbarb = sbarb.to(self.device)
                yb = yb.to(self.device)

                Rb, u_b = self.residual(xb, sbarb)
                tau = self.tau()
                mse = ((Rb - tau * yb)**2).mean()

                # u2, J2 = self.reg_terms(xb)
                # reg = cfg.lambda_u * u2.mean() + cfg.lambda_J * J2.mean()
                # loss = mse + reg 
                loss = mse

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()

                tot_loss += loss.item()

            if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
                tau_val = float(self.tau().detach().cpu())
                msg = f"[epoch {epoch:04d}] train_loss={tot_loss/len(train_loader):.4e}"
                msg += f" tau={tau_val:.4e}"
                print(msg)

        if best_state is not None:
            self.model.load_state_dict(best_state['model'])
            if self.tau_param is not None and best_state['tau_param'] is not None:
                self.tau_param.data.copy_(best_state['tau_param'])

        return {
            'model': self.model,
            'tau': float(self.tau().detach().cpu()),
            'cfg': self.cfg,
        }
#%%

# =========================
# Target preparation helpers
# =========================

def prepare_targets(g: np.ndarray,
                    mode: str = "linearized"):
    assert mode in {"linearized", "exact"}
    # if mode == "linearized":
    #     y_tr = center_g(g_train)
    #     y_va = center_g(g_val) if g_val is not None else None
    # else:
    # Compute r0 from g using stable log-sum-exp normalization over dataset.
    g_t = torch.as_tensor(g, dtype=torch.float64)
    lZ = logmeanexp(g_t, dim=0)  # log mean exp
    r0 = torch.exp(g_t - lZ).to(torch.float32).numpy()
    y = r0 - 1.0
    return y.astype(np.float32)

#%%
def make_linear_translation_data(
    N0: int = 20000,
    N: int = 20000,
    d: int = 2,
    T: int = 100,
    sigma: float = 1.0,
    discrete: bool = True,
    shift_per_unit: float = 1.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    v = np.zeros(d, dtype=np.float32)
    v[0] = float(shift_per_unit)
    X0 = rng.normal(0.0, sigma, size=(N0, d)).astype(np.float32)
    if discrete:
        ts = rng.integers(low=0, high=T + 1, size=N).astype(np.float32)
    else:
        ts = rng.random(N).astype(np.float32) * float(T)
    eps = rng.normal(0.0, sigma, size=(N, d)).astype(np.float32)
    X = ts[:, None] * v[None, :] + eps
    EX0, EX = X0.mean(0), X.mean(0)
    print(
        f"[sanity] mean(X0)[0]={EX0[0]:+.3f}, mean(X)[0]={EX[0]:+.3f}, expected ≈ {(T/2.0)*shift_per_unit:+.3f}"
    )
    return X0, X, v
#%%
# =========================
# Example usage (placeholder)
# =========================
# Toy shapes; replace with real data arrays
D = 2
# TODO set up linear translation of gaussian toy dataset
# X_train = np.random.randn(N_tr, D).astype(np.float32)
# X_val = np.random.randn(N_va, D).astype(np.float32)
# TODO implement the score model, these are just random values now
X0, X, v = make_linear_translation_data(N0=20000, N=20000, d=D, T=100, sigma=1.0, discrete=True, shift_per_unit=1.0, seed=0)
X = torch.tensor(X, dtype=torch.float32, device='cuda')
X0 = torch.tensor(X0, dtype=torch.float32, device='cuda')
#%%
n_epochs = 500

sbar = ScoreMatching(
    snet=make_mlp(in_dim=D, out_dim=D, width=128, depth=3).to(device='cuda'),
    alpha=0.1,
    sigma=0.1,
    eta=0.05,
    D=2,
    T=100,
    device='cuda'
)
s0 = ScoreMatching(
    snet=make_mlp(in_dim=D, out_dim=D, width=128, depth=3).to(device='cuda'),
    alpha=0.1,  # weight on divergence term
    sigma=0.1,
    eta=0.05,
    D=2,
    T=100,
    device='cuda'
)

optim_sbar = torch.optim.Adam(sbar.parameters(), lr=1e-4)
optim_s0 = torch.optim.Adam(s0.parameters(), lr=1e-4)
lambda_mag = 0.1
#%%
for i in range(n_epochs):
    # Forward passes
    sbar_loss = sbar(X)
    sbar_out = sbar.snet(X)

    # Match only magnitudes (L2 norms), not directions / per-component values
    sbar_norm = sbar_out.norm(dim=1)
    loss = sbar_loss
    optim_sbar.zero_grad()
    loss.backward()
    optim_sbar.step()

    if i%50 == 0:
        print(f'[{i}] sbar_loss: {sbar_loss.item():.4f}')
#%%
for i in range(n_epochs):
    # Forward passes
    s0_loss = s0(X0)
    s0_out = s0.snet(X)

    # # Recompute (or reuse detached) for s0 update, treat sbar as constant
    loss0 = s0_loss
    optim_s0.zero_grad()
    loss0.backward()
    optim_s0.step()
    
    if i%50 == 0:
        print(f'[{i}] s0_loss: {s0_loss.item():.4f}')


#%%
# Create a scalar parameter gamma_s0 and gamma_sbar that linearly scales their respective vector fields
# so that the mean norm of the fields is approximately equal. Freeze the vector fields so we learn
# only the scalars.
with torch.no_grad():
    sbar_ = sbar.snet(X).detach()
    s0_   = s0.snet(X).detach()
    mu_bar = sbar_.norm(dim=1).mean()
    mu_0   = s0_.norm(dim=1).mean()

    # Option A: fix gamma_bar=1
    gamma_0   = (mu_bar / mu_0)

#%%
# Freeze the score models
for param in sbar.parameters():
    param.requires_grad = False
for param in s0.parameters():
    param.requires_grad = False
#%%
# Score matched G potential field, i.e. train potential G so that grad G
g = make_mlp(in_dim=D, out_dim=D, width=128, depth=3).to(device='cuda')

optim_g = torch.optim.Adam(g.parameters(), lr=1e-4)
#%%
n_epochs = 2500
for i in range(n_epochs):
    X.requires_grad_(True)
    g_grad = torch.autograd.grad(g(X).sum(), X, create_graph=True)[0]
    loss = (g_grad - (s0.snet(X)*gamma_0 - sbar.snet(X))).pow(2).mean()
    if i%50==0:
        print(i,loss.item())
    optim_g.zero_grad()
    loss.backward()
    optim_g.step()

#%%
# Make a grid of points spanning the min and max of X plus 10%
x_min, x_max = X[:, 0].min().item(), X[:, 0].max().item()
y_min, y_max = X[:, 1].min().item(), X[:, 1].max().item()
x_grid = np.linspace(x_min - 10.1, x_max + 10.1, 20)
y_grid = np.linspace(y_min - 10.1, y_max + 10.1, 20)
X1, Y1 = np.meshgrid(x_grid, y_grid)
XY = np.stack([X1.ravel(), Y1.ravel()], axis=-1)
XY = torch.tensor(XY, dtype=torch.float32, device='cuda', requires_grad=True)
#%%
# Compute gradients
g_grads = torch.autograd.grad(g(XY).sum(), XY, create_graph=True)[0]
g_grads = g_grads.detach().cpu().numpy()
# Plot the quiver
#%%
# Plot a quiver of grad(g) arrows on a grid above a scatter plot of the data
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.scatter(X[:, 0].detach().cpu().numpy(), X[:, 1].detach().cpu().numpy(), s=1)

ax.quiver(X1, Y1, g_grads[:, 0].reshape(X1.shape), g_grads[:, 1].reshape(Y1.shape), color='r')
#%%
# Plot a quiver of sbar arrows on a grid
with torch.no_grad():
    sbars = sbar.snet(XY)
    sbars = sbars.cpu().numpy()
with torch.no_grad():
    s0s = s0.snet(XY)
    s0s = s0s.cpu().numpy()
#%%
plt.quiver(X1, Y1, 
           sbars[:, 0].reshape(X1.shape), 
           sbars[:, 1].reshape(Y1.shape), 
           color='b')
#%% 
# Plot a quiver of s0 arrows on a grid
plt.quiver(X1, Y1, 
           s0s[:, 0].reshape(X1.shape), 
           s0s[:, 1].reshape(Y1.shape), 
           color='g')

#%%
# Plot the s0-sbar arrows
plt.quiver(X1, Y1,
           (s0s*gamma_0.item() - sbars)[:, 0].reshape(X1.shape),
           (s0s*gamma_0.item() - sbars)[:, 1].reshape(Y1.shape),
           color='m')
# plt.quiver(X1, Y1, 
#            s0s[:, 0].reshape(X1.shape), 
#            s0s[:, 1].reshape(Y1.shape), 
#            color='g')
# plt.quiver(X1, Y1, 
#            sbars[:, 0].reshape(X1.shape), 
#            sbars[:, 1].reshape(Y1.shape), 
#            color='b')
#%%
# Prepare targets
target_mode = "linearized"  # or "exact"
g_vals = g(X).detach().cpu().numpy()
y = prepare_targets(g_vals)

# Datasets
sbar_vals = sbar.snet(X).detach().cpu().numpy()
ds = BarPDataset(X, sbar_vals, y)

# Train
cfg = TrainConfig(d=D, potential_flow=False, target=target_mode)
trainer = SingleModelTrainer(cfg)
#%%
result = trainer.train(ds, None, verbose=True)

print("Training complete. tau=", result['tau'])
#%%
phi = trainer.model
# %%
# Plot a quiver of the grad(potential)
phi_grads = phi(XY).detach().cpu().numpy()
# Plot the quiver
x_ = X.detach().cpu().numpy()
x0_ = X0.detach().cpu().numpy()
plt.scatter(x_[:, 0], x_[:, 1], s=1, alpha=.2, c='grey')
# plt.scatter(x0_[:, 0], x0_[:, 1], s=1)
plt.quiver(X1, Y1, 
           phi_grads[:, 0].reshape(X1.shape), 
           phi_grads[:, 1].reshape(Y1.shape), 
           color='blue')
# %%
# Simulate trajectories of points starting at X0 following the vector field
n_steps = 500
dt = 50

# Start from X0 on device
x = X.detach().clone()[:150]          # (N0, D) torch tensor on cuda
trajectories_list = []

tau = trainer.tau().item()

for _ in range(n_steps):
    v = tau*phi(x)                  # vector field (N0, D)
    x = x + dt * v               # Euler step
    trajectories_list.append(x.detach().cpu().numpy())

# Stack into a single numpy array: shape (n_steps, N0, D)
trajectories = np.stack(trajectories_list, axis=0)

#%%
# Plot the trajectories, colored by time
import matplotlib.pyplot as plt

# Stack trajectory list -> (n_steps, N0, D)
traj = np.stack(trajectories, axis=0)          # shape (T, N, 2)
T, N, D = traj.shape
times = np.arange(T)

# (Optional) subsample points for clarity
max_points = 400
if N > max_points:
    idx = np.random.choice(N, max_points, replace=False)
else:
    idx = np.arange(N)

traj_sub = traj[:, idx, :]                     # (T, N_sub, 2)

# Build arrays for scatter
pts = traj_sub.reshape(T * len(idx), D)
t_rep = np.repeat(times, len(idx))             # time label per point

plt.figure(figsize=(7, 6))
plt.scatter(x_[:,0], x_[:,1], c='grey', alpha=.2)
sc = plt.scatter(pts[:, 0], pts[:, 1], c=t_rep, s=4, cmap='viridis', norm=Normalize(vmin=0, vmax=T - 1))
plt.colorbar(sc, label='time step')


# Overlay starting positions
start_pts = traj_sub[0]
plt.scatter(start_pts[:, 0], start_pts[:, 1], c='red', s=8, label='start', alpha=0.7)

plt.title('Trajectories colored by time')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# %%
