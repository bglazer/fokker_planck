# %%
# # =====================
# Experiment / diagnostics (consolidated)
# =====================
import math
import numpy as np
import torch
from mmd import WeakContinuityTrainer, Config, DriftNet
from synthetic_data_utils import (
    make_linear_translation_data,
    make_linear_B_data,
    make_double_well_data,
    make_branching_data,
    make_spiral_manifold_noise_data,
    B_spiral_sink_2d,
)
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from weak_flow_util import (
    pca_fit,
    rk4_forward_step,
    rk4_reverse_step,
    logpt_reverse,
    drift_batch,
    generate_mixture_from_field,
    pca_project,
    pca_unproject,
)

# %%
# Generate synthetic data for testing
TEST_ID = 4  # 1: translation (baseline), 2: linear B, 3: double-well, 4: branching, 5: spiral+noise

d = 2
T = 4.0

if TEST_ID == 1:
    X0, X, v_true = make_linear_translation_data(
        N0=20000,
        N=40000,
        d=d,
        T=int(T),
        sigma=1.0,
        shift_per_unit=1.0,
        seed=42,
    )
    u_true_fn = None

elif TEST_ID == 2:
    B2 = B_spiral_sink_2d(omega=1.2, rho=-0.2)
    B = np.zeros((d, d), dtype=np.float32)
    B[:2, :2] = B2
    X0, X, ts, u_true_fn = make_linear_B_data(
        B,
        N0=20000,
        N=40000,
        T=T,
        sigma0=1.0,
        seed=42,
    )

elif TEST_ID == 3:
    X0, X, ts, u_true_fn = make_double_well_data(
        d=d,
        alpha=0.5,
        N0=20000,
        N=40000,
        T=T,
        sigma0=0.5,
        seed=42,
    )

elif TEST_ID == 4:
    X0, X, ts, u_true_fn = make_branching_data(
        d=d,
        v_root=0.8,
        v_branch=1.2,
        xsplit=1.0,
        beta=3.0,
        N0=20000,
        N=40000,
        T=T,
        sigma0=0.5,
        seed=42,
    )

elif TEST_ID == 5:
    d = 20
    X0, X, ts, u_true_fn = make_spiral_manifold_noise_data(
        d=d,
        omega=1.0,
        rho=-0.1,
        N0=20000,
        N=40000,
        T=6.0,
        sigma0=0.5,
        seed=42,
    )
    T = 6.0
else:
    raise ValueError("TEST_ID must be in {1,2,3,4,5}")


# %%
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


# Standardize X and X0
standardizer = ZScoreStandardizer().fit(X0, X)
X0 = standardizer.transform(X0)
X = standardizer.transform(X)
# %%
# --- trainer config ---
device = 'cuda'
X = torch.from_numpy(X.astype(np.float32)).to(device)
X0 = torch.from_numpy(X0.astype(np.float32)).to(device)
#%%
cfg = Config(
    d = d,
    n_tests = 256,
    x_scale = 1,
    t_scale = 1,
    time_feat = 32,
    width = 128,
    depth = 3,
    batch_p = 1024,
    batch_p0 = 1024,
    K_t = 16,
    lr = 1e-3,
    weight_decay = 0.000001,
    lambda_cont = 1,
    lambda_bound = 1,
    lambda_smooth_u = 0.001,
    lambda_entropy_t = 0.001,
    device = device
)
#%%
drift = DriftNet(d=cfg.d, width=cfg.width, depth=cfg.depth)
trainer = WeakContinuityTrainer(drift, cfg)
# %%
num_steps = 1500
start = time.time()
for step in range(num_steps):
    X_idxs = torch.randint(0, X.shape[0], (cfg.batch_p,), device=trainer.device)
    X0_idxs = torch.randint(0, X0.shape[0], (cfg.batch_p0,), device=trainer.device)
    x_batch = X[X_idxs]
    x0_batch = X0[X0_idxs]
    loss, stats = trainer.step(x_batch, x0_batch)
    if step % 100 == 0:
        print(step, stats)
end = time.time()
print(f"Elapsed time: {end - start:.2f} seconds")

# %%
# --- Visualization ---
Xnp = X.detach().cpu().numpy()
X0np = X0.detach().cpu().numpy()
X_all = np.vstack([X0np, Xnp])
mean, comps = pca_fit(X_all, n=2)
Z0 = pca_project(X0np, mean, comps)
Z = pca_project(Xnp, mean, comps)

# %%
# Vector field on PCA plane
nq = 25
marg = 0.00
mins = np.quantile(Z, q=marg, axis=0)
maxs = np.quantile(Z, q=1 - marg, axis=0)
gx = np.linspace(mins[0], maxs[0], nq)
gy = np.linspace(mins[1], maxs[1], nq)
GX, GY = np.meshgrid(gx, gy)
G2 = np.stack([GX.ravel(), GY.ravel()], axis=1)
G_full = pca_unproject(G2, mean, comps)
G_full_t = torch.tensor(G_full, dtype=torch.float32).to(device)

U_full = drift(G_full_t).detach().cpu().numpy() 
U2 = U_full @ comps.T
U2 = U2 / (np.linalg.norm(U2, axis=1, keepdims=True) + 1e-8) * 0.2

plt.figure(figsize=(12, 12))
idx0 = np.random.choice(Z0.shape[0], size=min(5000, Z0.shape[0]), replace=False)
idx = np.random.choice(Z.shape[0], size=min(5000, Z.shape[0]), replace=False)
plt.scatter(Z0[idx0, 0], Z0[idx0, 1], s=5, alpha=0.35, label="X0 (p0)")
plt.scatter(Z[idx, 0], Z[idx, 1], s=5, alpha=0.35, label="X (mixture)")
plt.quiver(
    G2[:, 0],
    G2[:, 1],
    U2[:, 0],
    U2[:, 1],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="purple",
    label="u_hat",
)

plt.legend()
plt.title("Projected drift field (PCA 2D) with X0 and X")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

#%%
# Simple simulation of a sample of the initial X0 points moving under the drift field
n_simulations = 100
n_steps = 500
dt = .01

# Initial sample
X0_sim0 = X0[torch.randperm(X0.shape[0], device=X0.device)[:n_simulations]]
# List to store trajectory (time dimension first)
traj = [X0_sim0.clone()]

with torch.no_grad():
    x = X0_sim0.clone()
    for _ in range(n_steps):
        x = x + dt * drift(x)
        traj.append(x.clone())

# Tensor of shape (n_steps+1, n_simulations, d)
X0_sim_traj = torch.stack(traj, dim=0)

# Also keep a flattened version for downstream PCA code expecting (N,d)
X0_sim = X0_sim_traj.reshape(-1, X0_sim_traj.shape[-1])

# Time stamps per row in flattened trajectory (optional for coloring)
t_sim = torch.linspace(0, n_steps * dt, n_steps + 1, device=X0.device).repeat_interleave(n_simulations).cpu().numpy()

# %%
# Scatter plot of the simulation, colored by time, overlaid on the PCA projection
Z0_sim = pca_project(X0_sim.detach().cpu().numpy(), mean, comps)
plt.figure(figsize=(6, 6))
plt.scatter(Z[:, 0], Z[:, 1], s=5, alpha=0.35, color="blue", label="X (mixture)")
sc = plt.scatter(
    Z0_sim[:, 0], Z0_sim[:, 1], c=t_sim, s=8, cmap="viridis", alpha=0.8
)
# plt.scatter(X0np[:, 0], X0np[:, 1], s=5, alpha=0.35, color="orange", label="X0 (p0)")
cb = plt.colorbar(sc)
cb.set_label("time")
plt.title("PCA of X0 simulation colored by time")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
# %%
