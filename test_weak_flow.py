# %%
# # =====================
# Experiment / diagnostics (consolidated)
# =====================
import math
import numpy as np
import torch
from weak_flow import WeakFlowTrainer, Config, PotentialNet
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from weak_flow_util import (
    rk4_forward_step,
    rk4_reverse_step,
    logpt_reverse,
    drift_batch,
    generate_mixture_from_field,
    pca_project,
    pca_unproject,
)


# %%
def roundtrip_error(
    potential: PotentialNet, X: torch.Tensor, t: float, steps: int = 80
) -> torch.Tensor:
    x0_hat = X.detach()
    ds = float(t) / max(1, int(steps))
    for _ in range(max(1, int(steps))):
        x0_hat = rk4_reverse_step(potential, x0_hat, ds)
    x_fwd = x0_hat.detach()
    for _ in range(max(1, int(steps))):
        x_fwd = rk4_forward_step(potential, x_fwd, ds)
    return ((x_fwd - X.detach()) ** 2).sum(dim=1).sqrt()


# =====================
# Synthetic data generators
# =====================


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


# ---- Test 2: Linear drift u(x)=B x ----


def make_linear_B_data(
    B: np.ndarray,
    N0: int = 20000,
    N: int = 40000,
    T: float = 4.0,
    K: int = 64,
    sigma0: float = 1.0,
    seed: int = 0,
):
    d = B.shape[0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B_t = torch.from_numpy(B.astype(np.float32)).to(dev)

    def u_fn(x: torch.Tensor) -> torch.Tensor:
        return x @ B_t.T

    X0, X, ts = generate_mixture_from_field(
        u_fn, d=d, N0=N0, N=N, T=T, K=K, sigma=sigma0, seed=seed
    )
    return X0, X, ts, u_fn


# helper constructors for B (2D blocks)


def B_rotation_2d(omega: float) -> np.ndarray:
    return np.array([[0.0, -omega], [omega, 0.0]], dtype=np.float32)


def B_sink_source_2d(lmbda: float) -> np.ndarray:
    return np.array([[lmbda, 0.0], [0.0, lmbda]], dtype=np.float32)


def B_saddle_2d(lpos: float, lneg: float) -> np.ndarray:
    return np.array([[lpos, 0.0], [0.0, -abs(lneg)]], dtype=np.float32)


def B_spiral_sink_2d(omega: float, rho: float) -> np.ndarray:
    return np.array([[rho, -omega], [omega, rho]], dtype=np.float32)


# ---- Test 3: Nonlinear gradient flow (double-well) ----


def make_double_well_data(
    d: int = 2,
    alpha: float = 0.5,
    N0: int = 20000,
    N: int = 40000,
    T: float = 4.0,
    K: int = 64,
    sigma0: float = 0.5,
    seed: int = 0,
):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def u_fn(x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, :1]
        rest = x[:, 1:]
        g1 = x1 * (x1 * x1 - 1.0)
        if rest.shape[1] > 0:
            grest = alpha * rest
            return torch.cat([g1, grest], dim=1)
        else:
            return g1

    X0, X, ts = generate_mixture_from_field(
        u_fn, d=d, N0=N0, N=N, T=T, K=K, sigma=sigma0, seed=seed
    )
    return X0, X, ts, u_fn


# ---- Test 4: Branching drift (Y-shaped) ----


def make_branching_data(
    d: int = 2,
    v_root: float = 1.0,
    v_branch: float = 1.0,
    xsplit: float = 1.0,
    beta: float = 3.0,
    N0: int = 20000,
    N: int = 40000,
    T: float = 4.0,
    K: int = 64,
    sigma0: float = 0.5,
    seed: int = 0,
):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e1 = torch.zeros(d, device=dev)
    e1[0] = 1.0
    e1 = e1.view(1, -1)
    vA = torch.zeros(d, device=dev)
    vA[:2] = torch.tensor([1.0, 1.0], device=dev)
    vA = (v_branch * vA).view(1, -1)
    vB = torch.zeros(d, device=dev)
    vB[:2] = torch.tensor([1.0, -1.0], device=dev)
    vB = (v_branch * vB).view(1, -1)

    def u_fn(x):
        s = torch.sigmoid(beta * (x[:, :1] - xsplit))
        base = (v_root * x[:, :1]) * e1
        side = torch.where(x[:, 1:2] >= 0, s * vA, s * vB)
        return base + side

    X0, X, ts = generate_mixture_from_field(
        u_fn, d=d, N0=N0, N=N, T=T, K=K, sigma=sigma0, seed=seed
    )
    return X0, X, ts, u_fn


# ---- Test 5: Spiral dynamics embedded in R^d with nuisance dims ----


def make_spiral_manifold_noise_data(
    d: int = 10,
    omega: float = 1.0,
    rho: float = -0.2,
    N0: int = 20000,
    N: int = 40000,
    T: float = 6.0,
    K: int = 96,
    sigma0: float = 0.5,
    seed: int = 0,
):
    assert d >= 2
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = torch.tensor([[rho, -omega], [omega, rho]], device=dev)

    def u_fn(x: torch.Tensor) -> torch.Tensor:
        x01 = x[:, :2]
        u01 = x01 @ M.T
        if d > 2:
            zeros = torch.zeros(x.size(0), d - 2, device=dev)
            return torch.cat([u01, zeros], dim=1)
        else:
            return u01

    X0, X, ts = generate_mixture_from_field(
        u_fn, d=d, N0=N0, N=N, T=T, K=K, sigma=sigma0, seed=seed
    )
    return X0, X, ts, u_fn


# %%
# --- choose which synthetic to run ---
TEST_ID = 3  # 1: translation (baseline), 2: linear B, 3: double-well, 4: branching, 5: spiral+noise

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


# Auto tuner (unchanged API, but uses trainer's EMA metrics)
class AutoTuner:
    def __init__(
        self,
        u2_band=(5.0, 25.0),
        g2_band=(1.0, 10.0),
        w_u_bounds=(1e-5, 1e-1),
        w_s_bounds=(1e-3, 1.0),
        gap_target=2.0,
    ):
        self.u2_lo, self.u2_hi = u2_band
        self.g2_lo, self.g2_hi = g2_band
        self.wu_lo, self.wu_hi = w_u_bounds
        self.ws_lo, self.ws_hi = w_s_bounds
        self.gap_target = gap_target

    def step(
        self, trainer: WeakFlowTrainer, gap_d_ma: float, U2_ma: float, G2_ma: float
    ):
        cfg = trainer.cfg
        changed = {}
        if U2_ma > self.u2_hi:
            cfg.drift_l2_weight = min(cfg.drift_l2_weight * 2.0, self.wu_hi)
            changed["drift_l2_weight"] = cfg.drift_l2_weight
        elif U2_ma < self.u2_lo and gap_d_ma > self.gap_target:
            cfg.drift_l2_weight = max(cfg.drift_l2_weight / 1.5, self.wu_lo)
            changed["drift_l2_weight"] = cfg.drift_l2_weight
        if G2_ma > self.g2_hi:
            cfg.sobolev_weight = min(cfg.sobolev_weight * 1.5, self.ws_hi)
            changed["sobolev_weight"] = cfg.sobolev_weight
            cfg.diffusion_D = max(getattr(cfg, "diffusion_D", 0.0), 1e-3)
            changed["diffusion_D"] = cfg.diffusion_D
        elif G2_ma < self.g2_lo and gap_d_ma > self.gap_target:
            cfg.sobolev_weight = max(cfg.sobolev_weight / 1.5, self.ws_lo)
            changed["sobolev_weight"] = cfg.sobolev_weight
        if (
            gap_d_ma < self.gap_target
            and (self.u2_lo <= U2_ma <= self.u2_hi)
            and (self.g2_lo <= G2_ma <= self.g2_hi)
        ):
            cfg.T = min(cfg.T * 1.25, 4.0)
            changed["T"] = cfg.T
        return changed


autotuner = AutoTuner()

# Standardize X and X0
standardizer = ZScoreStandardizer().fit(X0, X)
X0 = standardizer.transform(X0)
X = standardizer.transform(X)
# %%
# --- trainer config ---
cfg = Config(
    d=d,
    T=0.25,
    order=1,
    steps=1500,
    batch_size=1024,
    n_critic=1,
    lr_potential=1e-4,
    lr_critic=1e-4,
    sobolev_weight=0.2,
    drift_l2_weight=1e-3,
    critic_width=128,
    critic_depth=3,
    spectral_norm_critic=True,
    potential_width=256,
    potential_depth=4,
    mixed_precision=False,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    log_every=50,
    diffusion_D=0.0,
    lap_probes=2,
)
# %%
trainer = WeakFlowTrainer(cfg, X0, X)
# %%
# --- Training loop with cached metrics (no extra model evals) ---
start = time.time()
for step in range(1, cfg.steps + 1):
    gap_c, sp, G2_ma = trainer.critic_step()
    gap_d, sm, loss_d, U2_ma = trainer.drift_step()

    if step % cfg.log_every == 0:
        with torch.no_grad():
            dev = trainer.device
            xb = torch.from_numpy(X[:2048]).float().to(dev)
            x0b = torch.from_numpy(X0[:2048]).float().to(dev)
            Ep = trainer.critic(xb).mean().item()
            Ep0 = trainer.critic(x0b).mean().item()
            updated_params = autotuner.step(
                trainer, trainer.gap_d_ma or gap_d, U2_ma, G2_ma
            )
        print(
            f"[{step:05d}] gap_c={gap_c:+.3e} gap_d={gap_d:+.3e} sob={sp:.2e} smooth={sm:.2e} "
            f"T_now={trainer.cfg.T:.2f} ord={trainer.cfg.order} Ep={Ep:+.2e} Ep0={Ep0:+.2e} u2_ma={U2_ma:.2e} g2_ma={G2_ma:.2e}"
        )
        print(f"param updates: {updated_params}")
end = time.time()
print(f"Elapsed time: {end - start:.2f} seconds")
# %%
potential, critic = trainer.potential.eval(), trainer.critic.eval()
# %%
# --- Visualization ---
X_all = np.vstack([X0, X])
mean, comps = pca_fit(X_all, n=2)
Z0 = pca_project(X0, mean, comps)
Z = pca_project(X, mean, comps)
# Simulation snapshots
N_plot = min(4000, X0.shape[0])
idx0_vis = np.random.choice(X0.shape[0], size=N_plot, replace=False)
X0_vis_t = (
    torch.from_numpy(X0[idx0_vis]).float().to(next(iter(potential.parameters())).device)
)
K_snap = 20
snaps = pushforward_snapshots(potential, X0_vis_t, T=T, K=K_snap)
plt.figure(figsize=(6, 6))
colors = cm.viridis(np.linspace(0, 1, len(snaps)))
for k, Yk in enumerate(snaps):
    Zk = pca_project(Yk.detach().cpu().numpy(), mean, comps)
    alpha = 0.12 if 0 < k < len(snaps) - 1 else 0.4
    plt.scatter(
        Zk[:, 0],
        Zk[:, 1],
        s=5,
        color=colors[k],
        alpha=alpha,
        label=(
            None if k not in (0, len(snaps) - 1) else ("t=0" if k == 0 else f"t={T}")
        ),
    )
idx_vis = np.random.choice(X.shape[0], size=N_plot, replace=False)
Z_vis = pca_project(X[idx_vis], mean, comps)
plt.scatter(Z_vis[:, 0], Z_vis[:, 1], s=5, color="grey", alpha=0.1, label="X (actual)")
plt.title("Forward pushforward snapshots (PCA plane)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
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

U_full = drift_batch(potential, G_full)
U2 = U_full @ comps.T
U2 = U2 / (np.linalg.norm(U2, axis=1, keepdims=True) + 1e-8) * 0.2

Utrue2 = None
if u_true_fn is not None:
    dev = next(iter(potential.parameters())).device
    with torch.no_grad():
        G_full_t = torch.from_numpy(G_full).float().to(dev)
        U_true_full = u_true_fn(G_full_t).detach().cpu().numpy()
    Utrue2 = U_true_full @ comps.T
    Utrue2 = Utrue2 / (np.linalg.norm(Utrue2, axis=1, keepdims=True) + 1e-8) * 0.2

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
if Utrue2 is not None:
    plt.quiver(
        G2[:, 0],
        G2[:, 1],
        Utrue2[:, 0],
        Utrue2[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        alpha=0.8,
        label="u_true",
    )
plt.legend()
plt.title("Projected drift field (PCA 2D) with X0 and X")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
# %%
# --- Field metrics if u_true_fn is available ---
if u_true_fn is not None:
    dev = next(iter(potential.parameters())).device
    pts = (
        torch.from_numpy(X[np.random.choice(X.shape[0], size=5000, replace=False)])
        .float()
        .to(dev)
    )
    u_hat = potential.drift(pts)
    u_true = u_true_fn(pts)
    mse = torch.mean((u_hat - u_true) ** 2).item()
    cos = torch.mean(
        torch.sum(u_hat * u_true, dim=1)
        / (u_hat.norm(dim=1) * u_true.norm(dim=1) + 1e-8)
    ).item()
    print(f"[metrics] MSE(u): {mse:.4e} | mean cos(u, u_true): {cos:.4f}")
# %%
# --- t*(x) inference demo (valid for D=0) ---
mu0_t = (
    torch.from_numpy(X0.mean(0)).float().to(next(iter(potential.parameters())).device)
)
X0_t = torch.from_numpy(X0).float().to(mu0_t.device)
cov0_t = torch.cov(X0_t.T) + 1e-6 * torch.eye(X0_t.size(1), device=mu0_t.device)
L0 = torch.linalg.cholesky(cov0_t)
inv_cov0 = torch.cholesky_inverse(L0)
const0 = -0.5 * X0_t.size(1) * math.log(2 * math.pi) - torch.log(torch.diag(L0)).sum()


def logp0_gaussian(x: torch.Tensor) -> torch.Tensor:
    xc = x - mu0_t
    return const0 - 0.5 * (xc * (xc @ inv_cov0)).sum(dim=1, keepdim=True)


@torch.no_grad()
def argmax_t_for_batch(
    potential: PotentialNet,
    logp0_fn,
    X_t: torch.Tensor,
    T: float,
    K: int = 64,
    n_probe: int = 2,
):
    K = int(K)
    ts = torch.linspace(0.0, float(T), steps=K + 1, device=X_t.device)
    vals = [
        logpt_reverse(
            potential, logp0_fn, X_t, float(tval.item()), steps=K, n_probe=n_probe
        )
        for tval in ts
    ]
    L = torch.cat(vals, dim=1)
    idx = torch.argmax(L, dim=1)
    t_star = ts[idx].unsqueeze(1)
    logp_star = L.gather(1, idx.unsqueeze(1))
    return t_star, logp_star, ts, L


K_GEN = 64
B_eval = min(8000, X.shape[0])
idx_eval = np.random.choice(X.shape[0], size=B_eval, replace=False)
X_eval_t = torch.from_numpy(X[idx_eval]).float().to(mu0_t.device)
_t_star, _logp_star, ts_grid, Lgrid = argmax_t_for_batch(
    potential, logp0_gaussian, X_eval_t, T=T, K=K_GEN, n_probe=2
)
t_star_np = _t_star.squeeze(1).cpu().numpy()
plt.figure(figsize=(6, 4))
plt.hist(t_star_np, bins=30, density=True, alpha=0.8)
plt.xlabel(r"$t^*(x)$")
plt.ylabel("density")
plt.title("Inferred latent times $t^*(x)$ for a subset of X")
plt.tight_layout()
plt.show()
# %%
Z_eval = pca_project(X[idx_eval], mean, comps)
plt.figure(figsize=(6, 6))
sc = plt.scatter(
    Z_eval[:, 0], Z_eval[:, 1], c=t_star_np, s=8, cmap="viridis", alpha=0.8
)
cb = plt.colorbar(sc)
cb.set_label(r"$t^*(x)$")
plt.title("PCA of X colored by inferred $t^*(x)$")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# %%
