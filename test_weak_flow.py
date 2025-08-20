# %%
import numpy as np
import torch
from weak_flow import WeakFlowTrainer, Config  # from the file you already have
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from weak_flow import PotentialNet, pushforward_snapshots, logpt_reverse
import math
import time

# %%
# ==== Visual diagnostics for weak-form training (no long ODE runs) ====
# --- helpers ---
def pca_fit(X, n=2):
    Xc = X - X.mean(0, keepdims=True)
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    comps = VT[:n, :]  # (n, d)
    mean = X.mean(0)
    return mean, comps  # row-wise components


def pca_project(X, mean, comps):
    return (X - mean) @ comps.T  # (N, n)


def pca_unproject(Z, mean, comps):
    # Z: (..., n) -> (..., d)
    return Z @ comps + mean


@torch.no_grad()
def drift_batch(potential, X, device=None, bs=4096, freeze_params=True):
    """
    Evaluate u(x) = ∇_x phi_theta(x) in batches.
    Keeps grad enabled w.r.t. x; optionally freezes params so no param grads are tracked.
    """
    device = device or next(iter(potential.parameters())).device
    was_training = potential.training
    potential.eval()

    # Optionally freeze parameters to avoid tracking grads w.r.t. theta
    param_flags = None
    if freeze_params:
        param_flags = [p.requires_grad for p in potential.parameters()]
        for p in potential.parameters():
            p.requires_grad_(False)

    out = []
    # IMPORTANT: do NOT wrap this in torch.no_grad(); we need grads w.r.t. x
    with torch.set_grad_enabled(True):
        for i in range(0, X.shape[0], bs):
            xb = torch.from_numpy(X[i : i + bs]).float().to(device)
            xb.requires_grad_(True)
            phi = potential(xb)  # (B,1)
            u = torch.autograd.grad(phi.sum(), xb)[0]  # (B,d)
            out.append(u.detach().cpu().numpy())

    # Restore parameter flags if we changed them
    if freeze_params and param_flags is not None:
        for p, flag in zip(potential.parameters(), param_flags):
            p.requires_grad_(flag)

    if was_training:
        potential.train()

    return np.concatenate(out, axis=0)


def L_f_pointwise(critic, potential, X, device=None, bs=4096):
    device = device or next(iter(potential.parameters())).device
    vals = []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i : i + bs]).float().to(device)
        xb.requires_grad_(True)
        f = critic(xb)  # (B,1)
        gradf = torch.autograd.grad(f.sum(), xb, create_graph=False)[0]  # (B,d)
        xb2 = xb.clone().detach().requires_grad_(True)
        phi = potential(xb2)
        u = torch.autograd.grad(phi.sum(), xb2, create_graph=False)[0]
        vals.append((u * gradf).sum(1, keepdim=True).detach().cpu().numpy())
    return np.concatenate(vals, axis=0)  # (N,1)


def L2_f_pointwise(critic, potential, X, device=None, bs=2048):
    device = device or next(iter(potential.parameters())).device
    vals = []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i : i + bs]).float().to(device)
        xb.requires_grad_(True)
        # L f
        f = critic(xb)
        gradf = torch.autograd.grad(f.sum(), xb, create_graph=True)[0]
        phi = potential(xb)
        u = torch.autograd.grad(phi.sum(), xb, create_graph=True)[0]
        L1 = (u * gradf).sum(1, keepdim=True)
        # ∇(L1) · u
        gradL1 = torch.autograd.grad(L1.sum(), xb, create_graph=False)[0]
        vals.append((u * gradL1).sum(1, keepdim=True).detach().cpu().numpy())
    return np.concatenate(vals, axis=0)


def RT_apply_numpy(critic, potential, X, T, order=1, device=None, bs=4096):
    device = device or next(iter(potential.parameters())).device
    out = []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i : i + bs]).float().to(device)
        fb = critic(xb).detach().cpu().numpy()
        if order >= 1:
            L1 = L_f_pointwise(critic, potential, X[i : i + bs], device=device, bs=bs)
            fb = fb + 0.5 * T * L1
        if order >= 2:
            L2 = L2_f_pointwise(
                critic, potential, X[i : i + bs], device=device, bs=max(1024, bs // 2)
            )
            fb = fb + (T**2 / 6.0) * L2
        out.append(fb)
    return np.concatenate(out, axis=0)  # (N,1)


def make_linear_translation_data(
    N0=20000, N=20000, d=2, T=100, sigma=1.0, discrete=True, shift_per_unit=1.0, seed=0
):
    """
    p0 = N(0, sigma^2 I_d)
    For t in [0,T], p_t = N(t * v, sigma^2 I_d), where v = shift_per_unit * e1.
    If discrete=True, t is uniform on {0,1,...,T}; else t ~ Uniform[0,T] (continuous).
    Returns X0 ~ p0, X ~ time-mixture of {p_t}.
    """
    rng = np.random.default_rng(seed)
    v = np.zeros(d, dtype=np.float32)
    v[0] = float(shift_per_unit)

    # Initial samples
    X0 = rng.normal(0.0, sigma, size=(N0, d)).astype(np.float32)

    # Time samples
    if discrete:
        ts = rng.integers(low=0, high=T + 1, size=N).astype(np.float32)  # {0,...,T}
    else:
        ts = rng.random(N).astype(np.float32) * float(T)  # [0, T)

    # Mixture samples
    eps = rng.normal(0.0, sigma, size=(N, d)).astype(np.float32)
    X = ts[:, None] * v[None, :] + eps

    # Quick sanity checks
    E_X0 = X0.mean(0)
    E_X = X.mean(0)
    # Theoretical mean for the mixture along v-direction:
    if discrete:
        E_t = T / 2.0
    else:
        E_t = T / 2.0
    print(
        f"[sanity] mean(X0)[0]={E_X0[0]:+.3f}, mean(X)[0]={E_X[0]:+.3f}, expected ≈ {E_t*shift_per_unit:+.3f}"
    )
    return X0, X, v


# %%
# --- generate data ---
d = 500
T = 4
X0, X, v_true = make_linear_translation_data(
    N0=20000, N=40000, d=d, T=T, sigma=1.0, discrete=False, shift_per_unit=1.0, seed=42
)
# %%
# --- trainer config ---
cfg = Config(
    d=d,
    T=T,
    order=1,
    steps=500,
    batch_size=1024,
    n_critic=1,
    lr_potential=5e-4,
    lr_critic=1e-4,
    sobolev_weight=5e-2,
    drift_l2_weight=1e-4,
    critic_width=128,
    critic_depth=3,
    spectral_norm_critic=True,
    potential_width=256,
    potential_depth=4,
    mixed_precision=False,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    log_every=50,
)
trainer = WeakFlowTrainer(cfg, X0, X)
# %%
# Training
# T-continuation
import time
start = time.time()
T_target = cfg.T
trainer.cfg.T = min(0.5, 0.25 * T_target)
for step in range(1, cfg.steps + 1):
    gap_c, sp = trainer.critic_step()
    gap_d, sm, loss_d = trainer.drift_step()
    if step % 500 == 0:
        trainer.cfg.T = min(T_target, trainer.cfg.T * 1.25)
    if step % cfg.log_every == 0:
        with torch.no_grad():
            dev = trainer.device
            xb = torch.from_numpy(X[:2048]).float().to(dev)
            x0b = torch.from_numpy(X0[:2048]).float().to(dev)
            Ep = trainer.critic(xb).mean().item()
            Ep0 = trainer.critic(x0b).mean().item()
        print(
            f"[{step:05d}] gap_c={gap_c:+.3e} gap_d={gap_d:+.3e} sob={sp:.2e} smooth={sm:.2e} "
            f"T_now={trainer.cfg.T:.2f} ord={trainer.cfg.order} Ep={Ep:+.2e} Ep0={Ep0:+.2e}"
        )
end = time.time()
print(f"Training completed in {end - start:.2f} seconds")
# %%
potential, critic = trainer.potential.eval(), trainer.critic.eval()
#%%
# --- 1) Vector field in PCA plane ---
X_all = np.vstack([X0, X])
mean, comps = pca_fit(X_all, n=2)
Z0 = pca_project(X0, mean, comps)
Z = pca_project(X, mean, comps)
nq = 25
marg = 0.05
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
plt.figure(figsize=(6, 6))
idx0 = np.random.choice(Z0.shape[0], size=min(5000, Z0.shape[0]), replace=False)
idx = np.random.choice(Z.shape[0], size=min(5000, Z.shape[0]), replace=False)
plt.scatter(Z0[idx0, 0], Z0[idx0, 1], s=5, alpha=0.35, label="X0 (p0)")
plt.scatter(Z[idx, 0], Z[idx, 1], s=5, alpha=0.35, label="X (mixture)")
plt.quiver(
    G2[:, 0], G2[:, 1], U2[:, 0], U2[:, 1], angles="xy", scale_units="xy", scale=1
)
plt.legend()
plt.title("Projected drift field (PCA 2D) with X0 and X")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
#%%
# --- 2) Forward simulation snapshots ---

N_plot = min(400, X0.shape[0])
idx0_vis = np.random.choice(X0.shape[0], size=N_plot, replace=False)
X0_vis_t = (
    torch.from_numpy(X0[idx0_vis]).float().to(next(iter(potential.parameters())).device)
)
K = 20
snaps = pushforward_snapshots(potential, X0_vis_t, T=T, K=K, steps_per_unit=40)
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
        label=None if k not in (0, len(snaps) - 1) else ("t=0" if k == 0 else f"t={T}"),
    )
# use ScalarMappable for the colorbar without extra scatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sm = ScalarMappable(cmap="viridis", norm=Normalize(vmin=0, vmax=len(snaps)-1))
sm.set_array([])
# cbar = plt.colorbar(sm, shrink=0.8)
# cbar.set_label("time slice index (0→T)")
plt.title("Forward pushforward snapshots (PCA plane)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
#%%
# --- 3) Inference of t*(x) = argmax_t log p_t(x) ---
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
    n_grid: int = 64,
    steps: int = 80,
    n_probe: int = 2,
    refine: bool = True,
):
    ts = torch.linspace(0.0, float(T), steps=n_grid, device=X_t.device)
    vals = [
        logpt_reverse(
            potential, logp0_fn, X_t, float(tval.item()), steps=steps, n_probe=n_probe
        )
        for tval in ts
    ]
    L = torch.cat(vals, dim=1)
    idx = torch.argmax(L, dim=1)
    t_star = ts[idx].unsqueeze(1)
    logp_star = L.gather(1, idx.unsqueeze(1))
    if refine and n_grid >= 3:
        i0 = torch.clamp(idx - 1, 0, n_grid - 1)
        i2 = torch.clamp(idx + 1, 0, n_grid - 1)
        need = (i2 - i0) == 2
        if need.any():
            t1 = ts[i0][need]
            t2 = ts[idx][need]
            t3 = ts[i2][need]
            L1 = L[need, i0[need]]
            L2 = L[need, idx[need]]
            L3 = L[need, i2[need]]
            denom = t1 - 2 * t2 + t3
            denom[denom.abs() < 1e-12] = 1e-12
            t_vert = (t2 + 0.5 * ((t1 - t3) * (L1 - L3)) / denom).clamp(t1, t3)
            lp_ref = logpt_reverse(
                potential, logp0_fn, X_t[need], t_vert, steps=steps, n_probe=n_probe
            )
            t_star[need] = t_vert.unsqueeze(1)
            logp_star[need] = lp_ref
    return t_star, logp_star, ts, L


B_eval = min(8000, X.shape[0])
idx_eval = np.random.choice(X.shape[0], size=B_eval, replace=False)
X_eval_t = torch.from_numpy(X[idx_eval]).float().to(mu0_t.device)
_t_star, _logp_star, ts_grid, Lgrid = argmax_t_for_batch(
    potential, logp0_gaussian, X_eval_t, T=T, n_grid=64, steps=80, n_probe=2
)
t_star_np = _t_star.squeeze(1).cpu().numpy()
plt.figure(figsize=(6, 4))
plt.hist(t_star_np, bins=30, density=True, alpha=0.8)
plt.xlabel(r"$t^*(x)$")
plt.ylabel("density")
plt.title("Inferred latent times $t^*(x)$ for a subset of X")
plt.tight_layout()
plt.show()
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
