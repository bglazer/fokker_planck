#%%
import numpy as np

def make_linear_translation_data(
    N0=20000, N=20000, d=2, T=4, sigma=1.0,
    discrete=True, shift_per_unit=1.0, seed=0
):
    """
    p0 = N(0, sigma^2 I_d)
    For t in [0,T], p_t = N(t * v, sigma^2 I_d), where v = shift_per_unit * e1.
    If discrete=True, t is uniform on {0,1,...,T}; else t ~ Uniform[0,T] (continuous).
    Returns X0 ~ p0, X ~ time-mixture of {p_t}.
    """
    rng = np.random.default_rng(seed)
    v = np.zeros(d, dtype=np.float32); v[0] = float(shift_per_unit)

    # Initial samples
    X0 = rng.normal(0.0, sigma, size=(N0, d)).astype(np.float32)

    # Time samples
    if discrete:
        ts = rng.integers(low=0, high=T+1, size=N).astype(np.float32)  # {0,...,T}
    else:
        ts = rng.random(N).astype(np.float32) * float(T)               # [0, T)

    # Mixture samples
    eps = rng.normal(0.0, sigma, size=(N, d)).astype(np.float32)
    X  = ts[:, None] * v[None, :] + eps

    # Quick sanity checks
    E_X0 = X0.mean(0)
    E_X  = X.mean(0)
    # Theoretical mean for the mixture along v-direction:
    if discrete:
        E_t = T/2.0
    else:
        E_t = T/2.0
    print(f"[sanity] mean(X0)[0]={E_X0[0]:+.3f}, mean(X)[0]={E_X[0]:+.3f}, expected ≈ {E_t*shift_per_unit:+.3f}")
    return X0, X, v

#%%
import torch
from weak_flow import WeakFlowTrainer, Config  # from the file you already have

# --- generate data ---
d   = 2         # you can set higher, e.g., 50 or 1000; plotting uses PCA anyway
T   = 4         # try 2–4 for order=1; or use order=2 later
X0, X, v_true = make_linear_translation_data(N0=20000, N=40000, d=d, T=T,
                                             sigma=1.0, discrete=True, shift_per_unit=1.0, seed=42)

# --- trainer with safer defaults for this adversarial fit ---
cfg = Config(
    d=d, T=T, order=1,
    steps=8000, batch_size=1024,
    n_critic=1,                     # let drift keep up
    lr_potential=5e-4, lr_critic=1e-4,
    sobolev_weight=5e-2,            # regularize critic a bit more
    drift_l2_weight=1e-4,           # lighter smoothing at start
    critic_width=128, critic_depth=3, spectral_norm_critic=True,
    potential_width=256, potential_depth=4,
    mixed_precision=False, device=("cuda" if torch.cuda.is_available() else "cpu"),
    log_every=200
)

trainer = WeakFlowTrainer(cfg, X0, X)
# T-continuation: start smaller, ramp up
T_target = cfg.T
trainer.cfg.T = min(0.5, 0.25*T_target)
for step in range(1, cfg.steps + 1):
    # critic
    gap_c, sp = trainer.critic_step()
    # drift
    gap_d, sm, loss_d = trainer.drift_step()

    # ramp T and optionally switch to order=2 halfway
    if step % 500 == 0:
        trainer.cfg.T = min(T_target, trainer.cfg.T * 1.25)
    if step == 4000:
        trainer.cfg.order = 2

    if step % cfg.log_every == 0:
        with torch.no_grad():
            # tiny debug batch
            dev = trainer.device
            xb  = torch.from_numpy(X[:2048]).float().to(dev)
            x0b = torch.from_numpy(X0[:2048]).float().to(dev)
            Ep  = trainer.critic(xb).mean().item()
            Ep0 = trainer.critic(x0b).mean().item()
        print(f"[{step:05d}] gap_c={gap_c:+.3e} gap_d={gap_d:+.3e} sob={sp:.2e} smooth={sm:.2e} "
              f"T_now={trainer.cfg.T:.2f} ord={trainer.cfg.order} Ep={Ep:+.2e} Ep0={Ep0:+.2e}")

print("Done.")
potential, critic = trainer.potential.eval(), trainer.critic.eval()

# Use potential.drift(torch.tensor([...])) to evaluate u(x)
#%%
# ==== Visual diagnostics for weak-form training (no long ODE runs) ====
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- helpers ---
def pca_fit(X, n=2):
    Xc = X - X.mean(0, keepdims=True)
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    comps = VT[:n, :]           # (n, d)
    mean = X.mean(0)
    return mean, comps          # row-wise components

def pca_project(X, mean, comps):
    return (X - mean) @ comps.T # (N, n)

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
            xb = torch.from_numpy(X[i:i+bs]).float().to(device)
            xb.requires_grad_(True)
            phi = potential(xb)                                # (B,1)
            u = torch.autograd.grad(phi.sum(), xb)[0]          # (B,d)
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
        xb = torch.from_numpy(X[i:i+bs]).float().to(device)
        xb.requires_grad_(True)
        f = critic(xb)                       # (B,1)
        gradf = torch.autograd.grad(f.sum(), xb, create_graph=False)[0]  # (B,d)
        xb2 = xb.clone().detach().requires_grad_(True)
        phi = potential(xb2)
        u = torch.autograd.grad(phi.sum(), xb2, create_graph=False)[0]
        vals.append((u*gradf).sum(1, keepdim=True).detach().cpu().numpy())
    return np.concatenate(vals, axis=0)  # (N,1)

def L2_f_pointwise(critic, potential, X, device=None, bs=2048):
    device = device or next(iter(potential.parameters())).device
    vals = []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(device)
        xb.requires_grad_(True)
        # L f
        f = critic(xb)
        gradf = torch.autograd.grad(f.sum(), xb, create_graph=True)[0]
        phi = potential(xb)
        u = torch.autograd.grad(phi.sum(), xb, create_graph=True)[0]
        L1 = (u*gradf).sum(1, keepdim=True)
        # ∇(L1) · u
        gradL1 = torch.autograd.grad(L1.sum(), xb, create_graph=False)[0]
        vals.append((u*gradL1).sum(1, keepdim=True).detach().cpu().numpy())
    return np.concatenate(vals, axis=0)

def RT_apply_numpy(critic, potential, X, T, order=1, device=None, bs=4096):
    device = device or next(iter(potential.parameters())).device
    out = []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(device)
        fb = critic(xb).detach().cpu().numpy()
        if order >= 1:
            L1 = L_f_pointwise(critic, potential, X[i:i+bs], device=device, bs=bs)
            fb = fb + 0.5*T*L1
        if order >= 2:
            L2 = L2_f_pointwise(critic, potential, X[i:i+bs], device=device, bs=max(1024, bs//2))
            fb = fb + (T**2/6.0)*L2
        out.append(fb)
    return np.concatenate(out, axis=0)  # (N,1)

#%%
# --- prepare data/projection ---
X_all = np.vstack([X0, X])
mean, comps = pca_fit(X_all, n=2)
Z0 = pca_project(X0, mean, comps)   # (N0,2)
Z  = pca_project(X,  mean, comps)   # (N,2)
#%%
# --- 1) Vector field in PCA plane ---
# grid in 2D PCA, then unproject to full space to evaluate u, re-project arrows
nq = 25
marg = 0.05
mins = np.quantile(Z,  q=marg, axis=0)
maxs = np.quantile(Z,  q=1-marg, axis=0)
gx = np.linspace(mins[0], maxs[0], nq)
gy = np.linspace(mins[1], maxs[1], nq)
GX, GY = np.meshgrid(gx, gy)
G2 = np.stack([GX.ravel(), GY.ravel()], axis=1)           # (nq^2, 2)
G_full = pca_unproject(G2, mean, comps)                   # (nq^2, d)

U_full = drift_batch(potential, G_full)                   # (nq^2, d)
# project drift to PCA plane: u2 = (U_full) @ comps.T (since comps rows are PC axes)
U2 = U_full @ comps.T                                     # (nq^2, 2)
U2 = U2 / (np.linalg.norm(U2, axis=1, keepdims=True) + 1e-8) * 0.2  # scale arrows

plt.figure(figsize=(6,6))
idx0 = np.random.choice(Z0.shape[0], size=min(5000, Z0.shape[0]), replace=False)
idx  = np.random.choice(Z.shape[0],  size=min(5000,  Z.shape[0]),  replace=False)
plt.scatter(Z0[idx0,0], Z0[idx0,1], s=5, alpha=0.35, label="X0 (p0)")
plt.scatter(Z[idx,0],   Z[idx,1],   s=5, alpha=0.35, label="X (mixture)")
plt.quiver(G2[:,0], G2[:,1], U2[:,0], U2[:,1], angles='xy', scale_units='xy', scale=1)
plt.legend()
plt.title("Projected drift field (PCA 2D) with X0 and X")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.show()

#%%
# --- 2) Critic agreement: f(X) vs R_T f(X0) ---
with torch.no_grad():
    f_X  = critic(torch.from_numpy(X).float().to(next(iter(potential.parameters())).device)).cpu().numpy()
RTf_X0 = RT_apply_numpy(critic, potential, X0, T, order=order)

plt.figure(figsize=(6,4))
plt.hist(f_X.ravel(),  bins=64, density=True, alpha=0.6, label="f(X)   (p)")
plt.hist(RTf_X0.ravel(), bins=64, density=True, alpha=0.6, label="R_T f(X0) (p0)")
plt.title("Critic distributions: target vs. pushed-forward")
plt.xlabel("value"); plt.ylabel("density")
plt.legend(); plt.tight_layout(); plt.show()
#%%
# Optional: QQ-style plot to see alignment
q = np.linspace(0.01, 0.99, 99)
q1 = np.quantile(f_X.ravel(), q)
q2 = np.quantile(RTf_X0.ravel(), q)
plt.figure(figsize=(4,4))
plt.plot(q1, q2, lw=2)
lo = min(q1.min(), q2.min()); hi = max(q1.max(), q2.max())
plt.plot([lo,hi],[lo,hi],'--', lw=1)
plt.title("QQ plot: f(X) vs R_T f(X0)")
plt.xlabel("Quantiles of f(X)"); plt.ylabel("Quantiles of R_T f(X0)")
plt.tight_layout(); plt.show()
#%%
# --- 3) Potential contours in PCA plane (only if u=∇phi) ---
# Evaluate phi on the same grid (via unprojected points)
with torch.no_grad():
    phi_grid = []
    dev = next(iter(potential.parameters())).device
    for i in range(0, G_full.shape[0], 4096):
        xb = torch.from_numpy(G_full[i:i+4096]).float().to(dev)
        phi_grid.append(potential(xb).cpu().numpy())
    PHI = np.concatenate(phi_grid, axis=0).reshape(nq, nq)

plt.figure(figsize=(6,6))
plt.contourf(GX, GY, PHI, levels=20, alpha=0.8)
plt.quiver(G2[:,0], G2[:,1], U2[:,0], U2[:,1], color='k', angles='xy', scale_units='xy', scale=1, width=0.003)
plt.title("Potential (contours) and projected drift (arrows)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.show()


# %%
