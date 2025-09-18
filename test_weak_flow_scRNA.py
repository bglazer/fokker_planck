# %%
# === Environment ===
# pip install scanpy scvelo matplotlib torch numpy
import numpy as np
import scanpy as sc
import scvelo as scv
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# Your Weakflow implementation should be importable as a module `weakflow`
# (the canvas file you created). If it's a local file, ensure it's on PYTHONPATH.
from weak_flow import (
    Config,
    WeakFlowTrainer,
)

from weak_flow_util import (
    pca_fit,
    pca_project,
    pca_unproject,
    drift_batch,
    pushforward_snapshots,
    logpt_reverse,
)

# %%
scv.settings.verbosity = 2
scv.settings.presenter_view = True
sc.settings.set_figure_params(figsize=(5, 5), dpi=100)

# %%
# === 1) Load a widely used benchmark dataset (Dentate Gyrus) ===
# Two popular choices ship with scVelo:
#   - scv.datasets.dentategyrus()            (Hochgerner et al. 2018; P12/P35)
#   - scv.datasets.dentategyrus_lamanno()    (La Manno et al. 2018; P0/P5)
# We'll use the first (good quality + tutorial coverage).
adata = scv.datasets.dentategyrus()  # downloads an .h5ad with spliced/unspliced counts

# --- move this block UP, do it right after loading the dataset ---
# Keep a copy with all genes for marker scoring
adata_full = adata.copy()

# Normalize on the full gene set WITHOUT subsetting to HVGs
# (scanpy's simple log1p normalization is fine for scoring)
sc.pp.normalize_total(adata_full, target_sum=1e4)
sc.pp.log1p(adata_full)

# Improved, bias-corrected and smoothed marker scoring

# 1) Curate broader panels and match case-insensitively to the full gene set
full_idx_upper = {g.upper(): g for g in adata_full.var_names}

stem_markers = [
    "SOX2",
    "NES",
    "HOPX",
    "EGFR",
    "VIM",
    "FABP7",
    "PROM1",
    "MSI1",
    "PAX6",
    "HES1",
    "ID3",
    "ID4",
]
nb_markers = ["DCX", "PROX1", "NEUROD1", "TUBB3", "STMN2", "MAP2", "RBFOX3", "CALB2"]


def present(markers):
    return [
        full_idx_upper[g] for g in (m.upper() for m in markers) if g in full_idx_upper
    ]


stem_present_full = present(stem_markers)
nb_present_full = present(nb_markers)


# 2) Score with Scanpy's score_genes (controls for expression level via matched control genes)
def score_or_zeros(adata, genes, name):
    if len(genes) >= 3:
        sc.tl.score_genes(
            adata, gene_list=genes, score_name=name, ctrl_size=50, use_raw=False
        )
    elif len(genes) > 0:
        sc.tl.score_genes(
            adata,
            gene_list=genes,
            score_name=name,
            ctrl_size=min(10, len(adata.var_names) // 50),
            use_raw=False,
        )
    else:
        adata.obs[name] = 0.0


score_or_zeros(adata_full, stem_present_full, "stem_score_raw")
score_or_zeros(adata_full, nb_present_full, "nb_score_raw")

# 3) Smooth scores on kNN graph to reduce sparsity/noise
if "connectivities" not in adata_full.obsp:
    sc.pp.pca(adata_full, n_comps=min(50, adata_full.n_vars - 1))
    sc.pp.neighbors(
        adata_full, n_neighbors=30, n_pcs=min(30, adata_full.obsm["X_pca"].shape[1])
    )

C = adata_full.obsp["connectivities"]
deg = np.asarray(C.sum(1)).ravel()

stem_vec = adata_full.obs["stem_score_raw"].to_numpy(dtype=np.float32)
nb_vec = adata_full.obs["nb_score_raw"].to_numpy(dtype=np.float32)

# add a self-loop implicitly by +score and +1 to degree
stem_sm = (C @ stem_vec + stem_vec) / (deg + 1.0 + 1e-8)
nb_sm = (C @ nb_vec + nb_vec) / (deg + 1.0 + 1e-8)


# 4) Z-normalize and combine into a single progenitor-vs-neuroblast axis
def zscore(x):
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)


stem_z = zscore(stem_sm)
nb_z = zscore(nb_sm)

# Higher = more stem/progenitor-like; lower = more neuroblast/neuronal-like
stem_minus_nb = (stem_z - nb_z).astype(np.float32)
# %%
q = 0.80
thr = np.quantile(stem_minus_nb, q)
# %%
X0_mask = stem_minus_nb >= thr
# %%
# Put the mask back on the working AnnData (same cells, same order)
adata.obs["weakflow_X0"] = X0_mask
print(
    f"[marker scoring] {X0_mask.sum()} X0 cells | stem markers present: {stem_present_full} | nb markers present: {nb_present_full}"
)

# --- then continue with your existing scVelo preprocessing on `adata` ---
scv.pp.filter_and_normalize(adata, min_shared_cells=20, n_top_genes=2000)
scv.pp.moments(adata, n_neighbors=30, n_pcs=30)

print(
    f"Selected {X0_mask.sum()} / {adata.n_obs} cells as X0 (top {int(q*100)}% by stem-minus-neuroblast score)."
)
# %%
# Optional: visualize on UMAP with colors
sc.pp.neighbors(adata, use_rep="X_pca")  # graph for UMAP
sc.tl.umap(adata, min_dist=0.5)
sc.pl.umap(adata, color=["weakflow_X0"], wspace=0.5, s=3)

# %%
# === 3) Build features for Weakflow and standardize ===
# We’ll use the first 50 PCs as coordinates (d = 50). You can try HVG counts as well,
# but PCs are a strong, denoised Euclidean representation that many trajectory tools use.
Z = adata.obsm["X_pca"].astype(np.float32)  # (cells, 50)
Z_ = Z[~X0_mask, :]
Z0 = Z[X0_mask, :]
Z = Z_
d = Z.shape[1]

# Z-score standardization (no whitening)
mu = Z.mean(axis=0, keepdims=True).astype(np.float32)
std = Z.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
Z_std = (Z - mu) / std

# Partition: X0 = progenitor pool; X = whole snapshot (mixture over latent times)
# X0_np = Z_std[X0_mask, :]
# X_np = Z_std

# print("Shapes | X0:", X0_np.shape, " X:", X_np.shape)

# %%
# === 4) Train Weakflow on (X0, X) ===
# Choose a conservative horizon T and order=1 to keep the weak surrogate stable.
from weak_flow_util import ZScoreStandardizer, AutoTuner

autotuner = AutoTuner(
    u2_band=(5, 25),
    g2_band=(1, 10),
    w_u_bounds=(1e-5, 1),
    w_s_bounds=(1e-2, 1e2),
    gap_target=1,
)

# Standardize X and X0
standardizer = ZScoreStandardizer().fit(Z0, Z)
X0 = standardizer.transform(Z0)
X = standardizer.transform(Z)

# %%
# --- trainer config ---
cfg = Config(
    d=d,
    T=.3,
    order=1,
    steps=500,
    batch_size=512,
    n_critic=1,
    lr_potential=5e-4,
    lr_critic=1e-4,
    sobolev_weight=.2,
    drift_l2_weight=1e-3,
    critic_width=32,
    critic_depth=2,
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
# Training
start = time.time()
cfg.steps = 500

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
# === 5) Visualize the learned drift on the PCA plane ===
# We'll project to the first two PCs (which are already part of Z) and plot a quiver.
# To cover the full scatter extent, we grid across the convex hull-ish quantiles.

# Recompute PCA fit only to get a convenience projector (identity on the first 2 PCs in this case).
# If you later switch features (e.g., HVGs), these helpers still work.
mean_2d, comps_2d = pca_fit(Z, n=2)
Z2 = pca_project(Z, mean_2d, comps_2d)
Z02 = pca_project(Z0, mean_2d, comps_2d)
Z2_all = Z2

nq = 35
marg = 0.00
lo = np.quantile(Z2_all, q=marg, axis=0)
hi = np.quantile(Z2_all, q=1 - marg, axis=0)
pad = 0.05 * (hi - lo)
lo -= pad
hi += pad
gx, gy = np.linspace(lo[0], hi[0], nq), np.linspace(lo[1], hi[1], nq)
GX, GY = np.meshgrid(gx, gy)
G2 = np.stack([GX.ravel(), GY.ravel()], axis=1)
# Back to the model feature space (50D) along the 2D PCA plane
G_full = pca_unproject(G2, mean_2d, comps_2d)  # still in 50D PCA space
# Standardize as the model expects
G_full_std = (G_full - mu) / std

# Evaluate drift u on the grid (in standardized 50D space), then project arrows back to the 2D plane
U_full = drift_batch(potential, G_full_std)  # (nq^2, 50)
U2 = U_full @ comps_2d.T
U2 = U2 / (np.linalg.norm(U2, axis=1, keepdims=True) + 1e-8) * 0.25  # scale arrows

plt.figure(figsize=(10, 10))
idx = np.random.choice(Z2.shape[0], size=min(6000, Z2.shape[0]), replace=False)
plt.scatter(Z2[idx, 0], Z2[idx, 1], s=4, alpha=0.25, label="all cells")
plt.scatter(Z02[:, 0], Z02[:, 1], s=6, alpha=0.6, label="X0 (stem/prog)")
plt.quiver(
    G2[:, 0], G2[:, 1], U2[:, 0], U2[:, 1], angles="xy", scale_units="xy", scale=1
)
plt.legend()
plt.title("Weakflow drift on DG PCs (quiver in PC1–PC2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()
# %%
# Simulate pushforward
# --- Forward simulation and overlay on observed data (PC1–PC2 plane) ---

# Use current (possibly autotuned) horizon
T_sim = 1
K_snap = 50  # number of snapshot slices (inclusive of t=0 and t=T_sim)
N_seed = min(4000, X0.shape[0])
device = next(potential.parameters()).device

# Seed trajectories from inferred X0 (standardized model space)
idx0 = np.random.choice(X0.shape[0], size=N_seed, replace=False)
X0_seed_t = torch.from_numpy(X0[idx0]).float().to(device)

# Simulate pushforward snapshots in standardized space
snaps = pushforward_snapshots(potential, X0_seed_t, T=T_sim, K=K_snap)
# snaps is a list of length K_snap+1 (if implemented that way) or K_snap; treat generically
# We will map each snapshot back to original (unstandardized) PCA space for plotting.

# We already have:
#  - mu, std (broadcast 1xd) to unstandardize
#  - mean_2d, comps_2d from earlier pca_fit on raw Z (unstandardized 50D PCs)
#  - Z2 (or Z2_all) = projected real cells on first two PCs

plt.figure(figsize=(6, 6))

# Background: real observed cells (subsample for speed)
bg_n = min(6000, Z2.shape[0])
bg_idx = np.random.choice(Z2.shape[0], size=bg_n, replace=False)
plt.scatter(
    Z2[bg_idx, 0], Z2[bg_idx, 1], s=4, color="lightgrey", alpha=0.6, label="Observed"
)

colors = cm.viridis(np.linspace(0, 1, len(snaps)))
for k, Yk in enumerate(snaps):
    Yk_np = Yk.detach().cpu().numpy()  # standardized
    Yk_unstd = Yk_np * std + mu  # back to original 50D PCA coordinates
    Zk2 = pca_project(Yk_unstd, mean_2d, comps_2d)
    alpha = 0.10 if 0 < k < len(snaps) - 1 else 0.45
    lbl = None
    if k == 0:
        lbl = "t=0"
    elif k == len(snaps) - 1:
        lbl = f"t={T_sim:.2f}"
    plt.scatter(Zk2[:, 0], Zk2[:, 1], s=5, color=colors[k], alpha=alpha, label=lbl)

plt.title("Weakflow forward pushforward (seeded from X0)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

