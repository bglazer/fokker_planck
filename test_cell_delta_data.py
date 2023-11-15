#%%
%load_ext autoreload
%autoreload 2
#%%
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import scanpy as sc
import torch.distributions as D
from celldelta import CellDelta
from sklearn.decomposition import PCA

#%%
device = 'cuda:1'

#%%
genotype='wildtype'
dataset = 'net'
adata = sc.read_h5ad(f'{genotype}_{dataset}.h5ad')
nmp_cell_mask = adata.obs['cell_type'] == 'NMP'
n_genes = 10
X = adata.X.toarray()
X = torch.tensor(X, device=device, dtype=torch.float32)
# Get a boolean mask for the top n_genes by variance
top_var_genes = torch.argsort(X.var(dim=0))[-n_genes:]
top_var_gene_mask = torch.zeros(X.shape[1], dtype=torch.bool)
top_var_gene_mask[top_var_genes] = True

X0 = X[nmp_cell_mask,:][:,top_var_gene_mask].clone().detach()
X = X[:,top_var_gene_mask].clone().detach()

#%%
ts = torch.linspace(0, 1, 100, device=device)*100
ts.requires_grad = True
n_samples = 10_000
zero = torch.zeros(1, requires_grad=True).to(device)

#%%
# Initialize the model
# TODO add weight decay or dropout to the model
celldelta = CellDelta(input_dim=n_genes, 
                      ux_hidden_dim=64, ux_layers=2,
                      pxt_hidden_dim=64, pxt_layers=2,
                      device=device)

#%%
# Train the model
losses = celldelta.optimize(X, X0, ts, pxt_lr=1e-3, ux_lr=1e-3, 
                            n_epochs=1500, n_samples=n_samples, 
                            mcmc_steps=1, mcmc_step_size=.5,
                            verbose=True)
#%%
l_fps = losses['l_fp']
l_self_pxs = losses['l_self_px']
l_self_p0s = losses['l_self_p0']

fig, axs = plt.subplots(3, 1, figsize=(10,10))
start_skip = 10
axs[0].plot(l_fps[start_skip:], label='l_fp')
axs[1].plot(l_self_pxs[start_skip:], label='l_self_pxs')
axs[2].plot(l_self_p0s[start_skip:], label='l_self_p0')
[axs[i].set_xlabel('Epoch') for i in range(len(axs))]
[axs[i].set_ylabel('Loss') for i in range(len(axs))]
[axs[i].legend() for i in range(len(axs))]
fig.suptitle('Loss curves')
fig.tight_layout()

# %%
xs = X.clone().detach()
pxts = celldelta.pxt(xs, ts).squeeze().T.cpu().detach().numpy()
uxs = celldelta.ux(xs).squeeze().cpu().detach().numpy()
_, pxt_dts = celldelta.pxt.dx_dt(xs, ts)
pxt_dts = pxt_dts.detach().cpu().numpy()[:,:,0]

xs = xs.squeeze().cpu().detach().numpy()

viridis = matplotlib.colormaps.get_cmap('viridis')
#%%
# Setup the PCA using the saved components and mean
pca = PCA()
# pca.components_ = adata.uns['PCs']
# pca.mean_ = adata.uns['pca_mean']
x_proj = pca.fit_transform(X.cpu().numpy())
x0_proj = pca.transform(X0.cpu().numpy())

#%%
ts_np = ts.detach().cpu().numpy()
# Plot the pseudotime i.e. the timestep of maximum probability for each cell
pxt_norm = pxts
plt.scatter(x_proj[:,0], x_proj[:,1], c=ts_np[pxts.argmax(axis=1)], cmap=viridis, s=1)
plt.colorbar()

# Plot the predicted p(x,t) for each cell at each timestep t
fig, axs = plt.subplots(5,1, figsize=(5,20)) 
for i in range(5):
    t = pxts.shape[1]//5 * i
    axs[i].set_title(f't={ts_np[t]:.2f}')
    axs[i].scatter(x_proj[:,0], x_proj[:,1], c=pxts[:,t], cmap=viridis, s=1, vmin=pxts.min(), vmax=pxts.max())
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    plt.colorbar(axs[i].collections[0], ax=axs[i])
plt.tight_layout()
# Mark the initial points with a black triangle
x0_proj = pca.transform(X0.cpu().numpy())
plt.scatter(x0_proj[:,0], x0_proj[:,1], color='black', alpha=.5, s=9.5, marker='^')

# %%
# Simulate the stochastic differential equation using the Euler-Maruyama method
# with the learned drift term u(x)
x = X0.clone().detach()
tsim = torch.linspace(0, 1, 100, device=device, requires_grad=False)
zero_boundary = True

xts = celldelta.simulate(x, tsim, zero_boundary=zero_boundary)

#%%
# Plot trajectory of a single cell
fig, axs = plt.subplots(3, 3, figsize=(10,10))
tsim_np = tsim.cpu().detach().numpy()
for i in range(3):
    for j in range(3):
        cell = np.random.randint(0, xts.shape[1])
        traj_proj = pca.transform(xts[:,cell])
        # Plot the sequence of points in the trajectory, colored by their timestep
        axs[i,j].scatter(x_proj[:,0], x_proj[:,1], c='grey', s=.5, alpha=.5)
        axs[i,j].scatter(traj_proj[:,0], traj_proj[:,1], c=tsim_np, cmap=viridis, s=1.5)
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])
        # axs[i,j].scatter(x0[:,0], x0[:,1], color='black', alpha=1, s=.5)
plt.tight_layout()

# %%
# Scatter plot of all the simulation trajectories
xts_flat = xts.reshape(-1, xts.shape[-1])
random_idxs = np.random.randint(0, xts_flat.shape[0], 50_000)
xts_flat = xts_flat[random_idxs,:]
xts_proj = pca.transform(xts_flat)
plt.scatter(xts_proj[:,0], xts_proj[:,1], s=.5, alpha=.5)
plt.scatter(x_proj[:,0], x_proj[:,1], c=ts_np[pxts.argmax(axis=1)], cmap=viridis, s=1)

# %%
# Contour plot of the density of the simulation trajectories
# plt.scatter(x_proj[:,0], x_proj[:,1], c=tsnp[np.exp(pxts).argmax(axis=1)], cmap=viridis, s=1)
from scipy.stats import gaussian_kde

fig, axs = plt.subplots(1, 2, figsize=(10,5))

# Create a contour plot of the density of the simulation trajectories
x = xts_flat[:,0]
y = xts_flat[:,1]
kde = gaussian_kde(xts_flat.T)
x, y = np.meshgrid(np.linspace(x.min(), x.max(), 20), 
                   np.linspace(y.min(), y.max(), 20))
z = kde(np.vstack([x.ravel(), y.ravel()]))
axs[0].tricontourf(x.ravel(), y.ravel(), z, cmap='viridis')
axs[0].set_title('Simulation density')

# Compute the density of the data
data_kde = gaussian_kde(x_proj.T)
data_z = data_kde(np.vstack([x.ravel(), y.ravel()]))
axs[1].tricontourf(x.ravel(), y.ravel(), data_z, cmap='viridis')
plt.tight_layout()
axs[1].set_title('Data density')

# %%
