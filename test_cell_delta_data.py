#%%
%load_ext autoreload
%autoreload 2
#%%
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import scanpy as sc
from celldelta import CellDelta
from sklearn.decomposition import PCA
import time
import torch.distributions as D
from umap import UMAP

#%%
device = 'cuda:0'

#%%
genotype='wildtype'
dataset = 'net'
adata = sc.read_h5ad(f'{genotype}_{dataset}.h5ad')
nmp_cell_mask = adata.obs['cell_type'] == 'NMP'
n_genes = 50
X = adata.X.toarray()
X = torch.tensor(X, device=device, dtype=torch.float32)
# Get a boolean mask for the top n_genes by variance
top_var_genes = torch.argsort(X.var(dim=0))[-n_genes:]
top_var_gene_mask = torch.zeros(X.shape[1], dtype=torch.bool)
top_var_gene_mask[top_var_genes] = True

X0 = X[nmp_cell_mask,:][:,top_var_gene_mask].clone().detach()
X = X[:,top_var_gene_mask].clone().detach()
#%%
# Setup the PCA using the saved components and mean
pca = PCA()
# pca.components_ = adata.uns['PCs']
# pca.mean_ = adata.uns['pca_mean']
x_proj = pca.fit_transform(X.cpu().numpy())
x0_proj = pca.transform(X0.cpu().numpy())

#%%
ts = torch.linspace(0, 1, 100, device=device)*100
ts.requires_grad = True
n_samples = 10_000
zero = torch.zeros(1, requires_grad=True).to(device)

#%%
# Initialize the model
celldelta = CellDelta(input_dim=n_genes, 
                      ux_hidden_dim=64, ux_layers=2,
                      pxt_hidden_dim=64, pxt_layers=2,
                      device=device)
mean = X.mean(dim=0)
cov = np.cov(X.cpu().numpy().T)
cov = cov + np.eye(n_genes)*1e-3
cov = torch.tensor(cov, dtype=torch.float32).to(device)
noise = D.MultivariateNormal(mean, cov)
mean0 = X0.mean(dim=0)
cov0 = np.cov(X0.cpu().numpy().T)
cov0 = cov0 + np.eye(n_genes)*1e-3
cov0 = torch.tensor(cov0, dtype=torch.float32).to(device)
noise0 = D.MultivariateNormal(mean0, cov0)

#%%
# Train the model
# Get the start time
start = time.time()
losses = celldelta.optimize_initial_conditions(X0, ts, 
                                               n_epochs=1500,
                                               p0_noise=noise0,
                                               verbose=True)
end = time.time()
print(f'Training took {end-start:.2f} seconds')
#%%
start = time.time()
losses = celldelta.optimize(X, X0, ts, pxt_lr=1e-3, ux_lr=1e-3, 
                            p0_alpha=10, fokker_planck_alpha=100,
                            n_epochs=500, n_samples=n_samples, 
                            px_noise=noise, p0_noise=noise0,
                            verbose=True)
# Get the end time
end = time.time()
print(f'Training took {end-start:.2f} seconds')

#%%
start = time.time()
_ = celldelta.optimize_fokker_planck(X0, ts,
                                     pxt_lr=1e-3, ux_lr=1e-3,
                                     fokker_planck_alpha=10000,
                                     px=False, ux=True,
                                     n_epochs=500, n_samples=n_samples,
                                     verbose=True)
end = time.time()
print(f'Fokker-Planck optimization took {end-start:.2f} seconds')
#%%
l_fps = losses['l_fp']
l_pxs = losses['l_nce_px']
l_p0s = losses['l_nce_p0']

fig, axs = plt.subplots(3, 1, figsize=(10,10))
start_skip = 10
axs[0].plot(l_fps[start_skip:], label='l_fp')
axs[1].plot(l_pxs[start_skip:], label='l_pxs')
axs[2].plot(l_p0s[start_skip:], label='l_p0')
[axs[i].set_xlabel('Epoch') for i in range(len(axs))]
[axs[i].set_ylabel('Loss') for i in range(len(axs))]
[axs[i].legend() for i in range(len(axs))]
fig.suptitle('Loss curves')
fig.tight_layout()

# %%
xs = X.clone().detach()
pxts = celldelta.pxt(xs, ts).squeeze().T.cpu().detach().numpy()
uxs = celldelta.ux(xs).squeeze().cpu().detach().numpy()
# _, pxt_dts = celldelta.pxt.dx_dt(xs, ts)
# pxt_dts = pxt_dts.detach().cpu().numpy()[:,:,0]

xs = xs.squeeze().cpu().detach().numpy()

viridis = matplotlib.colormaps.get_cmap('viridis')

# Plot the probability of each cell at t=0
plt.title('p(x,0)')
plt.scatter(x_proj[:,0], x_proj[:,1], c=pxts[:,0], cmap=viridis, s=1)
# Remove the axis ticks 
plt.xticks([])
plt.yticks([]);

#%%
# Plot the pseudotime i.e. the timestep of maximum probability for each cell
ts_np = ts.detach().cpu().numpy()
pxt_norm = pxts
plt.title('Pseudotime, max_t p(x,t)')
plt.scatter(x_proj[:,0], x_proj[:,1], c=ts_np[pxts.argmax(axis=1)], cmap=viridis, s=1, 
            vmin=ts[0], vmax=ts[-1])
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.xlabel('PC1')
plt.ylabel('PC2');
#%%
# Calculate the psuedotime an alternate way, by finding cells with the maximum p(x,t) at each t
# TODO implement this

#%%
# Compute the UMAP projection of the data
# umap = UMAP()
# umap_proj = umap.fit_transform(X.cpu().numpy())
# # Plot the UMAP projection, colored by pseudotime
# plt.scatter(umap_proj[:,0], umap_proj[:,1], c=ts_np[pxts.argmax(axis=1)], cmap=viridis, s=.5)
# plt.colorbar()
# plt.title('UMAP projection, colored by pseudotime')
#%%
# Plot the predicted p(x,t) for each cell at each timestep t
fig, axs = plt.subplots(6,2, figsize=(10,20)) 
# Get 5 equally spaced timesteps with the first and last timesteps included
for i,t in enumerate(np.linspace(0, len(ts_np)-1, 6, dtype=int)):
    axs[i,0].set_title(f't={int(ts_np[t]):d}')
    axs[i,0].scatter(x_proj[:,0], x_proj[:,1], c=pxts[:,t], cmap=viridis, s=1, vmin=pxts.min(), vmax=pxts.max())
    axs[i,0].set_xticks([])
    axs[i,0].set_yticks([])
    plt.colorbar(axs[i,0].collections[0], ax=axs[i,0])
    axs[i,1].set_title(f't={int(ts_np[t]):d}')
    axs[i,1].scatter(x_proj[:,0], x_proj[:,1], c=pxts[:,t], cmap=viridis, s=1)
    axs[i,1].set_xticks([])
    axs[i,1].set_yticks([])
    plt.colorbar(axs[i,1].collections[0], ax=axs[i,1])
plt.tight_layout()
# Mark the initial points with a black triangle
# x0_proj = pca.transform(X0.cpu().numpy())
# plt.scatter(x0_proj[:,0], x0_proj[:,1], color='black', alpha=.5, s=9.5, marker='^')
#%%
plt.plot(pxts.mean(axis=0))

# %%
# Simulate the stochastic differential equation using the Euler-Maruyama method
# with the learned drift term u(x)
x = X0.clone().detach()

zero_boundary = False
xts = celldelta.simulate(x, ts, zero_boundary=zero_boundary, sigma=0.0, ux_alpha=1.0)

#%%
# Plot trajectory of a single cell
fig, axs = plt.subplots(3, 3, figsize=(10,10))
tsim_np = ts.cpu().detach().numpy()
for i in range(3):
    for j in range(3):
        cell = np.random.randint(0, xts.shape[1])
        traj_proj = pca.transform(xts[:,cell])
        # Plot the data distribution
        axs[i,j].scatter(x_proj[:,0], x_proj[:,1], c='grey', s=.5, alpha=.5)
        # Plot the sequence of points in the trajectory, colored by their timestep, connected by lines
        axs[i,j].plot(traj_proj[:,0], traj_proj[:,1], c='black', alpha=.5)
        axs[i,j].scatter(traj_proj[:,0], traj_proj[:,1], c=tsim_np[:], cmap=viridis, s=1.5)
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])
        # axs[i,j].scatter(x0[:,0], x0[:,1], color='black', alpha=1, s=.5)
plt.tight_layout()

#%%
# Plot arrows pointing in the direction of the drift term u(x)
# Select a random subset of cells
n_cells = 200
random_cells = X[torch.randperm(X.shape[0])[:n_cells],:]
# Get the drift term u(x) for each cell
uxs = celldelta.ux(random_cells)
# Add the uxs to the random_cells
random_drifts = random_cells + uxs
# Project the random_cells and random_drifts onto the PCA components
random_cells_proj = pca.transform(random_cells.detach().cpu().numpy())
random_drifts_proj = pca.transform(random_drifts.detach().cpu().numpy())
# Plot all the cells
plt.scatter(x_proj[:,0], x_proj[:,1], c='grey', s=.5, alpha=.5)
# Plot the random cells
plt.scatter(random_cells_proj[:,0], random_cells_proj[:,1], c='black', s=1)
# Plot the random drifts as arrows from the random cells
plt.quiver(random_cells_proj[:,0], random_cells_proj[:,1], 
           random_drifts_proj[:,0]-random_cells_proj[:,0], 
           random_drifts_proj[:,1]-random_cells_proj[:,1], 
           color='red', alpha=.5, scale=.1, scale_units='xy',
           angles='xy', width=.002)
plt.xticks([])
plt.yticks([]);


# %%
# Scatter plot of all the simulation trajectories
xts_flat = xts.reshape(-1, xts.shape[-1])
random_idxs = np.random.randint(0, xts_flat.shape[0], 50_000)
xts_flat = xts_flat[random_idxs,:]
t_idxs = ts.repeat((xts.shape[1])).reshape(-1)[random_idxs]
xts_proj = pca.transform(xts_flat)
# plt.scatter(x_proj[:,0], x_proj[:,1], c=ts_np[pxts.argmax(axis=1)], cmap='b', alpha=.1, s=1)
plt.scatter(xts_proj[:,0], xts_proj[:,1], s=.5, alpha=.5)
plt.scatter(x_proj[:,0], x_proj[:,1], c='black', cmap=viridis, alpha=.1, s=1)
plt.xticks([])
plt.yticks([])

#%%
# Plot the sum of the p(x,t) at each timestep
plt.title('Sum of p(x,t)')
plt.plot(pxts.sum(axis=0))

#%% 
# Histogram of the argmax_t p(x,t) for each cell
plt.title('Histogram of argmax_t p(x,t) for each cell')
plt.hist(pxts.argmax(axis=1), bins=30);

# %%
# Contour plot of the density of the simulation trajectories
# plt.scatter(x_proj[:,0], x_proj[:,1], c=tsnp[np.exp(pxts).argmax(axis=1)], cmap=viridis, s=1)
# from scipy.stats import gaussian_kde

fig, axs = plt.subplots(1, 2, figsize=(10,5))

xmin = (-.1,-.1)
xmax = (x_proj[:,0].max(), x_proj[:,1].max())

xts_proj = pca.transform(xts_flat)

# Create a contour plot of the density of the simulation trajectories
# # kde = gaussian_kde(xts_proj.T)
# x, y = np.meshgrid(np.linspace(xmin[0], xmax[0], 20), 
#                    np.linspace(xmin[1], xmax[1], 20))
# z = kde(np.vstack([x.ravel(), y.ravel()])).reshape(x.shape) 
# axs[0].contour(x, y, z, cmap='viridis', extent=(xmin[0], xmax[0], xmin[1], xmax[1]))
axs[0].set_title('Simulation density')
inside = (xts_proj[:,0] < xmax[0]) & (xts_proj[:,1] < xmax[1])
t_idxs = t_idxs.cpu().detach().numpy()
axs[0].scatter(xts_proj[inside,0], xts_proj[inside,1], c=t_idxs[inside], alpha=.1, s=1)

# Compute the density of the data
# data_kde = gaussian_kde(x_proj.T)
# data_z = data_kde(np.vstack([x_proj[:,0].ravel(), x_proj[:,1].ravel()])).reshape(x_proj.shape[0])
# axs[1].contour(x, y, data_z, cmap='viridis')
axs[1].scatter(x_proj[:,0], x_proj[:,1], c='black', alpha=.1, s=1)
plt.tight_layout()
axs[1].set_title('Data density')


# %%
