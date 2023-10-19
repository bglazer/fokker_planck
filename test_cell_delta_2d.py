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
device = 'cuda:0'

#%%
genotype='wildtype'
dataset = 'net'
adata = sc.read_h5ad(f'{genotype}_{dataset}.h5ad')
nmp_cell_mask = adata.obs['cell_type'] == 'NMP'
X = adata.X.toarray()
X = torch.tensor(X, device=device, dtype=torch.float32)
X0 = X[nmp_cell_mask,:].clone().detach()
n_genes = 2 #X.shape[1]
# Get the top n_genes by total expression
top_gene_idxs = torch.argsort(X.var(dim=0))[-n_genes:]
X = X[:,top_gene_idxs]
X0 = X0[:, top_gene_idxs]

#%%
ts = torch.linspace(0, 1, 100, device=device, requires_grad=True)
epochs = 500
n_samples = 5000
hx = 1e-3
ht = ts[1] - ts[0]
zero = torch.zeros(1, requires_grad=True).to(device)

noise = D.MultivariateNormal(loc=torch.ones(n_genes, device=device)*X.mean(), 
                             covariance_matrix=torch.eye(n_genes, device=device)*X.var()*3)

#%%
# Initialize the model
# TODO add weight decay or dropout to the model
ux_dropout = 0
pxt_dropout = 0

celldelta = CellDelta(input_dim=n_genes, 
                      ux_hidden_dim=64, ux_layers=2, ux_dropout=ux_dropout,
                      pxt_hidden_dim=64, pxt_layers=2, pxt_dropout=pxt_dropout,
                      noise=noise, device=device)

#%%
# Train the model
losses = celldelta.optimize(X, X0, ts, restart=True, pxt_lr=5e-4, ux_lr=5e-4, alpha_fp=1, 
                            n_epochs=epochs, n_samples=n_samples, hx=hx, verbose=True)
#%%
l_fps = losses['l_fp']
l_nce_pxs = losses['l_nce_px']
l_nce_p0s = losses['l_nce_p0']

fig, axs = plt.subplots(3, 1, figsize=(10,10))
axs[0].plot(l_fps[10:], label='l_fp')
axs[1].plot(l_nce_pxs[10:], label='l_nce_pxs')
axs[2].plot(l_nce_p0s[10:], label='l_nce_p0')
[axs[i].set_xlabel('Epoch') for i in range(len(axs))]
[axs[i].set_ylabel('Loss') for i in range(len(axs))]
[axs[i].legend() for i in range(len(axs))]
fig.suptitle('Loss curves')
fig.tight_layout()

# %%
viridis = matplotlib.colormaps.get_cmap('viridis')
greys = matplotlib.colormaps.get_cmap('Greys')
purples = matplotlib.colormaps.get_cmap('Purples')

xs = X.clone().detach()
pxts = celldelta.pxt(xs, ts).squeeze().T.cpu().detach().numpy()
uxs = celldelta.ux(xs).squeeze().cpu().detach().numpy()
_,pxt_dts = celldelta.pxt.dx_dt(xs, ts)
pxt_dts = pxt_dts.detach().cpu().numpy()[:,:,0]

xs = xs.squeeze().cpu().detach().numpy()

#%%
# Setup the PCA using the saved components and mean
# pca = PCA()
# pca.components_ = adata.uns['PCs']
# pca.mean_ = adata.uns['pca_mean']
# x_proj = pca.transform(X.cpu().numpy())
x_proj = X.cpu().numpy()
x0 = X0.cpu().numpy()

#%%
# Plot the pseudotime i.e. the timestep of maximum probability for each cell
plt.title('Pseudotime')
tsnp = ts.detach().cpu().numpy()
# Mark the initial points with a black triangle
plt.scatter(xs[:,0], xs[:,1], c=tsnp[np.exp(pxts).argmax(axis=1)], cmap=viridis, s=1)
# plt.scatter(x0[:,0], x0[:,1], color='black', alpha=.5, s=9.5, marker='^')

# %%
# Simulate the stochastic differential equation using the Euler-Maruyama method
# with the learned drift term u(x)
x = X0.clone().detach()
tsim = torch.linspace(0, 1, 1000, device=device, requires_grad=False)
xts = np.zeros((len(tsim), x.shape[0], x.shape[1]))
ht = tsim[1] - tsim[0]
zero_boundary = True

for i in range(len(tsim)):
    # Compute the drift term
    u = celldelta.ux(x)
    # Compute the diffusion term
    # Generate a set of random numbers
    dW = torch.randn_like(x) * torch.sqrt(ht)
    sigma = torch.ones_like(x)*0
    # Compute the change in x
    dx = u * ht + sigma * dW
    # print(f'{float(u.mean()):.5f}, {float(ht):.3f}, {float(dW.mean()): .5f}, {float(dx.mean()): .5f}')
    dx = dx.squeeze(0)
    # Update x
    x = x + dx
    if zero_boundary:
        x[x < 0] = 0
    xts[i,:,:] = x.cpu().detach().numpy()
#%%
# Plot trajectory of a single cell
fig, axs = plt.subplots(3, 3, figsize=(10,10))
tsim_np = tsim.cpu().detach().numpy()
for i in range(3):
    for j in range(3):
        cell = np.random.randint(0, xts.shape[1])
        # traj_proj = pca.transform(xts[cell,:])
        traj_proj = xts[:,cell,:]
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
plt.scatter(xts_flat[:,0], xts_flat[:,1], s=.5, alpha=.5)
plt.scatter(x_proj[:,0], x_proj[:,1], c=tsnp[np.exp(pxts).argmax(axis=1)], cmap=viridis, s=1)

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
