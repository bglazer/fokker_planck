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
#%%
device = 'cuda:0'

## %%
# Generate simple test data
# Initial distribution
# X0 = torch.randn((1000, 1), device=device)
# Final distribution is initial distribution plus another normal shifted by +4
# X1 = torch.randn((1000, 1), device=device)+4
# X = torch.vstack((X0, X1))
# %%
genotype='wildtype'
dataset = 'net'
adata = sc.read_h5ad(f'{genotype}_{dataset}.h5ad')
nmp_cell_mask = adata.obs['cell_type'] == 'NMP'
gene = 'POU5F1'
X = adata[:, adata.var_names == gene].X.toarray()
X = torch.tensor(X, device=device, dtype=torch.float32, requires_grad=True)
X0 = adata[nmp_cell_mask, adata.var_names == gene].X.toarray()
X0 = torch.tensor(X0, device=device, dtype=torch.float32, requires_grad=True)

#%%
# Plot the two distributions
_=plt.hist(X0.data.cpu().numpy(), bins=30, alpha=.3, label='X0')
# _=plt.hist(X1.cpu().numpy(), bins=30, alpha=.3, label='X1')
_=plt.hist(X.data.cpu().numpy(), bins=30, alpha=.3, label='X')
plt.legend()

#%%
ts = torch.linspace(0, 1, 100, device=device, requires_grad=True)
epochs = 500
n_samples = 1000
hx = 1e-3
ht = ts[1] - ts[0]
zero = torch.zeros(1, requires_grad=True).to(device)

#%%
# Initialize the model
ux_dropout = 0
pxt_dropout = 0

celldelta = CellDelta(input_dim=1, device=device,
                      ux_hidden_dim=64, ux_layers=2, 
                      pxt_hidden_dim=64, pxt_layers=2)

#%%
# Train the model
losses = celldelta.optimize(X, X0, ts,
                            pxt_lr=1e-3, ux_lr=1e-3, 
                            alpha_fp=1, n_epochs=epochs, n_samples=n_samples, 
                            verbose=True)

#%%
l_fps = losses['l_fp']
l_self_pxs = losses['l_self_px']
l_self_p0s = losses['l_self_p0']

fig, axs = plt.subplots(3, 1, figsize=(10,10))
axs[0].plot(l_fps[10:], label='l_fp')
axs[1].plot(l_self_pxs[10:], label='l_self_pxs')
axs[2].plot(l_self_p0s[10:], label='l_self_p0')
[axs[i].set_xlabel('Epoch') for i in range(len(axs))]
[axs[i].set_ylabel('Loss') for i in range(len(axs))]
[axs[i].legend() for i in range(len(axs))]
fig.suptitle('Loss curves')
fig.tight_layout()

# %%
viridis = matplotlib.colormaps.get_cmap('viridis')
greys = matplotlib.colormaps.get_cmap('Greys')
purples = matplotlib.colormaps.get_cmap('Purples')
low = float(X.min())
high = float(X.max())
l = low-.25*(high-low) 
h = high+.25*(high-low)
xs = torch.arange(0, h, .01, device=device)[:,None]

pxts = torch.exp(celldelta.pxt(xs, ts)).squeeze().T.cpu().detach().numpy()
uxs = celldelta.ux(xs).squeeze().cpu().detach().numpy()

# TODO need to calculate the gradient of u(x) and p(x,t) wrt x
# up_dx, up_dt = celldelta.ux.dx_dt(xs)
# _,pxt_dts = celldelta.pxt.dx_dt(xs, ts)
# up_dx = up_dx.detach().cpu().numpy()[:,:,0]
# pxt_dts = pxt_dts.detach().cpu().numpy()[:,:,0]

xs = xs.squeeze().cpu().detach().numpy()
#%%
# Plot the predicted p(x,t) at each timestep t
plt.title('p(x,t)')
x_density, x_bins = np.histogram(X.detach().cpu().numpy(), bins=30, density=True)
w = x_bins[1] - x_bins[0]
plt.bar(height=x_density, x=x_bins[:-1], width=w, alpha=.3, label='X')
plt.hist(X0.detach().cpu().numpy(), bins=30, alpha=.3, label='X0', density=True, color='orange')
for i in range(0, ts.shape[0], int(len(ts)/10)):
    t = ts[i]
    z = pxts[:,i].sum()
    plt.plot(xs, pxts[:,i], c=viridis(float(t)))
plt.xlabel('x')
plt.ylabel('p(x,t)')
# Add a colorbar to show the timestep
sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
plt.colorbar(sm, label='timestep (t)')

#%%
# Plot the Fokker Planck terms
# for i in range(0,len(ts),len(ts)//10):
#     plt.plot(xs, pxt_dts[i,:], c='r')
#     plt.plot(xs, up_dx[i,:], c='blue')
# labels = ['d/dt p(x,t)', 'd/dx u(x) p(x,t)']
# plt.legend(labels)

# %%
# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
plt.title('Cumulative mean of p(x,t)')
sim_cum_pxt = pxts.cumsum(axis=1) / np.arange(1, ts.shape[0]+1)
plt.imshow(sim_cum_pxt, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('x')
plt.yticks(ticks=np.linspace(0, sim_cum_pxt.shape[0], 10), 
           labels=np.round(np.linspace(xs.min(), xs.max(), 10), 2))
plt.xlabel('timestep (t)')
plt.xticks(ticks=np.linspace(0, sim_cum_pxt.shape[1], 10),
              labels=np.round(np.linspace(0, 1, 10), 2))
plt.colorbar()


#%%
# This is the individual p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
plt.title('p(x,t) at each timestep t')
plt.imshow(pxts, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('x')
plt.xlabel('timestep (t)')
plt.colorbar()
# %%
# Plot the u(x) term for all x
fig, ax1 = plt.subplots(1,1, figsize=(10,5))
plt.title('u(x) vs p(x)')
ax1.plot(xs, uxs, c='blue', alpha=.6)
# ax1.plot(xs, uxs, label='u(x)')
# Add vertical and horizontal grid lines
ax1.grid()
ax1.set_ylabel('u(x)')
ax1.set_xlabel('x')
ax1.axhline(0, c='r', alpha=.5)
ax2 = ax1.twinx()
data_dist, data_bins = np.histogram(X.detach().cpu().numpy().flatten(), bins=30, density=True)
w = data_bins[1] - data_bins[0]
ax2.bar(data_bins[:-1], data_dist, width=w, alpha=.3, label='Data')
ax2.set_ylabel('p(x)')
fig.legend()
# %%
# Simulate the stochastic differential equation using the Euler-Maruyama method
# with the learned drift term u(x)
x = X0.clone().detach()
tsim = torch.linspace(0, 1, 100, device=device, requires_grad=False)
zero_boundary = True

xts = celldelta.simulate(x, tsim, zero_boundary=zero_boundary)

#%%
# Plot the resulting probability densities at each timestep
low = float(xs.min())
high = float(xs.max())
bins = np.linspace(low, high, 60)
w = bins[1] - bins[0]
for i in range(0, tsim.shape[0], tsim.shape[0]//10):
    t=tsim[i]
    heights,bins = np.histogram(xts[i,:], 
                                bins=bins,
                                density=True)
    plt.bar(bins[:-1], heights, width=w, color=viridis(float(t)), alpha=.2)

for i in range(0, pxts.shape[1], pxts.shape[1]//10):
    plt.plot(xs, pxts[:,i], color='blue', alpha=.2)

labels = ['Simulation', 'Fokker-Planck theoretical']
artists = [plt.Line2D([0], [0], color=c, alpha=.2) for c in ['red', 'blue']]
plt.legend(artists, labels)
plt.title('p(x,t)')

#%%
# Plot the cumulative distribution of the simulated data at each timestep
sim_pxts = np.zeros((bins.shape[0]-1, tsim.shape[0]))
for i in range(0, tsim.shape[0]):
    heights,bins = np.histogram(xts[i,:], 
                                bins=bins,
                                density=True)
    sim_pxts[:,i] = heights

sim_cum_pxt = sim_pxts.cumsum(axis=1) / np.arange(1, sim_pxts.shape[1]+1)[None,:]

# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
plt.title('Cumulative mean of simulated p(x,t)')
plt.imshow(sim_cum_pxt, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('x')
plt.xlabel('timestep (t)')
plt.colorbar() 

#%%
# Plot the final cumulative distribution of simulations versus the data distribution
plt.title('Final cumulative distribution of simulations vs data distribution')
# Remove timesteps where the simulation diverged
# i.e. the simulation went outside the range of the data (>2stds away from the data mean)
std = X.detach().cpu().numpy().std()*2
good_runs = ((np.abs(xts - X.detach().cpu().numpy().mean()) > std).sum(axis=0) == 0)
print(f'Fraction of good runs: {good_runs.sum()/good_runs.shape[0]:.2f}')
sim_mean_dist, sim_mean_bins = np.histogram(xts[:,good_runs].flatten(), bins=30, density=True)
w = sim_mean_bins[1] - sim_mean_bins[0]
plt.bar(sim_mean_bins[:-1], sim_mean_dist, width=w, alpha=.3, label='Simulation')

plt.bar(data_bins[:-1], data_dist, width=w, alpha=.3, label='Data')
plt.ylabel('p(x)')
plt.xlabel(f'Expression {gene}')
plt.legend();
# %%
