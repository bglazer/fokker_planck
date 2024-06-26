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
import time
#%%
tonp = lambda x: x.detach().cpu().numpy()
#%%
# Create a celldelta model with a very simple u(x) model
# u(x) is just a fixed linear function of x: [2*x_0, 3*x_1]
class Ux(torch.nn.Module):
    def __init__(self):
        super(Ux, self).__init__()
        self.ux = torch.tensor([[2, 3]], dtype=torch.float32, device='cpu')

    def forward(self, x):
        return x**2 * self.ux + x[:,0,None]*x[:,1,None]
    
celldelta = CellDelta(input_dim=2,
                      ux_hidden_dim=1,
                      ux_layers=1,
                      pxt_hidden_dim=1,
                      pxt_layers=1,
                      device='cpu')
celldelta.ux.model = Ux()
#%%
# Compute the Fokker-Planck term for a simple u(x) model
# two dimensional x that's just two sequences from 1 to 10
x = torch.arange(1, 11, dtype=torch.float32, device='cpu').repeat(2,1).T
x.requires_grad = True
ts = torch.linspace(0, 1, 10, device='cpu', requires_grad=True)
dq_dx, dq_dt = celldelta.pxt.dx_dt(x, ts)
ux, du_dx = celldelta.ux.div(x)
d_dx = ((dq_dx * ux).sum(dim=2) + du_dx)[...,None]
# Print the divergence of the drift term u(x)
print(du_dx)

# %%
# One dimensional Gaussian sequence
# Generate sequence of distributions
# X_t = Normal(1+t) 
device = 'cuda:1'
N = 1000
tsteps = 100
X = torch.randn((N, tsteps), device=device)
X_t = X + torch.arange(0, tsteps, device=device)
X0 = X_t[:,0, None]
X = X_t.flatten()[:,None]
#%%
# Plot the sequence of distributions
viridis = matplotlib.colormaps.get_cmap('viridis')
greys = matplotlib.colormaps.get_cmap('Greys')
purples = matplotlib.colormaps.get_cmap('Purples')
for t in np.arange(0, tsteps, tsteps//20):
    xt = X_t[:,t]
    _ = plt.hist(xt.data.cpu().numpy(), bins=30, alpha=.6, label=t, density=True, color=viridis(float(t)/tsteps))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
#%%
ts = torch.linspace(0, 1, 100, device=device)*100
epochs = 500
n_samples = 1000

#%%
# Initialize the model
celldelta = CellDelta(input_dim=1, device=device,
                      ux_hidden_dim=64, ux_layers=2, 
                      pxt_hidden_dim=64, pxt_layers=2)
mean = X.mean(dim=0, keepdim=False)
cov = np.cov(X.cpu().numpy().T)
cov = cov + np.eye(1)*1e-3
cov = torch.tensor(cov, dtype=torch.float32).to(device)
noise = D.MultivariateNormal(mean, cov)
mean0 = X0.mean(dim=0, keepdim=False)
cov0 = np.cov(X0.cpu().numpy().T)
cov0 = cov0 + np.eye(1)*1e-3
cov0 = torch.tensor(cov0, dtype=torch.float32).to(device)
noise0 = D.MultivariateNormal(mean0, cov0)

#%%
# Train the model
losses = celldelta.optimize_initial_conditions(X0, ts, p0_noise=noise0, 
                                               n_epochs=500, 
                                               verbose=True)
#%%
p0_alpha = 0
fokker_planck_alpha = 1
losses = celldelta.optimize(X, X0, ts,
                            pxt_lr=1e-3, ux_lr=1e-4, 
                            n_epochs=1500, n_samples=n_samples, 
                            px_noise=noise, p0_noise=noise0, 
                            fokker_planck_alpha=fokker_planck_alpha,
                            p0_alpha=p0_alpha, 
                            verbose=True)
#%%
# Optimize the Fokker-Planck term
fokker_planck_alpha = 1
_ = celldelta.optimize_fokker_planck(X, ts,
                                     pxt_lr=0, ux_lr=1e-4,
                                     fokker_planck_alpha=fokker_planck_alpha,
                                     px=False, ux=True,
                                     noise=None,
                                     n_epochs=1500, 
                                     n_samples=n_samples,
                                     verbose=True)

#%%
l_fps = losses['l_fp']
l_pxs = losses['l_nce_px']
l_p0s = losses['l_nce_p0']

fig, axs = plt.subplots(3, 1, figsize=(10,10))
axs[0].plot(l_fps[10:], label='l_fp')
axs[1].plot(l_pxs[10:], label='l_pxs')
axs[2].plot(l_p0s[10:], label='l_p0')
[axs[i].set_xlabel('Epoch') for i in range(len(axs))]
[axs[i].set_ylabel('Loss') for i in range(len(axs))]
[axs[i].legend() for i in range(len(axs))]
fig.suptitle('Loss curves')
fig.tight_layout()

# %%
# Plot the learned sequence of distributions
low = float(X.min())
high = float(X.max())
l = low-.25*(high-low) 
h = high+.25*(high-low)
xs = torch.arange(l, h, .01, device=device)[:,None]

pxts = np.exp(celldelta.pxt.log_pxt(xs, ts).squeeze().T.cpu().detach().numpy())
uxs = celldelta.ux(xs).squeeze().cpu().detach().numpy()

# TODO need to calculate the gradient of u(x) and p(x,t) wrt x
# up_dx, up_dt = celldelta.ux.dx_dt(xs)
# _,pxt_dts = celldelta.pxt.dx_dt(xs, ts)
# up_dx = up_dx.detach().cpu().numpy()[:,:,0]
# pxt_dts = pxt_dts.detach().cpu().numpy()[:,:,0]

xs = xs.squeeze().cpu().detach().numpy()

# Plot the predicted p(x,t) at each timestep t
plt.title('p(x,t)')
x_density, x_bins = np.histogram(X.detach().cpu().numpy(), bins=30, density=True)
w = x_bins[1] - x_bins[0]
plt.bar(height=x_density, x=x_bins[:-1], width=w, alpha=.3, label='X', color='orange')
heights, edges = np.histogram(X0.detach().cpu().numpy(), bins=30)
heights = heights / heights.sum()
w = edges[1] - edges[0]
plt.bar(edges[:-1], heights, width=w, alpha=.3, label='X0', color='blue')
# Get a list of timesteps to plot, with first and last timestep included
for i in np.linspace(0, len(ts)-1, 10, dtype=int):
    t = ts[i]
    z = pxts[:,i].sum()
    plt.plot(xs, pxts[:,i], c=viridis(float(t/ts.max())), alpha=.5)
plt.xlabel('x')
plt.ylabel('p(x,t)')
# Add a colorbar to show the timestep
sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
plt.colorbar(sm, label='timestep (t)')
plt.legend()
#%%
plt.plot(pxts.max(axis=0), marker='o', markersize=2, c='blue', label='max p(x,t)')
plt.ylabel('max p(x,t)', c='blue')

# Make a second y axis
plt.twinx()
plt.plot(pxts.sum(axis=0), marker='o', markersize=2, c='orange', label='sum p(x,t)')
plt.ylabel('sum p(x,t)', c='orange')
print(f'Min sum pxt {pxts.sum(axis=0).min():.2e} t={pxts.sum(0).argmin()}')
print(f'Max sum pxt {pxts.sum(axis=0).min():.2e} t={pxts.sum(0).argmin()}')
#%%
# Plot the Fokker Planck terms
# X.requires_grad = True
# dq_dx, dq_dt = celldelta.pxt.dx_dt(X, ts)
# ux, du_dx = celldelta.ux.dx(X)
# d_dx = ((dq_dx * ux).sum(dim=2) + du_dx)[...,None]
# for i in range(0,len(ts),len(ts)//10):
#     plt.plot(X, dq_dt[i,:], c='r')
#     plt.plot(X, dq_dx[i,:], c='blue')
# labels = ['d/dt p(x,t)', 'd/dx u(x) p(x,t)']
# plt.legend(labels)
# X.requires_grad = False

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
xs = torch.arange(l, h, .01, device=device)[:,None]
uxs = celldelta.ux(xs).squeeze().cpu().detach().numpy()
xs = xs.squeeze().cpu().detach().numpy()
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
data_dist, data_bins = np.histogram(X.detach().cpu().numpy().flatten(), bins=150, density=True)
w = data_bins[1] - data_bins[0]
ax2.bar(data_bins[:-1], data_dist, width=w, alpha=.3, label='Data')
ax2.set_ylabel('p(x)')
fig.legend()
# %%
# Simulate the stochastic differential equation using the Euler-Maruyama method
# with the learned drift term u(x)
x = X0.clone().detach()
zero_boundary = True
ts.requires_grad = False
xts = celldelta.simulate(x, ts, zero_boundary=zero_boundary, sigma=0, ux_alpha=1)

#%%
# Plot the resulting probability densities at each timestep
low = float(xs.min())
high = float(xs.max())
bins = np.linspace(low, high, 60)
w = bins[1] - bins[0]
for i in range(0, ts.shape[0], ts.shape[0]//10):
    t=ts[i]
    heights,bins = np.histogram(xts[i,:], 
                                bins=bins,
                                density=True)
    plt.bar(bins[:-1], heights, width=w, color=viridis(float(t)/max(ts).item()), alpha=.2)

for i in range(0, pxts.shape[1], pxts.shape[1]//10):
    plt.plot(xs, pxts[:,i], color='blue', alpha=.2)

labels = ['Simulation', 'Fokker-Planck theoretical']
artists = [plt.Line2D([0], [0], color=c, alpha=.2) for c in ['red', 'blue']]
plt.legend(artists, labels)
plt.title('p(x,t)')

#%%
# Plot the cumulative distribution of the simulated data at each timestep
sim_pxts = np.zeros((bins.shape[0]-1, ts.shape[0]))
for i in range(0, ts.shape[0]):
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
sim_mean_dist, sim_mean_bins = np.histogram(xts.flatten(), bins=data_bins, density=True)
w = sim_mean_bins[1] - sim_mean_bins[0]
plt.bar(sim_mean_bins[:-1], sim_mean_dist, width=w, alpha=.3, label='Simulation')

plt.bar(data_bins[:-1], data_dist, width=w, alpha=.3, label='Data')
plt.ylabel('p(x)')
plt.xlabel('x')
plt.legend();
# %%
# Plot the marginalized p(x) for each x
plt.bar(data_bins[:-1], data_dist, width=w, alpha=.5, label='Data', color='green')
plt.plot(xs, pxts.mean(axis=1), alpha=.4, label='Estimated p(x,t)', color='purple')
plt.legend()
#%%
# Plot the histogram of differences from the true pseudotime
fig, axs = plt.subplots(4,1, figsize=(10,20))
pXt = celldelta.pxt(X, ts).squeeze().cpu().detach().numpy()
pt = pXt.argmax(axis=0)
true_pt = np.tile(np.arange(100), N)
d = np.abs(pt-true_pt).flatten()
axs[0].hist(d, bins=np.arange(0,d.max()))
axs[0].set_title('Count of differences of estimated versus true pseudotime')

# Make a empirical cumulative distribution of the differences
d = np.sort(d)
n = d.shape[0]
axs[1].plot(d, np.arange(0,n)/n)
axs[1].axhline(0.95, c='red', linestyle='--', linewidth=1)
pct95 = d[int(.95*n)]
axs[1].axvline(pct95, c='red', linestyle='--', linewidth=1)
# Label the 95th percentile, pad the label so it doesn't overlap with the line
axs[1].text(pct95+d.max()/100, 0, f'95th pct={pct95:d}')
axs[1].set_title('Empirical CDF of differences of estimated versus true pseudotime')

# Plot the number of incorrect pseudotimes at each timestep
incorrect_pct = (np.abs(pt-true_pt) > 0).reshape(N, -1).mean(axis=0)
incorrect_mag = np.abs(pt-true_pt).reshape(N, -1).sum(axis=0)
axs[2].plot(incorrect_mag, c='orange')
# Plot the incorrect count on a new axis
axs2 = axs[2].twinx()
axs2.plot(incorrect_pct, c='blue')
axs2.set_title('Number/percent of incorrect pseudotimes at each timestep')

# Plot the mean pseudotime of each set of points
axs[3].plot(pt.reshape(N, -1).mean(axis=0), c='green')
axs[3].set_title('Mean pseudotime of each set of points')

#%%
# Higher Dimensional Gaussians
#####################################################
# Sequence of distributions
# X_t = Normal(1+t) 
N = 99
tsteps = 100
d = 5
ts = torch.linspace(0, 1, tsteps, device=device)*100
X = torch.randn((N, d, tsteps), device=device)
# Create a sequence of steps to add to each timestep
v = torch.arange(0, tsteps, device=device)
# Repeat the sequence for each dimension of the data
v = v.repeat((1, d, 1))
# Zero out the first dimension
v[:,0] = 0
X_t = X + v
X0 = X_t[:, :, 0] 
X_t = X_t.permute(0, 2, 1)
X = X_t.reshape((N*tsteps, d))
# Compute PCA of the data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_proj = pca.fit_transform(X.cpu().numpy())
# %%
# Initialize the model
celldelta = CellDelta(input_dim=d, device=device,
                      ux_hidden_dim=64, ux_layers=2, 
                      pxt_hidden_dim=64, pxt_layers=2)

mean = X.mean(dim=0, keepdim=False)
cov = np.cov(X.cpu().numpy().T)
cov = cov + np.eye(d)*1e-3
cov = torch.tensor(cov, dtype=torch.float32).to(device)
noise = D.MultivariateNormal(mean, cov)
mean0 = X0.mean(dim=0, keepdim=False)
cov0 = np.cov(X0.cpu().numpy().T)
cov0 = cov0 + np.eye(d)*1e-3
cov0 = torch.tensor(cov0, dtype=torch.float32).to(device)
noise0 = D.MultivariateNormal(mean0, cov0)

#%%
# Train the model
losses = celldelta.optimize_initial_conditions(X0, ts, p0_noise=noise0, 
                                               n_epochs=1000, 
                                               verbose=True)
#%%
# for i in range(10):
start = time.time()
p0_alpha = 0
fokker_planck_alpha = 1
losses = celldelta.optimize(X, X0, ts,
                            pxt_lr=1e-4, ux_lr=1e-4, 
                            n_epochs=2500, n_samples=n_samples, 
                            px_noise=noise, p0_noise=noise0, 
                            fokker_planck_alpha=fokker_planck_alpha,
                            p0_alpha=p0_alpha, 
                            verbose=True)
end = time.time()
print(f'CellDelta optimization took {end-start:.2f} seconds')

#%%
start = time.time()
# NOTE This works much better with ux_lr having a lower learning rate
# X_noisy = noise.sample(sample_shape=(n_samples*100,))
fokker_planck_alpha = 1
_ = celldelta.optimize_fokker_planck(X, ts,
                                     pxt_lr=0, ux_lr=1e-4,
                                     fokker_planck_alpha=fokker_planck_alpha,
                                     px=False, ux=True,
                                     noise=.1,
                                     n_epochs=1500, 
                                     n_samples=n_samples,
                                     verbose=True)

end = time.time()
print(f'Fokker-Planck optimization took {end-start:.2f} seconds')

# %%
xs = X.clone().detach()
pxts = celldelta.pxt(xs, ts).squeeze().T.cpu().detach().numpy()
uxs = celldelta.ux(xs).squeeze().cpu().detach().numpy()
# _, pxt_dts = celldelta.pxt.dx_dt(xs, ts)
# pxt_dts = pxt_dts.detach().cpu().numpy()[:,:,0]

xs = xs.squeeze().cpu().detach().numpy()
#%%
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
plt.title('Pseudotime, max_t p(x,t)')
plt.scatter(x_proj[:,0], x_proj[:,1], c=ts_np[pxts.argmax(axis=1)], cmap=viridis, s=1, 
            vmin=ts[0], vmax=ts[-1])
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.xlabel('PC1')
plt.ylabel('PC2');
#%%
# Plot the histogram of differences from the true pseudotime
fig, axs = plt.subplots(2,1, figsize=(10,10))
pt = pxts.argmax(axis=1).reshape((N,-1))
true_pt = np.arange(0,ts.shape[0],dtype=int).repeat(N).reshape(tsteps,N).T
d = np.abs(pt-true_pt).flatten()
axs[0].hist(d, bins=np.arange(0,d.max()))
axs[0].set_title('Count of differences of estimated versus true pseudotime')

# Make a empirical cumulative distribution of the differences
d = np.abs(pt-true_pt).flatten()
d = np.sort(d)
n = d.shape[0]
axs[1].plot(d, np.arange(0,n)/n)
axs[1].axhline(0.95, c='red', linestyle='--', linewidth=1)
pct95 = d[np.where(np.arange(0,n)/n > .95)[0][0]]
axs[1].axvline(pct95, c='red', linestyle='--', linewidth=1)
# Label the 95th percentile, pad the label so it doesn't overlap with the line
axs[1].text(pct95+d.max()/100, 0, f'95th pct={pct95:d}')
axs[1].set_title('Empirical CDF of differences of estimated versus true pseudotime')

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
plt.plot(pxts.mean(axis=0), marker='o', markersize=2, c='blue')
plt.ylabel('mean p(x,t)', c='blue')
plt.twinx()
plt.plot(pxts.max(axis=0), marker='o', markersize=2, c='orange')
plt.ylabel('max p(x,t)', c='orange')

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
n_cells = 303
random_idxs = torch.randperm(X.shape[0])[:n_cells]
random_cells = X[random_idxs,:]
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

for i in range(n_cells):
    plt.arrow(random_cells_proj[i,0], random_cells_proj[i,1],
              (random_drifts_proj[i,0] - random_cells_proj[i,0])*5,
              (random_drifts_proj[i,1] - random_cells_proj[i,1])*5,
              color='red', alpha=.5, width=.002)
#%%
X.requires_grad = True
dq_dx, dq_dt = celldelta.pxt.dx_dt(X, ts)
ux, du_dx = celldelta.ux.div(X)
d_dx = ((dq_dx * ux).sum(dim=2) + du_dx)[...,None]
X.requires_grad = False

#%%
fig,axs = plt.subplots(5,1, figsize=(7,15))
tis = np.linspace(0, len(ts)-1, 5, dtype=int)
dq_dt_min = dq_dt.min().item()
dq_dt_max = dq_dt.max().item()
for i in range(5):
    ti = tis[i]
    dqdti = dq_dt[ti]
    dqdti_mag = torch.norm(dqdti, dim=1).detach().cpu().numpy()

    axs[i].set_title(f't={int(ts[ti].item())}')
    axs[i].scatter(x_proj[:,0], x_proj[:,1], c=dqdti_mag, cmap='viridis',s=1)
                #    vmin=dq_dt_min, vmax=dq_dt_max)

    plt.colorbar(axs[i].collections[0], ax=axs[i])
plt.tight_layout()

#%%
plt.scatter(x_proj[:,0], x_proj[:,1], c=tonp(du_dx), cmap='viridis',s=1)
plt.colorbar()

#%%
plt.scatter(x_proj[:,0], x_proj[:,1], c=tonp((ux**2).sum(1)), cmap='viridis',s=1)
plt.colorbar()

#%%
fig,axs = plt.subplots(5,2, figsize=(15,15))
tis = np.linspace(0, len(ts)-1, 5, dtype=int)
for i in range(5):
    ti = tis[i]
    dqdxi = (dq_dx[ti].sum(1))# * ux).sum(1) + du_dx
    dqdxi_mag = dqdxi.detach().cpu().numpy()

    axs[i][0].set_title(f't={int(ts[ti].item())}')
    axs[i][0].scatter(x_proj[:,0], x_proj[:,1], c=dqdxi_mag, cmap='viridis',s=1)
    plt.colorbar(axs[i][0].collections[0], ax=axs[i][0])
    dqdti = dq_dt[ti]
    dqdti_mag = torch.norm(dqdti, dim=1).detach().cpu().numpy()

    axs[i][1].set_title(f't={int(ts[ti].item())}')
    axs[i][1].scatter(x_proj[:,0], x_proj[:,1], c=dqdti_mag, cmap='viridis',s=1)

    plt.colorbar(axs[i][1].collections[0], ax=axs[i][1])

plt.tight_layout()

# %%
#%%
# for i in range(0,len(ts),len(ts)//10):
#     plt.plot(X, dq_dt[i,:], c='r')
#     plt.plot(X, dq_dx[i,:], c='blue')
# labels = ['d/dt p(x,t)', 'd/dx u(x) p(x,t)']
# plt.legend(labels)

# %%
# Scatter plot of all the simulation trajectories
xts_flat = xts.reshape(-1, xts.shape[-1])
random_idxs = np.random.randint(0, xts_flat.shape[0], 50_000)
xts_flat = xts_flat[random_idxs,:]
t_idxs = ts.repeat((xts.shape[1])).reshape(-1)[random_idxs]
xts_proj = pca.transform(xts_flat)
# plt.scatter(x_proj[:,0], x_proj[:,1], c=ts_np[pxts.argmax(axis=1)], cmap='b', alpha=.1, s=1)
plt.scatter(x_proj[:,0], x_proj[:,1], c='black', cmap=viridis, alpha=1, s=1)
plt.scatter(xts_proj[:,0], xts_proj[:,1], s=.5, alpha=.1)

#%%
# Plot the sum of the p(x,t) at each timestep
plt.title('Sum of p(x,t)')
plt.plot(pxts.sum(axis=0))

# %%
