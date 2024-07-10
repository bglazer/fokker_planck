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
X0_mask = torch.zeros_like(X_t, dtype=bool)
X0_mask[:,0] = True
X0_mask = X0_mask.flatten()
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
                                               n_epochs=1500,
                                               scale=ts.shape[0], 
                                               verbose=True)
#%%
p0_alpha = None
fokker_planck_alpha = 1
losses = celldelta.optimize(X, X0_mask, ts,
                            pxt_lr=1e-4, ux_lr=1e-4, 
                            n_epochs=2500, n_samples=n_samples, 
                            px_noise=noise, p0_noise=noise0, 
                            fokker_planck_alpha=fokker_planck_alpha,
                            p0_alpha=p0_alpha, 
                            verbose=True)

#%%
# Optimize the Fokker-Planck term
fokker_planck_alpha = 1000
_ = celldelta.optimize_fokker_planck(X, ts,
                                     ux_lr=1e-4,
                                     fokker_planck_alpha=fokker_planck_alpha,
                                     noise=None,
                                     n_epochs=1000, 
                                     n_samples=n_samples,
                                     verbose=True)


# %%
# Calculate the u(x) and p(x,t) for each cell 
low = float(X.min())
high = float(X.max())
l = low-.25*(high-low) 
h = high+.25*(high-low)
xs = torch.arange(l, h, .01, device=device)[:,None]

pxts = np.exp(celldelta.pxt.log_pxt(xs, ts).squeeze().T.cpu().detach().numpy())
uxs = celldelta.ux(xs).squeeze().cpu().detach().numpy()
xs = xs.squeeze().cpu().detach().numpy()
#%%
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
plt.colorbar(sm, label='timestep (t)', ax=plt.gca())
plt.legend();

# %%
# Plot the u(x) term for all x
xs = torch.arange(l, h, .01, device=device)[:,None]
uxs = celldelta.ux(xs).squeeze().cpu().detach().numpy()
xs = xs.squeeze().cpu().detach().numpy()
fig, ax1 = plt.subplots(1,1, figsize=(10,5))
plt.title('u(x) vs p(x)')
ax1.plot(xs[1000:], uxs[1000:], c='blue', alpha=.6)
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

#%%
plt.plot(pxts.max(axis=0), marker='o', markersize=2, c='blue', label='max p(x,t)')
plt.ylabel('max p(x,t)', c='blue')

# Make a second y axis
plt.twinx()
plt.plot(pxts.sum(axis=0), marker='o', markersize=2, c='orange', label='sum p(x,t)')
plt.ylabel('sum p(x,t)', c='orange')
print(f'Min sum pxt {pxts.sum(axis=0).min():.2e} t={pxts.sum(0).argmin()}')
print(f'Max sum pxt {pxts.sum(axis=0).min():.2e} t={pxts.sum(0).argmin()}')

# %%
# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].set_title('Cumulative mean of p(x,t)')
sim_cum_pxt = pxts.cumsum(axis=1) / np.arange(1, ts.shape[0]+1)
axs[0].imshow(sim_cum_pxt, aspect='auto', interpolation='none', cmap='viridis')
axs[0].set_ylabel('x')
axs[0].set_yticks(ticks=np.linspace(0, sim_cum_pxt.shape[0], 10), 
           labels=np.round(np.linspace(xs.min(), xs.max(), 10), 2))
axs[0].set_xlabel('timestep (t)')
axs[0].set_xticks(ticks=np.linspace(0, sim_cum_pxt.shape[1], 10),
              labels=np.round(np.linspace(0, 1, 10), 2))
axs[0].set_yticks(ticks=np.linspace(0, sim_cum_pxt.shape[0], 10),
                  labels=np.round(np.linspace(xs.min(), xs.max(), 10), 0))

# plt.colorbar(axs[0].collections[0], ax=axs[0])

# This is the individual p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
axs[1].set_title('p(x,t) at each timestep t')
axs[1].imshow(pxts, aspect='auto', interpolation='none', cmap='viridis')
axs[1].set_ylabel('x')
axs[1].set_xlabel('timestep (t)')

# %%
# Simulate the stochastic differential equation using the Euler-Maruyama method
# with the learned drift term u(x)
x = X0.clone().detach()
zero_boundary = True
ts.requires_grad = False
xts = celldelta.simulate(x, ts, zero_boundary=zero_boundary, sigma=0)

#%%
# Plot the resulting probability densities at each timestep
low = float(xs.min())
high = float(xs.max())
bins = np.linspace(low, high, 60)
w = bins[1] - bins[0]
for i in np.linspace(0, ts.shape[0]-1, 20, dtype=int):
    t=ts[i]
    heights,bins = np.histogram(xts[i,:], 
                                bins=bins,
                                density=True)
    plt.bar(bins[:-1], heights, width=w, color=viridis(float(t)/max(ts).item()), alpha=.2)

for i in np.linspace(0, ts.shape[0]-1, 20, dtype=int):
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
#%%
# Plot the dq_dx  and dq_dt terms for the Fokker-Planck equation
xs = torch.arange(l, h, .01, device=device, requires_grad=True)[:,None]
dq_dx, dq_dt = celldelta.pxt.dx_dt(xs, ts)
du_dx = celldelta.ux.div(xs)[1]
uxs = celldelta.ux(xs)
d_dx = ((dq_dx * uxs).sum(2) + du_dx)[...,None]
fp_err = (d_dx + dq_dt.sum(0))
dq_dx = tonp(dq_dx)
du_dx = tonp(du_dx)
dq_dt = tonp(dq_dt)
d_dx = tonp(d_dx)
div_x = celldelta.ux.div(xs)[1]
div_x = tonp(div_x)
xs = tonp(xs)
uxs = tonp(uxs)
fp_err = tonp(fp_err)
miny, maxy = min(dq_dx.min(), dq_dt.min())*1.1, max(dq_dx.max(), dq_dt.max())*1.1
reds = matplotlib.cm.get_cmap('Reds')

tsi = np.linspace(0, len(ts)-1, 5, dtype=int)
fig, axs = plt.subplots(len(tsi), 1, figsize=(6,5*len(tsi)))

span = slice(2974,(2974*2))

for i in range(5):
    # axs[i].plot(xs[span], (d_dx[tsi[i]])[span], 
    #             label='d_dx', 
    #             alpha=1, c='red', linewidth=.5)
    # axs[i].plot(xs[span], (dq_dt[tsi[i]])[span], 
    #             label='dq_dt', 
    #             alpha=1, c='blue', linewidth=.5)
    # axs[i].plot(xs[span], (dq_dx[tsi[i]])[span],
    #             label='dq_dx', alpha=1, c='green', linewidth=.5)
    # axs[i].plot(xs[span], (uxs[span]),
    #             label='ux', alpha=1, c='orange', linewidth=.5)
    axs[i].plot(xs[span], (dq_dx*uxs)[tsi[i]][span],
                label='dq_dx * u(x)', 
                alpha=1, c='black', linewidth=.5)
    axs[i].plot(xs[span], (d_dx[tsi[i]] + dq_dt[tsi[i]])[span],
                label='Fokker-Planck error', 
                alpha=1, c='magenta', linewidth=.5)
    # Summed FP error
    # axs[i].plot(xs[span], (((d_dx + dq_dt)**2).mean(0))[span],
    #             label='Fokker-Planck error', 
    #             alpha=1, c='purple', linewidth=.5)
    # Cumulative FP error
    # axs[i].plot(xs[span], np.cumsum(np.abs((fp_err[tsi[i]])[span]))/np.sum(np.abs(fp_err[tsi[i]][span])),
    #             label='Cumulative Fokker-Planck error', 
    #             alpha=1, c='brown', linewidth=.5)
    axs[i].plot(xs[span], div_x[span],
                label='div u(x)', 
                alpha=1, c='blue', linewidth=.5)
    axs[i].axhline(0, c='grey', alpha=.6, linewidth=.4)
    axs[i].axvline(0, c='grey', alpha=.6, linewidth=.4)
# Put the legend on each plot
for ax in axs:
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


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
#%%
# Higher Dimensional Gaussians
#####################################################
# Sequence of distributions
# X_t = Normal(1+t) 
device = 'cuda:1'
N = 99
tsteps = 100
d = 50
tscale = 10
ts = torch.linspace(0, 1, tsteps, device=device)*tscale
ts_np = ts.cpu().detach().numpy()
X = torch.randn((N, d, tsteps), device=device)
# Create a sequence of steps to add to each timestep
v = torch.arange(0, tsteps, device=device)
# Repeat the sequence for each dimension of the data
v = v.repeat((1, d, 1))
# Zero out the first dimension
v[:,0] = 0
X_t = X + v
# Change X_t to shape (N, tsteps, d)
X_t = X_t.permute(0, 2, 1)
# Flatten X_t to shape (N*tsteps, d)
X = X_t.reshape((N*tsteps, d))
X0 = X_t[:, 0, :] 
X0_mask = torch.zeros((N,tsteps), dtype=bool)
X0_mask[:,0] = True
X0_mask = X0_mask.flatten()
# Compute PCA of the data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_proj = pca.fit_transform(X.cpu().numpy())
# %%
# Initialize the model
celldelta = CellDelta(input_dim=d, device=device,
                      ux_hidden_dim=64, ux_layers=2, 
                      pxt_hidden_dim=64, pxt_layers=2,
                      batch_norm=True
                      )

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
losses = celldelta.optimize_initial_conditions(X0, ts, p0_noise=noise, 
                                               scale=1/ts.shape[0],
                                               pxt_lr=1e-3,
                                               n_epochs=600, 
                                               verbose=True)
#%%
start = time.time()
n_samples = 1000
p0_alpha = 1
fokker_planck_alpha = 1
l_consistency_alpha = None
pt_alpha = None
ux_lr  = 1e-3
pxt_lr = 1e-2

celldelta.pxt.model.set_tscale(1)

losses = celldelta.optimize(X=X, 
                            X0=X0, 
                            X0_mask=X0_mask,
                            ts=ts, 
                            pxt_lr=pxt_lr, 
                            ux_lr=ux_lr,
                            n_epochs=5000, 
                            n_samples=n_samples, 
                            p0_noise=noise0,
                            px_noise=noise, 
                            fokker_planck_alpha=fokker_planck_alpha,
                            p0_alpha=p0_alpha, 
                            pt_alpha=pt_alpha,
                            l_consistency_alpha=l_consistency_alpha,
                            verbose=True)

end = time.time()
print(f'Time elapsed: {end-start:.2f}s')
#%%
fokker_planck_alpha = 10

_=celldelta.optimize_fokker_planck(X, ts,
                                   ux_lr=1e-4,
                                   fokker_planck_alpha=fokker_planck_alpha,
                                   noise=None,
                                   n_epochs=500, 
                                   n_samples=n_samples,
                                   verbose=True)

#%%
start = time.time()
n_samples = 1000
p0_alpha = 1
fokker_planck_alpha = 1
l_consistency_alpha = None
ux_lr  = 1e-3
pxt_lr = 1e-3

losses = celldelta.optimize(X=X, 
                            X0=X0, 
                            ts=ts, 
                            pxt_lr=pxt_lr, 
                            ux_lr=ux_lr,
                            n_epochs=5000, 
                            n_samples=n_samples, 
                            p0_noise=noise0,
                            px_noise=noise, 
                            fokker_planck_alpha=fokker_planck_alpha,
                            p0_alpha=p0_alpha, 
                            l_consistency_alpha=l_consistency_alpha,
                            verbose=True)

end = time.time()
print(f'Time elapsed: {end-start:.2f}s')


#%%
fokker_planck_alpha = 1000
for fp_noise_scale in np.linspace(0.01, 10, 10):
    losses = celldelta.optimize_fokker_planck(X, ts,
                                              ux_lr=1e-4,
                                              fokker_planck_alpha=fokker_planck_alpha,
                                              noise=fp_noise_scale,
                                              n_epochs=1000, 
                                              n_samples=n_samples,
                                              verbose=True)
# plt.plot(losses['l_fp'], label='l_fp')
# plt.plot(losses['l_fp0'], label='l_fp0')

# %%
xs = X.clone().detach()
pxts = celldelta.pxt(xs, ts).squeeze().T.cpu().detach().numpy()
uxs = celldelta.ux(xs).squeeze().cpu().detach().numpy()
# _, pxt_dts = celldelta.pxt.dx_dt(xs, ts)
# pxt_dts = pxt_dts.detach().cpu().numpy()[:,:,0]

xs = xs.squeeze().cpu().detach().numpy()
#%%
# Plot the pseudotime i.e. the timestep of maximum probability for each cell
viridis = matplotlib.colormaps.get_cmap('viridis')
fig, axs = plt.subplots(3,1, figsize=(10,15))

ts_np = ts.detach().cpu().numpy()
axs[0].set_title('Pseudotime, max_t p(x,t)')
axs[0].scatter(x_proj[:,0], x_proj[:,1], c=ts_np[pxts.argmax(axis=1)], cmap=viridis, s=1, 
            vmin=ts[0], vmax=ts[-1])
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2');
plt.colorbar(axs[0].collections[0], ax=axs[0])

# Plot the histogram of differences from the true pseudotime
pt = pxts.argmax(axis=1).reshape((N,-1))
true_pt = np.arange(0,ts.shape[0],dtype=int).repeat(N).reshape(tsteps,N).T
diffs = (pt-true_pt).flatten()
diffs = np.sort(diffs)
axs[1].hist(diffs, bins=np.arange(diffs.min(),diffs.max()))
axs[1].set_title('Count of differences of estimated versus true pseudotime')
# Set the ticks of the x-axis to be in the middle of the bins
axs[1].set_xticks(np.arange(diffs.min(),diffs.max())+0.5, 
                  labels=np.arange(diffs.min(),diffs.max()), rotation=90)

# Make a empirical cumulative distribution of the differences
abs_diffs = np.sort(np.abs(diffs))
n = abs_diffs.shape[0]
axs[2].plot(abs_diffs, np.arange(0,n)/n)
axs[2].axhline(0.95, c='red', linestyle='--', linewidth=1)
pct95 = abs_diffs[np.where(np.arange(0,n)/n > .95)[0][0]]
axs[2].axvline(pct95, c='red', linestyle='--', linewidth=1)
# Label the 95th percentile, pad the label so it doesn't overlap with the line
axs[2].text(pct95+abs_diffs.max()/100, 0, f'95th pct={pct95:d}')
axs[2].set_title('Empirical CDF of absolute differences of estimated versus true pseudotime')

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
#
# Plot the p(x,t) for a new timestep at -1
# Xn1 = torch.randn((N, d), device=device)-1
# pxtn1 = celldelta.pxt(Xn1, ts).squeeze().T.cpu().detach().numpy()
# x_proj_n1 = pca.transform(Xn1.cpu().numpy())
# plt.scatter(x_proj[:,0], x_proj[:,1], c=pxts[:,0], cmap=viridis, s=1)
# plt.scatter(x_proj_n1[:,0], x_proj_n1[:,1], c=pxtn1[:,0], cmap=viridis, s=10, marker='x')
# plt.colorbar()

#%%
plt.plot(pxts.mean(axis=0), marker='o', markersize=2, c='blue')
plt.ylabel('mean p(x,t)', c='blue')
plt.twinx()
plt.plot(pxts.max(axis=0), marker='o', markersize=2, c='orange')
plt.ylabel('max p(x,t)', c='orange')
#%%
# Plot arrows pointing in the direction of the drift term u(x)
# Select a random subset of cells
n_cells = 503

random_idxs = torch.randperm(X.shape[0])[:n_cells]
random_cells = X[random_idxs,:]
# Get the drift term u(x) for each cell
ux = celldelta.ux(X)
uxs = ux[random_idxs,:]
 
# Add the uxs to the random_cells
random_drifts = random_cells + uxs
# Project the random_cells and random_drifts onto the PCA components
random_cells_proj = pca.transform(random_cells.detach().cpu().numpy())
random_drifts_proj = pca.transform(random_drifts.detach().cpu().numpy())
fig, axs = plt.subplots(1,2, figsize=(20,10))
# Plot all the cells
axs[0].scatter(x_proj[:,0], x_proj[:,1], c='grey', s=.5, alpha=.5)
# Plot the random cells
axs[0].scatter(random_cells_proj[:,0], random_cells_proj[:,1], c='black', s=1)
# Scaling factor for the arrow length
arrow_scale = .5

# Plot the random drifts as arrows from the random cells
for i in range(n_cells):
    axs[0].arrow(random_cells_proj[i,0], random_cells_proj[i,1],
                 (random_drifts_proj[i,0] - random_cells_proj[i,0])*arrow_scale,
                 (random_drifts_proj[i,1] - random_cells_proj[i,1])*arrow_scale,
                 color='red', alpha=.5, width=.002)

noise_vectors = False
if noise_vectors:
    noise_sample_size = 200
    ux_noise_scale = 10
    ux_noise = D.MultivariateNormal(torch.zeros(d, device=device), 
                                    torch.eye(d, device=device)*ux_noise_scale)
    random_idxs = torch.randperm(X.shape[0])[:noise_sample_size]
    noise_sample = X[random_idxs,:] + ux_noise.sample(sample_shape=(noise_sample_size,))
    noise_sample_proj = pca.transform(noise_sample.detach().cpu().numpy())
    noise_ux = celldelta.ux(noise_sample)
    noise_sample_drifts = noise_sample + noise_ux
    noise_sample_drifts_proj = pca.transform(noise_sample_drifts.detach().cpu().numpy())
    for i in range(noise_sample_size):
        axs[0].arrow(noise_sample_proj[i,0], noise_sample_proj[i,1],
                (noise_sample_drifts_proj[i,0] - noise_sample_proj[i,0])*arrow_scale,
                (noise_sample_drifts_proj[i,1] - noise_sample_proj[i,1])*arrow_scale,
                color='blue', alpha=.5, width=.002)
axs[1].scatter(x_proj[:,0], x_proj[:,1], c=tonp((ux**2).sum(1)), cmap='viridis',s=4)
axs[1].set_title('magnitude ux')
# Add a colorbar of the cmap from axs[1]
plt.colorbar(axs[1].collections[0], ax=axs[1])
print('UX mean magnitude:')
for i in range(ux.shape[1]):
    print(f'{(ux**2).mean(0)[i].item():.4f}')# %%
#%%
# Simulate the stochastic differential equation using the Euler-Maruyama method
# with the learned drift term u(x)
x0 = X_t[:,0,:].clone().detach()
x = x0

zero_boundary = False
max_t = 1
sigma=0.0
xts = celldelta.simulate(x, tsim=torch.linspace(0, max_t, 100, device=device)*tscale,
                         zero_boundary=zero_boundary, sigma=sigma)

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

# %%
# Scatter plot of all the simulation trajectories
scatter = True
if scatter:
    xts_flat = xts.reshape(-1, xts.shape[-1])
    random_idxs = np.random.randint(0, xts_flat.shape[0], 50_000)
    xts_flat = xts_flat[random_idxs,:]
    t_idxs = tonp(ts.tile((xts.shape[:2])).T.flatten()[random_idxs])
    xts_proj = pca.transform(xts_flat)
    # plt.scatter(x_proj[:,0], x_proj[:,1], c=ts_np[pxts.argmax(axis=1)], cmap='b', alpha=.1, s=1)
    plt.scatter(x_proj[:,0], x_proj[:,1], c='black', alpha=1, s=1)
    plt.scatter(xts_proj[:,0], xts_proj[:,1], s=.5, alpha=.5, c=t_idxs, cmap=viridis)
#%%
X.requires_grad = True
dq_dx, dq_dt = celldelta.pxt.dx_dt(X, ts)
ux, du_dx = celldelta.ux.div(X)
d_dx = ((dq_dx * ux).sum(dim=2) + du_dx)[...,None]
X.requires_grad = False
#%%0
x0n1 = X_t[:,:1].reshape((-1,d)).clone().detach()
# append X0 to the end of the sequence
# x0n1 = torch.cat((X0, x0n1), dim=0)
x0_proj = pca.transform(x0n1.cpu().numpy())
pxt0 = celldelta.pxt(x0n1, ts).squeeze().T.cpu().detach().numpy()
n_cells = 20
random_idxs = torch.randperm(x0n1.shape[0])[:n_cells]
random_cells = x0n1[random_idxs,:]
tis = np.linspace(0, len(ts)-1, 5, dtype=int)
fig, axs = plt.subplots(5,1, figsize=(7,15))
for i in range(5):
    # Get the drift term u(x) for each cell
    dq_dxs = dq_dx[:,random_idxs]#*ux[random_idxs]
    # Add the uxs to the random_cells
    random_drifts = random_cells + dq_dxs[tis[i]]
    # Project the random_cells and random_drifts onto the PCA components
    random_cells_proj = pca.transform(random_cells.detach().cpu().numpy())
    random_drifts_proj = pca.transform(random_drifts.detach().cpu().numpy())
    # Plot all the cells
    axs[i].scatter(x0_proj[:,0], x0_proj[:,1], c=pxt0[:,tis[i]], s=4, alpha=1)
    # Plot the random cells
    # axs[i].scatter(random_cells_proj[:,0], random_cells_proj[:,1], c='black', s=1)
    # Plot the random drifts as arrows from the random cells

    for j in range(n_cells):
        axs[i].arrow(random_cells_proj[j,0], random_cells_proj[j,1],
                (random_drifts_proj[j,0] - random_cells_proj[j,0])*5,
                (random_drifts_proj[j,1] - random_cells_proj[j,1])*5,
                color='red', alpha=.5, width=.002, head_width=.2)
    axs[i].set_title(f'dq_dx, t={int(ts[tis[i]].item())}')
    # Add a colorbar to each axis
    plt.colorbar(axs[i].collections[0], ax=axs[i])
plt.tight_layout()

#%%
# Plot the Fokker-Planck error term for each cell
fig, axs = plt.subplots(5,1, figsize=(7,15))
tis = np.linspace(0, len(ts)-1, 5, dtype=int)
for i in range(5):
    ti = tis[i]
    d_dx_ti = d_dx[ti]
    dq_dti = dq_dt[ti]
    fp_err = tonp(d_dx_ti+dq_dti)**2

    axs[i].set_title(f'Fokker-Planck error t={int(ts[ti].item())}')
    axs[i].scatter(x_proj[:,0], x_proj[:,1], c=fp_err, cmap='viridis',s=1,alpha=1)
    plt.colorbar(axs[i].collections[0], ax=axs[i])
plt.tight_layout()

#%%
plt.scatter(x_proj[:,0], x_proj[:,1], c=tonp(du_dx), cmap='viridis',s=1)
plt.title('div_ux')
plt.colorbar()

#%%
plt.scatter(x_proj[:,0], x_proj[:,1], c=tonp((ux**2).sum(1)), cmap='viridis',s=1)
plt.title('magnitude ux')
plt.colorbar()
print('UX mean magnitude:', [f'{x:.5}' for x in tonp(ux**2).mean(0)])

#%%
fig,axs = plt.subplots(6,2, figsize=(15,15))
tis = np.linspace(0, len(ts)-1, 5, dtype=int)
for i in range(5):
    ti = tis[i]
    dqdxi = (dq_dx[ti] * ux).sum(1) + du_dx
    dqdxi_mag = dqdxi.detach().cpu().numpy()

    axs[i][0].set_title(f'dq_dx t={int(ts[ti].item())}')
    axs[i][0].scatter(x_proj[:,0], x_proj[:,1], c=dqdxi_mag, cmap='viridis',s=1)
    plt.colorbar(axs[i][0].collections[0], ax=axs[i][0])
    dqdti = dq_dt[ti]
    dqdti_mag = torch.norm(dqdti, dim=1).detach().cpu().numpy()

    axs[i][1].set_title(f'dq_dt t={int(ts[ti].item())}')
    axs[i][1].scatter(x_proj[:,0], x_proj[:,1], c=dqdti_mag, cmap='viridis',s=1)

    plt.colorbar(axs[i][1].collections[0], ax=axs[i][1])

axs[5][0].scatter(x_proj[:,0], x_proj[:,1], c=tonp((dq_dx**2).sum((0,2))), cmap='viridis',s=1)
plt.colorbar(axs[5][0].collections[0], ax=axs[5][0])
axs[5][1].scatter(x_proj[:,0], x_proj[:,1], c=tonp((dq_dt**2).sum(0)), cmap='viridis',s=1)
plt.colorbar(axs[5][1].collections[0], ax=axs[5][1])

plt.tight_layout()

# %%
# Compute the gradient of the drift term u(x) with respect to the fokker-planck term
x.requires_grad = True
ts.requires_grad = True
dq_dx, dq_dt = celldelta.pxt.dx_dt(x, ts)
ux, div_ux = celldelta.ux.div(x)
ux.retain_grad()
dq_dx.retain_grad()
d_dx = ((dq_dx * ux).sum(dim=2) + div_ux)[...,None]

# Enforce that dq_dt = -dx, i.e. that both sides of the fokker planck equation are equal
l_fp = ((d_dx + dq_dt)**2).mean()
x.requires_grad = False
ts.requires_grad = False

l_fp.backward()
ux_grad = ux.grad
dq_dx_grad = dq_dx.grad
print(ux_grad.mean().item(), ux_grad.std().item())
print(dq_dx_grad.mean().item(), dq_dx_grad.std().item())
# %%
