#%%
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
# import scanpy as sc
import torch.distributions as D
from 



#%%
# genotype='wildtype'
# dataset = 'net'
# adata = sc.read_h5ad(f'../../data/{genotype}_{dataset}.h5ad')
# data = adata.layers['raw']
# X = data.toarray()
device = 'cuda:0'

#%%
# Generate simple test data
# Initial distribution
X0 = torch.randn((1000, 1), device=device)
# Final distribution is initial distribution plus another normal shifted by +4
X1 = torch.randn((1000, 1), device=device)+4
X = torch.vstack((X0, X1))
# Plot the two distributions
_=plt.hist(X0.cpu().numpy(), bins=30, alpha=.3, label='X0')
# _=plt.hist(X1.cpu().numpy(), bins=30, alpha=.3, label='X1')
_=plt.hist(X.cpu().numpy(), bins=30, alpha=.3, label='X')
plt.legend()
#%%
ts = torch.linspace(0, 1, 100, device=device, requires_grad=False)
epochs = 1500
n_samples = 1000
hx = 1e-3
ht = ts[1] - ts[0]
zero = torch.zeros(1, requires_grad=False).to(device)

noise = D.MultivariateNormal(loc=torch.ones(1, device=device)*2, 
                             covariance_matrix=torch.eye(1, device=device)*6)


#%%




#%%
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
low = float(X.min())
high = float(X.max())
l = low-.25*(high-low) 
h = high+.25*(high-low)
xs = torch.arange(l, h, .01, device=device)[:,None]

pxts = pxt(xs, ts).squeeze().T.cpu().detach().numpy()
uxs = ux(xs).squeeze().cpu().detach().numpy()
up_dx = (ux(xs+hx) * pxt(xs+hx, ts) - ux(xs-hx) * pxt(xs-hx, ts))/(2*hx)
pxt_dts = pxt.dt(xs, ts)
up_dx = up_dx.detach().cpu().numpy()[:,:,0]
pxt_dts = pxt_dts.detach().cpu().numpy()[:,:,0]

xs = xs.squeeze().cpu().detach().numpy()
#%%
# Plot the predicted p(x,t) at each timestep t
plt.title('p(x,t)')
# plt.hist(X.cpu().numpy(), bins=30, alpha=.3, label='X', density=True)
# plt.hist(X0.cpu().numpy(), bins=30, alpha=.3, label='X0')
# plt.plot(xs, pD.pdf(xs), c='k', alpha=1)
# plt.plot(xs, pD0.pdf(xs), c='k', alpha=1)
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
for i in range(0,len(ts),len(ts)//10):
    plt.plot(xs, pxt_dts[i,:], c='r')
    plt.plot(xs, up_dx[i,:], c='blue')
labels = ['d/dt p(x,t)', 'd/dx u(x) p(x,t)']
plt.legend(labels)

# %%
# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
plt.title('Cumulative mean of p(x,t)')
sim_cum_pxt = pxts.cumsum(axis=1) / np.arange(1, ts.shape[0]+1)
plt.imshow(sim_cum_pxt, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('x')
plt.xlabel('timestep (t)')
plt.colorbar()
# %%
# This plots the error of the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
pxs = pD.pdf(xs)
plt.title('Error of cumulative mean of p(x,t)')
plt.imshow(pxs[:,None] - sim_cum_pxt, aspect='auto', interpolation='none', cmap='RdBu')
plt.ylabel('x')
plt.xlabel('timestep (t)')
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
# ax2.plot(xs, pD.pdf(xs), c='k', alpha=1, label='p(x)')
ax2.set_ylabel('p(x)')
fig.legend()
# %%
# Simulate the stochastic differential equation using the Euler-Maruyama method
# with the learned drift term u(x)
x = X0.clone().detach()
xts = []
ts = torch.linspace(0, 1, 100, device=device, requires_grad=False)
ht = ts[1] - ts[0]
for i in range(len(ts)):
    # Compute the drift term
    u = ux(x)
    # Compute the diffusion term
    # Generate a set of random numbers
    dW = torch.randn_like(x) * torch.sqrt(ht)
    sigma = torch.ones_like(x)
    # Compute the change in x
    dx = u * ht + sigma * dW
    # print(f'{float(u.mean()):.5f}, {float(ht):.3f}, {float(dW.mean()): .5f}, {float(dx.mean()): .5f}')
    dx = dx.squeeze(0)
    # Update x
    x = x + dx
    xts.append(x.cpu().detach().numpy())
xts = np.concatenate(xts, axis=1)
#%%
# Plot the resulting probability densities at each timestep
low = float(xs.min())
high = float(xs.max())
bins = np.linspace(low, high, 60)
w = bins[1] - bins[0]
for i in range(0, ts.shape[0], ts.shape[0]//10):
    t=ts[i]
    heights,bins = np.histogram(xts[:,i], 
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
sim_pxts = np.zeros((bins.shape[0]-1, ts.shape[0]))
for i in range(0, ts.shape[0]):
    heights,bins = np.histogram(xts[:,i], 
                                bins=bins,
                                density=True)
    sim_pxts[:,i] = heights

sim_cum_pxt = sim_pxts.cumsum(axis=1) / np.arange(1, sim_pxts.shape[1]+1)[None,:]

# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
plt.title('Cumulative mean of p(x,t)')
plt.imshow(sim_cum_pxt, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('x')
plt.xlabel('timestep (t)')
plt.colorbar() 

#%%
# Plot the final cumulative distribution of simulations versus the data distribution
plt.title('Final cumulative distribution of simulations vs data distribution')
z = sim_cum_pxt[:,-1].sum()
plt.plot(bins[:-1], sim_cum_pxt[:,-1]/z, label='Simulation')
z = pD.pdf(bins[:-1]).sum()
plt.plot(bins[:-1], pD.pdf(bins[:-1])/z, label='Data')
plt.legend()
plt.ylabel('Cumulative distribution')
plt.xlabel('x')


# %%
# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
pX = pD.pdf(bins[:-1])
sim_cum_pxt_err = (pX[:,None] - sim_cum_pxt)**2

plt.title('Error of Cumulative mean of p(x,t)\n'
          f'Error ∫pxt(x,t)dt = {l_nce_pxs:.5f}\n'
          f'Error Simulation = {sim_cum_pxt_err[:,-1].mean():.5f}')
plt.imshow(sim_cum_pxt_err, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('x')
plt.xlabel('timestep (t)')
plt.colorbar() 
# %%
