#%%
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
# import scanpy as sc
from scipy.stats import rv_histogram
from torch.nn import Linear, ReLU
import torch.nn as nn
import torch.distributions as D

#%%
# Define models 

# Generic Dense Multi-Layer Perceptron (MLP), which is just a stack of linear layers with ReLU activations
# input_dim: dimension of input
# output_dim: dimension of output
# hidden_dim: dimension of hidden layers
# num_layers: number of hidden layers
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, input_bias=True):
        super(MLP, self).__init__()
        layers = []
        layers.append(Linear(input_dim, hidden_dim, bias=input_bias))
        layers.append(ReLU())
        for i in range(num_layers - 1):
            layers.append(Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(ReLU())
            # TODO do we need batch norm here?
        layers.append(Linear(hidden_dim, output_dim, bias=False))
        # layers.append(LeakyReLU())
        # Register the layers as a module of the model
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class NCE(nn.Module):
    def __init__(self, model, noise):
        super(NCE, self).__init__()
        # The normalizing constant logZ(θ)        
        self.c = nn.Parameter(torch.tensor([1.], requires_grad=True))

        self.model = model
        self.noise = noise

    def classify(self, x, ts):
        logp_x = self.log_px(x, ts)  # logp(x)
        logq_x = self.noise.log_prob(x).unsqueeze(1)  # logq(x)

        # Classification of noise vs target
        r_x = torch.sigmoid(logp_x - logq_x)
        return r_x
    
    def loss(self, x, ts, n_samples):
        # Generate samples from noise
        y = self.noise.sample((n_samples,))

        logp_x = self.model.log_px(x, ts)  # logp(x)
        logq_x = self.noise.log_prob(x).unsqueeze(1)  # logq(x)
        logp_y = self.model.log_px(y, ts)  # logp(y)
        logq_y = self.noise.log_prob(y).unsqueeze(1)  # logq(y)

        value_x = logp_x - torch.logsumexp(torch.cat([logp_x, logq_x], dim=1), dim=1, keepdim=True)  # logp(x)/(logp(x) + logq(x))
        value_y = logq_y - torch.logsumexp(torch.cat([logp_y, logq_y], dim=1), dim=1, keepdim=True)  # logq(y)/(logp(y) + logq(y))

        v = value_x.mean() + value_y.mean()

        # Classification of noise vs target
        r_x = torch.sigmoid(logp_x - logq_x)
        r_y = torch.sigmoid(logq_y - logp_y)

        # Compute the classification accuracy
        acc = ((r_x > 1/2).sum() + (r_y > 1/2).sum()).cpu().numpy() / (len(x) + len(y))
        
        return -v, acc

    def log_density(self, x, ts):
        return self.model.log_px(x, ts) - self.noise.log_prob(x).unsqueeze(1) - self.c
        
# Define models for the Fokker-Planck equation
# The u(x) drift term of the Fokker-Planck equation, modeled by a neural network
class Ux(torch.nn.Module):
    def __init__(self, hidden_dim, n_layers, device):
        super(Ux, self).__init__()
        # The drift term of the Fokker-Planck equation, modeled by a neural network
        self.model = MLP(1, 1, hidden_dim, n_layers).to(device)

    def forward(self, x):
        return self.model(x)
    
    # Compute the derivative of u(x) with respect to x
    def dx(self, x, ts, hx=1e-3):
        xgrad = (self(x+hx, ts) - self(x-hx, ts))/(2*hx)
        return xgrad

# The p(x,t) term of the Fokker-Planck equation, modeled by a neural network
class Pxt(torch.nn.Module):
    def __init__(self, hidden_dim, n_layers, device):
        super(Pxt, self).__init__()
        self.device = device

        self.model = MLP(2, 1, hidden_dim, n_layers).to(device)

    def log_pxt(self, x, ts):
        ts = ts if ts is None else ts
        # Repeat the x and t vectors for each timestep in the ts range
        xs = x.repeat((1, ts.shape[0])).T.unsqueeze(2)
        ts_ = ts.repeat((x.shape[0],1)).T.unsqueeze(2)
        # Concatentate them together to match the input the MLP model
        xts = torch.concatenate((xs,ts_), dim=2)
        log_ps = self.model(xts)
        # Ensure that the sum of the log_pxts is 1 for every t
        # log_ps = (log_ps.transpose(1,0) - torch.logsumexp(log_ps, dim=1).detach()).transpose(0,1)
        return log_ps
    
    def pxt(self, x, ts):
        return torch.exp(self.log_pxt(x, ts))

    def forward(self, x, ts):
        return self.pxt(x, ts)
    
    # Marginalize out the t dimension to get p(x)
    def log_px(self, x, ts):
        return torch.logsumexp(self.log_pxt(x, ts), dim=0) - torch.log(torch.tensor(ts.shape[0], device=self.device, dtype=torch.float32))
    
    # Compute the partial derivative of p(x,t) with respect to x
    def dx(self, x, ts, hx=1e-3):
        xgrad = (self.pxt(x+hx, ts) - self.pxt(x-hx, ts))/(2*hx)
        return xgrad

    # Compute the partial derivative of p(x,t) with respect to t
    def dt(self, x, ts, ht=1e-3):
        tgrad = (self.pxt(x, ts+ht) - self.pxt(x, ts-ht))/(2*ht)
        return tgrad

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

#%%
# Initialize the neural networks
pxt = Pxt(hidden_dim=20, n_layers=3, device=device)
ux = Ux(hidden_dim=20, n_layers=3, device=device)
noise = D.MultivariateNormal(loc=torch.ones(1, device=device)*2, 
                             covariance_matrix=torch.eye(1, device=device)*6)
nce = NCE(pxt, noise)

# Initialize the weights of pxt to be positive
# with torch.no_grad():
#     for param in pxt.parameters():
#         param.copy_(torch.abs(param)/10)
# Initialize the optimizers
pxt_optimizer = torch.optim.Adam(pxt.parameters(), lr=5e-4)
ux_optimizer = torch.optim.Adam(ux.parameters(), lr=1e-3)

#%%
l_nce_pxs = np.zeros(epochs)
l_p0s = np.zeros(epochs)
l_fps = np.zeros(epochs)
l_us = np.zeros(epochs)
l_pxt_dx = np.zeros(epochs)
for epoch in range(epochs):
    # Sample from the data distribution
    rand_idxs = torch.randperm(len(X))
    x = X[rand_idxs[:n_samples]]
    x0 = X0

    pxt_optimizer.zero_grad()
    ux_optimizer.zero_grad()

    # This is the initial condition p(x, t=0)=p_D0 
    l_nce_p0, acc_p0 = nce.loss(x0, ts=zero, n_samples=1000)
    l_nce_p0.backward()

    # p(x) marginalized over t, NCE loss
    l_nce_px, acc_px = nce.loss(x, ts=ts, n_samples=1000)
    l_nce_px.backward()

    # Record the losses
    l_nce_pxs[epoch] = float(l_nce_px.mean())
    l_p0s[epoch] = float(l_nce_p0.mean())   

    low = float(X.min())
    high = float(X.max())
    l = low-.25*(high-low) 
    h = high+.25*(high-low)
    x = torch.arange(l, h, .01, device=device)[:,None]

    # This is the calculation of the term that ensures the
    # derivatives match the Fokker-Planck equation
    # d/dx p(x,t) = -d/dt (u(x) p(x,t))
    up_dx = (ux(x+hx) * pxt(x+hx, ts) - ux(x-hx) * pxt(x-hx, ts))/(2*hx)
    pxt_dts = pxt.dt(x, ts)
    l_fp = ((pxt_dts + up_dx)**2).mean()

    l_fp.backward()

    # Take a gradient step
    pxt_optimizer.step()
    # ux_optimizer.step()
    l_fps[epoch] = float(l_fp.mean())
    # l_us[epoch] = float(l_u)
    print(f'{epoch} l_nce_px={float(l_nce_px):.5f}, acc_px={float(acc_px):.4f}, '
          f'l_nce_p0={float(l_nce_p0):.5f}, acc_p0={float(acc_p0):.4f}, '
          f'l_fp={float(l_fp):.5f}')



#%%
fig, axs = plt.subplots(3, 1, figsize=(10,10))
axs[0].plot(l_fps[10:], label='l_fp')
axs[1].plot(l_nce_pxs[10:], label='l_nce_pxs')
axs[2].plot(l_p0s[10:], label='l_nce_p0')
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
