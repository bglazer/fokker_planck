#%%
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
# import scanpy as sc
from scipy.stats import rv_histogram
from torch.nn import Linear, ReLU

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

#%%
# genotype='wildtype'
# dataset = 'net'
# adata = sc.read_h5ad(f'../../data/{genotype}_{dataset}.h5ad')
# data = adata.layers['raw']
# X = data.toarray()
device = 'cuda:0'

#%%
# This generates a scipy continuous distribution from a histogram
def gen_p(x, bins=100):
    hist = np.histogram(x, bins=bins, density=True)
    phist = rv_histogram(hist, density=True)
    return phist

# %%
# The u(x) drift term of the Fokker-Planck equation, modeled by a neural network
class Ux(torch.nn.Module):
    def __init__(self, hidden_dim, n_layers, device):
        super(Ux, self).__init__()
        # The drift term of the Fokker-Planck equation, modeled by a neural network
        self.model = MLP(1, 1, hidden_dim, n_layers).to(device)

    def forward(self, x):
        return self.model(x)
    
    # Compute the derivative of u(x) with respect to x
    def dx(self, x, hx=1e-3):
        xgrad = (self(x+hx) - self(x-hx))/(2*hx)
        return xgrad

# The p(x,t) term of the Fokker-Planck equation, modeled by a neural network
class Pxt(torch.nn.Module):
    def __init__(self, hidden_dim, n_layers, device):
        super(Pxt, self).__init__()
        self.model = MLP(2, 1, hidden_dim, n_layers).to(device)
        self.device = device

    def p(self, x, ts):
        xs = x.repeat((1,ts.shape[0])).T.unsqueeze(2)
        tss = ts.repeat((x.shape[0],1)).T.unsqueeze(2).to(device)
        xts = torch.concatenate((xs,tss), dim=2)
        ps = torch.exp(pxt.model(xts))
        return ps

    # Compute the probability density p(x,t) using the neural network
    def forward(self, x, ts):
        p = self.p(x, ts)
        return p
    
    # Compute the partial derivative of p(x,t) with respect to x
    def dx(self, x, ts, hx=1e-3):
        xgrad = (self.p(x+hx, ts) - self.p(x-hx, ts))/(2*hx)
        return xgrad

    # Compute the partial derivative of p(x,t) with respect to t
    def dt(self, x, ts, ht=1e-3):
        tgrad = (self.p(x, ts+ht) - self.p(x, ts-ht))/(2*ht)
        return tgrad

# %%

#%%
# Generate simple test data
# Initial distribution
X0 = torch.randn((1000, 1), device=device)
# Final distribution is initial distribution plus another normal shifted by +4
X1 = torch.randn((1000, 1), device=device)+4
X1 = torch.vstack((X0, X1))
# Plot the two distributions
_=plt.hist(X0.cpu().numpy(), bins=30, alpha=.3)
_=plt.hist(X1.cpu().numpy(), bins=30, alpha=.3)
#%%
epochs = 500
steps = 1
hx = 1e-3
ht = 1e-3
#%%
# Generate a continuous distribution from the data
pD = gen_p(X1.cpu().numpy(), bins=100)
# Initialize the neural networks
pxt = Pxt(20, 3, device)
ux = Ux(20, 3, device)
# Initialize the optimizers
pxt_optimizer = torch.optim.Adam(pxt.parameters(), lr=1e-3)
ux_optimizer = torch.optim.Adam(ux.parameters(), lr=1e-3)
#%%
ts = torch.linspace(0, 1, 100, device=device, requires_grad=False)
# ## %%
# # NOTE: This is a pre-training step to get p(x, t=0) to match the initial condition
# # train  p(x, t=0)=p_D0 for all x in the overall distribution
pD0 = gen_p(X0.cpu().numpy(), bins=100)
x0 = torch.linspace(X1.min(), X1.max(), 1000, device=device, requires_grad=False)[:,None]
pX_D0 = torch.tensor(pD0.pdf(x0.cpu().numpy()), device=device, dtype=torch.float32, requires_grad=False)
pxt0_optimizer = torch.optim.Adam(pxt.parameters(), lr=1e-3)
zero = torch.zeros(1)
for i in range(500):
    pxt0_optimizer.zero_grad()
    l_pD0 = ((pxt(x0, ts=zero) - pX_D0)**2).mean()
    l_pD0.backward()
    # torch.nn.utils.clip_grad_norm_(pxt.parameters(), .001)
    pxt0_optimizer.step()
    print(f'{i} l_pD0={float(l_pD0.mean()):.5f}')
#%%
ts = torch.linspace(0, 1, 100, device=device, requires_grad=False)
ht = ts[1] - ts[0]

l_fps = np.zeros(epochs)
l_Spxts = np.zeros(epochs)
l_p0s = np.zeros(epochs)
for epoch in range(1000):
    # Sample from the data distribution
    x = pD.rvs(size=1000)
    px = torch.tensor(pD.pdf(x), device=device, dtype=torch.float32, requires_grad=False)
    x = torch.tensor(x, device=device, dtype=torch.float32, requires_grad=False)[:,None]

    pxt_optimizer.zero_grad()
    ux_optimizer.zero_grad()
    # This is the initial condition p(x, t=0)=p_D0 
    l_p0 = ((pxt(x0, ts=zero) - pX_D0)**2).mean()
    l_p0.backward()

    # This is the marginal p(x) = int p(x,t) dt
    Spxt = pxt(x, ts).sum(dim=0) * ht
    # Ensure that the marginal p(x) matches the data distribution
    l_Spxt = ((Spxt - px)**2).mean()
    l_Spxt.backward()

    # This is the calculation of the term that ensures the
    # derivatives match the Fokker-Planck equation
    # d/dx p(x,t) = -d/dt (u(x) p(x,t))
    up_dx = (ux(x+hx) * pxt(x+hx, ts) - ux(x-hx) * pxt(x-hx, ts))/(2*hx)
    pxt_dts = pxt.dt(x, ts)
    l_fp = ((pxt_dts + up_dx)**2).mean()

    l_fp.backward()

    pxt_optimizer.step()
    ux_optimizer.step()

    print(f'{i} l_fp={float(l_fp.mean()):.5f}, l_px={float(l_Spxt.mean()):.5f}, l_p0={float(l_p0.mean()):.5f}')
    l_fps[epoch] = float(l_fp.mean())
    l_Spxts[epoch] = float(l_Spxt.mean())
    l_p0s[epoch] = float(l_p0.mean())   
#%%
plt.plot(l_fps[10:], label='l_fp')
plt.plot(l_Spxts[10:], label='l_Spxt')
plt.plot(l_p0s[10:], label='l_p0')
plt.legend()

# %%
# Plot the predicted p(x,t) at each timestep t
colors = matplotlib.colormaps.get_cmap('viridis')
xs = torch.arange(X1.min(), X1.max(), .01, device=device)[:,None]


pxst = pxt(xs, ts).squeeze().T.cpu().detach().numpy()
uxs = ux(xs).squeeze().cpu().detach().numpy()
xs = xs.squeeze().cpu().detach().numpy()
plt.plot(xs, pD.pdf(xs), c='k', alpha=1)
plt.plot(xs, pD0.pdf(xs), c='k', alpha=1)
for i in range(0, ts.shape[0], 10):
    t = ts[i]
    plt.plot(xs, pxst[:,i], c=colors(float(t)))
# Plot the data distribution
# plt.colorbar()

# %%
# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
cum_pxst = pxst.cumsum(axis=1) / np.arange(1, ts.shape[0]+1)
plt.imshow(cum_pxst, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
# %%
# This plots the error of the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
pxs = pD.pdf(xs)
plt.imshow(pxs[:,None] - cum_pxst, aspect='auto', interpolation='none', cmap='RdBu')
plt.colorbar()

#%%
# This is the individual p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
plt.imshow(pxst, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
# %%
# Plot the u(x) term for all x
fig, ax1 = plt.subplots(1,1, figsize=(10,5))
ax1.plot(xs, uxs, label='u(x)')
# Add vertical and horizontal grid lines
ax1.grid()
ax1.set_ylabel('u(x)')
ax1.set_xlabel('x')
ax2 = ax1.twinx()
ax2.plot(xs, pD.pdf(xs), c='k', alpha=1, label='p(x)')
ax2.set_ylabel('p(x)')
fig.legend()
# %%
