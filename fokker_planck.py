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

    def p(self, x, t):
        # Convert scalar t to a tensor of the same shape as x, for input to the model
        t_ = torch.ones(x.shape, device=self.device, dtype=torch.float32)*t
        # I use exp(model) instead of model output directly because 
        # I want to ensure that p(x,t) is always positive
        return torch.exp(self.model(torch.hstack((x, t_))))

    # Compute the probability density p(x,t) using the neural network
    def forward(self, x, t):
        p = self.p(x, t)
        return p
    
    # Compute the partial derivative of p(x,t) with respect to x
    def dx(self, x, t, hx=1e-3):
        xgrad = (self.p(x+hx, t) - self.p(x-hx, t))/(2*hx)
        return xgrad

    # Compute the partial derivative of p(x,t) with respect to t
    def dt(self, x, t, ht=1e-3):
        tgrad = (self.p(x, t+ht) - self.p(x, t-ht))/(2*ht)
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

ts = torch.linspace(0, 1, 100, device=device, requires_grad=False)
# ## %%
# # NOTE: This is a pre-training step to get p(x, t=0) to match the initial condition
# # train  p(x, t=0)=p_D0 for all x in the overall distribution
pD0 = gen_p(X0.cpu().numpy(), bins=100)
x0 = torch.linspace(X1.min(), X1.max(), 1000, device=device, requires_grad=False)[:,None]
pX_D0 = torch.tensor(pD0.pdf(x0.cpu().numpy()), device=device, dtype=torch.float32, requires_grad=False)
pxt0_optimizer = torch.optim.Adam(pxt.parameters(), lr=1e-3)
for i in range(500):
    pxt0_optimizer.zero_grad()
    l_pD0 = ((pxt(x0, t=0) - pX_D0)**2).mean()
    l_pD0.backward()
    # torch.nn.utils.clip_grad_norm_(pxt.parameters(), .001)
    pxt0_optimizer.step()
    print(f'{i} l_pD0={float(l_pD0.mean()):.5f}')
#%%
for i in range(1000):
    x = pD.rvs(size=1000)
    px = torch.tensor(pD.pdf(x), device=device, dtype=torch.float32, requires_grad=False)
    x = torch.tensor(x, device=device, dtype=torch.float32, requires_grad=False)[:,None]

    pxt_optimizer.zero_grad()
    ux_optimizer.zero_grad()
    l_p0 = ((pxt(x0, t=0) - pX_D0)**2).mean()
    l_p0.backward()

    Spxt = torch.zeros_like(x, device=device, requires_grad=True)
    ht = ts[1] - ts[0]
    for t in ts:
        Spxt = Spxt + pxt(x, t) * ht
    l_Spxt = ((Spxt - px)**2).mean()
    l_Spxt.backward()

    l_fp = torch.zeros(1, device=device, requires_grad=True)
    # This is the term that ensures the derivatives match the Fokker-Planck equation
    # up is just the product of u(x) and p(x,t)
    up = lambda x, t: ux(x) * pxt(x, t)

    for t in ts:
        # This is: d/dx (u(x) * p(x,t))
        up_dx = -(up(x+hx,t) - up(x-hx,t))/(2*hx)
        # This loss should enforce: d/dt p(x,t) == -d/dx (u(x) * p(x,t))
        l_fp_t = (pxt.dt(x, t) - up_dx)**2
        l_fp = l_fp + l_fp_t.mean()
    l_fp = l_fp / len(ts)
    l_fp.backward()

    pxt_optimizer.step()
    ux_optimizer.step()

    print(f'{i} l_fp={float(l_fp.mean()):.5f}, l_px={float(l_Spxt.mean()):.5f}, l_p0={float(l_p0.mean()):.5f}')

# %%
# Plot the predicted p(x,t) at each timestep t
colors = matplotlib.colormaps.get_cmap('viridis')
xs = torch.arange(X1.min(), X1.max(), .01, device=device)[:,None]

plt.plot(xs.cpu().numpy(), pD.pdf(xs.cpu().numpy()), c='k', alpha=1)
plt.plot(xs.cpu().numpy(), pD0.pdf(xs.cpu().numpy()), c='k', alpha=1)

for t in torch.linspace(0, 1, 10, device=device):
    # print(t)
    pxst = pxt(xs, t)
    plt.plot(xs.cpu().numpy(), pxst.cpu().detach().numpy(), c=colors(float(t)))
# Plot the data distribution
# plt.colorbar()

# %%
# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
xs = torch.arange(X1.min(), X1.max(), .1, device=device)[:,None]
pxs = pD.pdf(xs.cpu().numpy())
ts = torch.linspace(0, 1, 100, device=device)
pxst = torch.zeros(xs.shape[0], ts.shape[0], device=device)
for i,t in enumerate(ts):
    pxst[:,i] = pxt(xs, t).squeeze()

# Cumulative mean of p(x,t) at each timestep t
cum_pxst = pxst.cumsum(dim=1) / torch.arange(1, ts.shape[0]+1, device=device)[None,:]
plt.imshow(cum_pxst.cpu().detach().numpy(), aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
# %%
# This plots the error of the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
xs = torch.arange(X1.min(), X1.max(), .1, device=device)[:,None]
pxs = pD.pdf(xs.cpu().numpy())
ts = torch.linspace(0, 1, 100, device=device)
pxst = torch.zeros(xs.shape[0], ts.shape[0], device=device)
for i,t in enumerate(ts):
    pxst[:,i] = pxt(xs, t).squeeze()

# Cumulative mean of p(x,t) at each timestep t
cum_pxst = pxst.cumsum(dim=1) / torch.arange(1, ts.shape[0]+1, device=device)[None,:]
plt.imshow(pxs - cum_pxst.cpu().detach().numpy(), aspect='auto', interpolation='none', cmap='RdBu')
plt.colorbar()

#%%
# This is the individual p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
plt.imshow(pxst.cpu().detach().numpy(), aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
# %%
# Plot the u(x) term for all x
plt.plot(xs.cpu().detach().numpy(), ux(xs).cpu().detach().numpy())
# %%
