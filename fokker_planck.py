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
        t_ = torch.ones(x.shape, device=self.device)*t
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

    # Integral of (d/dx p(x,t)) over t
    # TODO instead of using a for loop, should use a combined
    # tensor across all timesteps. Not sure if this will 
    # consume too much GPU memory though.
    # xs = x.repeat((1,ts.shape[0])).T.unsqueeze(2)
    # tss = ts.repeat((x.shape[0],1)).T.unsqueeze(2).to(device)
    # xts = torch.concatenate((xs,tss), dim=2)
    # ps = pxt.model(xts)
    def Sdx_dt(self, x, hx=1e-3, steps=100):
        ts = torch.linspace(0, 1, steps, device=self.device)
        ht = ts[1] - ts[0]
        sum = torch.zeros(x.shape, device=self.device, requires_grad=True)
        for t in ts:
            dx = self.dx(x, t, hx=hx)
            sum = sum + (dx*ht)
        return sum

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
pxt_optimizer = torch.optim.Adam(pxt.parameters(), lr=1e-4)
ux_optimizer = torch.optim.Adam(ux.parameters(), lr=1e-3)
pD0 = torch.tensor(pD.pdf(X0.cpu().numpy()), device=device, dtype=torch.float32, requires_grad=False)

ts = torch.linspace(0, 1, 100, device=device, requires_grad=False)
# %%
# pxt0_optimizer = torch.optim.Adam(pxt.parameters(), lr=1e-3)
# for i in range(epochs):
#     pxt0_optimizer.zero_grad()
#     l_pD0 = ((pxt(X0, t=0) - pD0)**2).mean()
#     l_pD0.backward()
#     # torch.nn.utils.clip_grad_norm_(pxt.parameters(), .001)
#     pxt0_optimizer.step()
#     print(f'{i} l_pD0={float(l_pD0.mean()):.5f}')
#%%
for i in range(epochs):
    pxt_optimizer.zero_grad()
    ux_optimizer.zero_grad()
    x = pD.rvs(size=(1000,1))
    px = torch.tensor(pD.pdf(x), device=device, dtype=torch.float32, requires_grad=False)
    x = torch.tensor(x, device=device, dtype=torch.float32, requires_grad=False)
    ppx = (pxt(x, t=0) - pxt(x, t=1) - ux(x) * pxt.Sdx_dt(x)) / (ux.dx(x))
    
    l_px = ((ppx - px)**2).mean() #+ ((pxt(X0, t=0) - pD0)**2).mean()
    l_px.backward()
    torch.nn.utils.clip_grad_norm_(pxt.parameters(), .001)
    pxt_optimizer.step()

    l_fp = torch.zeros(1, device=device, requires_grad=True)
    up = lambda x, t: ux(x) * pxt(x, t)
    for t in ts:
        up_dx = -(up(x+hx,t) - up(x-hx,t))/(2*hx)
        l_fp_t = (pxt.dt(x, t) - up_dx)**2
        l_fp = l_fp + l_fp_t.mean()
    l_fp = l_fp / len(ts)
    l_fp.backward()

    torch.nn.utils.clip_grad_norm_(ux.parameters(), .001)
    ux_optimizer.step()

    # print(f'pxt0={float(pxt(x, t=0).mean()):.5f}, pxt1={float(pxt(x, t=1).mean()):.5f}')
    # print(f'ux={float(ux(x).mean()):.5f}')
    # print(f'Spxt_dx_dt={float(Spxt_dx_dt(pxt, x).mean()):.5f}')
    # print(f'ux.dx={float(ux.dx(x).mean()):.5f}')
    # print(f'pxt.dx={float(pxt.dx(x, t=0).mean()):.5f}')
    # print(f'pxt.dt={float(pxt.dt(x, t=0).mean()):.5f}')
    # print(f'ppx={float(ppx.mean()):.5f}')
    print(f'{i} l_fp={float(l_fp.mean()):.5f}, l_px={float(l_px.mean()):.5f}')
    # print('---')

#%%
xs = torch.arange(X1.min(), X1.max(), .01, device=device)[:,None]
ppxs = (pxt(xs, t=0) - pxt(xs, t=1) - ux(xs) * pxt.Sdx_dt(xs)) / (ux.dx(xs))
px = torch.tensor(pD.pdf(xs.detach().cpu().numpy()), device=device, dtype=torch.float32, requires_grad=False)
plt.plot(xs.detach().cpu().numpy(), ppxs.cpu().detach().numpy(), label='Model')
plt.plot(xs.detach().cpu().numpy(), px.cpu().numpy(), label='Data')
plt.legend()

# %%
colors = matplotlib.colormaps.get_cmap('viridis')
xs = torch.arange(X1.min(), X1.max(), .01, device=device)[:,None]
for t in torch.linspace(0, 1, 100, device=device):
    # print(t)
    pxst = pxt(xs, t)
    plt.plot(xs.cpu().numpy(), pxst.cpu().detach().numpy(), c=colors(float(t)))
    # plt.plot(xs.cpu().numpy(), pD.pdf(xs.cpu().numpy()), c='k', alpha=1)
# plt.colorbar()
#%%
ts = torch.linspace(0, 1, 100, device=device)
ppx = torch.zeros_like(xs)
for t in ts:
    pxst = pxt(xs, t)
    ppx = ppx + pxst/len(ts)
plt.plot(xs.cpu().numpy(), ppx.cpu().detach().numpy())
plt.plot(xs.cpu().numpy(), pD.pdf(xs.cpu().numpy()), c='k', alpha=1)
    
#%%
# Plot the cdf of the data and the final distribution
# plt.plot(X1.cpu().numpy(), px_cdf.cpu().numpy(), label='Data CDF')
# ppx_cdf = ppx.clip(0).cumsum(dim=0)/ppx.clip(0).sum()
# plt.plot(X1.cpu().numpy(), ppx_cdf.detach().cpu().numpy(), label='Model CDF')
# plt.legend()
# %%
xs = torch.arange(X1.min(), X1.max(), .1, device=device)[:,None]
ts = torch.linspace(0, 1, 100, device=device)
pxst = torch.zeros(xs.shape[0], ts.shape[0], device=device)
for i,t in enumerate(ts):
    pxst[:,i] = pxt(xs, t).squeeze()

# Cumulative mean of p(x,t) at each timestep t
cum_pxst = pxst.cumsum(dim=1) / torch.arange(1, ts.shape[0]+1, device=device)[None,:]
plt.imshow(cum_pxst.cpu().detach().numpy(), aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()

#%%
plt.imshow(pxst.cpu().detach().numpy(), aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()

# %%
# for x in xs:
#     print(float(ux(x)))
# %%
plt.plot(xs.cpu().detach().numpy(), ux(xs).cpu().detach().numpy())
# %%
