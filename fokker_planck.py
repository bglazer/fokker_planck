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
    def dx(self, x, ts, hx=1e-3):
        xgrad = (self(x+hx, ts) - self(x-hx, ts))/(2*hx)
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
hx = 1e-3
ht = 1e-3
#%%
# Generate a continuous distribution from the data
# Initialize the neural networks
pxt = Pxt(20, 3, device)
ux = Ux(200, 3, device)
# Initialize the optimizers
pxt_optimizer = torch.optim.Adam(pxt.parameters(), lr=1e-3)
ux_optimizer = torch.optim.Adam(ux.parameters(), lr=1e-3)
#%%
ts = torch.linspace(0, 1, 100, device=device, requires_grad=False)
# ## %%
# # NOTE: This is a pre-training step to get p(x, t=0) to match the initial condition
# # train  p(x, t=0)=p_D0 for all x in the overall distribution
pD = gen_p(X1.cpu().numpy(), bins=50)
pD0 = gen_p(X0.cpu().numpy(), bins=50)
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

l_Spxts = np.zeros(epochs)
l_p0s = np.zeros(epochs)
l_fps = np.zeros(epochs)
for epoch in range(epochs):
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
    l_Spxt = ((Spxt[:,0] - px)**2).mean()
    l_Spxt.backward()

    # Record the losses
    l_Spxts[epoch] = float(l_Spxt.mean())
    l_p0s[epoch] = float(l_p0.mean())   

    # This is the calculation of the term that ensures the
    # derivatives match the Fokker-Planck equation
    # d/dx p(x,t) = -d/dt (u(x) p(x,t))
    up_dx = (ux(x+hx) * pxt(x+hx, ts) - ux(x-hx) * pxt(x-hx, ts))/(2*hx)
    pxt_dts = pxt.dt(x, ts)
    l_fp = ((pxt_dts + up_dx)**2).mean()

    l_fp.backward()
    
    # Take a gradient step
    pxt_optimizer.step()
    ux_optimizer.step()
    print(f'{epoch} l_px={float(l_Spxt.mean()):.5f}, l_p0={float(l_p0.mean()):.5f}, '
          f'l_fp={float(l_fp.mean()):.5f}')
    l_fps[epoch] = float(l_fp.mean())


#%%
plt.title('Loss curves')
plt.plot(l_fps[10:], label='l_fp')
plt.plot(l_Spxts[10:], label='l_Spxt')
plt.plot(l_p0s[10:], label='l_p0')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# %%
viridis = matplotlib.colormaps.get_cmap('viridis')
greys = matplotlib.colormaps.get_cmap('Greys')
purples = matplotlib.colormaps.get_cmap('Purples')
low = float(X1.min())
high = float(X1.max())
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
plt.plot(xs, pD.pdf(xs), c='k', alpha=1)
plt.plot(xs, pD0.pdf(xs), c='k', alpha=1)
for i in range(0, ts.shape[0], int(len(ts)/10)):
    t = ts[i]
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
ax2.plot(xs, pD.pdf(xs), c='k', alpha=1, label='p(x)')
ax2.set_ylabel('p(x)')
fig.legend()
# %%
x = pD0.rvs(size=1000)
x = torch.tensor(x, device=device, dtype=torch.float32, requires_grad=False)[:,None]
xts = []
ts = torch.linspace(0, 1, 1500, device=device, requires_grad=False)
ht = ts[1] - ts[0]
for i in range(len(ts)):
    t = ts[i:i+1]
    # Compute the drift term
    u = ux(x)
    # Compute the diffusion term
    # Generate a set of random numbers
    dW = torch.randn_like(x) * torch.sqrt(ht)
    sigma = torch.ones_like(x)
    # Compute the change in x
    dx = u * ht + sigma * dW
    dx = dx.squeeze(0)
    # Update x
    x = x + dx
    xts.append(x.cpu().detach().numpy())
xts = np.concatenate(xts, axis=1)
#%%
# Plot the resulting probability densities at each timestep
low = float(xs.min())
high = float(xs.max())
bins = np.linspace(low, high, 30)
w = bins[1] - bins[0]
for i in range(0, ts.shape[0], ts.shape[0]//10):
    t = ts[i]
    heights,bins = np.histogram(xts[:,i], 
                                bins=bins,
                                density=True)
    plt.bar(bins[:-1], heights, width=w, color=viridis(float(t)), alpha=.2)

for i in range(0, pxts.shape[1], pxts.shape[1]//10):
    t = ts[i]
    plt.plot(xs, pxts[:,i], color='blue', alpha=.2)
labels = ['Simulation', 'Fokker-Planck theoretical']
artists = [plt.Line2D([0], [0], color=c, alpha=.2) for c in ['red', 'blue']]
plt.legend(artists, labels)
plt.title('p(x,t)')
# %%
# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
sim_pxts = np.zeros((bins.shape[0]-1, ts.shape[0]))
for i in range(0, ts.shape[0]):
    heights,bins = np.histogram(xts[:,i], 
                                bins=bins,
                                density=True)
    sim_pxts[:,i] = heights

plt.title('Cumulative mean of p(x,t)')
sim_cum_pxt = sim_pxts.cumsum(axis=1) / np.arange(1, sim_pxts.shape[1]+1)[None,:]
plt.imshow(sim_cum_pxt, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('x')
plt.xlabel('timestep (t)')
plt.colorbar() 
# %%
# This plots the cumulative mean of p(x,t) at each timestep t, going from t=0 (left) to t=1 (right)
# Higher probability is shown in yellow, lower probability is shown in blue
pX = pD.pdf(bins[:-1])
sim_cum_pxt_err = (pX[:,None] - sim_cum_pxt)**2

plt.title('Error of Cumulative mean of p(x,t)\n'
          f'Error ∫pxt(x,t)dt = {l_Spxt:.5f}\n'
          f'Error Simulation = {sim_cum_pxt_err.mean():.5f}')
plt.imshow(sim_cum_pxt_err, aspect='auto', interpolation='none', cmap='viridis')
plt.ylabel('x')
plt.xlabel('timestep (t)')
plt.colorbar() 
# %%
