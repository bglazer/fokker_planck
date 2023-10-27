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
from celldelta import MLP
from copy import deepcopy
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

#%%
# Plot the two distributions
_=plt.hist(X.data.cpu().numpy(), bins=30, alpha=.3, label='X')
plt.legend()

#%%
epochs = 500
n_samples = 1000
ts = torch.linspace(0, 1, 100, device=device, requires_grad=True)

#%%
class Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, device) -> None:
        super().__init__()
        self.nn = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=layers).to(device)
        self.device = device
        
    def sample(self, x0, n_steps, step_size, eps=None):
        """
        MCMC sampling of the model
        """
        samples = torch.zeros((n_steps, x0.shape[0], x0.shape[1]), device=self.device)
        for i in range(n_steps):
            # Sample from a normal distribution centered at
            # the current state
            d = torch.randn_like(x0)*step_size
            # Add the delta to the current state
            x0 = x0 + d
            # Calculate the acceptance probability
            p0 = self.log_prob(x0)
            p1 = self.log_prob(x0 + d)
            if eps is not None:
                p0 += eps
                p1 += eps
            # Accept or reject the new state
            accept = torch.rand_like(p0) < torch.exp(p1 - p0)
            # Update the state
            x0 = torch.where(accept, x0+d, x0)
            # Save the state
            samples[i] = x0
        samples = samples.reshape((-1, x0.shape[1]))
        # Randomly pick N points from the samples
        samples = samples[torch.randperm(len(samples))[:len(x0)]] 
        return samples
    
    def ux(self, x):
        """
        Gives the gradient of the probability distribution
        """
        # calculate the gradient of the log probability
        logp = self.log_prob(x)
        grad = torch.autograd.grad(logp.sum(), x)[0]
        return grad
    
    def log_prob(self, x):
        """
        Gives the log probability of x
        """
        return self.nn(x)

#%%
def self_loss(x, model, p_eps, x_eps, sample_steps=10):
    eps = torch.ones_like(x)*np.log(p_eps)

    y = model.sample(x, n_steps=sample_steps, step_size=x_eps, eps=p_eps) # y ~ q(y|x)
    
    logp_x = model.log_prob(x) # logp(x)
    logp_y = model.log_prob(y) # logp(y)
    # logq_x = logp_x + eps # logq(x)
    # logq_y = logp_y + eps # logq(y)

    l2 = np.log(2)
    lx = logp_x - torch.logaddexp(l2 + logp_x, eps)  # logp(x)/(log(2p(x)) + ε)
    ly = logp_x - torch.logaddexp(l2 + logp_y, eps)  # logp(x)/(log(2p(y)) + ε)
    leps = eps - torch.logaddexp(l2 + logp_y, eps)  # log(ε)/(log(2p(y)) + ε)
    # r_x = torch.sigmoid(logp_x - logq_x)
    # r_y = torch.sigmoid(logq_y - logp_y)
    # acc = ((r_x > 1/2).sum() + (r_y > 1/2).sum()).cpu().numpy() / (len(x) + len(y))

    loss = lx.mean() + ly.mean() + leps.mean()
    return -loss, 0

#%%
def nce_loss(x, model, noise):
    y = noise.sample(x.shape[0])

    logp_x = model.log_prob(x) # logp(x)
    logq_x = noise.log_prob(x) # logq(x)
    logp_y = model.log_prob(y) # logp(y)
    logq_y = noise.log_prob(y) # logq(y)

    value_x = logp_x - torch.logaddexp(logp_x, logq_x)  # logp(x)/(logp(x) + logq(x))
    value_y = logq_y - torch.logaddexp(logp_y, logq_y)  # logq(y)/(logp(y) + logq(y))

    v = value_x.mean() + value_y.mean()

    # Classification of noise vs target
    r_x = torch.sigmoid(logp_x - logq_x)
    r_y = torch.sigmoid(logq_y - logp_y)

    # Compute the classification accuracy
    acc = ((r_x > 1/2).sum() + (r_y > 1/2).sum()).cpu().numpy() / (len(x) + len(y))
    
    return -v, acc

#%%
# Initialize the model
model = Model(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
l_nces = []
accs = []
#%%
# Train the model
for i in range(2000):
    optimizer.zero_grad()
    x_eps = np.random.uniform(0, .2)
    p_eps = np.random.uniform(0, .2)
    l_nce, acc = self_loss(X, model, p_eps=p_eps, x_eps=x_eps)
    l_nce.backward()
    optimizer.step()
    l_nces.append(l_nce.item())
    accs.append(acc)
    print(f'epoch {i}: l_nce={l_nce.item():.4f} acc={acc:.4f}')

#%%
fig, axs = plt.subplots(2, 1, figsize=(10,10))
axs[0].plot(l_nces[10:], label='l_nce')
axs[1].plot(accs[10:], label='accuracy')
[axs[i].set_xlabel('Epoch') for i in range(len(axs))]
[axs[i].set_ylabel('Loss') for i in range(len(axs))]
[axs[i].legend() for i in range(len(axs))]
fig.suptitle('Loss curves')
fig.tight_layout()

# %%
# Plot the predicted p_hat(x) and true p(x)
low = float(X.min())
high = float(X.max())
l = low-.25*(high-low) 
h = high+.25*(high-low)
xs = torch.arange(0, h, .01, device=device)[:,None]

log_prob = model.log_prob(xs)
pxs = torch.exp(log_prob - torch.logsumexp(log_prob,dim=0)).squeeze().T.cpu().detach().numpy()

xs = xs.squeeze().cpu().detach().numpy()
plt.title('p(x,t)')
xheight, xbin = np.histogram(X.detach().cpu().numpy(), bins=30)
xheight = xheight / xheight.sum()
w = xbin[1] - xbin[0]
# Bucket the pxs to match the histogram
px_bucket = np.zeros_like(xheight)
for i in range(len(xheight)):
    px_bucket[i] = pxs[np.logical_and(xs >= xbin[i], xs < xbin[i+1])].sum()
plt.bar(height=xheight, x=xbin[:-1], width=w, alpha=.3, label='X')
plt.bar(x=xbin[:-1], height=px_bucket, width=w, alpha=.3, label='p(x)')
plt.xlabel('x')
plt.ylabel('p(x)')

# %%
adata = sc.read_h5ad(f'{genotype}_{dataset}.h5ad')
# Get a binary mask of the top 2 genes by variance
X = adata.X.toarray()
top_variance_gene_mask = np.zeros_like(adata.var_names, dtype=bool)
top_variance_gene_mask[np.argsort(X.var(axis=0))[-2:]] = True
X = adata[:, top_variance_gene_mask].X.toarray()
X = torch.tensor(X, device=device, dtype=torch.float32, requires_grad=True)
gene1, gene2 = adata.var_names[top_variance_gene_mask]

# %%
# Plot the two distributions separately
fig, axs = plt.subplots(1, 2, figsize=(10,5))
xnp = X.data.cpu().numpy()
axs[0].hist(xnp[:,0], bins=30, alpha=1)
axs[1].hist(xnp[:,1], bins=30, alpha=1)
axs[0].set_title(gene1)
axs[1].set_title(gene2)


# %%
# Initialize the model
model = Model(input_dim=2, 
              hidden_dim=64,
              layers=2,
              device=device)
# noise = PerturbationNoise(model, X, ts, perturbation=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
l_nces = []
accs = []
#%%
# Train the model
for i in range(2000):
    optimizer.zero_grad()
    x_eps = np.random.uniform(0, .2)
    p_eps = np.random.uniform(0, .2)
    l_nce, acc = self_loss(X, model, p_eps=p_eps, x_eps=x_eps)
    l_nce.backward()
    optimizer.step()
    l_nces.append(l_nce.item())
    accs.append(acc)
    print(f'epoch {i}: l_nce={l_nce.item():.4f} acc={acc:.4f}')
# %%
# Plot contours of log probability
# Make a meshgrid of the axes
x = np.linspace(xnp.min(), xnp.max(), 200)
y = np.linspace(xnp.min(), xnp.max(), 200)
xx, yy = np.meshgrid(x, y)
# Get the log_prob of every x,y pair
xy = torch.tensor(np.vstack((xx.flatten(), yy.flatten())).T, device=device, dtype=torch.float32)
log_prob = model.log_prob(xy)
# Normalize the log_probs
log_prob = log_prob - torch.logsumexp(log_prob, dim=0)
log_prob_np = log_prob.cpu().detach().numpy()
fig, axs = plt.subplots(1, 2, figsize=(10,5))
# Give both axes the same dimensions
axs[0].set_xlim(x.min(), x.max())
axs[0].set_ylim(y.min(), y.max())
axs[1].set_xlim(axs[0].get_xlim())
axs[1].set_ylim(axs[0].get_ylim())
axs[0].contourf(xx, yy, log_prob_np.reshape(xx.shape), alpha=.5)
# axs[0].scatter(xnp[:,0], xnp[:,1], alpha=.3, s=1, c='k')
# Count the number of points in each bin
xheight, xbin, ybin = np.histogram2d(xnp[:,0], xnp[:,1], bins=30)
xheight = xheight / xheight.sum()
xheight = xheight+1e-6
# Plot the contour
plt.contourf(xbin[:-1], ybin[:-1], np.log(xheight.T), alpha=.5)
# plt.scatter(xnp[:,0], xnp[:,1], alpha=.3, s=1, c='k')
axs[1].contourf(xbin[:-1], ybin[:-1], np.log(xheight.T), alpha=.5)
# axs[1].scatter(xnp[:,0], xnp[:,1], alpha=.3, s=1, c='k')

# log_prob = model.log_prob(xs)

# %%
# Generate the two half moons dataset
from sklearn.datasets import make_moons
Xnp, y = make_moons(n_samples=1000, noise=.05)
X = torch.tensor(Xnp, device=device, dtype=torch.float32, requires_grad=True)
# Plot the distribution
plt.scatter(Xnp[:,0], Xnp[:,1], c=y)
# %%
# Initialize the model
model = Model(input_dim=2, 
              hidden_dim=64,
              layers=2,
              device=device)
# noise = PerturbationNoise(model, X, ts, perturbation=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
l_nces = []
#%%
# Train the model
for i in range(2000):
    optimizer.zero_grad()
    x_eps = np.random.uniform(0, .01)
    p_eps = np.random.uniform(0, .2)
    l_nce, _ = self_loss(X, model, p_eps=p_eps, x_eps=x_eps, sample_steps=100)
    l_nce.backward()
    optimizer.step()
    l_nces.append(l_nce.item())
    print(f'epoch {i}: l_nce={l_nce.item():.4f}')

# %%
# Plot contours of log probability
# Make a meshgrid of the axes
x = np.linspace(Xnp.min(), Xnp.max(), 200)
y = np.linspace(Xnp.min(), Xnp.max(), 200)
xx, yy = np.meshgrid(x, y)
# Get the log_prob of every x,y pair
xy = torch.tensor(np.vstack((xx.flatten(), yy.flatten())).T, device=device, dtype=torch.float32)
log_prob = model.log_prob(xy)
# Normalize the log_probs
log_prob = log_prob - torch.logsumexp(log_prob, dim=0)
log_prob_np = log_prob.cpu().detach().numpy()
fig, axs = plt.subplots(1, 2, figsize=(10,5))
# Give both axes the same dimensions
axs[0].set_xlim(x.min(), x.max())
axs[0].set_ylim(y.min(), y.max())
axs[1].set_xlim(axs[0].get_xlim())
axs[1].set_ylim(axs[0].get_ylim())
axs[0].contourf(xx, yy, np.exp(log_prob_np.reshape(xx.shape)), alpha=.5, levels=30)
axs[0].scatter(Xnp[:,0], Xnp[:,1], alpha=.3, s=1, c='k')
# Count the number of points in each bin
xheight, xbin, ybin = np.histogram2d(Xnp[:,0], Xnp[:,1], bins=30)
xheight = xheight / xheight.sum()
xheight = xheight+1e-6
# Plot the contour
plt.contourf(xbin[:-1], ybin[:-1], np.log(xheight.T), alpha=.5)
plt.scatter(Xnp[:,0], Xnp[:,1], alpha=.3, s=1, c='k')

# %%
