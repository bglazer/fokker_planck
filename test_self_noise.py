#%%
%load_ext autoreload
%autoreload 2
#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import scanpy as sc
from celldelta import MLP
import copy

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
        with torch.no_grad():
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
            return x0   
    
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
# def self_loss(x, model, p_eps, mcmc_step, sample_steps=10):
#     logp_x = model.log_prob(x) # logp(x) 

#     y = model.sample(x, n_steps=sample_steps, step_size=mcmc_step, eps=None) # y ~ q(y|x)
#     logp_y = model.log_prob(y) # logp(y)

#     x_eps = -(torch.abs(torch.ones_like(x)*np.log(p_eps)) * logp_x).detach()
#     y_eps = -(torch.abs(torch.ones_like(x)*np.log(p_eps)) * logp_y).detach()

#     l2 = np.log(2)
#     lx = logp_x - torch.logaddexp(l2 + logp_x, x_eps)  # log(p(x)/(2p(x) + ε))
#     ly = logp_y - torch.logaddexp(l2 + logp_y, y_eps)  # log(p(y)/(2p(y) + ε))
#     leps = y_eps - torch.logaddexp(l2 + logp_y, y_eps)  # log(ε/(2p(y) + ε))
#     loss = lx.mean() + leps.mean() + ly.mean()

#     return lx, ly, leps, -loss

def self_loss(x, model, mcmc_step, sample_steps=10):
    logp_x = model.log_prob(x) # logp(x) 

    y = model.sample(x, n_steps=sample_steps, step_size=mcmc_step, eps=None) # y ~ q(y|x)
    logp_y = model.log_prob(y) # logp(y)

    zero = torch.zeros(1, device=x.device)

    lx = logp_x - torch.logaddexp(logp_x, zero)  # log(p(x)/(2p(x) + ε))
    ly = logp_y - torch.logaddexp(logp_y, zero)  # log(p(y)/(2p(y) + ε))
    leps = - torch.logaddexp(logp_y, zero)  # log(ε/(2p(y) + ε))
    loss = lx.mean() + leps.mean() + ly.mean()

    return lx, ly, leps, -loss

#%%
device = 'cuda:0'

# %%
genotype='wildtype'
dataset = 'net'
adata = sc.read_h5ad(f'{genotype}_{dataset}.h5ad')
gene = 'POU5F1'
X = adata[:, adata.var_names == gene].X.toarray()
X = torch.tensor(X, device=device, dtype=torch.float32, requires_grad=True)
Xnmp = adata[adata.obs['cell_type'] == 'NMP', adata.var_names == gene].X.toarray()
Xnmp = torch.tensor(Xnmp, device=device, dtype=torch.float32, requires_grad=True)
X = Xnmp
#%%
# Plot the distribution
_=plt.hist(X.data.cpu().numpy(), bins=30, alpha=.3, label='X')
plt.legend()

#%%
epochs = 500
n_samples = 1000

#%%
# Initialize the model
model = Model(input_dim=1, hidden_dim=64, layers=3, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
l_nces = []
#%%
# Train the model
for i in range(500):
    optimizer.zero_grad()
    mcmc_step = .25
    lx, ly, leps, l_nce = self_loss(Xnmp, model, mcmc_step=mcmc_step, sample_steps=1)
    l_nce.backward()
    optimizer.step()
    l_nces.append(l_nce.item())
    print(f'epoch {i}: l_nce={l_nce.item():.4f}, lx={lx.mean().item():.4f}, ly={ly.mean().item()}')


#%%
fig, axs = plt.subplots(1, 1, figsize=(10,5))
axs.plot(l_nces[10:], label='l_self')
axs.set_xlabel('Epoch')
axs.set_ylabel('Loss')
axs.legend()
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
fig, axs = plt.subplots(3, 1, figsize=(5,10))
xnp = X.data.cpu().numpy()
axs[0].hist(xnp[:,0], bins=30, alpha=1)
axs[1].hist(xnp[:,1], bins=30, alpha=1)
axs[0].set_title(gene1)
axs[1].set_title(gene2)
axs[2].scatter(xnp[:,0], xnp[:,1], alpha=.3, s=1, c='k')

# %%
# Initialize the model
model = Model(input_dim=2, 
              hidden_dim=64,
              layers=2,
              device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
l_nces = []
accs = []
#%%
# Train the model
for i in range(1500):
    optimizer.zero_grad()
    mcmc_step = .5
    lx, ly, leps, l_nce = self_loss(X, model, mcmc_step=mcmc_step, sample_steps=1)
    l_nce.backward()
    optimizer.step()
    l_nces.append(l_nce.item())
    print(f'epoch {i}: l_nce={l_nce.item():.4f}, lx={lx.mean().item():.4f}, ly={ly.mean().item()}')


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
axs[0].contourf(xx, yy, log_prob_np.reshape(xx.shape), alpha=.5, levels=30)
# axs[0].scatter(xnp[:,0], xnp[:,1], alpha=.3, s=1, c='k')
# Count the number of points in each bin
xheight, xbin, ybin = np.histogram2d(xnp[:,0], xnp[:,1], bins=30)
xheight = xheight / xheight.sum()
xheight = xheight+1e-6
# Plot the contour`
# plt.scatter(xnp[:,0], xnp[:,1], alpha=.3, s=1, c='k')
axs[1].contourf(xbin[:-1], ybin[:-1], np.log(xheight.T), alpha=.5, levels=30)
# axs[1].scatter(xnp[:,0], xnp[:,1], alpha=.3, s=1, c='k')

# log_prob = model.log_prob(xs)

# %%
######################
# TWO MOONS
######################
# Generate the two half moons dataset
from sklearn.datasets import make_moons
Xnp, y = make_moons(n_samples=1000, noise=.05)
X = torch.tensor(Xnp, device=device, dtype=torch.float32, requires_grad=True)
# Plot the distribution
plt.scatter(Xnp[:,0], Xnp[:,1], c=y, s=3, alpha=1)
# %%
# Initialize the model
model = Model(input_dim=2, 
              hidden_dim=64,
              layers=2,
              device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
l_nces = []
#%%
# Train the model
n_epochs = 1000
for i in range(n_epochs):
    optimizer.zero_grad()
    mcmc_step = .5
    lx, ly,_, l_nce = self_loss(X, model=model, mcmc_step=mcmc_step, sample_steps=1)
    l_nce.backward()
    optimizer.step()
    l_nces.append(l_nce.item())
    print(f'epoch {i}: l_nce={l_nce.item():.4f}, lx={lx.mean().item():.4f}, ly={ly.mean().item()} leps={leps.mean().item()}')
#%%
plt.plot(l_nces)

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
# log_prob = log_prob - torch.logsumexp(log_prob, dim=0)
log_prob_np = log_prob.cpu().detach().numpy()
fig, axs = plt.subplots(1, 2, figsize=(10,5))
# Give both axes the same dimensions
axs[0].set_xlim(x.min(), x.max())
axs[0].set_ylim(y.min(), y.max())
axs[1].set_xlim(axs[0].get_xlim())
axs[1].set_ylim(axs[0].get_ylim())
# axs[0].contourf(xx, yy, np.exp(log_prob_np.reshape(xx.shape)), alpha=.5, levels=30)
axs[0].contourf(xx, yy, log_prob_np.reshape(xx.shape), alpha=.5, levels=30)
# axs[0].scatter(Xnp[:,0], Xnp[:,1], alpha=.3, s=1, c='k')
# Count the number of points in each bin
xheight, xbin, ybin = np.histogram2d(Xnp[:,0], Xnp[:,1], 
                                    range=[[x.min(), x.max()], 
                                           [y.min(), y.max()]],
                                    bins=50)
xheight = xheight / xheight.sum()
xheight = xheight+1e-6
# Plot the contour
axs[1].contourf(xbin[:-1], ybin[:-1], np.log(xheight.T+1e-8), alpha=.5, levels=15)
axs[1].scatter(Xnp[:,0], Xnp[:,1], alpha=.3, s=1, c='k')

# %%
# Test if the model can learn the distribution of the NMP cell type versus the rest
# of the cell types in the single cell data with all the genes
Xnp = adata.X.toarray()
X = torch.tensor(Xnp, device=device, dtype=torch.float32, requires_grad=True)
nmp_cell_mask = adata.obs['cell_type'] == 'NMP'
Xnmp = X[nmp_cell_mask].detach()
# %%
# Calculate the PCA projection of the data
from sklearn.decomposition import PCA
pca = PCA()
pca.components_ = adata.uns['PCs']
pca.mean_ = adata.uns['pca_mean']
Xpca = pca.transform(Xnp)
Xnmp_pca = pca.transform(Xnmp.cpu().detach().numpy())
plt.scatter(Xpca[:,0], Xpca[:,1], alpha=.3, s=1, c='k')
plt.scatter(Xnmp_pca[:,0], Xnmp_pca[:,1], alpha=.3, s=1, c='r')
# Calculate a contour plot in 2d PCA space of the NMP cells
# Make a meshgrid of the axes
xnmp_height, xbin, ybin = np.histogram2d(Xnmp_pca[:,0], Xnmp_pca[:,1], bins=30, 
                                         range=[[Xpca[:,0].min(), Xpca[:,0].max()], 
                                                [Xpca[:,1].min(), Xpca[:,1].max()]])
plt.contourf(xbin[:-1], ybin[:-1], np.log(xnmp_height.T+1e-6), alpha=.5)
#%%
# Initialize the model
model = Model(input_dim=X.shape[1], 
              hidden_dim=256,
              layers=2,
              device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
l_nces = []
#%%
# Train the model
for i in range(1000):
    optimizer.zero_grad()
    mcmc_step = .5
    lx, ly, _, l_nce = self_loss(Xnmp, model, mcmc_step=mcmc_step, sample_steps=1)
    l_nce.backward()
    optimizer.step()
    l_nces.append(l_nce.item())
    print(f'epoch {i}: l_nce={l_nce.item():.4f}, lx={lx.mean().item():.4f}, ly={ly.mean().item()}')

# %%
# Scatter plot of the data with cells colored by log probability
log_prob = model.log_prob(X)
log_prob_np = log_prob.cpu().detach().numpy()
# Get the middle 99% of the log probabilities
qnt = np.quantile(log_prob_np, [.005, .995])
qnt_mask = np.logical_and(log_prob_np > qnt[0], log_prob_np < qnt[1]).flatten()
log_prob_mid = log_prob_np[qnt_mask]
prob_np = torch.exp(log_prob - torch.logsumexp(log_prob, dim=0)).detach().cpu().numpy()
plt.scatter(Xpca[qnt_mask,0], Xpca[qnt_mask,1], alpha=.3, s=1, c=log_prob_mid)
# %%
plt.hist(log_prob_np, density=True, bins=100, alpha=.3, label='All cells')
plt.hist(model.log_prob(Xnmp).detach().cpu().numpy(), 
         density=True, bins=100, alpha=.3, label='NMP cells');
plt.legend()
# %%
# Get the top N cells by probability, where N=number of NMP cells
# Check if the NMP cells are in the top N
nmp_count = np.sum(nmp_cell_mask)
top_n = np.argsort(log_prob_np.flatten())[-nmp_count:]
# Get the cell type of the top N cells
top_n_cell_type = adata.obs['cell_type'][top_n]
# Print the cell types counts of the top N cells
# Add a column with the percentage of each cell type in the top N
top_n_cell_pct = top_n_cell_type.value_counts() / nmp_count
print(top_n_cell_type.value_counts().to_frame().join(top_n_cell_pct.to_frame(), rsuffix='_pct'))
# %%
# Compare performance to using NCE loss
#%%
def nce_loss(x, model, noise):
    y = noise.sample((x.shape[0],)).to(x.device).detach()

    logp_x = model.log_prob(x).flatten() # logp(x)
    logq_x = noise.log_prob(x).detach() # logq(x)
    logp_y = model.log_prob(y).flatten() # logp(y)
    logq_y = noise.log_prob(y).detach() # logq(y)

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
# Use a Gaussian noise distribution, with covariance and mean estimated from the data
cov = torch.cov(X.T)
noise = torch.distributions.MultivariateNormal(loc=Xnmp.mean(dim=0), 
                                               covariance_matrix=cov)

#%%
# Initialize the model
model = Model(input_dim=X.shape[1], 
              hidden_dim=256,
              layers=2,
              device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
l_nces = []

#%%
# Train the model
for i in range(2500):
    optimizer.zero_grad()
    l_nce, acc = nce_loss(Xnmp, model, noise)
    l_nce.backward()
    optimizer.step()
    l_nces.append(l_nce.item())
    print(f'epoch {i}: l_nce={l_nce.item():.4f}, acc={acc:.4f}')
# %%
log_prob = model.log_prob(X)
log_prob_np = log_prob.cpu().detach().numpy()
nmp_count = np.sum(nmp_cell_mask)
top_n = np.argsort(log_prob_np.flatten())[-nmp_count:]
# Get the cell type of the top N cells
top_n_cell_type = adata.obs['cell_type'][top_n]
# Print the cell types counts of the top N cells
# Add a column with the percentage of each cell type in the top N
top_n_cell_pct = top_n_cell_type.value_counts() / nmp_count
print(top_n_cell_type.value_counts().to_frame().join(top_n_cell_pct.to_frame(), rsuffix='_pct'))
# %%
