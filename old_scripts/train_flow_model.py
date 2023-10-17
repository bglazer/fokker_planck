#%%
import scanpy as sc
import normflows as nf
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA

#%%
# adata = sc.read_h5ad('wildtype_net.h5ad')
# X = adata.X.toarray()
# num_genes = adata.shape[1]
X1 = np.random.normal(0, 1, size=(1000, 100))
X2 = np.random.normal(3, 1, size=(1000, 100))
X = np.concatenate((X1, X2), axis=0)
num_genes = X.shape[1]

# %%
# Set up model

# Define 2D Gaussian base distribution
base = nf.distributions.base.DiagGaussian(num_genes)

# Define list of flows
num_layers = 32
flows = []
for i in range(num_layers):
    # Neural network with two hidden layers 
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([num_genes//2, num_genes*4, num_genes*4, num_genes], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(num_genes, mode='swap'))
# Construct flow model
device = 'cuda:0'
#%%
latent_size = num_genes
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
for i in range(num_layers):
    s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
    t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    flows += [nf.flows.ActNorm(latent_size)]

# Set base distribuiton
q0 = nf.distributions.DiagGaussian(num_genes, trainable=False)
    
# Construct flow model
model = nf.NormalizingFlow(q0=q0, flows=flows).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

# %%
# Train model
model.train()
max_iter = 5000
num_samples = 1500
show_iter = 20

loss_hist = np.array([])

for it in range(max_iter):
    optimizer.zero_grad()
    
    # Get training samples
    rand_idxs = np.random.choice(X.shape[0], num_samples, replace=False)
    x = torch.tensor(X[rand_idxs], dtype=torch.float32).to(device)
    
    # Compute loss
    loss = model.forward_kld(x)
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

    if (it-1) % show_iter == 0:
        print('iter: {}, loss: {:.4f}'.format(it, float(loss)))
    
#%%
# Plot loss
plt.figure(figsize=(8, 4))
plt.plot(loss_hist[loss_hist < 1e3], label='loss')
plt.legend()
plt.show()

# %%
# Plot the learned distribution
model.eval()
x2 = np.random.normal(0,1.0,(1000,100))
zz = torch.tensor(x2, dtype=torch.float32).to(device)
log_prob = model.log_prob(zz)
#%%
# Normalize log_prob
valid_log_prob = ~(log_prob.isnan() | log_prob.isinf())
# log_prob = log_prob - torch.max(log_prob[valid_log_prob])
print('valid:',int(valid_log_prob.sum()))
log_prob[log_prob.isnan()] = 0
log_prob[log_prob.isnan()] = 0
log_prob[log_prob.isinf()] = 0
prob = torch.exp(log_prob.to('cpu'))
plt.hist(log_prob.to('cpu').data.numpy(), bins=100);
#%%
pca = PCA()
pca.fit(X)
proj = np.array(pca.transform(x2))[:,0:2]
# Set the PC mean and components
# pca.mean_ = adata.uns['pca_mean']
# pca.components_ = adata.uns['PCs']
#%%
prob[torch.isnan(prob)] = 0
# prob[torch.isinf(prob)] = 0
# prob[prob > 1] = 1
#%%
plt.figure(figsize=(15, 15))
plt.scatter(proj[:,0], proj[:,1], c=log_prob.data.to('cpu').numpy(), cmap='viridis',s=5)
plt.gca().set_aspect('equal', 'box')
plt.show()

# %%
samples, sample_log_prob = model.sample(5000)
sample_proj = pca.transform(samples.data.cpu().numpy())[:,:2]
X_proj = np.array(pca.transform(X))[:,0:2]

plt.scatter(X_proj[:,0], X_proj[:,1], s=1)
plt.scatter(sample_proj[:,0], sample_proj[:,1],s=1)
#%%
plt.hist(sample_log_prob.data.cpu().numpy(), bins=100);
# %%
