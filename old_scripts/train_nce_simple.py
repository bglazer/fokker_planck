#%%
%load_ext autoreload
%autoreload 2
#%%
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from noise_contrastive_model import NCE

device = 'cuda:0'

#%%
# Train the model to learn a one-dimensional mixture of two Gaussians
dim = 1

noise = D.MultivariateNormal(torch.ones(dim).to(device)*3, 10*torch.eye(dim).to(device))

model = NCE(noise=noise, dim=dim, hidden_dim=128, num_layers=2).to(device)

# Target is a mixture of two Gaussians, with mean 1 and 5 and std=1
target1 = np.random.multivariate_normal(np.ones(dim)*1, np.eye(dim), size=1000)
target2 = np.random.multivariate_normal(np.ones(dim)*5, np.eye(dim), size=1000)
target = np.concatenate((target1, target2), axis=0)
target = torch.tensor(target, dtype=torch.float32).to(device)

model.optimize(target, n_epochs=3000, n_samples=1000, lr=1e-3)

#%%
bins = 100
x = torch.linspace(-5, 10, bins).unsqueeze(1).to(device)
log_density_x = model.log_density(x)
p_x = torch.exp(log_density_x - torch.logsumexp(log_density_x, dim=0, keepdim=True))
plt.plot(x.cpu().numpy(), p_x.cpu().detach().numpy())
height, bins = np.histogram(target.cpu().numpy(), bins=bins)
height = height / height.sum()
w = bins[1] - bins[0]
plt.bar(x=bins[:-1], width=w, height=height, alpha=0.5)



#%% 
# Now a more complex example, with a 100-dimensional target distribution
dim = 100

noise = D.MultivariateNormal(torch.ones(dim).to(device)*2, 10*torch.eye(dim).to(device))

model = NCE(noise=noise, dim=dim, hidden_dim=128, num_layers=2).to(device)

# Target is a mixture of two Gaussians, with mean 1 and 5 and std=1
target1 = np.random.multivariate_normal(np.ones(dim)*1, np.eye(dim), size=1000)
target2 = np.random.multivariate_normal(np.ones(dim)*3, np.eye(dim), size=1000)
target = np.concatenate((target1, target2), axis=0)
target = torch.tensor(target, dtype=torch.float32).to(device)

#%%
# Train the model
model.optimize(target, n_epochs=3000, n_samples=1000, lr=1e-3)

#%%
# PCA projection of the learned distribution
model.eval()
# Generate samples from the noise distribution
new_noise = D.MultivariateNormal(torch.ones(dim).to(device)*2, 10*torch.eye(dim).to(device))
noise_samples = new_noise.sample((len(target),))
# Calculate the log probability of the target distribution
new_target1 = np.random.multivariate_normal(np.ones(dim)*1, np.eye(dim), size=1000)
new_target2 = np.random.multivariate_normal(np.ones(dim)*3, np.eye(dim), size=1000)
new_target = np.concatenate((new_target1, new_target2), axis=0)
new_target = torch.tensor(new_target, dtype=torch.float32).to(device)

print('Classification accuracy of noise vs new target: ', model.classify(new_target).mean().cpu().detach().numpy())

# Recover the density p using the relation p=q*exp(psi), where psi=log(p/q)
# and q is the noise distribution
combo = torch.cat((new_target, noise_samples), dim=0)
log_density = model.log_density(combo)
# Combine the target and noise samples
p = torch.exp(log_density - torch.logsumexp(log_density, dim=0, keepdim=True))

# Compute the PCA projection of the combined distribution
pca = PCA()
proj = pca.fit_transform(combo.cpu().numpy())
plt.figure(figsize=(5, 5))
plt.scatter(proj[:,0], proj[:,1], s=1, c=log_density.detach().cpu().numpy(), alpha=.2)
plt.colorbar()

#%%
# Calculate the log probability of the target distribution
# Plot the target distribution
height, bins = np.histogram(log_density.detach().cpu().numpy(), bins=100, density=True)
plt.bar(bins[:-1], height, alpha=0.5)
# %%
