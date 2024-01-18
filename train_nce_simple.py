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
# Target is a mixture of two Gaussians, with mean 1 and 5 and std=1
target1 = np.random.multivariate_normal(np.ones(dim)*1, np.eye(dim), size=1000)
target2 = np.random.multivariate_normal(np.ones(dim)*5, np.eye(dim), size=1000)
target = np.concatenate((target1, target2), axis=0)
target = torch.tensor(target, dtype=torch.float32).to(device)

mean = target.mean(dim=0)
cov = np.cov(target.cpu().numpy().T)
cov = cov + np.eye(dim)*1e-3
cov = torch.tensor(cov, dtype=torch.float32).to(device)
noise = D.MultivariateNormal(mean, cov)

model = NCE(noise=noise, dim=dim, hidden_dim=32, num_layers=2).to(device)

#%%
model.optimize(target, n_epochs=1000, n_samples=1000, lr=1e-3)

#%%
bins = 100
x = torch.linspace(-5, 10, bins).unsqueeze(1).to(device)
log_density_x = model(x)
p_x = torch.exp(log_density_x)
plt.plot(x.cpu().numpy(), p_x.cpu().detach().numpy())
# plt.plot(x.cpu().numpy(), log_density_x.cpu().detach().numpy())
height, bins = np.histogram(target.cpu().numpy(), bins=bins)
height = height / height.sum()
w = bins[1] - bins[0]
plt.bar(x=bins[:-1], width=w, height=height, alpha=0.5)

#%% 
# Now a more complex example, with a 100-dimensional target distribution
dim = 100

# Target is a mixture of two Gaussians, with mean 1 and 5 and std=1
target1 = np.random.multivariate_normal(np.ones(dim)*1, np.eye(dim), size=1000)
target2 = np.random.multivariate_normal(np.ones(dim)*3, np.eye(dim), size=1000)
target = np.concatenate((target1, target2), axis=0)
target = torch.tensor(target, dtype=torch.float32).to(device)

mean = target.mean(dim=0)
cov = np.cov(target.cpu().numpy().T)
cov = cov + np.eye(dim)*1e-3
cov = torch.tensor(cov, dtype=torch.float32).to(device)
noise = D.MultivariateNormal(mean, cov)

model = NCE(noise=noise, dim=dim, hidden_dim=128, num_layers=2).to(device)

#%%
# Train the model
model.optimize(target, n_epochs=1000, n_samples=1000, lr=1e-3)

#%%
# PCA projection of the learned distribution
model.eval()
# Generate samples from the noise distribution
new_noise = D.MultivariateNormal(torch.ones(dim).to(device)*2, 20*torch.eye(dim).to(device))
noise_samples = new_noise.sample((len(target),))
# Calculate the log probability of the target distribution
new_target1 = np.random.multivariate_normal(np.ones(dim)*1, np.eye(dim), size=1000)
new_target2 = np.random.multivariate_normal(np.ones(dim)*3, np.eye(dim), size=1000)
new_target = np.concatenate((new_target1, new_target2), axis=0)
new_target = torch.tensor(new_target, dtype=torch.float32).to(device)

print('Classification accuracy of noise vs new target: ', model.classify(new_target).mean().cpu().detach().numpy())

# Combine the target and noise samples
# Recover the density p using the relation p=q*exp(psi), where psi=log(p/q)
# and q is the noise distribution
combo = torch.cat((new_target, noise_samples), dim=0)
log_density = model(combo)
# p = torch.exp(log_density - torch.logsumexp(log_density, dim=0, keepdim=True))
#%%
# Compute the PCA projection of the combined distribution
pca = PCA()
proj = pca.fit_transform(combo.cpu().numpy())
plt.figure(figsize=(5, 5))
plt.scatter(proj[:,0], proj[:,1], s=1, c=log_density.detach().cpu().numpy(), alpha=.2)
plt.colorbar()

#%%
# Calculate the log probability of the target distribution
# Plot the target distribution
test_prob_np = model(new_target).cpu().detach().numpy()
noise_prob_np = model(noise_samples).cpu().detach().numpy()

plt.hist(test_prob_np, bins=100, alpha=.5, label='test', density=True)
plt.hist(noise_prob_np, bins=100, alpha=.5, label='noise', density=True)
plt.legend()


# %%
import scanpy as sc
data = sc.read_h5ad('wildtype_net.h5ad')
dim = data.shape[1]

n_samples = target.shape[0]
n_train = int(n_samples*.8)
train_target = target[:n_train]
test_target = target[n_train:]
#%%
# %%
# Train 1d on the most expressed gene
gene = 'POU5F1'
dim = 1
target = torch.tensor(data[:, data.var_names == gene].X.toarray(), device=device, dtype=torch.float32)
#%%
# Train the model
mean = target.mean(dim=0)
cov = np.cov(target.cpu().numpy().T)
cov = cov + np.eye(dim)*1e-3
cov = torch.tensor(cov, dtype=torch.float32).to(device)
noise = D.MultivariateNormal(mean, cov)
model = NCE(noise=noise, dim=dim, hidden_dim=128, num_layers=2).to(device)

model.optimize(target, n_epochs=1000, n_samples=1000, lr=1e-3)
# %%
plt.hist(target.cpu().numpy(), bins=100, alpha=.5, label='target', density=True)
xnp = np.linspace(0, target.max().item(), 100)
x=torch.tensor(xnp, device=device, dtype=torch.float32).unsqueeze(1)
log_density_np = model(x).cpu().detach().numpy()
prob_np = np.exp(log_density_np)
plt.plot(xnp, prob_np, label='model')
#%%
target = torch.tensor(data.X.toarray(), device=device, dtype=torch.float32)
dim = data.shape[1]

n_samples = target.shape[0]
n_train = int(n_samples*.8)
train_target = target[:n_train]
test_target = target[n_train:]

mean = target.mean(dim=0)
cov = np.cov(target.cpu().numpy().T)
cov = cov + np.eye(dim)*1e-3
cov = torch.tensor(cov, dtype=torch.float32).to(device)
noise = D.MultivariateNormal(mean, cov)

model = NCE(noise=noise, dim=dim, hidden_dim=128, num_layers=2).to(device)

# %%
model.optimize(train_target, n_epochs=1000, n_samples=1000, lr=1e-3)
# %%
model.eval()
# Generate samples from the noise distribution
new_noise = D.MultivariateNormal(torch.ones(dim).to(device)*2, 20*torch.eye(dim).to(device))
noise_samples = new_noise.sample((len(target),))
# Calculate the log probability of the target distribution
test_prob_np = model(test_target).cpu().detach().numpy()
noise_prob_np = model(noise_samples).cpu().detach().numpy()

plt.hist(test_prob_np, bins=100, alpha=.5, label='test', density=True)
plt.hist(noise_prob_np, bins=100, alpha=.5, label='noise', density=True)
plt.legend()
#%%
viridis = plt.get_cmap('viridis', 256)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
pca = pca.fit(target.cpu().numpy())
p_train = model(train_target).detach().cpu().numpy()
x_proj = pca.transform(train_target.cpu().numpy())
axs[0].scatter(x_proj[:,0], x_proj[:,1], c=p_train, cmap=viridis, s=1)
x_proj = pca.transform(test_target.cpu().numpy())
p_test = model(test_target).detach().cpu().numpy()
axs[1].scatter(x_proj[:,0], x_proj[:,1], c=p_test, cmap=viridis, s=1, 
               vmin=p_train.min(), vmax=p_train.max())
#%%
plt.hist(p_train, bins=100, alpha=.5, label='train', density=True)
plt.hist(p_test, bins=100, alpha=.5, label='test', density=True);
plt.hist(noise_prob_np, bins=100, alpha=.5, label='noise', density=True);
plt.legend()


# %%
