#%%
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms
import numpy as np
from tqdm.notebook import tqdm

#%%
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 1 input channel, 10 output channels
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 10 input channels, 20 output channels

        self.fc1 = nn.Linear(320, 50) # 320 = 20 * 4 * 4
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def mcmc(self, x0, n_steps, step_size, eps=None, verbose=False):
        """
        MCMC sampling of the model
        """
        with torch.no_grad():
            for i in tqdm(range(n_steps), disable=not verbose):
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
                # Make sure accept has the same number of dimensions as x0
                # so that it can be broadcasted
                new_size = tuple(accept.size()) + (1,) * (x0.dim() - accept.dim()) 
                accept = accept.view(new_size)
                # Update the state
                x0 = torch.where(accept, x0+d, x0)
            return x0   
    
    def langevin(self, x0, n_steps, step_size, eps=None, verbose=False):
        """
        Langevin sampling of the model
        """
        x0.requires_grad_(True)
        for i in tqdm(range(n_steps), disable=not verbose):
            # Sample from a normal distribution centered at
            # the current state
            noise = torch.randn_like(x0)*np.sqrt(step_size)
            # Get the gradient of the log probability
            # with respect to the current state
            grad = torch.autograd.grad(self.log_prob(x0).mean(), x0)[0]
            # Add the gradient to the current state
            x0 = x0 - step_size * grad + noise
        x0 = x0.detach()
        return x0
    
    def log_prob(self, x):
        """
        Gives the log probability of x
        """
        return self(x)

#%%
def self_loss(x, model, p_eps, mcmc_step, sample_steps=10):
    logp_x = model.log_prob(x).flatten() # logp(x) 

    y = model.mcmc(x, n_steps=sample_steps, step_size=mcmc_step, eps=p_eps) # y ~ q(y|x)
    logp_y = model.log_prob(y).flatten() # logp(y)

    x_eps_ = torch.abs(torch.ones_like(x)*np.log(p_eps)).transpose(-1,0)
    x_eps = -(x_eps_ * logp_x).detach()
    y_eps = -(x_eps_ * logp_y).detach()

    l2 = np.log(2)
    lx = logp_x - torch.logaddexp(l2 + logp_x, x_eps)  # log(p(x)/(2p(x) + ε))
    ly = logp_y - torch.logaddexp(l2 + logp_y, y_eps)  # log(p(y)/(2p(y) + ε))
    leps = y_eps - torch.logaddexp(l2 + logp_y, y_eps)  # log(ε/(2p(x) + ε))
    loss = lx.mean() + ly.mean() + leps.mean()

    return lx, ly, leps, -loss

#%%
# Load the MNIST Dataset
print('Loading MNIST dataset...')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('~/data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
print('Done loading.')
#%%
device = 'cuda:3'
# Initialize the model
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
l_selfs = []
#%%
# Train the model
model.train()
for i in range(1):
    for X,_ in trainloader:
        X = X.to(device)
        optimizer.zero_grad()
        mcmc_step = .1
        p_eps = .001
        lx, ly, _, l_self = self_loss(X, model, p_eps=p_eps, mcmc_step=mcmc_step, sample_steps=1)
        l_selfs.append(l_self.item())
        l_self.backward()
        optimizer.step()
        print(f'epoch {i}: l_self={l_self.item():.4f}, lx={lx.mean().item():.4f}, ly={ly.mean().item()}')

#%%
# Plot the loss
plt.plot(l_selfs)

#%%
# Sample from the model
print('Sampling from the model...', flush=True)
nrm = torch.distributions.Normal(0, 1)
Xnrm = nrm.sample((1000, 1, 28, 28)).to(device)
samples = model.mcmc(Xnrm, n_steps=5000, step_size=1e-3, eps=None, verbose=True)
print('Done sampling.') 

#%%
# Plot the samples
fig, ax = plt.subplots(4, 4)
subsamples = samples[np.random.randint(0, len(samples), 16)]
for i in range(4):
    for j in range(4):
        ax[i][j].imshow(subsamples[i*4+j, 0].cpu().numpy())
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
plt.tight_layout()

#%%
plt.hist(model.log_prob(samples).detach().cpu().numpy(), bins=10, label='samples logprob')
plt.hist(model.log_prob(X).detach().cpu().numpy(), bins=10, label='data logprob')
plt.legend()

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
# Test the model with NCE loss
# Initialize the model
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

class Noise():
    def __init__(self, X):
        flat = X.reshape(X.shape[0], -1)
        loc = flat.mean(dim=0)
        # cov = torch.cov(flat.T + torch.randn_like(flat.T))
        cov = torch.cov(flat.T) + torch.eye(flat.shape[1], device=flat.device)*1e-3
        self.normal = torch.distributions.MultivariateNormal(loc=loc, 
                                                             covariance_matrix=cov)

    def log_prob(self, x):
        return self.normal.log_prob(x.reshape(x.shape[0], -1))

    def sample(self, shape):
        s = self.normal.sample(shape)
        return s.reshape(shape + (1, 28, 28))

# Get a random sample of 10000 images from the dataloader
data_sample = trainset.data[np.random.randint(0, len(trainset), 10000)].float().to(device)
# Apply the transform to the data
data_sample = transforms.Normalize((0.5,), (0.5,))(data_sample)
noise = Noise(data_sample)
l_nces = []
#%%
# Train the model
model.train()
for i in range(1):
    for X,_ in trainloader:
        X = X.to(device)
        optimizer.zero_grad()
        l_nce, acc = nce_loss(X, model, noise)
        l_nces.append(l_nce.item())
        l_nce.backward()
        optimizer.step()
        print(f'epoch {i}: l_nce={l_nce.item():.4f}, acc={acc.item():.4f}')

#%%
# Sample from the model
print('Sampling from the model...')
nrm = torch.distributions.Normal(0, 1)
Xnrm = nrm.sample((1000, 1, 28, 28)).to(device)
samples = model.mcmc(Xnrm, n_steps=1000, step_size=1e-2, eps=None)
print('Done sampling.') 

#%%
# Plot the samples
fig, ax = plt.subplots(4, 4)
subsamples = samples[np.random.randint(0, len(samples), 16)]
for i in range(4):
    for j in range(4):
        ax[i][j].imshow(subsamples[i*4+j, 0].cpu().numpy())
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
plt.tight_layout()

# %%
