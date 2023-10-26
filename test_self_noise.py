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
    def __init__(self, device) -> None:
        super().__init__()
        self.nn = MLP(input_dim=1, hidden_dim=64, output_dim=1, num_layers=2).to(device)
        self.device = device
        
    def simulate(self, n_samples, ts):
        """
        Langevin sampling of distribution X at times ts
        """
        x = torch.randn((n_samples, 1), device=self.device)*6+2 #.requires_grad_(True)
        return x
        # xs = []
        # dt = float(ts[1] - ts[0])
        # for t in ts:
        #     x = x - self.ux(x)*dt + torch.randn_like(x)*np.sqrt(dt)
        #     xs.append(x)
        # return torch.cat(xs, dim=1)
    
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

class PerturbationNoise():
    """
    Noise distribution that is a version of the current model with perturbed parameters
    """
    def __init__(self, model, X, ts, perturbation=0.1):
        self.perturbation = perturbation
        self.model = model
        self.perturb_model = deepcopy(model)
        self.ts = ts
        self.X = X

    def perturb_(self):
        model_params = deepcopy(self.model.state_dict())
        with torch.no_grad():
            for param in model_params.values():
                param += torch.randn_like(param)*self.perturbation
        self.perturb_model.load_state_dict(model_params)

    def sample(self, n_samples):
        self.perturb_()
        samples = self.perturb_model.simulate(n_samples, self.ts)[len(self.ts)//2:].reshape((-1, self.X.shape[1]))
        samples = samples[:n_samples,:]
        samples = samples.to(self.model.device)
        return samples.detach()
    
    def log_prob(self, x):
        # Perturb the model parameters using dropout
        self.perturb_()
        log_prob = self.perturb_model.log_prob(x)
        return log_prob.detach()

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
noise = PerturbationNoise(model, X, ts, perturbation=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
l_nces = []
accs = []
#%%
# Train the model
for i in range(epochs):
    optimizer.zero_grad()
    l_nce, acc = nce_loss(X, model, noise)
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
plt.bar(height=xheight, x=xbin[:-1], width=w, alpha=.3, label='X')
plt.plot(xs, pxs)
plt.xlabel('x')
plt.ylabel('p(x)')


# %%
