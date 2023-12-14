import torch
import torch.nn as nn
import numpy as np
import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class NCE(nn.Module):
    def __init__(self, noise, dim, hidden_dim=128, num_layers=2):
        super(NCE, self).__init__()
        # The normalizing constant logZ(θ)        
        self.c = nn.Parameter(torch.tensor([1.], requires_grad=True))

        input_layer = [nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)]
        hidden_layers = []
        for i in range(num_layers):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.f = nn.Sequential(*input_layer, *hidden_layers, nn.Linear(hidden_dim, 1))
        self.noise = noise

    def forward(self, x):
        log_p = - self.f(x) - self.c
        return log_p

    def classify(self, x):
        logp_x = self.f(x)  # logp(x)
        logq_x = self.noise.log_prob(x).unsqueeze(1)  # logq(x)

        # Classification of noise vs target
        r_x = torch.sigmoid(logp_x - logq_x)
        return r_x
    
    def optimize(self, target, n_epochs=1000, n_samples=1000, lr=1e-3):
        optimizer = torch.optim.Adam(self.f.parameters(), lr=lr)

        for epoch in range(n_epochs):
            # Sample from the target data
            x = target[np.random.choice(len(target), n_samples, replace=False)]
            #  Generate samples from noise
            y = self.noise.sample((n_samples,))
            #  Train Energy-Based Model
            optimizer.zero_grad()

            logp_x = self.forward(x)  # logp(x)
            logq_x = self.noise.log_prob(x).unsqueeze(1)  # logq(x)
            logp_y = self.forward(y)  # logp(y)
            logq_y = self.noise.log_prob(y).unsqueeze(1)  # logq(y)

            value_x = logp_x - torch.logaddexp(logp_x, logq_x)  # logp(x)/(logp(x) + logq(x))
            value_y = logq_y - torch.logaddexp(logp_y, logq_y)  # logq(y)/(logp(y) + logq(y))

            v = value_x.mean() + value_y.mean()

            # Classification of noise vs target
            r_x = torch.sigmoid(logp_x - logq_x)
            r_y = torch.sigmoid(logq_y - logp_y)

            acc = ((r_x > 1/2).sum() + (r_y > 1/2).sum()).cpu().numpy() / (len(x) + len(y))
            
            (-v).backward()

            # Normalized gradient descent, according to:
            # https://blog.ml.cmu.edu/2021/11/05/analyzing-and-improving-the-optimization-landscape-of-noise-contrastive-estimation/
            # TODO figure out how to implement eNCE loss
            # for p in self.f.parameters():
            #     p.grad /= p.grad.norm()
            optimizer.step()  

            print(f"Epoch {epoch}, loss: {float(v)}, acc: {acc}")
    
    def log_density(self, x):
        return self.f(x) - self.noise.log_prob(x).unsqueeze(1) - self.c
        
