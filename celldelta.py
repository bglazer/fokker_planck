import torch
from torch import logsumexp
from torch.nn import Linear, ReLU
import torch.nn as nn
import numpy as np
from scipy.stats import gaussian_kde
from KDE import GaussianKDE
from copulas.multivariate.gaussian import GaussianMultivariate
from torch.distributions import Categorical
from copy import deepcopy
#%%
# Define models 

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, input_bias=True):
        """
        Generic Dense Multi-Layer Perceptron (MLP), which is just a stack of linear layers with ReLU activations
        
        Args:
            input_dim (int): dimension of input
            output_dim (int): dimension of output
            hidden_dim (int): dimension of hidden layers
            num_layers (int): number of hidden layers
            input_bias (bool, optional): whether to include a bias term in the input layer. Defaults to True.
        
        Returns:
            None
        """
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
        
class Ux(torch.nn.Module):
    """
    The u(x) drift term of the Fokker-Planck equation, modeled by a neural network
    """
    def __init__(self, input_dim, hidden_dim, n_layers):
        """
        Args:
            input_dim (int): The dimensionality of the data.
            hidden_dim (int): The number of hidden units in each layer.
            n_layers (int): The number of layers in the model.

        Returns:
            None
        """
        super(Ux, self).__init__()
        # The drift term of the Fokker-Planck equation, modeled by a neural network
        self.model = MLP(input_dim, input_dim, hidden_dim, n_layers)

    def forward(self, x):
        return self.model(x)
    
    # TODO batching of analytical differentiation
    def dx(self, x):
        """
        Compute the derivative of u(x) with respect to x
        """
        ux = self.model(x)
        dudx = torch.autograd.grad(outputs=ux, 
                                   inputs=x, 
                                   grad_outputs=torch.ones_like(ux),
                                   create_graph=True,
                                   retain_graph=True)[0]
        return dudx


    
# The p(x,t) term of the Fokker-Planck equation, modeled by a neural network
class Pxt(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        """
        The p(x,t) term of the Fokker-Planck equation, modeled by a neural network

        Args:
            input_dim (int): The dimensionality of the data.
            hidden_dim (int): The number of hidden units in each layer.
            n_layers (int): The number of layers in the model.
        
        Returns:
            None
        """
        super(Pxt, self).__init__()

        # Add another dimension to the input for time
        self.model = MLP(input_dim+1, 1, hidden_dim, n_layers)

    def xts(self, x, ts):
        # Repeat the x and t vectors for each timestep in the ts range
        xs = x.repeat((ts.shape[0],1,1,))
        ts_ = ts.repeat((x.shape[0],1)).T.unsqueeze(2)
        # Concatentate them together to match the input the MLP model
        xts = torch.concatenate((xs,ts_), dim=2)
        return xts

    def log_pxt(self, x, ts):
        """
        Compute the log probability of the data at the given time points and data points.

        Args:
            x (torch.tensor): The input data of shape (n_batch, n_cells, n_genes).
            ts (torch.tensor): The time points at which to evaluate the model of shape (n_timesteps,).
        
        Returns:
            #TODO make sure the shape is correct
            torch.tensor: The log probability of the data at each time and data point of shape (n_cells, n_timesteps).
        """

        xts = self.xts(x, ts)
        log_ps = self.model(xts)
        return log_ps
    
    def pxt(self, x, ts):
        """
        Exponentiate the log probability of the data at the given time points and data points.
        """
        return torch.exp(self.log_pxt(x, ts))

    def forward(self, x, ts):
        return self.log_pxt(x, ts)
    
    def log_px(self, x, ts):
        """
        Marginalize out the t dimension to get log(p(x))
        """
        return torch.logsumexp(self.log_pxt(x, ts), dim=0) - torch.log(torch.tensor(ts.shape[0], device=x.device, dtype=torch.float32))
    
    def dx_dt(self, x, ts):
        """
        Compute the partial derivative of log p(x,t) with respect to x

        Returns:
            torch.tensor: The partial derivative of log p(x,t) with respect to x
            torch.tensor: The partial derivative of log p(x,t) with respect to t
        """
        xts = self.xts(x, ts)
        log_pxt = self.model(xts)
        dpdx = torch.autograd.grad(outputs=log_pxt, 
                                   inputs=xts, 
                                   grad_outputs=torch.ones_like(log_pxt),
                                   create_graph=True,
                                   retain_graph=True)[0]
        # The :-1 is to get the gradient with respect to x
        # The -1 is to get the gradient with respect to t
        return dpdx[:,:,:-1], dpdx[:,:,-1].unsqueeze(2)
    
    def sample(self, x0, n_steps, step_size, eps=None):
        """
        MCMC sampling of the model
        """
        samples = torch.zeros((n_steps, x0.shape[0], x0.shape[1]), device=x0.device)
        for i in range(n_steps):
            # Sample from a normal distribution centered at
            # the current state
            d = torch.randn_like(x0)*step_size
            # Add the delta to the current state
            x0 = x0 + d
            # Calculate the acceptance probability
            p0 = self.log_prob(x0)
            p1 = self.log_prob(x0 + d)
            # Perturb the log probability if the eps parameter is given
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

class CellDelta(nn.Module):
    """
    CellDelta is a learned model of cell differentiation in single-cell RNA-seq single timepoint data.
    It models the developmental trajectory of cells as a driven stochastic process
    # TODO expand this docstring to describe the rationale of the model
    """
    def __init__(self, input_dim, 
                 ux_hidden_dim, ux_layers,
                 pxt_hidden_dim, pxt_layers,
                 device) -> None:
        """
        Initialize the CellDelta model with the given hyperparameters.

        Args:
            input_dim (int): The dimensionality of the input data.
            ux_hidden_dim (int): The number of hidden units in each layer for the UX model.
            ux_layers (int): The number of layers in the UX model.
            pxt_hidden_dim (int): The number of hidden units in each layer for the PXT model.
            pxt_layers (int): The number of layers in the PXT model.
            loss_type (str, optional): The type of loss to use for training. Defaults to 'nce'. Must be one of ('nce', 'ence', 'self').
            device (torch.device): The device to use for the model.

        Returns:
            None
        """
        super().__init__()
        self.ux = Ux(input_dim, ux_hidden_dim, ux_layers).to(device)
        self.pxt = Pxt(input_dim, pxt_hidden_dim, pxt_layers).to(device)
        self.device = device
        # Add the component models (ux, pxt, nce) to a module list
        self.models = torch.nn.ModuleDict({'ux':self.ux, 'pxt':self.pxt})
    
    def nce_loss(self, x, noise, ts):
        """
        Compute the Noise-Contrastive Estimation (NCE) loss for the given data.

        Args:
            x (torch.tensor): The input data of shape (n_cells, n_genes).
            ts (torch.tensor): The time points at which to evaluate the model of shape (n_timesteps,).
        
        Returns:
            torch.tensor: The NCE loss value.
            torch.tensor: The sample vs noise classification accuracy.
        """
        y = noise.sample((x.shape[0],))

        logp_x = self.pxt.log_px(x, ts)  # logp(x)
        logq_x = noise.log_prob(x).unsqueeze(1) # logq(x)
        logp_y = self.pxt.log_px(y, ts)  # logp(y)
        logq_y = noise.log_prob(y).unsqueeze(1) # logq(y)

        value_x = logp_x - torch.logaddexp(logp_x, logq_x)  # logp(x)/(logp(x) + logq(x))
        value_y = logq_y - torch.logaddexp(logp_y, logq_y)  # logq(y)/(logp(y) + logq(y))

        v = value_x.mean() + value_y.mean()

        # Classification of noise vs target
        r_x = torch.sigmoid(logp_x - logq_x)
        r_y = torch.sigmoid(logq_y - logp_y)

        # Compute the classification accuracy
        acc = ((r_x > 1/2).sum() + (r_y > 1/2).sum()).cpu().numpy() / (len(x) + len(y))
        
        return -v, acc

    def fokker_planck_loss(self, x, ts):
        """
        This is the calculation of the term that ensures the derivatives match the log scale Fokker-Planck equation
        q(x,t) = log p(x,t)
        d/dt q(x,t) = \sum_{i=1}^{N} -[u(x)*dq(x,t)/dx_i + du(x)/dx]

        u(x): The drift function
        dq_dx: The derivative of the log probability with respect to x.
        du_dx: The derivative of the drift term with respect to x.
        Args:
            x (torch.Tensor): The input data of shape (n_cells, n_genes).
            ts (torch.Tensor): The time points at which to evaluate the model of shape (n_timesteps,).
            u (torch.Tensor): The tensor representing the potential energy.
            dq_dx (torch.Tensor): The tensor representing the derivative of the log probability with respect to x.
            du_dx (torch.Tensor): The tensor representing the derivative of the potential energy with respect to x.

        Returns:
            torch.Tensor: The tensor representing loss enforcing constraint to the Fokker-Planck term.
        """
        x.requires_grad = True
        ts.requires_grad = True
        xts = self.pxt.xts(x, ts)
        px_ux = (self.pxt.model(xts) * self.ux(xts[:,:,:-1]))
        px_ux_dx = torch.autograd.grad(outputs=px_ux,
                                       inputs=xts, 
                                       grad_outputs=torch.ones_like(px_ux),
                                       create_graph=True,
                                       retain_graph=True)[0]
        
        # Calculate the derivative of the log probability with respect to x and t
        # dq_dt = Left hand side of the Fokker-Planck equation: d/dt q(x,t)
        dq_dx, dq_dt = self.pxt.dx_dt(x, ts)
        dx = px_ux_dx[:,:,:-1].sum(dim=2, keepdim=True)
            
        # Enforce that dq_dt = -dx, i.e. that both sides of the fokker planck equation are equal
        l_fp = ((dq_dt + dx)**2).mean()
        return l_fp


    def consistency_loss(self, x, ts):
        """
        Enforce that the p(x,t_i) ~= p(x,t_0) for all t_i in ts
        """
        zero = torch.zeros(1).to(self.device)
        sum_log_pxt = self.pxt.log_pxt(x, ts).mean(dim=1)
        sum_log_px0 = self.pxt.log_pxt(x, zero).mean(dim=1).detach()
        l_consistency = ((sum_log_pxt - sum_log_px0)**2).mean()
        return l_consistency

    def optimize(self, X, X0, ts, px_noise, p0_noise,
                 pxt_lr=5e-4, ux_lr=1e-3,  
                 n_epochs=100, n_samples=1000, verbose=False):
        """
        Optimize the cell delta model parameters using the provided training data.

        Args:
            X (torch.tensor): The input training data of shape (n_cells, n_genes).
            X0 (torch.tensor): The initial state of the cells of shape (n_cells, n_genes).
            ts (torch.tensor): The time points at which to evaluate the model of shape (n_timesteps,).
            px_noise (torch.distributions): The noise distribution to use for the NCE loss for the overall distribution.
            p0_noise (torch.distributions): The noise distribution to use for the NCE loss for the initial conditions.
            pxt_lr (float, optional): The learning rate for the PXT model. Defaults to 5e-4.
            ux_lr (float, optional): The learning rate for the UX model. Defaults to 1e-3.
            n_epochs (int, optional): The number of epochs to train for. Defaults to 100.
            n_samples (int, optional): The number of data samples to use in training for each epoch. Defaults to 1000.
            verbose (bool, optional): Whether to print the optimization progress. Defaults to False.

        Returns:
            dict: A dictionary containing the loss values for each epoch.
        """
        nce_loss = self.nce_loss

        self.pxt_optimizer = torch.optim.Adam(self.pxt.parameters(), lr=pxt_lr)
        self.ux_optimizer = torch.optim.Adam(self.ux.parameters(), lr=ux_lr)

        # Convenience variable for the time t=0
        zero = torch.zeros(1).to(self.device)
        
        l_nce_pxs = np.zeros(n_epochs)
        l_nce_p0s = np.zeros(n_epochs)
        l_fps = np.zeros(n_epochs)
        
        n_samples = min(n_samples, len(X))

        for epoch in range(n_epochs):
            # Sample from the data distribution
            rand_idxs = torch.randperm(len(X))[:n_samples]
            x = X[rand_idxs].clone().detach()
            x0 = X0

            self.pxt_optimizer.zero_grad()
            self.ux_optimizer.zero_grad()

            # Calculate the Noise-Constrastive Loss of the distribution
            # of p(x,t) marginalized over t: p(x) = \int p(x,t) dt
            l_nce_px, acc_px = nce_loss(x, px_noise, ts=ts)
            l_nce_px.backward()
            # l_nce_px = zero
            # acc_px = zero
            
            # Calculate the Noise-Constrastive Loss of the initial distribution
            l_nce_p0, acc_p0 = nce_loss(x0, p0_noise, ts=zero)
            l_nce_p0.backward()

            # Calculate the consistency loss
            # l_consistency = self.consistency_loss(x, ts)
            # l_consistency.backward()
            l_consistency = zero

            # Calculate the Fokker-Planck loss
            l_fp = self.fokker_planck_loss(x, ts)
            l_fp.backward()
            # l_fp = zero

            self.pxt_optimizer.step()
            self.ux_optimizer.step()

            # Record the losses
            l_nce_pxs[epoch] = float(l_nce_px.mean())
            l_nce_p0s[epoch] = float(l_nce_p0.mean())
            l_fps[epoch] = float(l_fp.mean())
            
            if verbose:
                print(f'{epoch} l_nce_px={float(l_nce_px):.5f}, acc_px={float(acc_px):.5f}, '
                    f'l_nce_p0={float(l_nce_p0):.5f}, acc_p0={float(acc_p0):.5f},'
                    f'l_fp={float(l_fp):.5f} l_consistency={float(l_consistency):.5f}')
                
        return {'l_nce_px': l_nce_pxs, 'l_nce_p0': l_nce_p0s, 'l_fp': l_fps}
    
    def simulate(self, X0, tsim, ux_alpha=1, zero_boundary=True):
        # Simulate the stochastic differential equation using the Euler-Maruyama method
        # with the learned drift term u(x)
        x = X0.clone().detach()
        tsim = torch.linspace(0, 1, 100, device=self.device, requires_grad=False)
        xts = torch.zeros((len(tsim), x.shape[0], x.shape[1]), device='cpu')
        ht = tsim[1] - tsim[0]
        zero_boundary = zero_boundary

        for i in range(len(tsim)):
            # Compute the drift term
            u = self.ux(x)*ux_alpha
            # Compute the diffusion term
            # Generate a set of random numbers
            dW = torch.randn_like(x) * torch.sqrt(ht)
            sigma = torch.ones_like(x)
            # Compute the change in x
            dx = u * ht + sigma * dW
            # print(f'{float(u.mean()):.5f}, {float(ht):.3f}, {float(dW.mean()): .5f}, {float(dx.mean()): .5f}')
            dx = dx.squeeze(0)
            # Update x
            x = x + dx
            if zero_boundary:
                x[x < 0] = 0
            xts[i,:,:] = x.cpu().detach()
        return xts