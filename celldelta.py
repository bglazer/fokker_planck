import torch
from torch.nn import Linear, ReLU
import torch.nn as nn
import numpy as np

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
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.5):
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
        # Apply dropout to the model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.model(x)
    
    def dx(self, x, ts, hx=1e-3):
        """
        Compute the derivative of u(x) with respect to x
        """
        xgrad = (self(x+hx, ts) - self(x-hx, ts))/(2*hx)
        return xgrad

# The p(x,t) term of the Fokker-Planck equation, modeled by a neural network
class Pxt(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.5):
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
        # Apply dropout to the model
        self.dropout = nn.Dropout(p=dropout)

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

        # Repeat the x and t vectors for each timestep in the ts range
        xs = x.repeat((ts.shape[0],1,1,))
        ts_ = ts.repeat((x.shape[0],1)).T.unsqueeze(2)
        # Concatentate them together to match the input the MLP model
        xts = torch.concatenate((xs,ts_), dim=2)
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
    
    def dx(self, x, ts, hx=1e-3):
        """
        Compute the partial derivative of p(x,t) with respect to x
        """
        xgrad = (self.pxt(x+hx, ts) - self.pxt(x-hx, ts))/(2*hx)
        return xgrad

    def dt(self, x, ts, ht=1e-3):
        """
        Compute the partial derivative of p(x,t) with respect to t
        """
        tgrad = (self.pxt(x, ts+ht) - self.pxt(x, ts-ht))/(2*ht)
        return tgrad
    
class CellDelta(nn.Module):
    """
    CellDelta is a learned model of cell differentiation in single-cell RNA-seq single timepoint data.
    It models the developmental trajectory of cells as a driven stochastic process
    # TODO expand this docstring to describe the rationale of the model
    """
    def __init__(self, input_dim, 
                 ux_hidden_dim, ux_layers, ux_dropout,
                 pxt_hidden_dim, pxt_layers, pxt_dropout,
                 noise, device) -> None:
        """
        Initialize the CellDelta model with the given hyperparameters.

        Args:
            input_dim (int): The dimensionality of the input data.
            ux_hidden_dim (int): The number of hidden units in each layer for the UX model.
            ux_layers (int): The number of layers in the UX model.
            ux_dropout (float): The dropout probability for the UX model.
            pxt_hidden_dim (int): The number of hidden units in each layer for the PXT model.
            pxt_layers (int): The number of layers in the PXT model.
            pxt_dropout (float): The dropout probability for the PXT model.
            noise (torch.distributions): The noise distribution to use for the NCE loss.
            device (torch.device): The device to use for the model.

        Returns:
            None
        """
        super().__init__()
        self.ux = Ux(input_dim, ux_hidden_dim, ux_layers, ux_dropout).to(device)
        self.pxt = Pxt(input_dim, pxt_hidden_dim, pxt_layers, pxt_dropout).to(device)
        self.noise = noise
        self.device = device
        # Add the component models (ux, pxt, nce) to a module list
        self.models = torch.nn.ModuleDict({'ux':self.ux, 'pxt':self.pxt})
    
    def nce_loss(self, x, ts):
        """
        Compute the Noise-Contrastive Estimation (NCE) loss for the given data.

        Args:
            x (torch.tensor): The input data of shape (n_cells, n_genes).
            ts (torch.tensor): The time points at which to evaluate the model of shape (n_timesteps,).
        
        Returns:
            torch.tensor: The NCE loss value.
            torch.tensor: The sample vs noise classification accuracy.
        """
        y = self.noise.sample((x.shape[0],))

        logp_x = self.pxt.log_px(x, ts)  # logp(x)
        logq_x = self.noise.log_prob(x).unsqueeze(1)  # logq(x)
        logp_y = self.pxt.log_px(y, ts)  # logp(y)
        logq_y = self.noise.log_prob(y).unsqueeze(1)  # logq(y)

        value_x = logp_x - torch.logsumexp(torch.cat([logp_x, logq_x], dim=1), dim=1, keepdim=True)  # logp(x)/(logp(x) + logq(x))
        value_y = logq_y - torch.logsumexp(torch.cat([logp_y, logq_y], dim=1), dim=1, keepdim=True)  # logq(y)/(logp(y) + logq(y))

        v = value_x.mean() + value_y.mean()

        # Classification of noise vs target
        r_x = torch.sigmoid(logp_x - logq_x)
        r_y = torch.sigmoid(logq_y - logp_y)

        # Compute the classification accuracy
        acc = ((r_x > 1/2).sum() + (r_y > 1/2).sum()).cpu().numpy() / (len(x) + len(y))
        
        return -v, acc

    def optimize(self, X, X0, ts, restart=True, pxt_lr=5e-4, ux_lr=1e-3, alpha_fp=1, n_epochs=100, n_samples=1000, hx=1e-3, verbose=False):
        """
        Optimize the cell delta model parameters using the provided training data.

        Args:
            X (torch.tensor): The input training data of shape (n_cells, n_genes).
            X0 (torch.tensor): The initial state of the cells of shape (n_cells, n_genes).
            ts (torch.tensor): The time points at which to evaluate the model of shape (n_timesteps,).
            restart (bool, optional): Whether to restart the optimization from scratch or continue from the current state. Defaults to True.
            pxt_lr (float, optional): The learning rate for the PXT model. Defaults to 5e-4.
            ux_lr (float, optional): The learning rate for the UX model. Defaults to 1e-3.
            n_epochs (int, optional): The number of epochs to train for. Defaults to 100.
            n_samples (int, optional): The number of data samples to use in training for each epoch. Defaults to 1000.
            hx (float, optional): The step size for the numerical differentiation. Defaults to 1e-3.
            verbose (bool, optional): Whether to print the optimization progress. Defaults to False.

        Returns:
            dict: A dictionary containing the loss values for each epoch.
        """
        if restart:
            self.pxt_optimizer = torch.optim.Adam(self.pxt.parameters(), lr=pxt_lr)
            self.ux_optimizer = torch.optim.Adam(self.ux.parameters(), lr=ux_lr)
        
        # Convenience variable for the time t=0
        zero = torch.zeros(1, requires_grad=False).to(self.device)
        
        l_nce_pxs = np.zeros(n_epochs)
        l_nce_p0s = np.zeros(n_epochs)
        l_fps = np.zeros(n_epochs)
        
        # For the sake of brevity in the code, we omit the self. prefix
        ux = self.ux
        pxt = self.pxt

        n_samples = min(n_samples, len(X))

        for epoch in range(n_epochs):
            # Sample from the data distribution
            rand_idxs = torch.randperm(len(X))[:n_samples]
            x = X[rand_idxs]
            x0 = X0

            self.pxt_optimizer.zero_grad()
            self.ux_optimizer.zero_grad()

            # Calculate the Noise-Constrastive Loss of the distribution
            # of p(x,t) marginalized over t: p(x) = \int p(x,t) dt
            l_nce_px, acc_px = self.nce_loss(x, ts=ts)
            l_nce_px.backward()
            
            # Calculate the Noise-Constrastive Loss of the initial distribution
            l_nce_p0, acc_p0 = self.nce_loss(x0, ts=zero)
            l_nce_p0.backward()

            # This is the calculation of the term that ensures the
            # derivatives match the Fokker-Planck equation
            # d/dt p(x,t) = \sum_{i=1}^{N} -d/dx_i (u(x) p(x,t))
            xs = x.repeat((ts.shape[0],1,1))
            ts_ = ts.repeat((x.shape[0],1)).T.unsqueeze(2)
            # Concatentate them together to match the input the MLP model
            xts = torch.concatenate((xs,ts_), dim=2)
            up_pxt = ux(x)*torch.exp(pxt.model(xts))
            up_dx = torch.autograd.grad(outputs=up_pxt, 
                                        inputs=xts, 
                                        grad_outputs=torch.ones_like(up_pxt),
                                        create_graph=True)[0]
            up_dx = up_dx[:,:,:-1].sum(dim=2, keepdim=True)
            pxt_dts = pxt.dt(x, ts)
            l_fp = ((pxt_dts + up_dx)**2).mean()*alpha_fp
            l_fp.backward()

            # Take a gradient step
            self.pxt_optimizer.step()
            self.ux_optimizer.step()

            # Record the losses
            l_nce_pxs[epoch] = float(l_nce_px.mean())
            l_nce_p0s[epoch] = float(l_nce_p0.mean())
            l_fps[epoch] = float(l_fp.mean())
            
            if verbose:
                print(f'{epoch} l_nce_px={float(l_nce_px):.5f}, acc_px={float(acc_px):.4f}, '
                    f'l_nce_p0={float(l_nce_p0):.5f}, acc_p0={float(acc_p0):.4f}, '
                    f'l_fp={float(l_fp):.5f}')
                
        return {'l_nce_px': l_nce_pxs, 'l_nce_p0': l_nce_p0s, 'l_fp': l_fps}