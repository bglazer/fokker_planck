import torch
from torch import logsumexp
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
    def __init__(self, input_dim, hidden_dim, n_layers, grad_mode='analytical', dropout=0.0):
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
        # if self.grad_mode not in ('analytic', 'numerical'):
        #     raise ValueError('grad_mode must be one of ("analytic", "numerical")')
        # self.grad_mode = grad_mode
        # if self.grad_mode == 'analytic':
        #     self.dx = self.dx_
        # elif self.grad_mode == 'numerical':
        #     self.dx = self.dx_num_


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

    # TODO implement batched numerical differentiation
    # TODO compare memory usage and speed of analytical vs numerical differentiation    
    # def dx_num_(self, x):
    #     """
    #     Compute the derivative of u(x) with respect to x
    #     """
    #     dudx = torch.zeros_like(x)
    #     x_up = x.clone().detach()
    #     x_dn = x.clone().detach()
    #     for i in range(x.shape[1]):
    #         x_up[:,i] += self.hx
    #         x_dn[:,i] -= self.hx
    #     ux_up = self.model(x_up)
    #     ux_dn = self.model(x_dn)
    #     dudx = (ux_up - ux_dn)/(2*self.hx)
    #     return dudx

    
# The p(x,t) term of the Fokker-Planck equation, modeled by a neural network
class Pxt(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
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

class SelfNoise(torch.nn.Module):
    """
    # TODO expand this description if this works well
    A noise distribution that is learned from the data
    """
    def __init__(self, celldelta, ts):
        super().__init__()
        self.celldelta = celldelta
        self.ts = ts
    
    def generate(self, x0, ts):
        """
        Generate a set of samples from the noise distribution, by simulating
        the Fokker-Planck equation with the learned drift term u(x)

        Sets the SelfNoise.xts attribute to the generated samples
        """
        # Simulate steps in the ODE using the learned drift term
        # The take all the samples and flatten them into a single dimension
        self.xts = self.celldelta.simulate(x0, ts).reshape((-1, x0.shape[1]))

    def sample(self, n_samples):
        """
        Sample from the learned noise distribution
        """
        rand_idxs = torch.randperm(len(self.xts))[:n_samples[0]]
        return self.xts[rand_idxs,:].detach().to(self.celldelta.device)
    
    def log_prob(self, x):
        """
        Compute the log probability of the given data under the learned noise distribution
        """
        return self.celldelta.pxt.log_px(x, ts=self.ts).squeeze(1)

class CellDelta(nn.Module):
    """
    CellDelta is a learned model of cell differentiation in single-cell RNA-seq single timepoint data.
    It models the developmental trajectory of cells as a driven stochastic process
    # TODO expand this docstring to describe the rationale of the model
    """
    def __init__(self, input_dim, 
                 ux_hidden_dim, ux_layers, ux_dropout,
                 pxt_hidden_dim, pxt_layers, pxt_dropout,
                 device, loss_type='nce', normalize_gradient=False) -> None:
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
            device (torch.device): The device to use for the model.

        Returns:
            None
        """
        super().__init__()
        if loss_type not in ('nce', 'ence'):
            raise ValueError('loss_type must be one of ("nce", "ence")')
        self.loss_type = loss_type
        self.normalize_gradient = normalize_gradient
        self.ux = Ux(input_dim, ux_hidden_dim, ux_layers, ux_dropout).to(device)
        self.pxt = Pxt(input_dim, pxt_hidden_dim, pxt_layers, pxt_dropout).to(device)
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
        logq_x = noise.log_prob(x).unsqueeze(1)  # logq(x)
        logp_y = self.pxt.log_px(y, ts)  # logp(y)
        logq_y = noise.log_prob(y).unsqueeze(1)  # logq(y)

        value_x = logp_x - torch.logsumexp(torch.cat([logp_x, logq_x], dim=1), dim=1, keepdim=True)  # logp(x)/(logp(x) + logq(x))
        value_y = logq_y - torch.logsumexp(torch.cat([logp_y, logq_y], dim=1), dim=1, keepdim=True)  # logq(y)/(logp(y) + logq(y))

        v = value_x.mean() + value_y.mean()

        # Classification of noise vs target
        r_x = torch.sigmoid(logp_x - logq_x)
        r_y = torch.sigmoid(logq_y - logp_y)

        # Compute the classification accuracy
        acc = ((r_x > 1/2).sum() + (r_y > 1/2).sum()).cpu().numpy() / (len(x) + len(y))
        
        return -v, acc
    
    def ence_loss(self, x, noise, ts):
        """
        Compute the eNCE loss for the given data.
        From https://arxiv.org/pdf/2110.11271.pdf

        Args:
            x (torch.tensor): The input data of shape (n_cells, n_genes).
            ts (torch.tensor): The time points at which to evaluate the model of shape (n_timesteps,).
        
        Returns:
            torch.tensor: The eNCE loss value.
            torch.tensor: The sample vs noise classification accuracy.
        """
        y = noise.sample((x.shape[0],))

        logp_x = self.pxt.log_px(x, ts)  # logp(x)
        logq_x = noise.log_prob(x).unsqueeze(1)  # logq(x)
        logp_y = self.pxt.log_px(y, ts)  # logp(y)
        logq_y = noise.log_prob(y).unsqueeze(1)  # logq(y)

        qx_px = logq_x - logp_x
        py_qy = logp_y - logq_y
        # Threshold the log density ratios to avoid large gradients
        if self.ratio_clip is not None:
            ratio_min, ratio_max = self.ratio_clip
            qx_px = torch.clamp(qx_px, min=ratio_min, max=ratio_max)
            py_qy = torch.clamp(py_qy, min=ratio_min, max=ratio_max)

        # c is the number of samples, it's used to normalize the logsumexp to get logmeanexp
        c = torch.log(torch.tensor(logp_x.shape[0],
                                    device=x.device, dtype=torch.float32, requires_grad=False))
        log_loss_x = logsumexp(1/2*(qx_px), dim=0) - c
        log_loss_y = logsumexp(1/2*(py_qy), dim=0) - c
        log_loss = (log_loss_x + log_loss_y).mean()
        
        # Classification of noise vs target
        # r_x = 1/2*qx_px
        # r_y = 1/2*py_qy

        # Compute the classification accuracy
        # acc = ((r_x > 1/2).sum() + (r_y > 1/2).sum()).cpu().numpy() / (len(x) + len(y))
        
        return log_loss, 0
    
    def fokker_planck_loss(self, x, ts, alpha_fp):
        """
        This is the calculation of the term that ensures the derivatives match the log scale Fokker-Planck equation
        q(x,t) = log p(x,t)
        d/dt q(x,t) = \sum_{i=1}^{N} -[u(x)*dq(x,t)/dx_i + du(x)/dx]

        Args:
            u (torch.Tensor): The tensor representing the potential energy.
            dq_dx (torch.Tensor): The tensor representing the derivative of the log probability with respect to x.
            du_dx (torch.Tensor): The tensor representing the derivative of the potential energy with respect to x.

        Returns:
            torch.Tensor: The tensor representing loss enforcing constraint to the Fokker-Planck term.
        """

        # Calculate the derivative of the log probability with respect to x and t
        # dq_dt = Left hand side of the Fokker-Planck equation: d/dt q(x,t)
        dq_dx, dq_dt = self.pxt.dx_dt(x, ts)
        du_dx = self.ux.dx(x)
        ux = self.ux(x)
        # Right hand side of the Fokker-Planck equation: \sum_{i=1}^{N} -[u(x)*dq(x,t)/dx_i + du(x)/dx]
        dxi = ux*dq_dx + du_dx
        # Sum over the samples
        dx = dxi.sum(dim=2, keepdim=True)
            
        # Enforce that dq_dt = -dx, i.e. that both sides of the fokker planck equation are equal
        l_fp = ((dq_dt + dx)**2).mean()*alpha_fp
        return l_fp

    def optimize(self, X, X0, ts, noise, restart=True, pxt_lr=5e-4, ux_lr=1e-3, alpha_fp=1, n_epochs=100, n_samples=1000, ratio_clip=(-10,10), verbose=False):
        """
        Optimize the cell delta model parameters using the provided training data.

        Args:
            X (torch.tensor): The input training data of shape (n_cells, n_genes).
            X0 (torch.tensor): The initial state of the cells of shape (n_cells, n_genes).
            ts (torch.tensor): The time points at which to evaluate the model of shape (n_timesteps,).
            noise (torch.distributions): The noise distribution to use for the NCE loss.
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
        if self.loss_type == 'nce':
            nce_loss = self.nce_loss
        if self.loss_type == 'ence':
            nce_loss = self.ence_loss

        self.pxt_optimizer = torch.optim.Adam(self.pxt.parameters(), lr=pxt_lr)
        self.ux_optimizer = torch.optim.Adam(self.ux.parameters(), lr=ux_lr)
        # if restart:
        #     if self.loss_type == 'ence':
        #         self.pxt_optimizer = torch.optim.SGD(self.pxt.parameters(), lr=pxt_lr)
            
        
        self.ratio_clip = ratio_clip

        # Convenience variable for the time t=0
        zero = torch.zeros(1, requires_grad=True).to(self.device)
        
        l_nce_pxs = np.zeros(n_epochs)
        l_nce_p0s = np.zeros(n_epochs)
        l_fps = np.zeros(n_epochs)
        
        # For the sake of brevity in the code, we omit the self prefix
        pxt = self.pxt

        n_samples = min(n_samples, len(X))

        for epoch in range(n_epochs):
            # Sample from the data distribution
            rand_idxs = torch.randperm(len(X))[:n_samples]
            x = X[rand_idxs].clone().detach()
            x.requires_grad=True
            x0 = X0

            # Generate a set of samples from the noise distribution
            # TODO only generate every N epochs?
            # noise.generate(x0, ts)

            self.pxt_optimizer.zero_grad()
            self.ux_optimizer.zero_grad()

            # Calculate the Noise-Constrastive Loss of the distribution
            # of p(x,t) marginalized over t: p(x) = \int p(x,t) dt
            l_nce_px, acc_px = nce_loss(x, noise, ts=ts)
            l_nce_px.backward()
            
            # Calculate the Noise-Constrastive Loss of the initial distribution
            l_nce_p0, acc_p0 = nce_loss(x0, noise, ts=zero)
            l_nce_p0.backward()

            l_fp = self.fokker_planck_loss(x, ts, alpha_fp)
            l_fp.backward()

            # Take a gradient step
            # Normalize the pxt gradients:
            if self.normalize_gradient:
                for p in pxt.parameters():
                    p.grad /= torch.linalg.vector_norm(p.grad) + 1e-8
            self.pxt_optimizer.step()
            self.ux_optimizer.step()

            # Record the losses
            l_nce_pxs[epoch] = float(l_nce_px.mean())
            l_nce_p0s[epoch] = float(l_nce_p0.mean())
            l_fps[epoch] = float(l_fp.mean())
            
            if verbose:
                print(f'{epoch} l_nce_px={float(l_nce_px):.5f}, acc_px={float(acc_px):.5f}, '
                    f'l_nce_p0={float(l_nce_p0):.5f}, acc_p0={float(acc_p0):.5f},'
                    f'l_fp={float(l_fp):.5f}')
                
        return {'l_nce_px': l_nce_pxs, 'l_nce_p0': l_nce_p0s, 'l_fp': l_fps}
    
    def simulate(self, X0, tsim, zero_boundary=True):
        # Simulate the stochastic differential equation using the Euler-Maruyama method
        # with the learned drift term u(x)
        x = X0.clone().detach()
        tsim = torch.linspace(0, 1, 100, device=self.device, requires_grad=False)
        xts = torch.zeros((len(tsim), x.shape[0], x.shape[1]), device='cpu')
        ht = tsim[1] - tsim[0]
        zero_boundary = zero_boundary

        for i in range(len(tsim)):
            # Compute the drift term
            u = self.ux(x)
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