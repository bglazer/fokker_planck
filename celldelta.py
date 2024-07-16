import torch
from torch.nn import Linear, LeakyReLU, BatchNorm1d
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Ux(torch.nn.Module):
    """
    The u(x) drift term of the Fokker-Planck equation, modeled by a neural network
    """
    def __init__(self, input_dim, hidden_dim, n_layers, batch_norm=False):
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
        layers = []
        # First Linear layer
        if batch_norm:
            layers.append(BatchNorm1d(input_dim, affine=True))
        layers.append(Linear(input_dim, hidden_dim, bias=True))
        if batch_norm:
            layers.append(BatchNorm1d(hidden_dim, affine=True))
        layers.append(LeakyReLU())
        for i in range(n_layers - 1):
            layers.append(Linear(hidden_dim, hidden_dim, bias=True))
            if batch_norm:
                layers.append(BatchNorm1d(hidden_dim, affine=True))
            layers.append(LeakyReLU())  
        # Last Linear layer without BatchNorm1d after it
        layers.append(Linear(hidden_dim, input_dim, bias=True))
        self.layers = torch.nn.Sequential(*layers)
        self.batch_norm = batch_norm
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        s = x.shape
        x = x.reshape(-1, x.shape[-1])
        x = self.model(x)
        # Reshape to restore time and data dimensions
        x = x.reshape(s[:-1] + (-1,))
        return x
    
    def div(self, x):
        """
        Compute the divergence of u(x) with respect to x. Divergence is the sum of the 
        partial derivatives of each component of u(x) with respect to each element of x.
        \sum_{i=1}^{N} du_i(x)/dx_i
        """
        u = self.model(x)
        div = torch.zeros_like(x[:,0])
        for i in range(u.shape[1]):
            div += torch.autograd.grad(u[:,i], x, 
                                       torch.ones_like(u[:,i]), 
                                       retain_graph=True, 
                                       create_graph=True)[0][:,i]
        return u, div

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
        layers = []
        # First Linear layer
        layers.append(Linear(input_dim+1, hidden_dim, bias=True))
        layers.append(LeakyReLU())
        for i in range(n_layers - 1):
            layers.append(Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(LeakyReLU())  
        # Last Linear layer without BatchNorm1d after it
        layers.append(Linear(hidden_dim, 1, bias=True))
        self.layers = torch.nn.Sequential(*layers)
        self.tscale = torch.ones(input_dim+1)
        self.tscale = torch.nn.Parameter(self.tscale, requires_grad=False)
        self.model = torch.nn.Sequential(*layers)

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
            torch.tensor: The log probability of the data at each time and data point of shape (n_cells, n_timesteps).
        """
        xts = self.xts(x, ts)
        log_ps = self.forward(xts)
        return log_ps
    
    def pxt(self, x, ts):
        """
        Exponentiate the log probability of the data at the given time points and data points.
        """
        return torch.exp(self.log_pxt(x, ts))

    def forward(self, xts):
        # Scale the weights linear layer connected to the last element of the input by the time scale
        x = xts
        layer0 = self.layers[0]
        w = layer0.weight
        b = layer0.bias
        s = x.shape
        # Flatten the time and data dimensions so that it's a single big batch
        x = x.reshape(-1, x.shape[-1])
        # First layer with scaling factor
        x = ((x*self.tscale) @ w.T + b)  

         # Compute the remaining layers
        for layer in self.layers[1:]: 
            x = layer(x)

        # Reshape to restore time and data dimensions
        x = x.reshape(s[:-1] + (-1,))
        return x
    
    def set_tscale(self, tscale):
        """
        Set the time scaling factor for the model
        """
        self.tscale[-1] = tscale
    
    def log_px(self, x, ts):
        """
        Marginalize out the t dimension to get log(p(x))
        """
        return torch.logsumexp(self.log_pxt(x, ts), dim=0) 

    def dx_dt(self, x, ts):
        """
        Compute the partial derivative of log p(x,t) with respect to x and to t

        Returns:
            torch.tensor: The partial derivative of log p(x,t) with respect to x
            torch.tensor: The partial derivative of log p(x,t) with respect to t
        """
        xts = self.xts(x, ts)
        log_pxt = self.forward(xts)
        dpdx = torch.autograd.grad(outputs=log_pxt, 
                                   inputs=xts, 
                                   grad_outputs=torch.ones_like(log_pxt), 
                                   create_graph=True,
                                   retain_graph=True)[0]
        # The last element of the gradient is the derivative with respect to t
        # The :-1 is to get the gradient with respect to x
        # The -1: is to get the gradient with respect to t
        return dpdx[...,:-1], dpdx[...,-1:]
    
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
                 ux_batch_norm=False,
                 device='cpu') -> None:
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
        self.ux = Ux(input_dim, ux_hidden_dim, ux_layers, ux_batch_norm).to(device)
        self.pxt = Pxt(input_dim, pxt_hidden_dim, pxt_layers).to(device)
        self.device = device
        # Add the component models (ux, pxt, nce) to a module list
        self.models = torch.nn.ModuleDict({'ux':self.ux, 'pxt':self.pxt})
    
    def nce_loss(self, x, noise, ts, scale):
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

        log_scale = torch.log(torch.tensor(scale))
        logp_x = self.pxt.log_px(x, ts) + log_scale # logp(x)
        logq_x = noise.log_prob(x).unsqueeze(1) # logq(x)
        logp_y = self.pxt.log_px(y, ts) + log_scale # logp(y)
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

    def pseudotime_loss(self, X, X0_mask, ts):
        """
        Penalize data points in the initial state that are not at the beginning of the pseudotime trajectory.
        Also penalize data points that are not in the initial state that have a pseudotime of 0.

        Args:
            X0 (torch.Tensor): The initial state of the cells of shape (n_cells, n_genes).
            ts (torch.Tensor): The time points at which to evaluate the model of shape (n_timesteps,).
        """

        log_pxt = self.pxt.log_pxt(X, ts)
        log_pxt0 = log_pxt[:, X0_mask]
        log_pxtn0 = log_pxt[:, ~X0_mask]
        # Calculate the pseudotime of the initial state
        # Pseudotime is the timestep of max probability
        pt0 = torch.argmax(log_pxt0, dim=0).flatten()
        ptn0 = torch.argmax(log_pxtn0, dim=0).flatten()

        # Penalize data points in the initial state that don't have a pseudotime of 0
        if (pt0 != 0).any():
            l_pt0 = log_pxt0[pt0[pt0!=0],pt0!=0].mean()
            l_pt0 -= log_pxt0[torch.zeros_like(pt0[pt0!=0]),pt0!=0].mean()
        else:
            l_pt0 = torch.zeros(1, requires_grad=True).to(self.device)

        # Penalize data points not in the initial state that have a pseudotime of 0
        if (pt0 == 0).any():
            l_ptn0 = log_pxtn0[:,ptn0==0].mean()/((ptn0!=0).sum()+1)
            print(((ptn0!=0).sum()+1).item())
        else:
            l_ptn0 = torch.zeros(1, requires_grad=True).to(self.device)
        
        return l_pt0, l_ptn0
    
    def log_cosine_similarity(self, log_p, log_q):
        dot_product = torch.logsumexp(log_p + log_q, dim=0)
        norm_p = 0.5 * torch.logsumexp(2 * log_p, dim=0)
        norm_q = 0.5 * torch.logsumexp(2 * log_q, dim=0)
        return dot_product - norm_p - norm_q

    def log_cosine_loss(self, X, ts):
        log_pxt = self.pxt.log_pxt(X, ts)
        loss = torch.zeros(1, requires_grad=True).to(self.device)
        for i in range(1, log_pxt.shape[0]-1):
            loss += self.log_cosine_similarity(log_pxt[i], log_pxt[i+1])
        return loss
    
    def jensen_shannon(self, d1, d2):
        """
        Compute the Jensen-Shannon divergence between two distributions using log_softmax,
        avoiding the use of exp for numerical stability.
        """
        # Compute log_softmax for d1 and d2
        log_d1 = F.log_softmax(d1, dim=-1)
        log_d2 = F.log_softmax(d2, dim=-1)
        
        # Compute log of the mixture distribution
        log_m = torch.logsumexp(torch.stack([log_d1, log_d2]), dim=0) - torch.log(torch.tensor(2.0))
        
        # Compute KL divergences in log space
        kl_d1_m = F.kl_div(log_m, log_d1, reduction='batchmean', log_target=True)
        kl_d2_m = F.kl_div(log_m, log_d2, reduction='batchmean', log_target=True)
        
        # Compute Jensen-Shannon divergence
        js_div = 0.5 * (kl_d1_m + kl_d2_m)
        
        return js_div
    
    def entropy_loss(self, X, ts):
        """
        Penalize the entropy of the pseudtimes at each time point, using 
        softmax to convert the log probabilities to pseudo-probabilities.
        """
        log_pxt = self.pxt.log_pxt(X, ts)
        pxt = F.log_softmax(log_pxt, dim=1)  
        
        loss = torch.zeros(1, requires_grad=True).to(self.device)
        for i in range(1,pxt.shape[0]-1):
            loss += F.kl_div(pxt[i], pxt[i+1], 
                             reduction='batchmean', 
                             log_target=True)
        
        return -loss
    
    def entropy_loss2(self, X, ts):
        return -F.softmax(self.pxt.log_pxt(X, ts), dim=0).var(0).mean()

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

        Returns:
            torch.Tensor: The tensor representing loss enforcing constraint to the Fokker-Planck term.
        """
        x.requires_grad = True
        ts.requires_grad = True
        
        dq_dx, dq_dt = self.pxt.dx_dt(x, ts)
        ux, div_ux = self.ux.div(x)

        d_dx = ((dq_dx * ux).sum(dim=2) + div_ux)[...,None]

        # Enforce that dq_dt = -dx, i.e. that both sides of the fokker planck equation are equal
        l_fp = ((d_dx + dq_dt)**2).mean()
        
        x.requires_grad = False
        ts.requires_grad = False
        
        return l_fp
    
    def max_consistency_loss(self, X, ts):
        """
        Ensure that each timepoint has a similar max probability
        """
        log_pxt = self.pxt.log_pxt(X, ts)
        max_log_pxt = log_pxt.max(dim=1)[0]
        l_max = (max_log_pxt - max_log_pxt.mean())**2
        return l_max.mean()
    
    def consistency_loss(self, X, ts):
        """
        Ensure that each timepoint has a similar mean probability to t=0
        """
        log_pxt = self.pxt.log_pxt(X, ts)
        l_cons = (log_pxt[1:].mean(1) - log_pxt[0].mean())**2
        return l_cons.mean()

    def optimize(self, X, X0, X0_mask, ts, px_noise, p0_noise, p0_alpha=1,
                 pxt_lr=5e-4, ux_lr=1e-3, fokker_planck_alpha=1, 
                 l_max_alpha=None, 
                 l_consistency_alpha=None,
                 pt_alpha=None,
                 entropy_alpha=None,
                 p_alpha=None,
                 n_epochs=100, n_samples=1000, verbose=False):
        """
        Optimize the cell delta model parameters using the provided training data.

        Args:
            X (torch.tensor): The input training data of shape (n_cells, n_genes).
            X0 (torch.tensor): The initial state of the cells of shape (n_cells, n_genes).
            ts (torch.tensor): The time points at which to evaluate the model of shape (n_timesteps,).
            px_noise (torch.distributions): The noise distribution to use for the NCE loss for the overall distribution.
            p0_noise (torch.distributions): The noise distribution to use for the NCE loss for the initial conditions.
            p0_alpha (float, optional): The weight to apply to the initial conditions loss. Defaults to 1.
            pxt_lr (float, optional): The learning rate for the PXT model. Defaults to 5e-4.
            ux_lr (float, optional): The learning rate for the UX model. Defaults to 1e-3.
            fokker_planck_alpha (bool, optional): The weight to apply to the Fokker-Planck loss term. Defaults to 1.            n_epochs (int, optional): The number of epochs to train for. Defaults to 100.
            n_samples (int, optional): The number of data samples to use in training for each epoch. Defaults to 1000.
            verbose (bool, optional): Whether to print the optimization progress. Defaults to False.

        Returns:
            dict: A dictionary containing the loss values for each epoch.
        """
        self.pxt_optimizer = torch.optim.Adam(self.pxt.parameters(), lr=pxt_lr, weight_decay=1e-3)
        self.ux_optimizer = torch.optim.Adam(self.ux.parameters(), lr=ux_lr, weight_decay=1e-3)

        # Convenience variable for the time t=0
        zero = torch.zeros(1).to(self.device)
        
        l_nce_pxs = np.zeros(n_epochs)
        l_nce_p0s = np.zeros(n_epochs)
        l_fps = np.zeros(n_epochs)
        
        n_samples = min(n_samples, len(X))

        for epoch in range(n_epochs):
            # Sample from the data distribution
            rand_idxs = torch.randperm(len(X))[:n_samples]
            # TODO do I need to clone here?
            x = X[rand_idxs].detach()
            x0 = X0.detach()

            self.pxt_optimizer.zero_grad()
            self.ux_optimizer.zero_grad()

            # Calculate the Noise-Constrastive Loss of the distribution
            # of p(x,t) marginalized over t: p(x) = \int p(x,t) dt
            if p_alpha is not None:
                l_nce_px, acc_px = self.nce_loss(x, px_noise, ts=ts, scale=1/ts.shape[0])
                l_nce_px.backward()
            else:
                l_nce_px = zero
                acc_px = zero
            
            # Calculate the Noise-Constrastive Loss of the initial distribution
            if p0_alpha is not None:
                l_nce_p0, acc_p0 = self.nce_loss(x0, p0_noise, ts=zero, scale=1)
                l_nce_p0 = l_nce_p0 * p0_alpha
                l_nce_p0.backward()
            else:
                l_nce_p0 = zero
                acc_p0 = zero
                       
            if fokker_planck_alpha is not None:
                # Calculate the Fokker-Planck loss
                l_fp = self.fokker_planck_loss(x, ts)*fokker_planck_alpha
                l_fp.backward()
            else:
                l_fp = zero

            if l_max_alpha is not None:
                l_max = self.max_consistency_loss(x, ts)
                l_max.backward()
            else:
                l_max = zero

            if l_consistency_alpha is not None:
                l_cons = self.consistency_loss(x, ts)*l_consistency_alpha
                l_cons.backward()
            else:
                l_cons = zero

            if pt_alpha is not None:
                l_pt0, l_ptn0 = self.pseudotime_loss(X, X0_mask, ts)
                l_pt0 = l_pt0 * pt_alpha
                l_ptn0 = l_ptn0 * pt_alpha
                l_pt0.backward(retain_graph=True)
                l_ptn0.backward()
            else:
                l_pt0 = zero
                l_ptn0 = zero

            if entropy_alpha is not None:
                l_entropy = self.log_cosine_loss(X, ts)*entropy_alpha
                l_entropy.backward()
            else:
                l_entropy = zero

            # ux_total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in self.ux.model.parameters() if p.requires_grad]), 2)
            # pxt_total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in self.pxt.model.parameters() if p.requires_grad]), 2)
            # print(f'ux total norm: {ux_total_norm}, pxt total norm: {pxt_total_norm}')

            # torch.nn.utils.clip_grad_norm_(self.ux.model.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(self.pxt.model.parameters(), max_norm=1.0)

            self.pxt_optimizer.step()
            self.ux_optimizer.step()

            # Record the losses
            l_nce_pxs[epoch] = float(l_nce_px.mean())
            l_fps[epoch] = float(l_fp.mean())
            
            if verbose:
                print(f'{epoch:6d} l_nce_px={float(l_nce_px):.5f}, acc_px={float(acc_px):.5f}, '
                    f'l_nce_p0={float(l_nce_p0): .5f}, '
                    f'acc_p0={float(acc_p0): .5f}, '
                    f'l_fp={float(l_fp):.5f}, '
                    f'l_entropy={float(l_entropy):.5f}, '
                    f'l_cons={float(l_cons):.5f}, '
                    f'l_pt0={float(l_pt0):.5f}, '
                    f'l_ptn0={float(l_ptn0):.5f} '
                    )
                
        return {'l_nce_px': l_nce_pxs, 'l_nce_p0': l_nce_p0s, 'l_fp': l_fps}
    
    def optimize_fokker_planck(self, X, ts, 
                               ux_lr=1e-3, fokker_planck_alpha=1,
                               noise=None,
                               n_epochs=100, n_samples=1000, verbose=False):
        """
        Optimize the Fokker-Planck component of the loss independently of the NCE component.
        """
        self.ux_optimizer = torch.optim.Adam(self.ux.parameters(), lr=ux_lr)

        l_fps = np.zeros(n_epochs)
        l_fp0s = np.zeros(n_epochs)
        
        n_samples = min(n_samples, len(X))

        if noise is not None:
            # Create a Gaussian distribution
            normal = torch.distributions.Normal(loc=torch.zeros(n_samples,device=self.device), 
                                                scale=torch.ones(n_samples,device=self.device)*noise)

        for epoch in range(n_epochs):
            # Sample from the data distribution
            rand_idxs = torch.randperm(len(X))[:n_samples]
            x = X[rand_idxs].clone().detach()
            if noise is not None:
                # Add Gaussian noise to the data
                x = x + normal.sample().unsqueeze(1)

            self.pxt_optimizer.zero_grad()
            self.ux_optimizer.zero_grad()

            # Calculate the Fokker-Planck loss
            l_fp = self.fokker_planck_loss(x, ts)*fokker_planck_alpha
            l_fp.backward()

            # Fokker-Planck loss at t=0
            # l_fp0 = self.fokker_planck_loss(x, ts[:1])*fokker_planck_alpha
            # l_fp0.backward()
            l_fp0 = torch.zeros(1).to(self.device)

            self.ux_optimizer.step()

            # Record the losses
            l_fps[epoch] = float(l_fp.mean())
            l_fp0s[epoch] = float(l_fp0.mean())
            
            if verbose:
                print(f'{epoch} l_fp={float(l_fp):.5f}, l_fp0={float(l_fp0):.5f}')
                
        return {'l_fp': l_fps, 'l_fp0': l_fp0s}
    
    def optimize_initial_conditions(self, X0, ts, p0_noise, pxt_lr=1e-3,
                                    n_epochs=100, verbose=False, scale=1):
        """
        Optimize the initial conditions of the model.
        """
        nce_loss = self.nce_loss

        self.pxt_optimizer = torch.optim.Adam(self.pxt.parameters(), lr=pxt_lr)

        l_nce_p0s = np.zeros(n_epochs)
        
        for epoch in range(n_epochs):
            ti = torch.randint(0, len(ts), (1,)).item()
            t = ts[ti,None].detach()

            self.pxt_optimizer.zero_grad()
            
            # Fit every time to the initial distribution
            l_nce_p0, acc_p0 = nce_loss(X0, p0_noise, ts=t, scale=scale)
            l_nce_p0.backward()

            self.pxt_optimizer.step()

            # Record the losses
            l_nce_p0s[epoch] = float(l_nce_p0.mean())
            
            if verbose:
                print(f'{epoch} l_nce_p0={float(l_nce_p0):.5f}, acc_p0={float(acc_p0):.5f}')
                
        return {'l_nce_p0': l_nce_p0s}

    def simulate(self, X0, tsim, sigma=1, zero_boundary=True):
        """
        Simulate the stochastic differential equation using the Euler-Maruyama method
        with the learned drift term u(x)
        """
        x = X0.clone().detach()
        xts = torch.zeros((len(tsim), x.shape[0], x.shape[1]), device='cpu')
        ht = tsim[1] - tsim[0]
        zero_boundary = zero_boundary
        # sigma = torch.ones_like(x)*sigma
        
        for i in range(len(tsim)):
            # Compute the drift term
            u = self.ux(x)
            # Compute the diffusion term
            # Generate a set of random numbers
            dW = torch.randn_like(x) * torch.sqrt(ht)
            # Compute the change in x
            dx = u * ht + sigma * dW
            # dx = dx.squeeze(0)
            # Update x
            x = x + dx
            if zero_boundary:
                x[x < 0] = 0.0
            xts[i,:,:] = x.cpu().detach()
        return xts