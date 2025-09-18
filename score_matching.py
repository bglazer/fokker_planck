import torch
import torch.nn as nn

class ScoreMatching(nn.Module):
    def __init__(self, snet, alpha, sigma, eta, D, T, device='cpu'):
        super(ScoreMatching, self).__init__()

        print('Score Matching by JT.')

        self.snet = snet
        
        # other hyperparams
        self.D = D
                
        self.sigma = sigma
        
        self.T = T
        
        self.alpha = alpha
        
        self.eta = eta
        
        self.device = device    
    
    def sample_base(self, x_1, sigma=1.0):
        # Uniform over [-1, 1]**D
        return torch.randn_like(x_1, device=self.device) * sigma
    
    def langevine_dynamics(self, x):
        for t in range(self.T):
            x = x + self.alpha * self.snet(x) + self.eta * torch.randn_like(x, device=self.device)
        return x

    def forward(self, x, reduction='mean'):
        # =====Score Matching
        # sample noise
        epsilon = torch.randn_like(x, device=self.device)

        # =====
        # calculate the noisy data
        tilde_x = x + self.sigma * epsilon

        # =====
        # calculate the score model
        s = self.snet(tilde_x)
        
        # =====LOSS: the Score Matching Loss
        SM_loss = (1. / (2. * self.sigma)) * ((s + epsilon)**2.).sum(-1) # in order to keep the Langevine dynamics unchanged, we do not use \tilde{s} = -sigma * s but we use \tilde{s} = sigma * s
        
        # Final LOSS
        if reduction == 'sum':
            loss = SM_loss.sum()
        else:
            loss = SM_loss.mean()

        return loss

    def sample(self,  batch_size=64, sigma=1.0):
        # sample x_0
        x_ = torch.empty(batch_size, self.D, device=self.device)
        x = self.sample_base(x_, sigma=sigma)
        
        # run langevine dynamics
        x = self.langevine_dynamics(x)
        
        return x