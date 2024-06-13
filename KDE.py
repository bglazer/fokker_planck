import torch
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution
import numpy as np

class GaussianKDE(Distribution):
    def __init__(self, X, bw, chunk_size=None):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        self.X = X
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims, device=X.device),
                                      covariance_matrix=torch.eye(self.dims, device=X.device))
        self.chunk_size = chunk_size

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.

        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.X.detach()
        diffs = (X.unsqueeze(1) - Y) / self.bw
        log_p_norms = self.mvn.log_prob(diffs)
        log_probs = (-self.dims) * np.log(self.bw) + \
                    torch.logsumexp(log_p_norms, dim=0) - np.log(self.n)

        return log_probs

    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated

        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        if self.chunk_size is not None:
            log_prob = torch.zeros((Y.shape[0],), device=Y.device, dtype=Y.dtype)
            
            X = self.X.detach()
            Y = Y.detach()
            for x_start in range(0, X.shape[0], self.chunk_size):
                x_end = x_start + self.chunk_size
                X_chunk = X[x_start:x_end]
                for y_start in range(0, Y.shape[0], self.chunk_size):
                    y_end = y_start + self.chunk_size
                    Y_chunk = Y[y_start:y_end]
                    log_prob[y_start:y_end] = self.score_samples(Y_chunk, X_chunk)

        else:
            log_prob = self.score_samples(Y).sum(dim=0)

        return log_prob