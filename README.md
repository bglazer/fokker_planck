# Fokker-Planck Constrained Density Estimation for Single Cell Dynamics
## Mathematical statement of the problem
Given a sample $\hat x$ from a distribution $p(X)$, we assume that a subset of that sample $\hat x_0$ is the drawn from an initial state distribution $p_0(X)$. We assume that $p(X)$ is the mean density of a time varying distribution. 
$$p(x) = \frac{1}{T}\int_{0}^{T} p(x,t) dt$$ We assume that $X_t$ is determined by a stochastic differential equation:
$$dX_t = u(x)dt + \sigma dW$$ where 
* u(x) is a time-independent drift term
* $\sigma$ is a constant diffusion coefficient
* W is Wiener process (Gaussian noise)

The evolution of p(x,t) then follows a Fokker-Planck equation
$$\frac{\partial p(x,t)}{\partial t} = -\nabla \cdot (u(x)p(x,t)) + \Delta Dp(x,t)$$ where: 
* $\nabla$ is the divergence w.r.t. $x$ 
* $\Delta$ is the Laplacian
* $D$ is a constant diffusion term

We train a neural network $f_p(x,t)$ (`pxt` in the code) to learn the initial distribution $p(\hat x_0)$ and $p(\hat x) = \frac{1}{T} \sum_0^{T} p(\hat x, t)$ and  using noise contrastive estimation [1]. However, this is prone to learning a degenerate solution with $p(x,0)= p(x)$ and $p(x, t \neq 0) =0$, so we add a loss term to enforce a "consistency" condition $$\mathcal{L}_{cons} = \left\|\sum_{\hat x}p(x, t_i) - \sum_{\hat x}p(x, t_j)\right\|^2\forall i,j$$ 

However, the real goal is to learn the vector field that drives the dynamics, i.e. the $u(x)$ above. To do this, we model the drift as another neural network $f_u$ (`ux` in the code) add the Fokker-Planck equation as a loss term:
$$\mathcal{L}_{FP} = \left\|\frac{\partial p(x,t)}{\partial t} + \nabla \cdot (u(x)p(x,t))\right\|^2$$
assuming that the diffusion term is constant. 
**NOTE**: I'm just now realizing that excluding the diffusion might be a mistake. Not sure about this.

So, the overall loss is then: 
$$ \mathcal{L}_p + \mathcal{L}_{p0} + \mathcal{L}_{FP} + \mathcal{L}_{cons} $$
where :
* $\mathcal{L}_p$ is the NCE loss of estimating $p(\hat x)$
* $\mathcal{L}_p0$ is the NCE loss of estimating $p(\hat x_0)$
* $\mathcal{L}_{FP}$ is the Fokker-Planck consistency loss
* $\mathcal{L}_{cons}$ is the time density consistency loss

[1] https://proceedings.mlr.press/v9/gutmann10a.html

## Implementation

## Biological background
Single Cell RNA sequencing (scRNAseq) is a method for measuring the RNA content of individual cells. Briefly, RNA content gives us an idea of the state and function of particular cells, i.e. we expect a skin cell to have different RNA than a liver cell. This might be obvious but we can also tease apart much subtler distinctions such as which tumor cells might respond to a particular treatment. Further, it's known that cells can change their RNA state in response to their environment or during development. Often we know that cells begin in some "progenitor" or initial state and then stochastically transition to new states. We would like to create a machine learning model to help understand how different genes affect the paths cells take in "RNA-space" and potentially how we could perturb specific genes to cause the cells to adopt specific fates. Imagine reversing treatment resistance in tumor cells or inducing tissue repair rather than scarring or fibrosis.


However, scRNAseq data presents several challenges. First, it is destructive; we have to destroy cells to measure their RNA. We can't trace RNA levels in a single cell across time, we only get one snapshot of each cell. Second, it's high dimensional but quite sparse and noisy due to technical limitations of RNA capture and sequencing. Typical data has about 20k-30k genes and anywhere between 5k-100k cells. Time series data (multiple samples measured at different time points) is rare given the cost of scRNAseq and given the destructive measurement method, we can't map one cell at time $t_0$ to another cell at $t_1$. Instead we often assume that a single scRNAseq experiment contains a sample of cells that span the space of trajectories. That is, some cells are in the initial state, some cells are actively transitioning from initial to final state, and some cells are fully transitioned into another state. Typically we only know which cells are in the initial state.







