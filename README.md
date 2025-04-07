# Fokker-Planck Constrained Density Estimation for Single Cell Dynamics
## Mathematical statement of the problem
Given a sample $\hat x$ from a distribution $p(X)$, we assume that a subset of that sample $\hat x_0$ is the drawn from an initial state distribution $p_0(X)$. We assume that $p(X)$ is the mean density of a time varying distribution. 

$$p(x) = \frac{1}{T}\int_{0}^{T} p(x,t) dt$$ 

We assume that $X_t$ is determined by a stochastic differential equation:
$$dX_t = u(x)dt + \sigma dW$$ where 
* u(x) is a time-independent drift term
* $\sigma$ is a constant diffusion coefficient
* W is Wiener process (Gaussian noise)

The evolution of p(x,t) then follows a Fokker-Planck equation

$$\frac{\partial p(x,t)}{\partial t} = -\nabla \cdot (u(x)p(x,t)) + \Delta Dp(x,t)$$ 

where: 
* $\nabla$ is the divergence w.r.t. $x$ 
* $\Delta$ is the Laplacian
* $D$ is a constant diffusion term

We train a neural network $f_p(x,t)$ (`pxt` in the code) to learn the initial distribution $p(\hat x_0)$ and $p(\hat x) = \frac{1}{T} \sum_0^{T} p(\hat x, t)$ and  using noise contrastive estimation [1]. However, this is prone to learning a degenerate solution with $p(x,0)= p(x)$ and $p(x, t \neq 0) =0$, so we add a loss term to enforce a "consistency" condition: 

$$L_c = \left\|\sum_{\hat x}p(x, t_i) - \sum_{\hat x}p(x, t_j)\right\|^2 \forall t_i, t_j$$

While knowing the $p(x,t)$ is useful, our primary goal is to learn the vector field that drives the dynamics, i.e. the $u(x)$ above. To do this, we model the drift as another neural network $f_u$ (`ux` in the code) add the Fokker-Planck equation as a loss term:

$$L_{FP} = \left\|\frac{\partial p(x,t)}{\partial t} + \nabla \cdot (u(x)p(x,t))\right\|^2$$

assuming that the diffusion term is constant. 

**NOTE**: I'm just now realizing that excluding the diffusion might be a mistake. Not sure about this.

So, the overall loss is then: 

$$L_{p} + L_{p0} + L_{FP} + L_{c}$$

where :
* $L_p$ is the NCE loss of estimating $p(\hat x)$
* $L_{p0}$ is the NCE loss of estimating $p(\hat x_0)$
* $L_{FP}$ is the Fokker-Planck consistency loss
* $L_{cons}$ is the time density consistency loss

[1] https://proceedings.mlr.press/v9/gutmann10a.html

## Implementation

This architecture is implemented in PyTorch. I was stuck on version 1.9, due to an old CUDA driver on my lab's server. 

`celldelta.py` contains the implementation. 

`Ux` is a simple feedforward neural network that computes the vector field at each x. The only tricky part is computing the divergence of `Ux`, which requires D backwards passes to compute the partial derivative w.r.t. each $x_i$. 

`Pxt` is another feedforward neural network that models the time-varying log probability distribution $log\ p(x,t)$. We compute p(x,t) by concatenating x and t and passing that vector through the network. There is an additional scaling factor for t, as I found that the model struggled to learn time dependence when the dimension of x was much greater than 1 due to the decreasing influence of the single time dimension. Increasing the time scaling factor improved training. The other obvious option, just increasing the (arbitrary) time scale from 0-1 to 0-T, would change the scale of the Fokker-Planck time derivative.

`CellDelta` is the overall model architecture. The only important methods here are:
* `consistency_loss` - ensures approximately equal total density across our sample at all time points
* `fokker_planck_loss` - ensures vector field and probability density follows the Fokker-Planck equation 
* `optimize` - combines the loss functions and steps the optimizer
* `optimize_initial_conditions` - learns the "initial" state distribution

There are a lot of other "candidate" loss functions in `CellDelta`. Please ignore these. As I was thrashing around trying to figure out how to make this converge  I ended up leaving the remnants of many bad ideas here.

## Problems with the model

In `test_drift_learning.py` you can find a sequence of tests that gradually increases in complexity. First I define a custom non-learned `Ux` that is a very simple function of two variables. I did this to confirm that I was computing the divergence correctly, as this was an early issue with my code.

(Line 45) I then create a one-dimensional sequence of Gaussians with increasing means. This is the simplest synthetic test data that I could think of, and ***I think*** the model should work in this case. There are several diagnostic plots that help illustrate the model's behavior. I also included a simulation of points moving on the learned vector field. The goal is to get a histogram of mean locations of points that matches the "true" distribution as well as the learned $p(x,t)$. I also describe the "pseudotime" here, which is simply the time of maximum density for each x. Pseudotime is the term of art from computational biology.

(Line 398) ***Here is where things break***. This is testing the ability of the model to recover the probability distribution sequence and vector field for a Gaussian with linearly increasing mean. The code has dimension, `d=50`, but I would suggest reducing this to around `5` as that's when the model really starts to struggle. 

In higher dimensions, I think the model is still able to learn an approximately correct sequence of probability distributions. I judge this based on the pseudotime progression, i.e. the points generated by the Nth Gaussian have maximum density at time t=N. However, I'm not totally sure how to diagnose whether or not I'm learning a good distribution with NCE. 

***BUT the model always fails to learn a vector field that is consistent with the learned distributions***. I'm unsure why this happens and is the impetus for the comments on HN. I started to try to diagnose this problem in more depth, but then I had to complete my dissertation, so this got pushed back until now. Consequently, I haven't looked at this seriously in several months. I suspect the problem lies in trying to backpropagate through the derivative terms in the FP equation, which leads to a weird or badly conditioned loss landscape.

## Biological background
Single Cell RNA sequencing (scRNAseq) is a method for measuring the RNA content of individual cells. Briefly, RNA content gives us an idea of the state and function of particular cells, i.e. we expect a skin cell to have different RNA than a liver cell. This might be obvious but we can also tease apart much subtler distinctions such as which tumor cells might respond to a particular treatment. Further, it's known that cells can change their RNA state in response to their environment or during development. Often we know that cells begin in some "progenitor" or initial state and then stochastically transition to new states. We would like to create a machine learning model to help understand how different genes affect the paths cells take in "RNA-space" and potentially how we could perturb specific genes to cause the cells to adopt specific fates. Imagine reversing treatment resistance in tumor cells or inducing tissue repair rather than scarring or fibrosis.


However, scRNAseq data presents several challenges. First, it is destructive; we have to destroy cells to measure their RNA. We can't trace RNA levels in a single cell across time, we only get one snapshot of each cell. Second, it's high dimensional but quite sparse and noisy due to technical limitations of RNA capture and sequencing. Typical data has about 20k-30k genes and anywhere between 5k-100k cells. Time series data (multiple samples measured at different time points) is rare given the cost of scRNAseq and given the destructive measurement method, we can't map one cell at time $t_0$ to another cell at $t_1$. Instead we often assume that a single scRNAseq experiment contains a sample of cells that span the space of trajectories. That is, some cells are in the initial state, some cells are actively transitioning from initial to final state, and some cells are fully transitioned into another state. Typically we only know which cells are in the initial state.







