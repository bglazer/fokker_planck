# import torch.optim as optim

# # Define the loss function
# def loss_fn(x_grad):
#     return (x_grad - 10)**2

# # Define u(x) and pxt(x,t)
# def u(x):
#     return x**3


# # Define x and t
# x = torch.tensor([1.0], requires_grad=True)

# # Define the optimizer
# optimizer = optim.Adam([x], lr=0.01)

# # Perform optimization
# for i in range(300):
#     optimizer.zero_grad()
#     y = u(x)
#     x_grad = torch.autograd.grad(y, x, create_graph=True)[0]
#     loss = loss_fn(x_grad)
#     loss.backward()
#     optimizer.step()
#     print(float(x_grad), float(x), 3*float(x)**2)

#%%
import torch
from celldelta import CellDelta

X = torch.ones((1,1))*3
X.requires_grad=True
ts = torch.linspace(0, 1, 10, requires_grad=True)
noise = torch.distributions.MultivariateNormal(loc=torch.ones(1)*2, 
                             covariance_matrix=torch.eye(1)*6)

cd = CellDelta(input_dim=1, 
               ux_hidden_dim=10, ux_layers=2, ux_dropout=0,
               pxt_hidden_dim=10, pxt_layers=2, pxt_dropout=0,
               noise=noise, device='cpu',)
cd.ux_optimizer = torch.optim.Adam(cd.ux.parameters(), lr=0.001)
cd.pxt_optimizer = torch.optim.Adam(cd.ux.parameters(), lr=0.001)
hx = 1e-3

#%%
n_samples = 10
for epoch in range(10):
    rand_idxs = torch.randperm(len(x))
    x = X[rand_idxs].clone().detach()
    x.requires_grad=True
    # x = X[rand_idxs]
    # x0 = X
    cd.pxt_optimizer.zero_grad()
    cd.ux_optimizer.zero_grad()

    l_nce_p, acc_p = cd.nce_loss(x, ts=ts)
    l_nce_p.backward()

    l_nce_p0, acc_p = cd.nce_loss(x, ts=torch.zeros(1))
    l_nce_p0.backward()

    xts = cd.pxt.xts(x, ts)
    log_pxt = cd.pxt.model(xts)
    dq = torch.autograd.grad(outputs=log_pxt, 
                            inputs=xts, 
                            grad_outputs=torch.ones_like(log_pxt),
                            create_graph=True,
                            )[0]
    dqdx = dq[:,:,:-1]
    dqdt = dq[:,:,-1].unsqueeze(2)
    ux = cd.ux.model(x)
    dudx = torch.autograd.grad(outputs=ux, 
                            inputs=x, 
                            grad_outputs=torch.ones_like(ux),
                            create_graph=True,
                            )[0]
    dxi = ux*dqdx + dudx
    dx = dxi.sum(dim=2, keepdim=True)

    l_fp = ((dqdt + dx)**2).mean()
    print(epoch)
    l_fp.backward()
    print(float(l_fp))
    cd.pxt_optimizer.step()
    cd.ux_optimizer.step()
# %%
