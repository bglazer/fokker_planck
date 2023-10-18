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

x = torch.ones(1)*3
x.requires_grad=True
ts = torch.linspace(0, 1, 10, requires_grad=True)

cd = CellDelta(1, 1, 1, noise=None, ux_dropout=0, pxt_dropout=0, device='cpu')
cd.ux_optimizer = torch.optim.Adam(cd.ux.parameters(), lr=0.001)
hx = 1e-3

#%%
for i in range(100):
    xs = x.repeat((ts.shape[0],1,1))
    ts_ = ts.repeat((x.shape[0],1)).T.unsqueeze(2)
    # Concatentate them together to match the input the MLP model
    xts = torch.concatenate((xs,ts_), dim=2)
    up_pxt = cd.ux(x)*torch.exp(cd.pxt.model(xts))
    # xts.requires_grad=True
    up_dx = torch.autograd.grad(outputs=up_pxt, 
                                inputs=xts, 
                                grad_outputs=torch.ones_like(up_pxt),
                                create_graph=True)[0]
    up_dx = up_dx[:,:,:-1].sum(dim=2, keepdim=True)
    up_dx_num = (cd.ux(x+hx) * cd.pxt.pxt(x+hx, ts) - \
                 cd.ux(x-hx) * cd.pxt.pxt(x-hx, ts))/(2*hx)
    pxt = torch.exp(cd.pxt.model(xts))
    pxt_dts = torch.autograd.grad(outputs=pxt,
                                  inputs=xts,
                                  grad_outputs=torch.ones_like(pxt),
                                  create_graph=True)[0]
    pxt_dts = pxt_dts[:,:,-1]
    pxt_dts_num = cd.pxt.dt(x, ts)
    l_fp = ((pxt_dts + up_dx)**2).mean()
    l_fp.backward()

    mse_pxt_num = ((pxt_dts-pxt_dts_num.squeeze(2))**2).mean()
    mse_up_dx_num = ((up_dx.squeeze(2)-up_dx_num.squeeze(2))**2).mean()
    print(float(x), l_fp.item(), float(mse_pxt_num), float(mse_up_dx_num))
    cd.ux_optimizer.step()