import torch
import torch.nn as nn
from hqsnet import utils
from hqsnet import loss as losslayer
from myutils.array import make_imshowable as mims
import matplotlib.pyplot as plt

class ClassicalOptimization(nn.Module):
    def __init__(self, image_dims, y_zf, device):
        super(ClassicalOptimization, self).__init__()
        self.image_dims = image_dims
        self.device = device

        self.x = nn.Parameter(torch.FloatTensor(1, self.image_dims[0], self.image_dims[1], 2))
        self.x.requires_grad = True
        self.x.data = y_zf.clone() #+ torch.normal(0, 1, size=self.x.size()).to(device)

    def forward(self):
        return self.x

def gradient_descent(x_prev, until_convergence, max_iters, tol, w_coeff, tv_coeff, lmbda, lr, device, mask):
    loss_list = []
    prev_loss = torch.tensor(0.).float().to(device)
    classical_loss = torch.tensor(1e10).float().to(device)
    iters = 0

    model = ClassicalOptimization(mask.shape, x_prev, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    while True:
        if until_convergence:
            if torch.abs(prev_loss - classical_loss) < tol or iters > max_iters:
                break
        else:
            if iters > max_iters:
                break
        prev_loss = classical_loss
        optimizer.zero_grad()
        z = model().float().to(device)

        classical_loss, dc_loss, reg_loss = losslayer.loss_with_reg(z, x_prev, w_coeff, tv_coeff, lmbda, device)
        loss_list.append(classical_loss.item())
        classical_loss.backward()
        optimizer.step()
        iters += 1
        # print(iters)
        
    print('gd iters:', iters)
    return z.detach(), loss_list     

def calc_dc(y, x):
    l2 = torch.nn.MSELoss(reduction='sum')
    return l2(y, x)

def hqsplitting(xdata, mask, w_coeff, tv_coeff, lmbda, device, until_convergence, K, lr, max_iters=1000, tol=1e-8):
    y = xdata.clone()
    y_zf = utils.ifft(y)
    y, y_zf = utils.scale(y, y_zf)
    x = y_zf 
    
    final_losses = []
    final_dcs = []
    final_regs = []
    metrics = []
    for iteration in range(K):
        print('iteration:', iteration, lmbda, w_coeff, tv_coeff, lr)
        #  z-minimization
        z, loss_list = gradient_descent(x, until_convergence, max_iters, tol, w_coeff, tv_coeff, lmbda, lr, device, mask)
        # plt.imshow(mims(z.cpu().detach().numpy()), cmap='gray')
        # plt.title('z')
        # plt.show()

        # plt.plot(loss_list)
        # plt.title('z minimization loss')
        # plt.show()
            
        # x-minimization
        z_ksp = utils.fft(z)
        x_ksp = losslayer.data_consistency(z_ksp, y, mask, lmbda=lmbda)
        x = utils.ifft(x_ksp)
        # plt.imshow(mims(x.cpu().detach().numpy()), cmap='gray')
        # plt.title('x')
        # plt.show()
        
        # print('x', x)
        # print('y', y)
        final_l, final_dc, final_reg = losslayer.final_loss(x, y, mask, w_coeff, tv_coeff, device)
        final_losses.append(final_l.item())
        final_dcs.append(final_dc.item())
        final_regs.append(final_reg.item())
        # plt.plot(final_losses)
        # plt.show()
    return x, final_losses, final_dcs, final_regs
