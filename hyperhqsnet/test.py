import torch
from . import utils, model
from . import loss as losslayer
import numpy as np
import myutils

def test(model_path, xdata, alpha, config, device):
    dataset = config['dataset']
    maskname = config['maskname']
    n_hidden = int(config['n_hidden'])
    lmbda = float(config['lmbda'])
    K = int(config['K'])
    learn_reg_coeff = config['learn_reg_coeff']
    if learn_reg_coeff == 'True':
        learn_reg_coeff = 1
    recon_type = config['recon_type']

    mask = utils.get_mask(maskname)
    mask = torch.tensor(mask, requires_grad=False).float().to(device)

    if recon_type == 'unroll':
        network = model.HQSNet(K, mask, lmbda, learn_reg_coeff, device, n_hidden).to(device) 
    else:
        network = model.Unet(device, learn_reg_coeff, nh=n_hidden).to(device) 

    network = myutils.io.load_checkpoint(network, model_path)

    alpha = torch.tensor(alpha).to(device).float()
    return test_hqsnet(network, xdata, device, alpha, mask)

def test_hqsnet(trained_model, xdata, device, hyperparams, mask):
    recons = []
    losses = []
    for i in range(len(xdata)):
        y = torch.as_tensor(xdata[i:i+1]).to(device).float()
        zf = utils.ifft(y)
        if True:
            y, zf = utils.scale(y, zf)

        print(hyperparams)
        pred, cap_reg = trained_model(zf, y, hyperparams)
        loss, _ = losslayer.unsup_loss(pred, y, mask, hyperparams, device, cap_reg)
        recons.append(pred.cpu().detach().numpy())
        losses.append(cap_reg.item())

    preds = np.array(recons).squeeze()
    return preds, losses
