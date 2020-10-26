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
    recon_type = config['recon_type']
    reg_types = config['reg_types'].strip('][').split(', ')
    reg_types = [s.strip('\'') for s in reg_types]
    # range_restrict = True if config['range_restrict'] == 'True' else False
    range_restrict = True

    mask = utils.get_mask(maskname)
    mask = torch.tensor(mask, requires_grad=False).float().to(device)

    if recon_type == 'unroll':
        network = model.HQSNet(K, mask, lmbda, len(reg_types), device, n_hidden).to(device) 
    else:
        network = model.Unet(device, len(reg_types), nh=n_hidden).to(device) 

    network = myutils.io.load_checkpoint(network, model_path)

    alpha = torch.tensor(alpha).to(device).float()
    return test_hqsnet(network, xdata, device, alpha, mask, reg_types, range_restrict)

def test_hqsnet(trained_model, xdata, device, hyperparams, mask, reg_types, range_restrict):
    recons = []
    losses = []
    for i in range(len(xdata)):
        y = torch.as_tensor(xdata[i:i+1]).to(device).float()
        zf = utils.ifft(y)
        y, zf = utils.scale(y, zf)

        pred, cap_reg = trained_model(zf, y, hyperparams)
        loss, _ = losslayer.unsup_loss(pred, y, mask, hyperparams, device, reg_types, cap_reg, range_restrict)
        recons.append(pred.cpu().detach().numpy())
        losses.append(cap_reg.item())

    preds = np.array(recons).squeeze()
    return preds, losses
