import torch
from . import utils, model, train
from . import loss as losslayer
import numpy as np
import myutils

def tester(model_path, xdata, alpha, config, device, n_hyp_layers, batch_size=8):
    maskname = config['maskname']
    n_hidden = int(config['n_hidden'])
    lmbda = float(config['lmbda'])
    K = int(config['K'])
    recon_type = config['recon_type']
    reg_types = config['reg_types'].strip('][').split(', ')
    reg_types = [s.strip('\'') for s in reg_types]
    range_restrict = True if config['range_restrict'] == 'True' else False
    alpha_bound = [float(i) for i in config['alpha_bound'].strip('][').split(', ')]
    beta_bound = [float(i) for i in config['beta_bound'].strip('][').split(', ')]

    # valset = dataset.Dataset(xdata, gt_data)
    # params = {'batch_size': batch_size,
    #      'shuffle': False,
    #      'num_workers': 4}
    # dataloader = torch.utils.data.DataLoader(valset, **params)

    mask = utils.get_mask(maskname)
    mask = torch.tensor(mask, requires_grad=False).float().to(device)

    if recon_type == 'unroll':
        network = model.HQSNet(K, mask, lmbda, len(reg_types), device, n_hidden).to(device) 
    else:
        network = model.Unet(device, len(reg_types), n_hyp_layers, alpha_bound, beta_bound, nh=n_hidden).to(device) 

    network = myutils.io.load_checkpoint(network, model_path)

    alpha = torch.tensor(alpha).to(device).float()
    # return train.test(network, dataloader, mask, device, alpha_bound, beta_bound, topK, reg_types, range_restrict, alpha, evaluate=True)
    return test_hqsnet(network, xdata, device, alpha, mask, reg_types, range_restrict)

def test_hqsnet(trained_model, xdata, device, hyperparams, mask, reg_types, range_restrict):
    trained_model.eval()
    recons = []
    losses = []
    dcs = []
    cap_regs = []
    ws = []
    tvs = []
    for i in range(len(xdata)):
        y = torch.as_tensor(xdata[i:i+1]).to(device).float()
        zf = utils.ifft(y)
        y, zf = utils.scale(y, zf)

        pred, cap_reg = trained_model(zf, y, hyperparams)
        loss, dc, w, tv = losslayer.unsup_loss(pred, y, mask, hyperparams, device, reg_types, cap_reg, range_restrict)
        recons.append(pred.cpu().detach().numpy())
        losses.append(loss.item())
        dcs.append(dc.item())
        cap_regs.append(cap_reg.item())
        ws.append(w.item())
        tvs.append(tv.item())

    preds = np.array(recons)[:,0,...]
    return preds, losses, dcs, cap_regs, ws, tvs
