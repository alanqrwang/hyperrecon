import torch
from . import utils, model, train, dataset
from . import loss as losslayer
import numpy as np
import myutils

def tester(model_path, xdata, gt_data, conf, device, n_hyp_layers, n_samples=10):
    batch_size = len(xdata)

    ### Change later!!!
    conf['sampling'] = 'bestdc'

    maskname = conf['maskname']
    n_hidden = int(conf['n_hidden'])
    lmbda = float(conf['lmbda'])
    K = int(conf['K'])
    recon_type = conf['recon_type']
    reg_types = conf['reg_types'].strip('][').split(', ')
    reg_types = [s.strip('\'') for s in reg_types]
    range_restrict = True if conf['range_restrict'] == 'True' else False
    alpha_bound = [float(i) for i in conf['alpha_bound'].strip('][').split(', ')]
    beta_bound = [float(i) for i in conf['beta_bound'].strip('][').split(', ')]
    topK = None if conf['topK'] == 'None' else int(conf['topK'])

    alphas = np.linspace(alpha_bound[0], alpha_bound[1], n_samples)
    betas = np.linspace(beta_bound[0], beta_bound[1], n_samples)
    hps = np.stack(np.meshgrid(alphas, betas), -1).reshape(-1,2)

    valset = dataset.Dataset(xdata, gt_data)
    params = {'batch_size': batch_size,
         'shuffle': False,
         'num_workers': 0}
    dataloader = torch.utils.data.DataLoader(valset, **params)

    mask = utils.get_mask(maskname)
    mask = torch.tensor(mask, requires_grad=False).float().to(device)

    if recon_type == 'unroll':
        network = model.HQSNet(K, mask, lmbda, len(reg_types), device, n_hidden).to(device) 
    else:
        network = model.Unet(device, len(reg_types), n_hyp_layers, alpha_bound, beta_bound, nh=n_hidden).to(device) 

    network = myutils.io.load_checkpoint(network, model_path)
    criterion = losslayer.AmortizedLoss(reg_types, range_restrict, mask, conf['sampling'])

    return test(network, dataloader, conf, hps, topK, criterion=criterion, give_loss=True, give_metrics=True)

def test(trained_model, dataloader, conf, hps, topK, criterion=None, \
        give_loss=False, give_metrics=False, take_avg=True):
    trained_model.eval()

    res = {}
    recons = []
    losses = []
    dcs = []
    cap_regs = []
    ws = []
    tvs = []
    all_psnrs = []

    for h in hps:
        print(conf['device'])
        h = torch.tensor(h).to(conf['device']).float().reshape([-1, 2])
        for i, (y, gt) in enumerate(dataloader): 
            h = h.expand(len(y), -1)
            y = y.float().to(conf['device'])
            gt = gt.float().to(conf['device'])
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            pred, cap_reg = trained_model(zf, y, h)
            recons.append(pred.cpu().detach().numpy())
            if give_loss:
                assert criterion is not None, 'loss must be provided'
                loss, regs, _ = criterion(pred, y, h, cap_reg, topK, schedule=True)
                losses.append(loss.cpu().detach().numpy())
                dcs.append(regs['dc'].cpu().detach().numpy())
                cap_regs.append(regs['cap_reg'].cpu().detach().numpy())
                ws.append(regs['w'].cpu().detach().numpy())
                tvs.append(regs['tv'].cpu().detach().numpy())
            if give_metrics:
                psnrs = get_metrics(y, gt, pred, metric_type='relative psnr', device=conf['device'])
                all_psnrs.append(psnrs)


    res['recons'] = np.array(recons)
    if give_loss:
        res['loss'] = np.array(losses).mean(axis=1)
        res['dc'] = np.array(dcs).mean(axis=1)
        res['cap'] = np.array(cap_regs).mean(axis=1)
        res['w'] = np.array(ws).mean(axis=1)
        res['tv'] = np.array(tvs).mean(axis=1)
    if give_metrics:
        res['rpsnr'] = np.array(all_psnrs)

    return res

def get_metrics(y, gt, recons, metric_type, device, normalized=True, take_avg=True):
    metrics = []
    zf = utils.ifft(y)
    if normalized:
        recons = utils.normalize_recons(recons)
        gt = utils.normalize_recons(gt)
        zf = utils.normalize_recons(zf)
    else:
        recons = myutils.array.make_imshowable(recons)
        gt = myutils.array.make_imshowable(gt)
        zf = myutils.array.make_imshowable(zf)
    for i in range(len(recons)):
        metric = myutils.metrics.get_metric(recons[i], gt[i], metric_type, zero_filled=zf[i])
        metrics.append(metric)
    if take_avg:
        metrics = np.mean(metrics)
    return np.array(metrics)
