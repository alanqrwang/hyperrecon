"""
Test/inference functions for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 

"""
import torch
from . import utils, model, train, dataset
from . import loss as losslayer
import numpy as np
import myutils
import sys

def tester(model_path, xdata, gt_data, conf, device, hyparch, take_avg, n_grid=20):
    """Driver code for test-time inference.

    Performs inference for a given model.
    Model parameters are parsed from the model path.
    """
    batch_size = len(xdata)

    conf['sampling'] = 'uniform'

    n_hidden = int(conf['unet_hidden'])
    reg_types = conf['reg_types'].strip('][').split(', ')
    reg_types = [s.strip('\'') for s in reg_types]
    range_restrict = True if conf['range_restrict'] == 'True' else False
    bounds = [float(i) for i in conf['bounds'].strip('][').split(', ')]
    topK = None if conf['topK'] == 'None' else int(conf['topK'])

    alphas = np.linspace(bounds[0], bounds[1], n_grid)
    betas = np.linspace(bounds[2], bounds[3], n_grid)
    hps = np.stack(np.meshgrid(alphas, betas), -1).reshape(-1,2)

    valset = dataset.Dataset(xdata, gt_data)
    params = {'batch_size': batch_size,
         'shuffle': False,
         'num_workers': 0}
    dataloader = torch.utils.data.DataLoader(valset, **params)

    mask = utils.get_mask(conf['maskname'])
    mask = torch.tensor(mask, requires_grad=False).float().to(device)

    network = model.Unet(device, len(reg_types), hyparch, nh=n_hidden).to(device) 

    network = utils.load_checkpoint(network, model_path)
    criterion = losslayer.AmortizedLoss(reg_types, range_restrict, mask, conf['sampling'], evaluate=True)

    gr = False
    gl = True
    return test(network, dataloader, conf, hps, topK, take_avg, criterion=criterion, give_recons=gr, give_loss=gl, give_metrics=True)

def test(trained_model, dataloader, conf, hps, topK, take_avg, criterion=None, \
        give_recons=False, give_loss=False, give_metrics=False):
    """Testing for a fixed set of hyperparameter setting.

    Returns recons, losses, and metrics (if specified)
    """
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
        h = torch.tensor(h).to(conf['device']).float().reshape([-1, 2])
        for i, (y, gt) in enumerate(dataloader): 
            h = h.expand(len(y), -1)
            y = y.float().to(conf['device'])
            gt = gt.float().to(conf['device'])
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            pred, cap_reg = trained_model(zf, y, h)
            if give_recons:
                recons.append(pred.cpu().detach().numpy())
            if give_loss:
                assert criterion is not None, 'loss must be provided'
                loss, regs, _ = criterion(pred, y, h, cap_reg, topK, schedule=True)
                losses.append(loss.cpu().detach().numpy())
                dcs.append(regs['dc'].cpu().detach().numpy())
                cap_regs.append(regs['cap'].cpu().detach().numpy())
                tvs.append(regs['tv'].cpu().detach().numpy())
            if give_metrics:
                psnrs = get_metrics(y, gt, pred, metric_type='relative psnr', take_avg=take_avg)
                all_psnrs.append(psnrs)



    if give_recons:
        res['recons'] = np.array(recons)
    if give_loss:
        res['loss'] = np.array(losses)
        res['dc'] = np.array(dcs)
        res['cap'] = np.array(cap_regs)
        res['tv'] = np.array(tvs)
        if take_avg:
            res['loss'] = res['loss'].mean(axis=1)
            res['dc'] = res['dc'].mean(axis=1)
            res['cap'] = res['cap'].mean(axis=1)
            res['tv'] = res['tv'].mean(axis=1)

    if give_metrics:
        res['rpsnr'] = np.array(all_psnrs)
        if take_avg:
            res['rpsnr'] = res['rpsnr'].mean(axis=1)

    return res

def get_metrics(y, gt, recons, metric_type, take_avg, normalized=True):
    metrics = []
    zf = utils.ifft(y)
    if normalized:
        recons_pro = utils.normalize_recons(recons)
        gt_pro = utils.normalize_recons(gt)
        zf_pro = utils.normalize_recons(zf)
    else:
        recons_pro = myutils.array.make_imshowable(recons)
        gt_pro = myutils.array.make_imshowable(gt)
        zf_pro = myutils.array.make_imshowable(zf)
    for i in range(len(recons)):
        metric = myutils.metrics.get_metric(recons_pro[i], gt_pro[i], metric_type, zero_filled=zf_pro[i])
        metrics.append(metric)
    return np.array(metrics)
