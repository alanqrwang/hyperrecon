import torch
from . import utils, model, train, dataset
from . import loss as losslayer
import numpy as np
import myutils
import sys

def tester(model_path, xdata, gt_data, conf, device, n_hyp_layers, take_avg, n_grid=20):
    batch_size = len(xdata)

    ### Change later!!!
    conf['sampling'] = 'uniform'

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

    # if take_avg:
    if True:
        alphas = np.linspace(alpha_bound[0], alpha_bound[1], n_grid)
        betas = np.linspace(beta_bound[0], beta_bound[1], n_grid)
    else:
        alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
        betas =  [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
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
    criterion = losslayer.AmortizedLoss(reg_types, range_restrict, mask, conf['sampling'], evaluate=True)

    # gr = False if take_avg else True
    gr = True
    # gl = False if take_avg else True
    gl = True
    return test(network, dataloader, conf, hps, topK, take_avg, criterion=criterion, give_recons=gr, give_loss=gl, give_metrics=True)

def test(trained_model, dataloader, conf, hps, topK, take_avg, criterion=None, \
        give_recons=False, give_loss=False, give_metrics=False):
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
                # print(loss.shape)
                losses.append(loss.cpu().detach().numpy())
                dcs.append(regs['dc'].cpu().detach().numpy())
                cap_regs.append(regs['cap'].cpu().detach().numpy())
                # ws.append(regs['w'].cpu().detach().numpy())
                tvs.append(regs['tv'].cpu().detach().numpy())
            if give_metrics:
                psnrs = get_metrics(y, gt, pred, metric_type='relative psnr', take_avg=take_avg)
                all_psnrs.append(psnrs)



    if give_recons:
        res['recons'] = np.array(recons)
    if give_loss:
        res['loss'] = np.array(losses).mean(axis=1)
        # res['dc'] = np.array(dcs).mean(axis=1)
        res['dc'] = np.array(dcs)
        res['cap'] = np.array(cap_regs).mean(axis=1)
        # res['w'] = np.array(ws).mean(axis=1)
        res['tv'] = np.array(tvs).mean(axis=1)
    if give_metrics:
        # res['rpsnr'] = np.array(all_psnrs).mean(axis=1)
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
    # if take_avg:
    #     metrics = np.mean(metrics)
    return np.array(metrics)
