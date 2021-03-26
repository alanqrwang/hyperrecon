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

def tester(model_path, xdata, gt_data, conf, device, take_avg, n_grid=20, convert=False):
    """Driver code for test-time inference.

    Performs inference for a given model.
    Model parameters are parsed from the model path.
    """
    batch_size = len(xdata)

    n_hidden = conf['unet_hdim']
    if isinstance(conf['reg_types'], str):
        reg_types = conf['reg_types'].strip('][').split(', ')
        reg_types = [s.strip('\'') for s in reg_types]
    else:
        reg_types = conf['reg_types']
    if isinstance(conf['range_restrict'], str):
        range_restrict = True if conf['range_restrict'] == 'True' else False
    else:
        range_restrict = conf['range_restrict']
    if isinstance(conf['topK'], str):
        topK = None if conf['topK'] == 'None' else int(conf['topK'])
    else:
        topK = conf['topK']

    alphas = np.linspace(0, 1, n_grid)
    betas = np.linspace(0, 1, n_grid)
    hps = torch.tensor(np.stack(np.meshgrid(alphas, betas), -1).reshape(-1,2)).float().to(device)
    if convert:
        hps = utils.oldloss2newloss(hps)

    valset = dataset.Dataset(xdata, gt_data)
    params = {'batch_size': batch_size,
         'shuffle': False,
         'num_workers': 0}
    dataloader = torch.utils.data.DataLoader(valset, **params)

    mask = dataset.get_mask(4).to(device)

    num_hyperparams = len(reg_types) if conf['range_restrict'] else len(reg_types) + 1
    assert num_hyperparams == hps.shape[1], 'num_hyperparams %d, hps shape %d' % (num_hyperparams, hps.shape[1])
    network = model.Unet(device, num_hyperparams, conf['hnet_hdim'], conf['unet_hdim']).to(device) 

    network = utils.load_checkpoint(network, model_path)
    criterion = losslayer.AmortizedLoss(reg_types, range_restrict, conf['sampling'], topK, device, mask, take_avg=False)

    gr = False
    gl = True
    return test(network, dataloader, device, hps, take_avg, criterion=criterion, give_recons=gr, give_loss=gl, give_metrics=True)

def test(trained_model, dataloader, device, hps, take_avg, criterion=None, \
        give_recons=False, give_loss=False, give_metrics=False):
    """Testing for a fixed set of hyperparameter setting.

    Returns recons, losses, and metrics (if specified)
    For every sample in the dataloader, evaluates with all hyperparameters in hps.
    Batch size must match size of dataset (TODO change this)

    If take_avg is True, then returns [len(hps)]
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
        # print(h)
        for i, (y, gt) in enumerate(dataloader): 
            batch_h = h.expand(len(y), -1)
            y, gt = y.float().to(device), gt.float().to(device)
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            pred = trained_model(zf, batch_h)
            cap_reg = trained_model.get_l1_weight_penalty(len(y))
            if give_recons:
                recons.append(pred.cpu().detach().numpy())
            if give_loss:
                assert criterion is not None, 'loss must be provided'
                loss, regs, _ = criterion(pred, y, batch_h, cap_reg, hps.shape[1]-1)
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
            print(res['loss'].shape)
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
        metric = myutils.metrics.get_metric(recons_pro[i].cpu().detach().numpy(), gt_pro[i].cpu().detach().numpy(), metric_type, zero_filled=zf_pro[i].cpu().detach().numpy())
        metrics.append(metric)
    return np.array(metrics)


def baseline_test(model_path, xdata, gt_data, conf, device, take_avg, give_recons=True, give_loss=True, give_metrics=True):
    """Baselines test function"""
    network = model.BaseUnet().to(device) 
    network = utils.load_checkpoint(network, model_path)
    network.eval()

    mask = dataset.get_mask(4)
    mask = torch.tensor(mask, requires_grad=False).float().to(device)

    criterion = losslayer.AmortizedLoss(['cap', 'tv'], True, 'uhs', device, mask, evaluate=True)

    res = {}
    recons = []
    losses = []
    dcs = []
    cap_regs = []
    ws = []
    tvs = []
    all_psnrs = []

    h = torch.tensor([[0.,0.]]).to(device)
    cap_reg=torch.tensor(0.).to(device)
    for i in range(len(xdata)): 
        y = torch.tensor(xdata[i:i+1]).float().to(device)
        gt = torch.tensor(gt_data[i:i+1]).float().to(device)
        zf = utils.ifft(y)
        y, zf = utils.scale(y, zf)

        pred = network(zf)
        if give_recons:
            recons.append(pred.cpu().detach().numpy()[0])
        if give_loss:
            assert criterion is not None, 'loss must be provided'
            loss, regs, _ = criterion(pred, y, h, cap_reg, None, schedule=True)
            losses.append(loss.cpu().detach().numpy()[0])
            dcs.append(regs['dc'].cpu().detach().numpy()[0])
            # cap_regs.append(regs['cap'].cpu().detach().numpy()[0])
            tvs.append(regs['tv'].cpu().detach().numpy()[0])
        if give_metrics:
            psnrs = get_metrics(y, gt, pred, metric_type='relative ssim', take_avg=take_avg)
            all_psnrs.append(psnrs[0])



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
