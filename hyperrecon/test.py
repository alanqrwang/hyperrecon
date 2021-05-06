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
import pytorch_ssim
import glob
import os
import json

def get_everything(path, device, take_avg=True, \
                   metric_type='relative psnr', \
                   cp=None, n_grid=20, \
                   gt_data=None, xdata=None, test_data=True, normalized=True, maskname='4p2'):
    
    # Forward through latest available model
    if cp is None:
        glob_path = path.replace('[', '[()').replace(']', '()]').replace('()', '[]')
        model_paths = sorted(glob.glob(os.path.join(glob_path, 'checkpoints/model.*.h5')))
        model_path = model_paths[-1]
    # Or forward through specified epoch
    else:
        model_path = os.path.join(path, 'checkpoints/model.{epoch:04d}.h5'.format(epoch=cp))
        
    args_txtfile = os.path.join(path, 'args.txt')
    if os.path.exists(args_txtfile):
        with open(args_txtfile) as json_file:
            args = json.load(json_file)
    else:
        raise Exception('no args found')
    args['metric_type'] = metric_type
    args['normalized'] = normalized

    if not 'n_ch_out' in args:
        args['n_ch_out'] = 2

    if gt_data is None:
        if test_data:
            gt_data = dataset.get_test_gt('med')
            xdata = dataset.get_test_data('med')
            if args['n_ch_out'] == 2:
                gt_data = np.concatenate((gt_data, np.zeros(gt_data.shape)), axis=3)
                print('appended')
        else:
            gt_data = dataset.get_train_gt('med')
            xdata = dataset.get_train_data('med')

    result_dict = tester(model_path, xdata, gt_data, args, device, take_avg, n_grid, maskname)
    return result_dict

def tester(model_path, xdata, gt_data, conf, device, take_avg, n_grid=20, maskname='4p2', batch_size=8):
    """Driver code for test-time inference.

    Performs inference for a given model.
    Model parameters are parsed from the model path.
    """
    n_hidden = conf['unet_hdim']
    losses = conf['losses']
    range_restrict = conf['range_restrict']
    topK = conf['topK']
    hyperparameters = conf['hyperparameters']

    use_tanh = True if 'use_tanh' not in conf else conf['use_tanh']

    num_hyperparams = len(losses)-1 if range_restrict else len(losses)

    if hyperparameters is not None:
        hps = torch.tensor(np.repeat(hyperparameters, n_grid)).unsqueeze(1).float().to(device)
    elif len(losses) == 3:
        alphas = np.linspace(0, 1, n_grid)
        betas = np.linspace(0, 1, n_grid)
        hps = torch.tensor(np.stack(np.meshgrid(alphas, betas), -1).reshape(-1,2)).float().to(device)
        if not range_restrict:
            hps = utils.oldloss2newloss(hps)
    elif len(losses) == 2:
        hps = torch.linspace(0, 1, n_grid).view(-1, 1).float().to(device)
        if not range_restrict:
            hps = utils.oldloss2newloss(hps)

    valset = dataset.Dataset(xdata, gt_data)
    params = {'batch_size': batch_size,
         'shuffle': False,
         'num_workers': 0,
         'pin_memory': True}
    # dataloader = torch.utils.data.DataLoader(valset, **params)
    dataloader = {'x': xdata, 'gt': gt_data}

    mask = dataset.get_mask(maskname).to(device)

    if conf['hyperparameters'] is not None:
        network = model.Unet().to(device)
    else:
        network = model.HyperUnet(device, 
                         num_hyperparams, 
                         conf['hnet_hdim'], 
                         conf['unet_hdim'], \
                         hnet_norm=not conf['range_restrict'], \
                         n_ch_out=conf['n_ch_out'], \
                         use_tanh=use_tanh
                         ).to(device) 

    network = utils.load_checkpoint(network, model_path)
    
    criterion = losslayer.AmortizedLoss(losses + ['perc'], range_restrict, conf['sampling'], topK, device, mask, take_avg=False)

    gr = True
    gl = False
    return test(network, dataloader, device, hps, take_avg, conf['metric_type'], conf['normalized'], criterion=criterion, give_recons=gr, give_loss=gl, give_metrics=True)

def test(trained_model, dataloader, device, hps, take_avg, metric_type, normalized, criterion=None, \
        give_recons=False, give_loss=False, give_metrics=False):
    """Testing for a fixed set of hyperparameter setting.

    Returns recons, losses, and metrics (if specified)
    For every sample in the dataloader, evaluates with all hyperparameters in hps.
    Batch size must match size of dataset (TODO change this)

    If take_avg is True, then returns [len(hps)]
    """
    trained_model.eval()
    xdata = torch.tensor(dataloader['x'])
    gt_data = torch.tensor(dataloader['gt'])

    res = {}
    recons = []
    losses = []
    dcs = []
    cap_regs = []
    ws = []
    tvs = []
    all_psnrs = []
    all_rpsnrs = []
    all_mses = []
    all_ssims = []
    all_l1s = []
    all_percs = []

    for h in hps:
        print(h)
        # for i, (y, gt) in enumerate(dataloader): 
        #     y, gt = y.float().to(device), gt.float().to(device)
        y, gt = xdata.float().to(device), gt_data.float().to(device)
        zf = utils.ifft(y)
        y, zf = utils.scale(y, zf)
        batch_h = h.expand(len(y), -1)

        pred, cap_reg = trained_model(zf, batch_h)
        if give_recons:
            recons.append(pred.cpu().detach().numpy())
        if give_loss:
            assert criterion is not None, 'loss must be provided'
            loss, regs, _ = criterion(pred, y, batch_h, cap_reg, gt)
            losses.append(loss.cpu().detach().numpy())
            dcs.append(regs['dc'].cpu().detach().numpy())
            cap_regs.append(regs['cap'].cpu().detach().numpy())
            tvs.append(regs['tv'].cpu().detach().numpy())
        if give_metrics:
            psnrs = utils.get_metrics(gt, pred, zf, metric_type='psnr', normalized=normalized)
            rpsnrs = utils.get_metrics(gt, pred, zf, metric_type='relative psnr', normalized=normalized)
            mses = criterion.get_mse(gt, pred).detach().cpu().numpy()
            ssims = criterion.get_ssim(gt, pred).detach().cpu().numpy()
            l1s = criterion.get_l1(gt, pred).detach().cpu().numpy()
            percs = criterion.get_watson_dft(gt, pred).detach().cpu().numpy()
            all_psnrs.append(psnrs)
            all_rpsnrs.append(rpsnrs)
            all_mses.append(mses)
            all_ssims.append(ssims)
            all_l1s.append(l1s)
            all_percs.append(percs)



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
        res['psnr'] = np.array(all_psnrs)
        res['rpsnr'] = np.array(all_rpsnrs)
        res['mse'] = np.array(all_mses)
        res['ssim'] = np.array(all_ssims)
        res['l1'] = np.array(all_l1s)
        res['perc'] = np.array(all_percs)
        if take_avg:
            res['psnr'] = res['psnr'].mean(axis=1)
            res['rpsnr'] = res['rpsnr'].mean(axis=1)
            res['mse'] = res['mse'].mean(axis=1)
            res['ssim'] = res['ssim'].mean(axis=1)
            res['l1'] = res['l1'].mean(axis=1)
            res['perc'] = res['perc'].mean(axis=1)

    return res

def baseline_test(model_path, xdata, gt_data, device, take_avg, give_recons=True, give_loss=True, give_metrics=True):
    """Baselines test function"""
    network = model.BaseUnet(n_ch_out=2).to(device) 
    network = utils.load_checkpoint(network, model_path)
    for name, val in network.named_parameters():
        print(name, val.max(), val.min())
    network.eval()

    mask = dataset.get_mask(4)
    mask = torch.tensor(mask, requires_grad=False).float().to(device)

    criterion = losslayer.AmortizedLoss(['dc', 'cap', 'tv'], True, 'uhs', None, device, mask, take_avg=False)

    res = {}
    recons = []
    losses = []
    dcs = []
    cap_regs = []
    ws = []
    tvs = []
    all_psnrs = []

    h = torch.tensor([[0., 0.]]).to(device)
    cap_reg=torch.tensor(0.).to(device)
    for i in range(len(xdata)): 
        y = torch.tensor(xdata[i:i+1]).float().to(device)
        gt = torch.tensor(gt_data[i:i+1]).float().to(device)
        zf = utils.ifft(y)
        y, zf = utils.scale(y, zf)

        pred, _ = network(zf)
        if give_recons:
            recons.append(pred.cpu().detach().numpy()[0])
        if give_loss:
            assert criterion is not None, 'loss must be provided'
            loss, regs, _ = criterion(pred, y, h, cap_reg)
            losses.append(loss.cpu().detach().numpy()[0])
            # dcs.append(regs['dc'].cpu().detach().numpy()[0])
            # cap_regs.append(regs['cap'].cpu().detach().numpy()[0])
            # tvs.append(regs['tv'].cpu().detach().numpy()[0])
        if give_metrics:
            psnrs = utils.get_metrics(gt, pred, zf, metric_type='relative psnr', take_avg=take_avg, take_absval=True)
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
