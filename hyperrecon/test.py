"""
Test/inference functions for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 

"""
import torch
from . import utils, networks, dataset, metrics
from . import losses as losslayer
import numpy as np
import myutils
from glob import glob
import os
import json

def tester(path, device, take_avg, cp=None, n_grid=20, gt_data=None, xdata=None, test_data=True, normalized=True):
    """Driver code for test-time inference.

    Performs inference for a given model.
    Model parameters are parsed from the model path.
    """
    # Forward through latest available model
    if cp is None:
        glob_path = path.replace('[', '[()').replace(']', '()]').replace('()', '[]')
        model_paths = sorted(glob(os.path.join(glob_path, 'checkpoints/model.*.h5')))
        model_path = model_paths[-1]
    # Or forward through specified epoch
    else:
        model_path = os.path.join(path, 'checkpoints/model.{epoch:04d}.h5'.format(epoch=cp))
        
    if gt_data is None:
        if test_data:
            gt_data = dataset.get_test_gt('small')
            xdata = dataset.get_test_data('small')
        else:
            gt_data = dataset.get_train_gt('med')
            xdata = dataset.get_train_data('med')

    args_txtfile = os.path.join(path, 'args.txt')
    if os.path.exists(args_txtfile):
        with open(args_txtfile) as json_file:
            args = json.load(json_file)
    else:
        raise Exception('no args found')

    batch_size = len(xdata)

    n_hidden = args['unet_hdim']
    reg_types = args['reg_types']
    range_restrict = args['range_restrict']
    topK = args['topK']

    num_hyperparams = len(reg_types) if range_restrict else len(reg_types) + 1
    if len(reg_types) == 2:
        alphas = np.linspace(0, 1, n_grid)
        betas = np.linspace(0, 1, n_grid)
        hps = torch.tensor(np.stack(np.meshgrid(alphas, betas), -1).reshape(-1,2)).float().to(device)
        # hps = torch.tensor([[0.9, 0.1],
        #  [0.995, 0.6],
        #  [0.9, 0.2],
        #  [0.995, 0.5],
        #  [0.9, 0],
        #  [0.99, 0.7]]).float().to(device)
        # hps = utils.get_reference_hps(num_hyperparams, range_restrict).to(device)
        # print('ref', hps)
        # hps = hps.repeat(int(np.ceil(batch_size / len(hps))), 1)[:batch_size]
        if not range_restrict:
            hps = utils.oldloss2newloss(hps)
    elif len(reg_types) == 1:
        hps = torch.linspace(0, 1, n_grid).view(-1, 1).float().to(device)
        if not range_restrict:
            hps = utils.oldloss2newloss(hps)

    valset = dataset.Dataset(xdata, gt_data)
    params = {'batch_size': batch_size,
         'shuffle': False,
         'num_workers': 0,
         'pin_memory': True}
    dataloader = torch.utils.data.DataLoader(valset, **params)
    dataloader = {'x': xdata, 'gt': gt_data}

    mask = dataset.get_mask(4).to(device)

    network = networks.HyperUnet(num_hyperparams, not args['range_restrict'], args['unet_hdim'], hh=args['hnet_hdim'], use_tanh=args['use_tanh']).to(device) 

    network = utils.load_checkpoint(network, model_path)
    criterion = losslayer.AmortizedLoss(reg_types, range_restrict, args['sampling'], topK, device, mask, take_avg=False)

    gr = False
    gl = True
    return test(network, dataloader, device, hps, take_avg, criterion=criterion, give_recons=gr, give_loss=gl, give_metrics=True, normalized=normalized)

def test(trained_model, dataloader, device, hps, take_avg, criterion=None, \
        give_recons=False, give_loss=False, give_metrics=False, normalized=True):
    """Testing for a fixed set of hyperparameter setting.

    Returns recons, losses, and metrics (if specified)
    For every sample in the dataloader, evaluates with all hyperparameters in hps.
    Batch size must match size of dataset (TODO change this)

    If take_avg is True, then returns [len(hps)]
    """
    trained_model.eval()
    xdata = torch.tensor(dataloader['x'])
    gt_data = torch.tensor(dataloader['gt'])
    y, gt = xdata.float().to(device), gt_data.float().to(device)
    zf_abs = utils.ifft(y).norm(dim=-1, keepdim=True)
    print(zf_abs.shape, gt.shape)

    res = {}
    recons = []
    losses = []
    dcs = []
    cap_regs = []
    ws = []
    tvs = []
    all_psnrs = []
    all_rpsnrs = []
    all_zf_psnrs = []

    for h in hps:
        print(h)
        # for i, (y, gt) in enumerate(dataloader): 
        # y, gt = y.float().to(device), gt.float().to(device)
        batch_h = h.expand(len(y), -1)

        pred = trained_model(y, batch_h)
        cap_reg = trained_model.get_l1_weight_penalty()
        if give_recons:
            recons.append(pred.cpu().detach().numpy())
        if give_loss:
            loss, regs, _ = criterion(pred, y, batch_h, cap_reg)
            losses.append(loss.cpu().detach().numpy())
            dcs.append(regs['dc'].cpu().detach().numpy())
            cap_regs.append(regs['cap'].cpu().detach().numpy())
            tvs.append(regs['tv'].cpu().detach().numpy())
        if give_metrics:
            psnrs = metrics.get_metrics(gt, pred, zf_abs, 'psnr', normalized, reduction='none', )
            rpsnrs = metrics.get_metrics(gt, pred, zf_abs, 'relative psnr', normalized, reduction='none')
            zf_psnrs = metrics.get_metrics(gt, zf_abs, zf_abs, 'psnr', normalized, reduction='none')
            all_psnrs.append(psnrs)
            all_rpsnrs.append(rpsnrs)
            all_zf_psnrs.append(zf_psnrs)



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
        res['zf_psnr'] = np.array(all_zf_psnrs)
        if take_avg:
            res['psnr'] = res['psnr'].mean(axis=1)
            res['rpsnr'] = res['rpsnr'].mean(axis=1)
            res['zf_psnr'] = res['zf_psnr'].mean(axis=1)

    return res


def get_everything(path, device, take_avg=True, \
                   cp=None, n_grid=20, \
                   gt_data=None, xdata=None, test_data=True, convert=False, normalized=True):
    
    # Forward through latest available model
    if cp is None:
        glob_path = path.replace('[', '[()').replace(']', '()]').replace('()', '[]')
        model_paths = sorted(glob.glob(os.path.join(glob_path, 'checkpoints/model.*.h5')))
        model_path = model_paths[-1]
    # Or forward through specified epoch
    else:
        model_path = os.path.join(path, 'checkpoints/model.{epoch:04d}.h5'.format(epoch=cp))
        
    if gt_data is None:
        if test_data:
            gt_data = dataset.get_test_gt('small')
            xdata = dataset.get_test_data('small')
        else:
            gt_data = dataset.get_train_gt('med')
            xdata = dataset.get_train_data('med')

    args_txtfile = os.path.join(path, 'args.txt')
    if os.path.exists(args_txtfile):
        with open(args_txtfile) as json_file:
            args = json.load(json_file)
    else:
        raise Exception('no args found')

    result_dict = test.tester(model_path, xdata, gt_data, args, device, take_avg, n_grid, convert, normalized)
    return result_dict

def baseline_test(model_path, xdata, gt_data, device, take_avg, give_recons=True, give_loss=True, give_metrics=True):
    """Baselines test function"""
    network = model.BaseUnet().to(device) 
    network = utils.load_checkpoint(network, model_path)
    network.eval()

    mask = dataset.get_mask(4)
    mask = torch.tensor(mask, requires_grad=False).float().to(device)

    criterion = losslayer.AmortizedLoss(['cap', 'tv'], True, 'uhs', None, device, mask, take_avg=False)

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
            loss, regs, _ = criterion(pred, y, h, cap_reg, schedule=True)
            losses.append(loss.cpu().detach().numpy()[0])
            dcs.append(regs['dc'].cpu().detach().numpy()[0])
            # cap_regs.append(regs['cap'].cpu().detach().numpy()[0])
            tvs.append(regs['tv'].cpu().detach().numpy()[0])
        if give_metrics:
            psnrs = metrics.get_metrics(gt, pred, zf, metric_type='relative psnr', take_avg=take_avg)
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
