"""
Test/inference functions for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 

"""
import torch
import torchio as tio
from . import utils, model, train, dataset
from . import loss as losslayer
import numpy as np
import myutils
import sys
import pytorch_ssim
import glob
import os
import json
import matplotlib.pyplot as plt

def get_everything(path, device, \
                   cp=None, n_grid=20, \
                   gt_data=None, xdata=None, seg_data=None, normalized=True, rescale_in=False):
    
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
    args['normalized'] = normalized
    preprocess_dataset = True if 'preprocess_dataset' not in args else args['preprocess_dataset']

    if preprocess_dataset:
        testset = dataset.Dataset(xdata, gt_data, seg_data)
        params = {'batch_size': 8,
             'shuffle': False,
             'num_workers': 0,
             'pin_memory': True}
        out_shape = xdata.shape

    else:
        total_subjects = 1
        testset = dataset.VolumeDataset(args['data_path'], 
                'test', 
                total_subjects=total_subjects, 
                include_seg=True).get_tio_dataset()
        params = {'batch_size': 1,
             'shuffle': False,
             'num_workers': 0, 
             'pin_memory': True}
        out_shape = [total_subjects*24, 160, 224, 1]

    dataloader = torch.utils.data.DataLoader(testset, **params)

    losses = args['losses']
    range_restrict = args['range_restrict']
    topK = args['topK']
    hyperparameters = args['hyperparameters']
    maskname = args['undersampling_rate']

    use_tanh = True if 'use_tanh' not in args else args['use_tanh']
    img_dims = '256_256' if 'img_dims' not in args else args['img_dims']
    # args['rescale_in'] = False if 'rescale_in' not in args else args['rescale_in']
    args['rescale_in'] = rescale_in

    num_hyperparams = len(losses)-1 if range_restrict else len(losses)

    if hyperparameters is not None:
        hps = torch.tensor([hyperparameters]).unsqueeze(1).float().to(device)
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


    args['mask'] = dataset.get_mask(img_dims, maskname).to(device)
    args['device'] = device
    n_ch_in = 1 if args['rescale_in'] else 2

    print(n_ch_in)
    if args['hyperparameters'] is not None:
        network = model.Unet().to(device)
    else:
        network = model.HyperUnet(device, 
                         num_hyperparams, 
                         args['hnet_hdim'], 
                         args['unet_hdim'], \
                         hnet_norm=not args['range_restrict'], \
                         n_ch_in=n_ch_in,
                         n_ch_out=args['n_ch_out'], \
                         use_tanh=use_tanh
                         ).to(device) 

    network = utils.load_checkpoint(network, model_path)
    
    criterion = losslayer.AmortizedLoss(losses, range_restrict, args['sampling'], topK, device, args['mask'], take_avg=False)

    gr = False
    gl = True
    return test(network, dataloader, args, hps, args['normalized'], out_shape, criterion=criterion, give_recons=gr, give_losses=gl)

def prepare_batch(batch, args):
    if isinstance(batch, list):
        inputs = batch[0].float().to(args['device'])
        targets = batch[1].float().to(args['device'])
        segs = batch[2].float().to(args['device'])
    else: # Volume was loaded, so randomly take batch_size slices
        volume = batch['mri'][tio.DATA]
        seg_volume = batch['seg'][tio.DATA]

        num_slices = volume.shape[2]
        rand_ind = np.arange(num_slices, step=8)
        targets = volume[0, :, rand_ind].to(args['device']).permute(1, 2, 3, 0)
        targets = utils.rescale(targets)
        segs = seg_volume[0, :, rand_ind].to(args['device']).permute(1, 2, 3, 0)
        under_ksp = utils.undersample(targets, args['mask'])

        zf = utils.ifft(under_ksp).norm(2, dim=-1, keepdim=True)
        # y, zf = utils.scale(y, zf)
        zf = utils.rescale(zf)

    return zf, targets, segs

def test(trained_model, dataloader, args, hps, normalized, out_shape, criterion=None, \
        give_recons=False, give_losses=False):
    """Testing for a fixed set of hyperparameter setting.

    Returns recons, losses, and metrics (if specified)
    For every sample in the dataloader, evaluates with all hyperparameters in hps.
    Batch size must match size of dataset (TODO change this)

    If take_avg is True, then returns [len(hps)]
    """
    trained_model.eval()

    total_subjects = len(dataloader.dataset)

    res = {}
    if give_recons:
        res['recons'] = np.full((len(hps), *out_shape), np.nan)
        res['gts'] = np.full((len(hps), *out_shape), np.nan)
        # res['gt_segs'] = np.full((len(hps), total_subjects, *vol_shape), np.nan)
        # res['recon_segs'] = np.full((len(hps), total_subjects, *vol_shape), np.nan)
    if give_losses:
        res['losses'] = np.full((len(hps), out_shape[0]), np.nan)
        # res['dcs'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
        # res['cap_regs'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
        # res['ws'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
        # res['tvs'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
        res['psnrs'] = np.full((len(hps), out_shape[0]), np.nan)
        # all_rpsnrs = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
        res['mses'] = np.full((len(hps), out_shape[0]), np.nan)
        res['ssims'] = np.full((len(hps), out_shape[0]), np.nan)
        res['l1s'] = np.full((len(hps), out_shape[0]), np.nan)
        # res['percs'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
        # res['dices'] = np.full((len(hps), out_shape[0]), np.nan)

    for h, hp in enumerate(hps):
        print(hp)
        for i, batch in enumerate(dataloader): 
            zf, gt, seg = utils.prepare_batch(batch, args, split='test')
            y = None
            batch_size = len(zf)

            # zf = utils.ifft(y)
            # y, zf = utils.scale(y, zf)
            batch_h = hp.expand(batch_size, -1)
            with torch.set_grad_enabled(False):
                preds, cap_reg = trained_model(zf, batch_h)
                loss, regs, _ = criterion(preds, y, batch_h, cap_reg, gt)

            # fig, axes = plt.subplots(1, 3, figsize=(10, 6))
            # axes[0].imshow(zf[0, ..., 0].cpu().detach().numpy())
            # axes[1].imshow(gt[0, ..., 0].cpu().detach().numpy())
            # axes[2].imshow(preds[0, ..., 0].cpu().detach().numpy())
            # plt.show()
            if give_losses:
                assert criterion is not None, 'loss must be provided'
                res['losses'][h, i*batch_size:i*batch_size+len(preds)] = loss.cpu().detach().numpy()
                # dcs.append(regs['dc'].cpu().detach().numpy())
                # cap_regs.append(regs['cap'].cpu().detach().numpy())
                # tvs.append(regs['tv'].cpu().detach().numpy())
                res['psnrs'][h, i*batch_size:i*batch_size+len(preds)] = utils.get_metrics(gt, preds, zf, metric_type='psnr', normalized=normalized)
                # rpsnrs = utils.get_metrics(gt, preds, zf, metric_type='relative psnr', normalized=normalized)
                res['mses'][h, i*batch_size:i*batch_size+len(preds)] = criterion.get_mse(gt, preds).detach().cpu().numpy()
                res['ssims'][h, i*batch_size:i*batch_size+len(preds)] = criterion.get_ssim(gt, preds).detach().cpu().numpy()
                res['l1s'][h, i*batch_size:i*batch_size+len(preds)] = criterion.get_l1(gt, preds).detach().cpu().numpy()
                # percs = criterion.get_watson_dft(gt, preds).detach().cpu().numpy()
                # dices, pred_segs = criterion.get_dice(preds, gt, seg)#.detach().cpu().numpy()
                # res['dices'][h, i*batch_size:i*batch_size+len(preds)] = dices

            if give_recons:
                res['recons'][h, i*batch_size:i*batch_size+len(preds)] = preds.cpu().detach().numpy()
                res['gts'][h, i*batch_size:i*batch_size+len(preds)] = gt.cpu().detach().numpy()
                # res['recon_segs'][h, i*batch_size:i*batch_size+len(preds)] = recon_segs.cpu().detach().numpy()
                # res['gt_segs'][h, i*batch_size:i*batch_size+len(preds)] = seg.cpu().detach().numpy()


    if give_recons:
        assert np.isnan(np.sum(res['recons'])) == False, 'Missed some predictions'
        assert np.isnan(np.sum(res['gts'])) == False, 'Missed some gts'
    if give_losses:
        assert np.isnan(np.sum(res['losses'])) == False, 'Missed some predictions'
        assert np.isnan(np.sum(res['psnrs'])) == False, 'Missed some gts'
        assert np.isnan(np.sum(res['mses'])) == False, 'Missed some gts'
        assert np.isnan(np.sum(res['ssims'])) == False, 'Missed some gts'
        assert np.isnan(np.sum(res['l1s'])) == False, 'Missed some gts'
        # assert np.isnan(np.sum(res['dices'])) == False, 'Missed some gts'

    return res
