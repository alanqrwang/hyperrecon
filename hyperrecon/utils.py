"""
Utility functions for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
import torchio as tio
import numpy as np
import pickle
# import parse
import glob
from . import test, dataset
import myutils
import os

def add_bool_arg(parser, name, default=True):
    """Add boolean argument to argparse parser"""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})

def fft(x):
    """Normalized 2D Fast Fourier Transform"""
    # complex_x = torch.view_as_complex(x)
    # fft = torch.fft.fft2(complex_x,  norm='ortho')
    # return torch.view_as_real(fft) 
    if x.shape[-1] == 1:
        x = torch.cat((x, torch.zeros_like(x)), dim=3)
    return torch.fft(x, signal_ndim=2, normalized=True)

def ifft(x):
    """Normalized 2D Inverse Fast Fourier Transform"""
    # complex_x = torch.view_as_complex(x)
    # ifft = torch.fft.ifft2(complex_x, norm='ortho')
    # return torch.view_as_real(ifft) 
    return torch.ifft(x, signal_ndim=2, normalized=True)

def undersample(fullysampled, mask):
    '''Generate undersampled k-space data with given binary mask'''
    if fullysampled.shape[-1] == 1:
        fullysampled = torch.cat((fullysampled, torch.zeros_like(fullysampled)), dim=3)

    mask_expand = mask.unsqueeze(2)

    ksp = fft(fullysampled)
    under_ksp = ksp * mask_expand
    return under_ksp

def absval(arr):
    """
    Takes absolute value of last dimension, if complex.
    Input dims:  (N, l, w, 2)
    Output dims: (N, l, w)
    """
    assert arr.shape[-1] == 2 or arr.shape[-1] == 1
    if torch.is_tensor(arr):
        arr = arr.norm(dim=-1)
    else:
        arr = np.linalg.norm(arr, axis=-1)

    return arr

def scale(y, zf):
    """Scales inputs for numerical stability"""
    flat_yzf = torch.flatten(absval(zf), start_dim=1, end_dim=2)
    max_val_per_batch, _ = torch.max(flat_yzf, dim=1, keepdim=True) 

    # Handling the edge case of image of all 0's
    max_val_per_batch[max_val_per_batch==0] = 1
    
    y = y / max_val_per_batch.view(len(y), 1, 1, 1)
    zf = zf / max_val_per_batch.view(len(y), 1, 1, 1)
    return y, zf

def prepare_batch(batch, args, split='train'):
    if isinstance(batch, tuple) == 4:
        inputs = batch[0].float().to(args['device'])
        targets = batch[1].float().to(args['device'])
    else: # Volume was loaded, so randomly take batch_size slices
        volume = batch['mri'][tio.DATA]
        num_slices = volume.shape[2]
        if split == 'train':
            rand_ind = np.random.choice(num_slices, args['batch_size'], replace=False)
        else:
            rand_ind = np.arange(num_slices, step=8)

        targets = volume[0, :, rand_ind].to(args['device']).permute(1, 2, 3, 0)
        targets = rescale(targets)
        under_ksp = undersample(targets, args['mask'])

        zf = ifft(under_ksp)

        if args['rescale_in']:
            zf = zf.norm(2, dim=-1, keepdim=True)
            zf = rescale(zf)
        else:
            under_ksp, zf = scale(under_ksp, zf)

    return zf, targets, under_ksp

def nextPowerOf2(n):
    """Get next power of 2"""
    count = 0;

    if (n and not(n & (n - 1))):
        return n

    while( n != 0):
        n >>= 1
        count += 1

    return 1 << count;

def rescale(arr):
    """Rescales a batch of images into range [0, 1]

    arr: (batch_size, l, w, 2)
    """
    flat_arr = torch.flatten(arr, start_dim=1, end_dim=2)
    max_per_batch, _ = torch.max(flat_arr, dim=1, keepdim=True) 
    min_per_batch, _ = torch.min(flat_arr, dim=1, keepdim=True) 

    # Handling the edge case of image of all 0's
    max_per_batch[max_per_batch==0] = 1 

    max_per_batch = max_per_batch.view(len(arr), 1, 1, 1)
    min_per_batch = min_per_batch.view(len(arr), 1, 1, 1)

    return (arr - min_per_batch) / (max_per_batch - min_per_batch)

# def normalize_recons(recons):
#     recons = absval(recons)
#     recons = normalize(recons)
#     return recons

def count_parameters(model):
    """Count total parameters in model"""
    for name, val in model.named_parameters():
        print(name)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

######### Saving/Loading checkpoints ############
def load_checkpoint(model, path, optimizer=None, scheduler=None):
    print('Loading checkpoint from', path)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and scheduler is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        return model, optimizer, scheduler

    elif optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer

    else:
        return model

def save_checkpoint(epoch, model_state, optimizer_state, model_folder, scheduler=None):
    state = {
        'epoch': epoch,
        'state_dict': model_state,
        'optimizer' : optimizer_state
    }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()

    filename = os.path.join(model_folder, 'model.{epoch:04d}.h5')
    torch.save(state, filename.format(epoch=epoch))
    print('Saved checkpoint to', filename.format(epoch=epoch))

def save_loss(save_dir, logger, *metrics):
    """
    metrics (list of strings): e.g. ['loss_d', 'loss_g', 'rmse_test', 'mae_train']
    """
    for metric in metrics:
        metric_arr = logger[metric]
        np.savetxt(save_dir + '/{}.txt'.format(metric), metric_arr)

def get_args(path):
    args_txtfile = os.path.join(path, 'args.txt')
    if os.path.exists(args_txtfile):
        with open(args_txtfile) as json_file:
            config = json.load(json_file)
    return config

def get_metrics(gt, recons, zf, metric_type, normalized=True, take_absval=True):
    metrics = []
    if take_absval:
        recons = absval(recons)
        gt = absval(gt)
        zf = absval(zf)
    # if normalized:
    #     recons = rescale(recons)
    #     gt = rescale(gt)
    #     zf = rescale(zf)

    if len(recons.shape) > 2:
        for i in range(len(recons)):
            metric = myutils.metrics.get_metric(recons[i], gt[i], metric_type, zero_filled=zf[i])
            metrics.append(metric)
    else:
        metric = myutils.metrics.get_metric(recons, gt, metric_type)
        metrics.append(metric)

    return np.array(metrics)


def gather_baselines(device):
    base_psnrs = []
    base_dcs = []
    base_recons = []
    alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
    betas =  [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
    baseline_temp = '/share/sablab/nfs02/users/aw847/models/HQSplitting/hypernet-baselines/hp-baseline_{beta:01}_{alpha:01}_unet_0.0001_0_5_0_64/t1_4p2_unsup/model.0100.h5'

    gt_data = dataset.get_test_gt('small')
    xdata = dataset.get_test_data('small')
    hps = []
    for beta in betas:
        for alpha in alphas:
            hps.append([alpha, beta])
            baseline_path = baseline_temp.format(alpha=np.round(alpha, 3), beta=np.round(beta, 3))
            print(baseline_path)
            base = test.baseline_test(baseline_path, xdata, gt_data, device, take_avg=False)
            print(base['recons'].shape)
            print(base['rpsnr'].shape)
            print(base['dc'].shape)
            base_recons.append(base['recons'])
            base_psnrs.append(base['rpsnr'])
            base_dcs.append(base['dc'])

    return np.array(hps), np.array(base_psnrs), np.array(base_dcs), np.array(base_recons)

def newloss2oldloss(points):
    '''Convert new loss hps to old loss hps
    
    points: (N, 3)
    '''
    points_convert1 = points[:, 0] / (points[:, 0] + points[:, 1] + points[:, 2])
    points_convert2 = points[:, 1] / (points[:, 1] + points[:, 2])
    points_convert = torch.stack((points_convert1, points_convert2), dim=1).numpy()
    return points_convert
def oldloss2newloss(old_hps):
    '''Convert old loss hps to new loss hps
    
    old_hps: (N, 2)
    '''
    if old_hps.shape[1] == 1:
        a0 = 1-old_hps[:, 0]
        a1 = old_hps[:, 0]
        new_hps = torch.stack((a0, a1), dim=1)
    elif old_hps.shape[1] == 2:
        a0 = old_hps[:, 0]
        a1 = (1-old_hps[:, 0])*old_hps[:,1]
        a2 = (1-old_hps[:, 0])*(1-old_hps[:,1])
        new_hps = torch.stack((a0, a1, a2), dim=1)
    else:
        raise Exception('Unsupported num_hyperparams')
    return new_hps 

def get_reference_hps(num_hyperparams, range_restrict=True):
    hps_2 = torch.tensor([[0.9, 0.1],
                          [0.995, 0.6],
                          [0.9, 0.2],
                          [0.995, 0.5],
                          [0.9, 0],
                          [0.99, 0.7]]).float()
    hps_1 = torch.tensor([[0.5],
                          [0.6],
                          [0.7],
                          [0.8],
                          [0.9],
                          [0.99]]).float()
    if num_hyperparams == 2 and range_restrict:
        return hps_2
    elif num_hyperparams == 3 and not range_restrict:
        return oldloss2newloss(hps_2)
    elif num_hyperparams == 1 and range_restrict:
        return hps_1
    elif num_hyperparams == 2 and not range_restrict:
        return oldloss2newloss(hps_1)
    else:
        raise Exception('Error in num_hp and range_restrict combo')

