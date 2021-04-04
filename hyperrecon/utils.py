"""
Utility functions for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
import numpy as np
import os
import pickle
# import parse
import glob
from . import test, dataset
from hqsnet import test as hqstest
import myutils
import json

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
    return torch.fft(x, signal_ndim=2, normalized=True)

def ifft(x):
    """Normalized 2D Inverse Fast Fourier Transform"""
    # complex_x = torch.view_as_complex(x)
    # ifft = torch.fft.ifft2(complex_x, norm='ortho')
    # return torch.view_as_real(ifft) 
    return torch.ifft(x, signal_ndim=2, normalized=True)

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
    y = y / max_val_per_batch.view(len(y), 1, 1, 1)
    zf = zf / max_val_per_batch.view(len(y), 1, 1, 1)
    return y, zf

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
    """Rescales a batch of images into range [0, 1]"""
    if type(arr) is np.ndarray:
        if len(arr.shape) > 2:
            res = np.zeros(arr.shape)
            for i in range(len(arr)):
                res[i] = (arr[i] - np.min(arr[i])) / (np.max(arr[i]) - np.min(arr[i]))
            return res
        else:
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    else:
        if len(arr.shape) > 2:
            res = torch.zeros_like(arr)
            for i in range(len(arr)):
                res[i] = (arr[i] - torch.min(arr[i])) / (torch.max(arr[i]) - torch.min(arr[i]))
            return res
        else:
            return (arr - torch.min(arr)) / (torch.max(arr) - torch.min(arr))

def normalize_recons(recons):
    recons = absval(recons)
    recons = normalize(recons)
    return recons

def count_parameters(model):
    """Count total parameters in model"""
    for name, val in model.named_parameters():
        print(name)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

######### Saving/Loading checkpoints ############
def load_checkpoint(model, path, optimizer=None):
    print('Loading checkpoint from', path)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
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

def get_metrics(gt, recons, zf, metric_type, take_avg, normalized=True, take_absval=True):
    metrics = []
    if take_absval:
        recons = absval(recons)
        gt = absval(gt)
        zf = absval(zf)
    if normalized:
        recons = rescale(recons)
        gt = rescale(gt)
        zf = rescale(zf)

    if len(recons.shape) > 2:
        for i in range(len(recons)):
            metric = myutils.metrics.get_metric(recons[i], gt[i], metric_type, zero_filled=zf[i])
            metrics.append(metric)
    else:
        metric = myutils.metrics.get_metric(recons, gt, metric_type)
        metrics.append(metric)

    if take_avg:
        return np.array(metrics).mean()
    else:
        return np.array(metrics)

def get_everything(path, device, take_avg=True, \
                   metric_type='relative psnr', \
                   cp=None, n_grid=20, \
                   gt_data=None, xdata=None, test_data=True, convert=False, take_absval=True):
    
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
            gt_data = dataset.get_test_gt(old=True)
            xdata = dataset.get_test_data(old=True)
        else:
            gt_data = dataset.get_train_gt(old=True)
            xdata = datset.get_train_data(old=True)

    args_txtfile = os.path.join(path, 'args.txt')
    if os.path.exists(args_txtfile):
        with open(args_txtfile) as json_file:
            args = json.load(json_file)
    else:
        raise Exception('no args found')
    args['metric_type'] = metric_type
    args['take_absval'] = take_absval

    result_dict = test.tester(model_path, xdata, gt_data, args, device, take_avg, n_grid, convert)
    return result_dict

def gather_baselines(device):
    base_psnrs = []
    base_dcs = []
    base_recons = []
    alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
    betas =  [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
    baseline_temp = '/nfs02/users/aw847/models/HQSplitting/hypernet-baselines/hp-baseline_{beta:01}_{alpha:01}_unet_0.0001_0_5_0_64/t1_4p2_unsup/'

    N=10
    gt_data = dataset.get_test_gt(old=True)
    gt_data = gt_data[3:3+N]
    xdata = dataset.get_test_data(old=True)
    xdata = xdata[3:3+N]
    hps = []
    for beta in betas:
        for alpha in alphas:
            hps.append([alpha, beta])
            baseline_path = baseline_temp.format(alpha=np.round(alpha, 3), beta=np.round(beta, 3))
            base = get_everything(baseline_path, device, hypernet=False, metric_type='relative ssim', take_avg=False, gt_data=gt_data, xdata=xdata)
            print(base['recons'].shape)
            print(base['rpsnr'].shape)
            print(base['dc'].shape)
            base_recons.append(base['recons'])
            base_psnrs.append(base['rpsnr'])
            base_dcs.append(base['dc'])

    return np.array(hps), np.array(base_psnrs), np.array(base_dcs), np.array(base_recons)

def oldloss2newloss(old_hps):
    '''Convert old loss hps to new loss hps
    
    old_hps: (N, num_hyperparams)
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

