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

    if x.shape[-1] == 1:
        x = torch.cat((x, torch.zeros_like(x)), dim=-1)
    return torch.fft(x, signal_ndim=2, normalized=True)

def ifft(x):
    """Normalized 2D Inverse Fast Fourier Transform"""
    # complex_x = torch.view_as_complex(x)
    # ifft = torch.fft.ifft2(complex_x, norm='ortho')
    # return torch.view_as_real(ifft) 
    return torch.ifft(x, signal_ndim=2, normalized=True)

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

def gather_baselines(device):
    base_psnrs = []
    base_dcs = []
    base_recons = []
    alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
    betas =  [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
    baseline_temp = '/share/sablab/nfs02/users/aw847/models/HQSplitting/hypernet-baselines/hp-baseline_{beta:01}_{alpha:01}_unet_0.0001_0_5_0_64/t1_4p2_unsup/model.0100.h5'

    N=1
    gt_data = dataset.get_test_gt('med')
    gt_data = gt_data[3:3+N]
    xdata = dataset.get_test_data('med')
    xdata = xdata[3:3+N]
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

