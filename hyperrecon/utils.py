"""
Utility functions for HyperRecon
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
import torchio as tio
import numpy as np
import pickle
import glob
from . import test, dataset, model, layers
import myutils
import os
import matplotlib.pyplot as plt
from myutils.plot import plot_img

def add_bool_arg(parser, name, default=True):
    """Add boolean argument to argparse parser"""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})

def fft(x):
    """Normalized 2D Fast Fourier Transform

    x: input of shape (batch_size, n_ch, l, w)
    """
    # complex_x = torch.view_as_complex(x)
    # fft = torch.fft.fft2(complex_x,  norm='ortho')
    # return torch.view_as_real(fft) 
    if x.shape[-1] == 1:
        x = torch.cat((x, torch.zeros_like(x)), dim=3)
    x = torch.fft(x, signal_ndim=2, normalized=True)
    return x

def ifft(x):
    """Normalized 2D Inverse Fast Fourier Transform

    x: input of shape (batch_size, n_ch, l, w)
    """
    # complex_x = torch.view_as_complex(x)
    # ifft = torch.fft.ifft2(complex_x, norm='ortho')
    # return torch.view_as_real(ifft) 
    x = torch.ifft(x, signal_ndim=2, normalized=True)
    return x

def undersample(fullysampled, mask):
    '''Generate undersampled k-space data with given binary mask'''
    if fullysampled.shape[-1] == 1:
        fullysampled = torch.cat((fullysampled, torch.zeros_like(fullysampled)), dim=3)

    mask_expand = mask.unsqueeze(-1)
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
    if type(batch) == list:
        targets = batch[0].to(args['device']).float()
        segs = batch[1].to(args['device']).float()

    else:
        targets = batch.float().to(args['device'])
        segs = None

    under_ksp = undersample(targets, args['mask'])
    zf = ifft(under_ksp)

    if args['rescale_in']:
        zf = zf.norm(p=2, dim=-1, keepdim=True)
        zf = rescale(zf)
    else:
        under_ksp, zf = scale(under_ksp, zf)

    return zf, targets, under_ksp, segs

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

def remove_sequential(network, all_layers):
    for layer in network.children():
        if type(layer) == layers.BatchConv2d: # if sequential layer, apply recursively to layers in sequential layer
            all_layers.append(layer)
        if list(layer.children()) != []: # if leaf node, add it to list
            all_layers = remove_sequential(layer, all_layers)

    return all_layers

def count_parameters(network):
    """Count total parameters in model"""
    for name, val in network.named_parameters():
        print(name)

    main_param_count = 0
    all_layers = []
    all_layers = remove_sequential(network, all_layers)
    for l in all_layers:
        main_param_count += np.prod(l.get_weight_shape()) + np.prod(l.get_bias_shape())
    print('Number of main weights:', main_param_count)
    return sum(p.numel() for p in network.parameters() if p.requires_grad)

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
