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

def add_bool_arg(parser, name, default=True):
    """Add boolean argument to argparse parser"""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})

def fft(x):
    """Normalized 2D Fast Fourier Transform"""
    return torch.fft(x, signal_ndim=2, normalized=True)

def ifft(x):
    """Normalized 2D Inverse Fast Fourier Transform"""
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

def normalize(arr):
    """Normalizes a batch of images into range [0, 1]"""
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
    recons = _normalize(recons)
    return recons

def count_parameters(model):
    """Count total parameters in model"""
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

def save_checkpoint(epoch, model_state, optimizer_state, logger, model_folder, log_interval, scheduler=None):
    if epoch % log_interval == 0:
        state = {
            'epoch': epoch,
            'state_dict': model_state,
            'optimizer' : optimizer_state,
            'loss': logger['loss_train'],
            'val_loss': logger['loss_val']
        }
        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict(),

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

def get_metrics(gt, recons, metric_type, take_avg, normalized=True):
    metrics = []
    if normalized:
        recons_pro = normalize(recons)
        gt_pro = normalize(gt)
    else:
        recons_pro = myutils.array.make_imshowable(recons)
        gt_pro = myutils.array.make_imshowable(gt)

    if len(recons.shape) > 2:
        for i in range(len(recons)):
            metric = myutils.metrics.get_metric(recons_pro[i], gt_pro[i], metric_type)
            metrics.append(metric)
    else:
        metric = myutils.metrics.get_metric(recons_pro, gt_pro, metric_type)
        metrics.append(metric)

    if take_avg:
        return np.mean(np.array(metrics))
    else:
        return np.array(metrics)

def path2config(path, new_parse, device):
    '''
    Converts model path name to dictionary of parameters
    '''
    if new_parse:
        parse_format = '/nfs02/users/aw847/models/HyperHQSNet/{prefix}_{lr}_{batch_size}_{reg_types}_{unet_hidden}_{bounds}_{topK}_{range_restrict}/{filename}'
        # parse_format = '/nfs02/users/aw847/models/HyperHQSNet/{prefix}_{lr}_{batch_size}_{reg_types}_{unet_hidden}_{topK}_{range_restrict}/{filename}'
    else:
        parse_format = '/nfs02/users/aw847/models/HyperHQSNet/{prefix}_{recon_type}_{lr}_{batch_size}_{lmbda}_{K}_{reg_types}_{unet_hidden}_{alpha_bound}_{beta_bound}_{topK}_{range_restrict}/{dataset}_{maskname}/{filename}'
        # parse_format = '/nfs02/users/aw847/models/HyperHQSNet/{prefix}_{lr}_{batch_size}_{reg_types}_{unet_hidden}_{topK}_{range_restrict}/{filename}'
    config = parse.parse(parse_format, path)
    assert (config is not None), '\n parse_format is %s \n path is %s' % (parse_format, path)
    config = config.named
    config['device'] = device
    return config

def path2configlegacy(path):
    parse_format = '/nfs02/users/aw847/models/HQSplitting/{prefix}_{w_coeff}_{tv_coeff}_{recon_type}_{lr}_{lmbda}_{K}_{learn_reg_coeff}_{n_hidden}/{dataset}_{maskname}_{strategy}/{filename}'
#     parse_format = '/nfs02/users/aw847/models/HQSplitting/{prefix}_{tv_coeff}_{lr}_{lmbda}_{K}_{learn_reg_coeff}_{n_hidden}/{dataset}_{maskname}_{strategy}/{filename}'
    
    config = parse.parse(parse_format, path)
    assert (config is not None), '\n parse_format is %s \n path is %s' % (parse_format, path)
    return config

def get_everything(path, device, take_avg=True, recons_only=False, \
                   hypernet=True, metric_type='relative psnr', \
                   hyparch='small', cp=None, n_grid=20, new_parse=False, \
                   gt_data=None, xdata=None, test_data=True):
    mask = torch.tensor(dataset.get_mask(4)).to(device).float()
    
    # Forward through latest available model
    if cp is None:
        model_paths = sorted(glob.glob(os.path.join(path, 'model.*.h5')))
        model_path = model_paths[-1]
    # Or forward through specified epoch
    else:
        path = path.replace('[[]', '[').replace('[]]', ']')
        model_path = os.path.join(path, 'model.{epoch:04d}.h5'.format(epoch=cp))
        
    if gt_data is None:
        N=10
        if test_data:
            gt_data = dataset.get_test_gt(old=True)
            xdata = dataset.get_test_data(old=True)
        else:
            gt_data = dataset.get_train_gt(old=True)
            xdata = datset.get_train_data(old=True)

        gt_data = gt_data[3:3+N]
        xdata = xdata[3:3+N]
    if hypernet:
        config = path2config(model_path, new_parse, device)
        result_dict = test.tester(model_path, xdata, gt_data, config, device, hyparch, take_avg, n_grid, new_parse)
    else:
        config = path2configlegacy(model_path)
        result_dict = test.baseline_test(model_path, xdata, gt_data, config, device, take_avg)

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
    a0 = old_hps[:, 0]
    a1 = (1-old_hps[:, 0])*old_hps[:,1]
    a2 = (1-old_hps[:, 0])*(1-old_hps[:,1])
#     a3 = torch.zeros_like(old_hps[:, 0])
    return torch.stack((a0, a1, a2), dim=1)

def get_reference_hps(device):
    old_hps = torch.tensor([[0.9, 0.1],
     [0.995, 0.6],
     [0.9, 0.2],
     [0.995, 0.5],
     [0.9, 0],
     [0.99, 0.7]]).to(device).float()
    return oldloss2newloss(old_hps)

