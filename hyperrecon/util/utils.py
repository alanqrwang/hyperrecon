"""
Utility functions for HyperRecon
For more details, please read:
  Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
  "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
import numpy as np
import os
import json
from hyperrecon.model import layers

def fft(x):
  """Normalized 2D Fast Fourier Transform

  x: input of shape (batch_size, n_ch, l, w)
  """
  # complex_x = torch.view_as_complex(x)
  # fft = torch.fft.fft2(complex_x,  norm='ortho')
  # return torch.view_as_real(fft) 
  x = x.permute(0, 2, 3, 1)
  if x.shape[-1] == 1:
    x = torch.cat((x, torch.zeros_like(x)), dim=3)
  x = torch.fft(x, signal_ndim=2, normalized=True)
  x = x.permute(0, 3, 1, 2)
  return x

def ifft(x):
  """Normalized 2D Inverse Fast Fourier Transform

  x: input of shape (batch_size, n_ch, l, w)
  """
  # complex_x = torch.view_as_complex(x)
  # ifft = torch.fft.ifft2(complex_x, norm='ortho')
  # return torch.view_as_real(ifft) 
  x = x.permute(0, 2, 3, 1)
  x = torch.ifft(x, signal_ndim=2, normalized=True)
  x = x.permute(0, 3, 1, 2)
  return x

def linear_normalization(arr, new_range=(0, 1)):
  """Linearly normalizes a batch of images into new_range

  arr: (batch_size, n_ch, l, w)
  """
  flat_arr = torch.flatten(arr, start_dim=2, end_dim=3)
  max_per_batch, _ = torch.max(flat_arr, dim=2, keepdim=True) 
  min_per_batch, _ = torch.min(flat_arr, dim=2, keepdim=True) 

  # Handling the edge case of image of all 0's
  max_per_batch[max_per_batch==0] = 1 

  max_per_batch = max_per_batch.view(len(arr), 1, 1, 1)
  min_per_batch = min_per_batch.view(len(arr), 1, 1, 1)

  return (arr - min_per_batch) * (new_range[1]-new_range[0]) / (max_per_batch - min_per_batch) + new_range[0]

def gray2rgb(arr):
  '''Converts grayscale image to rgb by copying along channel dimension.'''
  return arr.repeat(1, 3, 1, 1)

def get_onehot(asegs):
  subset_regs = [[0,130,165,258],   # 0 and 5
                  [8,10,11,12,13,16,17,18,26,28,47,49,50,51,52,53,54,58,60,172,174], # 2
                  [2,7,41,46], # 3
                  [4,5,14,15,24,30,31,43,44,62,63,257],  # 4
                ]

  batch_size = asegs.shape[0]
  data_dims = asegs.shape[2:]

  n_classes = len(subset_regs) + 1 # ROIs plus non-ROI
  one_hot = torch.zeros(batch_size, n_classes, *data_dims)

  for i,subset in enumerate(subset_regs):
    combined_vol = torch.zeros(asegs.shape, dtype=bool).cuda()
    for j in range(len(subset)):
      combined_vol = combined_vol | (asegs == subset[j])
    one_hot[:,i:i+1,...] = combined_vol.long().float()

  mask = one_hot.sum(1).squeeze()
  ones = torch.ones_like(mask)
  non_roi = ones-mask
  one_hot[:,-1,...] = non_roi

  assert one_hot.sum(1).sum() == batch_size*np.prod(data_dims), 'One-hot encoding does not added up to 1'
  return one_hot

def remove_sequential(network, all_layers):
  for layer in network.children():
    if type(layer) == layers.BatchConv2d: # if sequential layer, apply recursively to layers in sequential layer
      all_layers.append(layer)
    if list(layer.children()) != []: # if leaf node, add it to list
      all_layers = remove_sequential(layer, all_layers)

  return all_layers

def summary(network):
  """Print model summary."""
  print('')
  print('Model Summary')
  print('---------------------------------------------------------------')
  for name, val in network.named_parameters():
    print(name)

  print('---------------------------------------------------------------')
  main_param_count = 0
  all_layers = remove_sequential(network, [])
  for l in all_layers:
    main_param_count += np.prod(l.get_kernel_shape()) + np.prod(l.get_bias_shape())
  print('Number of main weights:', main_param_count)
  print('Total parameters:', sum(p.numel() for p in network.parameters() if p.requires_grad))
  print('---------------------------------------------------------------')
  print('')

######### Saving/Loading checkpoints ############
def load_checkpoint(model, path, optimizer=None, scheduler=None, verbose=True):
  if verbose:
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

def save_checkpoint(epoch, model, optimizer, model_folder, scheduler=None):
  state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict()
  }
  if scheduler is not None:
    state['scheduler'] = scheduler.state_dict()

  filename = os.path.join(model_folder, 'model.{epoch:04d}.h5')
  torch.save(state, filename.format(epoch=epoch))
  print('Saved checkpoint to', filename.format(epoch=epoch))

def save_metrics(save_dir, logger, *metrics):
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




# def gather_baselines(device):
#   base_psnrs = []
#   base_dcs = []
#   base_recons = []
#   alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
#   betas =  [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
#   baseline_temp = '/share/sablab/nfs02/users/aw847/models/HQSplitting/hypernet-baselines/hp-baseline_{beta:01}_{alpha:01}_unet_0.0001_0_5_0_64/t1_4p2_unsup/model.0100.h5'

#   gt_data = brain.get_test_gt('small')
#   xdata = brain.get_test_data('small')
#   hps = []
#   for beta in betas:
#     for alpha in alphas:
#       hps.append([alpha, beta])
#       baseline_path = baseline_temp.format(alpha=np.round(alpha, 3), beta=np.round(beta, 3))
#       print(baseline_path)
#       base = test.baseline_test(baseline_path, xdata, gt_data, device, take_avg=False)
#       print(base['recons'].shape)
#       print(base['rpsnr'].shape)
#       print(base['dc'].shape)
#       base_recons.append(base['recons'])
#       base_psnrs.append(base['rpsnr'])
#       base_dcs.append(base['dc'])

#   return np.array(hps), np.array(base_psnrs), np.array(base_dcs), np.array(base_recons)

# def newloss2oldloss(points):
#   '''Convert new loss hps to old loss hps
  
#   points: (N, 3)
#   '''
#   points_convert1 = points[:, 0] / (points[:, 0] + points[:, 1] + points[:, 2])
#   points_convert2 = points[:, 1] / (points[:, 1] + points[:, 2])
#   points_convert = torch.stack((points_convert1, points_convert2), dim=1).numpy()
#   return points_convert
# def oldloss2newloss(old_hps):
#   '''Convert old loss hps to new loss hps
  
#   old_hps: (N, 2)
#   '''
#   if old_hps.shape[1] == 1:
#     a0 = 1-old_hps[:, 0]
#     a1 = old_hps[:, 0]
#     new_hps = torch.stack((a0, a1), dim=1)
#   elif old_hps.shape[1] == 2:
#     a0 = old_hps[:, 0]
#     a1 = (1-old_hps[:, 0])*old_hps[:,1]
#     a2 = (1-old_hps[:, 0])*(1-old_hps[:,1])
#     new_hps = torch.stack((a0, a1, a2), dim=1)
#   else:
#     raise Exception('Unsupported num_hyperparams')
#   return new_hps 

# def get_reference_hps(num_hyperparams, range_restrict=True):
#   hps_2 = torch.tensor([[0.9, 0.1],
#               [0.995, 0.6],
#               [0.9, 0.2],
#               [0.995, 0.5],
#               [0.9, 0],
#               [0.99, 0.7]]).float()
#   hps_1 = torch.tensor([[0.5],
#               [0.6],
#               [0.7],
#               [0.8],
#               [0.9],
#               [0.99]]).float()
#   if num_hyperparams == 2 and range_restrict:
#     return hps_2
#   elif num_hyperparams == 3 and not range_restrict:
#     return oldloss2newloss(hps_2)
#   elif num_hyperparams == 1 and range_restrict:
#     return hps_1
#   elif num_hyperparams == 2 and not range_restrict:
#     return oldloss2newloss(hps_1)
#   else:
#     raise Exception('Error in num_hp and range_restrict combo')

