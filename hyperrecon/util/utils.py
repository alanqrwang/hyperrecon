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