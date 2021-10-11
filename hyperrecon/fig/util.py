import numpy as np
import torch
from hyperrecon.model.unet import HyperUnet, Unet
from hyperrecon.model.unet_v2 import LastLayerHyperUnet
from hyperrecon.util import utils
import json
import os
from glob import glob

def _overlay_error(img, error):
  img_rgb = np.dstack((img, img, img))
  img_rgb[..., 0] += error * 10
  return img_rgb

def _extract_slices(imgs, slice_num):
  '''Collect slices from list of batch of images.
  
  Args:
    imgs: (num_hparams, bs, 1, n1, n2) 
    slice_num: slice index to collect
  
  Returns:
    res: (num_hparams, n1, n2)
  '''
  res = []
  for j in range(len(imgs)):
    pred_slice = imgs[j][slice_num,0]
    res.append(pred_slice)
  return np.array(res)

def _compute_pixel_range(imgs):
  '''Computes pixel-wise range.
  
  Args:
    imgs: (num_imgs, n1, n2)
  '''
  return np.ptp(imgs, axis=0)

def get_all_kernels_base(model):
  kernels = []
  kernels.append(model.dconv_down1[0].weight.detach().clone())
  kernels.append(model.dconv_down1[3].weight.detach().clone())
  kernels.append(model.dconv_down2[0].weight.detach().clone())
  kernels.append(model.dconv_down2[3].weight.detach().clone())
  kernels.append(model.dconv_down3[0].weight.detach().clone())
  kernels.append(model.dconv_down3[3].weight.detach().clone())
  kernels.append(model.dconv_down4[0].weight.detach().clone())
  kernels.append(model.dconv_down4[3].weight.detach().clone())
  kernels.append(model.dconv_up3[0].weight.detach().clone())
  kernels.append(model.dconv_up3[3].weight.detach().clone())
  kernels.append(model.dconv_up2[0].weight.detach().clone())
  kernels.append(model.dconv_up2[3].weight.detach().clone())
  kernels.append(model.dconv_up1[0].weight.detach().clone())
  kernels.append(model.dconv_up1[3].weight.detach().clone())
  kernels.append(model.conv_last.weight.detach().clone())

  return kernels


def get_all_kernels_hyp(model, hparam):
  def extract_kernel_hyp(model_layer, hyp_out):
    kernels = model_layer.get_kernel(hyp_out)
    kernels = kernels.reshape(model_layer.get_kernel_shape())    
    return kernels

  hnet = model.hnet
  hyp_out = hnet(torch.tensor(hparam).view(-1, 2))

  kernels = []
  kernels.append(extract_kernel_hyp(model.unet.dconv_down4[0], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.dconv_down4[3], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.dconv_down3[0], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.dconv_down3[3], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.dconv_down2[0], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.dconv_down2[3], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.dconv_down1[0], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.dconv_down1[3], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.dconv_up3[0], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.dconv_up3[3], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.dconv_up2[0], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.dconv_up2[3], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.dconv_up1[0], hyp_out))
  kernels.append(extract_kernel_hyp(model.unet.conv_last, hyp_out))

  return kernels

def extract_kernel_layer(model_path, hparam, layer_idx, hnet_hdim=None, arch=False):
  '''Extract kernel for given model_path at specific layer_idx.
  
  Returns:
    kernel: [out_channels, in_channels, ks, ks]
  '''
  if arch == 'baseline':
    model = Unet(
              in_ch=2,
              out_ch=1,
              h_ch=32,
              use_batchnorm=True
            )
    model = utils.load_checkpoint(model, model_path, verbose=False)
    model.eval()

    kernels = get_all_kernels_base(model)
  elif arch == 'last_layer_hyperunet':
    model = LastLayerHyperUnet(
      2,
      hnet_hdim,
      in_ch_main=2,
      out_ch_main=1,
      h_ch_main=32,
      residual=True,
      use_batchnorm=True,
    )
    model = utils.load_checkpoint(model, model_path, verbose=False)
    model.eval()
    kernels = get_all_kernels_hyp(model, hparam) 
  else:
    model = HyperUnet(
      2,
      hnet_hdim,
      in_ch_main=2,
      out_ch_main=1,
      h_ch_main=32,
      residual=True,
      use_batchnorm=True,
    )
    model = utils.load_checkpoint(model, model_path, verbose=False)
    model.eval()
    kernels = get_all_kernels_hyp(model, hparam) 
  
  return kernels[layer_idx]

def _collect_hypernet_subject(model_path, hparams, subject, cp):
  '''For subject, get reconstructions from hypernet for all hyperparams.
  
  Returns:
    gt: Ground truth of subject (bs, 1, n1, n2)
    zf: Zero-filled recon of subject (bs, 1, n1, n2)
    preds: Predictions of subject for each hyperparameter (num_hparams, bs, nch, n1, n2)
  '''
  gt_path = os.path.join(model_path, 'img/gtsub{}.npy'.format(subject))
  zf_paths = []
  for hparam in hparams:
    zf_path = os.path.join(model_path, 'img/zf{}sub{}.npy'.format(hparam, subject))
    if os.path.exists(zf_path):
      zf_paths.append(zf_path)
  if len(zf_paths) == 0:
    zf_paths = [os.path.join(model_path, 'img/zfsub{}.npy'.format(subject))]
  gt = np.load(gt_path)
  zfs = [np.linalg.norm(np.load(zf_path), axis=1, keepdims=True) for zf_path in zf_paths]
  preds = []
  for hparam in hparams:
    if cp is None:
      pred_path = os.path.join(model_path, 'img/pred{}sub{}.npy'.format(hparam, subject))
    else:
      pred_path = os.path.join(model_path, 'img/pred{}sub{}cp{:04d}.npy'.format(hparam, subject, cp))
    print('loading:', pred_path)
    preds.append(np.load(pred_path))
  return gt, zfs, preds

def _collect_base_subject(model_paths, hparams, subject, cps):
  '''For subject, get reconstructions from baseline for all hyperparams.
  
  The hparams are specified in the model_paths.
  Returns:
    gt: Ground truth of subject (bs, 1, n1, n2)
    zf: Zero-filled recon of subject (bs, 1, n1, n2)
    preds: Predictions of subject for each hyperparameter (num_hparams, bs, nch, n1, n2)
  '''
  if not isinstance(model_paths, (list, tuple)):
    model_paths = [model_paths]
  gt_path = os.path.join(model_paths[0], 'img/gtsub{}.npy'.format(subject))
  gt = np.load(gt_path)
  preds = []
  zfs = []
  for hparam, model_path, cp in zip(hparams, model_paths, cps):
    zf_path = os.path.join(model_path, 'img/zfsub{}.npy'.format(subject))
    zfs.append(np.linalg.norm(np.load(zf_path), axis=1, keepdims=True))
    pred_path = os.path.join(model_path, 'img/pred{}sub{}cp{:04d}.npy'.format(hparam, subject, cp))
    preds.append(np.load(pred_path))
  return gt, zfs, preds

def _parse_summary_json(model_path, metric_of_interest, split='test'):
  # Opening JSON file
  with open(os.path.join(model_path, 'summary_full.json')) as json_file:
    data = json.load(json_file)
    parsed = {}
  
    for key in data:
      if split in key and metric_of_interest in key:
        parts = key.split(':')
        if len(parts) > 1:
          metr, split, hp, sub = parts[0], parts[1], parts[2], parts[3]
        else:
          metr = key.split('.')[0]
          sub = key.split('sub')[-1]
          hp = key.split('sub')[0].split(split)[-1]

        if hp in parsed:
          parsed[hp].append(data[key])
        else:
          parsed[hp] = [data[key]]
  return parsed