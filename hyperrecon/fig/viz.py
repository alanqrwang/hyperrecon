import matplotlib.pyplot as plt
import numpy as np
from .util import _collect_base_subject, _collect_hypernet_subject, _extract_slices, _compute_pixel_range, _overlay_error, extract_kernel_layer
from .plot import _plot_img
from hyperrecon.util import metric
import os
from torchvision.utils import make_grid

def viz_pixel_range(paths, slice_idx, hparams, subject, cp, title, base=False, ax=None, rot90=True):
  '''Visualize pixel-wise range across hyperparameters.'''
  ax = ax or plt.gca()
  if base:
    assert isinstance(paths, list)
    assert len(paths) == len(hparams), 'Paths and hparams mismatch'
    _, _, preds = _collect_base_subject(paths, hparams, subject, cp)
  else:
    _, _, preds = _collect_hypernet_subject(paths, hparams, subject, cp)

  slices = _extract_slices(preds, slice_idx)
  error = _compute_pixel_range(slices)

  avg_error = np.mean(error)
  _plot_img(_overlay_error(slices[0], error), ax=ax, rot90=rot90, title=title, xlabel='MAE=' + str(np.round(avg_error, 3)))
  return avg_error

def viz_pairwise_errors(paths, slices, hparams, subject, cp, base=False):
  '''Visualize pairwise errors.'''
  if base:
    assert isinstance(paths, list)
    assert len(paths) == len(hparams), 'Paths and hparams mismatch'
    gt, _, preds = _collect_base_subject(paths, hparams, subject, cp)
  else:
    gt, _, preds = _collect_hypernet_subject(paths, hparams, subject, cp)
  gt_slices = gt[slices,0]
  for s in slices:
    slices = _extract_slices(preds, s)
    num_slices = len(slices)

    fig, axes = plt.subplots(num_slices, num_slices, figsize=(10, 15))
    [ax.set_axis_off() for ax in axes.ravel()]

    for i in range(num_slices):
      for j in range(i, num_slices):
        if i == j:
          pred_psnr = 'PSNR={:.04f}'.format(metric.psnr(gt_slices[i], slices[i]))
          print(pred_psnr)
          _plot_img(slices[i], ax=axes[i, j], rot90=True, title=hparams[i], xlabel=pred_psnr)
        else:
          error = np.abs(slices[i] - slices[j])
          _plot_img(_overlay_error(slices[i], error), ax=axes[i, j], rot90=True, title=np.round(np.mean(error), 3))
    fig.show()

def viz_all(paths, s, hparams, subject, cp, title, base=False, rot90=True):
  if base:
    gt, zf, preds = _collect_base_subject(paths, hparams, subject, cp)
  else:
    gt, zf, preds = _collect_hypernet_subject(paths, hparams, subject, cp)
  gt_slice = gt[s,0]
  zf_slice = _extract_slices(zf, s)[0]
  pred_slice = _extract_slices(preds, s)
  zf_psnr = 'PSNR={:.04f}'.format(metric.psnr(gt_slice, zf_slice))

  fig, axes = plt.subplots(1, len(hparams)+2, figsize=(17, 7))
  _plot_img(gt_slice, ax=axes[0], rot90=rot90, title='GT', ylabel=title)
  _plot_img(zf_slice, ax=axes[1], rot90=rot90, title='Input', xlabel=zf_psnr)
  for j in range(len(hparams)):
    pred_psnr = 'PSNR={:.04f}'.format(metric.psnr(gt_slice, pred_slice[j]))
    _plot_img(pred_slice[j], ax=axes[j+2], rot90=rot90, title=hparams[j], xlabel=pred_psnr, vlim=[0, 1])
  
  fig.tight_layout()

def viz_all_loupe(paths, s, hparams, subject, cp, title, base=False, rot90=True):
  if base:
    gt, zfs, preds = _collect_base_subject(paths, hparams, subject, cp)
  else:
    gt, zfs, preds = _collect_hypernet_subject(paths, hparams, subject, cp)
  gt_slice = gt[s,0]
  zf_slice = _extract_slices(zfs, s)
  pred_slice = _extract_slices(preds, s)

  fig, axes = plt.subplots(2, len(hparams)+1, figsize=(len(hparams)*4, 2*4))
  _plot_img(gt_slice, ax=axes[0,0], rot90=rot90, title='GT', ylabel=title+ ' zf')
  _plot_img(gt_slice, ax=axes[1,0], rot90=rot90, title='GT', ylabel=title + ' pred')
  for j in range(len(hparams)):
    zf_psnr = 'PSNR={:.04f}'.format(metric.psnr(gt_slice, zf_slice[j]))
    _plot_img(zf_slice[j], ax=axes[0, j+1], rot90=rot90, title=hparams[j], xlabel=zf_psnr)
    pred_psnr = 'PSNR={:.04f}'.format(metric.psnr(gt_slice, pred_slice[j]))
    _plot_img(pred_slice[j], ax=axes[1, j+1], rot90=rot90, title=hparams[j], xlabel=pred_psnr)
  
  fig.tight_layout()

def viz_trajnet(traj_path):
  recon_path = os.path.join(traj_path, 'img/recons.npy')
  recons = np.load(recon_path)
  for i in range(len(recons)):
    fig, axes = plt.subplots(3, 4, figsize=(12, 12))
    for j in range(recons.shape[1]):
      _plot_img(recons[i, j, 0], rot90=True, ax=axes[j//4, j%4])
    fig.show()
    plt.show()

def viz_trajnet_range(traj_path):
  recon_path = os.path.join(traj_path, 'img/recons.npy')
  recons = np.load(recon_path)
  for i in range(len(recons)):
    slices = recons[i, :, 0]
    error = _compute_pixel_range(slices)
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    _plot_img(_overlay_error(slices[0], error), ax=axes, rot90=True, title=np.round(np.mean(error), 3))
    plt.show()

def viz_real_imag_kernel(kernel, title=None):
  fig, axes = plt.subplots(1, 2, figsize=(16, 6))
  img = make_grid(kernel).permute(1, 2, 0).cpu().detach().numpy()
  _, im = _plot_img(img[..., 0], title='real', xlabel='MAE='+str(np.round(np.abs(img[..., 0]).sum(), 3)), ax=axes[0], vlim=[-1, 1]) 
  _, _  = _plot_img(img[..., 1], title='imag', xlabel='MAE='+str(np.round(np.abs(img[..., 1]).sum(), 3)), ax=axes[1], vlim=[-1, 1]) 
  
  cbar_ax = fig.add_axes([0.5, 0.5, 0.01, 0.1])

  fig.colorbar(im, cax=cbar_ax)
  if title is not None:
      fig.suptitle(title)
  fig.show()

def viz_kernel_last(kernel):
  pass

def viz_real_imag_kernel_0_1(path, layer_idx, title, base=False, dim=64):
  if base:
    kernels0 = extract_kernel_layer(path[0], None, layer_idx, arch='baseline')
    kernels1 = extract_kernel_layer(path[1], None, layer_idx, arch='baseline')
  else:
    kernels0 = extract_kernel_layer(path, [1., 0.], layer_idx, dim, arch='hyperunet')
    kernels1 = extract_kernel_layer(path, [0., 1.], layer_idx, dim, arch='hyperunet')

  mae = [(k0-k1).abs() for k0, k1 in zip(kernels0, kernels1)]

  if len(mae)-1 == layer_idx:
    viz_kernel_last(kernels0, title='{} hp=0, layer{}'.format(title, layer_idx))
    viz_kernel_last(kernels1, title='{} hp=1, layer{}'.format(title, layer_idx))
    viz_kernel_last(mae, title='{} MAE, layer{}'.format(title, layer_idx))
  else:
    viz_real_imag_kernel(kernels0, title='{} hp=0, layer{}'.format(title, layer_idx))
    viz_real_imag_kernel(kernels1, title='{} hp=1, layer{}'.format(title, layer_idx))
    viz_real_imag_kernel(mae, title='{} MAE, layer{}'.format(title, layer_idx))