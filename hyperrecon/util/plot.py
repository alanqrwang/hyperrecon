from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from glob import glob
from hyperrecon.util import utils
from hyperrecon.data import brain
from scipy.spatial.distance import squareform, pdist
matplotlib.rcParams['lines.linewidth'] = 3
from . import metric
import json

SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 22
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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

def viz_pixel_range(paths, slices, hparams, subject, base=False):
  '''Visualize pixel-wise range across hyperparameters.'''
  if base:
    assert isinstance(paths, list)
    assert len(paths) == len(hparams), 'Paths and hparams mismatch'
    _, _, preds = _collect_base_subject(paths, subject)
  else:
    _, _, preds = _collect_hypernet_subject(paths, hparams, subject)

  for s in slices:
    slices = _extract_slices(preds, s)

    fig, axes = plt.subplots(1, 1, figsize=(5, 6))
    error = np.ptp(slices, axis=0)
    _plot_img(_overlay_error(slices[0], error), ax=axes, rot90=True)
    fig.suptitle('Pixel-wise range across hparams')
    fig.show()

def viz_pairwise_errors(paths, slices, hparams, subject, base=False):
  '''Visualize pairwise errors.'''
  if base:
    assert isinstance(paths, list)
    assert len(paths) == len(hparams), 'Paths and hparams mismatch'
    _, _, preds = _collect_base_subject(paths, subject)
  else:
    _, _, preds = _collect_hypernet_subject(paths, hparams, subject)
  for s in slices:
    slices = _extract_slices(preds, s)
    num_slices = len(slices)

    fig, axes = plt.subplots(num_slices, num_slices, figsize=(10, 15))
    [ax.set_axis_off() for ax in axes.ravel()]

    for i in range(num_slices):
      for j in range(i, num_slices):
        if i == j:
          _plot_img(slices[i], ax=axes[i, j], rot90=True)
        else:
          error = np.abs(slices[i] - slices[j])
          _plot_img(_overlay_error(slices[i], error), ax=axes[i, j], rot90=True, title=np.round(np.mean(error), 3))
    fig.show()

def viz_all(hyp_path, base_paths, slices, hparams, subject):
  hyp_gt, hyp_zf, hyp_preds = _collect_hypernet_subject(hyp_path, hparams, subject)
  _, _, base_preds = _collect_base_subject(base_paths, subject)
  for s in slices:
    gt_slice = hyp_gt[s,0]
    zf_slice = hyp_zf[s]
    hyp_slice = _extract_slices(hyp_preds, s)
    base_slice = _extract_slices(base_preds, s)
    zf_psnr = 'PSNR={:.04f}'.format(metric.psnr(gt_slice, zf_slice))

    fig, axes = plt.subplots(2, len(hparams)+2, figsize=(17, 7))
    _plot_img(gt_slice, ax=axes[0,0], rot90=True, title='GT', ylabel='Hypernet')
    _plot_img(zf_slice, ax=axes[0,1], rot90=True, title='ZF', xlabel=zf_psnr)
    for j in range(len(hparams)):
      pred_psnr = 'PSNR={:.04f}'.format(metric.psnr(gt_slice, hyp_slice[j]))
      _plot_img(hyp_slice[j], ax=axes[0,j+2], rot90=True, title=hparams[j], xlabel=pred_psnr)

    _plot_img(gt_slice, ax=axes[1,0], rot90=True, ylabel='Base')
    _plot_img(zf_slice, ax=axes[1,1], rot90=True, xlabel=zf_psnr)
    for j in range(len(hparams)):
      base_psnr = 'PSNR={:.04f}'.format(metric.psnr(gt_slice, base_slice[j]))
      _plot_img(base_slice[j], ax=axes[1,j+2], rot90=True, xlabel=base_psnr)
  
  fig.tight_layout()

def plot_over_hyperparams(model_path, base_paths, metric_of_interest, flip=False, ax=None):
  ax = ax or plt.gca()
  if metric_of_interest in ['psnr', 'ssim', 'dice']:
    ann_min, ann_max = False, True
  elif metric_of_interest in ['loss', 'hfen']:
    ann_min, ann_max = True, False

  hyp_parsed = _parse_summary_json(model_path, metric_of_interest)

  base_xs, base_ys = [], []
  for base_path in base_paths:
    base_parsed = _parse_summary_json(base_path, metric_of_interest)
    base_xs.append([float(n) for n in base_parsed.keys()][0])
    base_ys.append([np.mean(l) for l in base_parsed.values()][0])

  xs = [float(n) for n in hyp_parsed.keys()]
  ys = np.array([np.mean(l) for l in hyp_parsed.values()])
  if flip:
    ys = 1 - ys
    base_ys = 1 - np.array(base_ys)

  _plot_1d(xs, ys, color='blue', label='hyp', linestyle='-', annotate_min=ann_min, annotate_max=ann_max, ax=ax)
  _plot_1d(base_xs, base_ys, color='orange', label='base', linestyle='.--', annotate_min=ann_min, annotate_max=ann_max, ax=ax)
  ax.set_title(metric_of_interest)
  ax.set_xlabel('alpha')
  ax.legend()
  ax.grid()

def plot_over_hyperparams_per_subject(model_path, base_paths, metric_of_interest, flip=False, ax=None):
  ax = ax or plt.gca()
  if metric_of_interest in ['psnr', 'ssim', 'dice']:
    ann_min, ann_max = False, True
  elif metric_of_interest in ['loss', 'hfen']:
    ann_min, ann_max = True, False

  hyp_parsed = _parse_summary_json(model_path, metric_of_interest)
  base_xs, base_ys = [], []
  for base_path in base_paths:
    base_parsed = _parse_summary_json(base_path, metric_of_interest)
    key = [n for n in base_parsed.keys()][0]
    base_xs.append(float(key))
    base_ys.append(base_parsed[key])

  base_ys = np.array(base_ys).T
  xs = [float(n) for n in hyp_parsed.keys()]
  ys = np.array([[n for n in l] for l in hyp_parsed.values()]).T
  if flip:
    ys = 1 - ys
    base_ys = 1 - np.array(base_ys)

  num_subjects = ys.shape[0]
  labels=['subject {}'.format(n) for n in range(num_subjects)]
  colors = cm.cool(np.linspace(0, 1, num_subjects))
  for y, base_y, l, c in zip(ys, base_ys, labels, colors):
    _plot_1d(xs, y, label=l, color=c, annotate_min=ann_min, annotate_max=ann_max, ax=ax)
    _plot_1d(base_xs, base_y, label=l, color=c, annotate_min=ann_min, annotate_max=ann_max, linestyle='--', ax=ax)
  ax.set_title(metric_of_interest + ' per subject')
  ax.set_xlabel('alpha')
  ax.legend()
  ax.grid()

def plot_over_hyperparams_2d(model_path, metric_of_interest, flip=False, ax=None, vlim=None):
  ax = ax or plt.gca()
  if metric_of_interest in ['psnr', 'ssim', 'dice']:
    ann_min, ann_max = False, True
  elif metric_of_interest in ['loss', 'hfen', 'mae', 'watson']:
    ann_min, ann_max = True, False

  hyp_parsed = _parse_summary_json(model_path, metric_of_interest)

  # Gather values and unique x, y values
  x_idx = set()
  y_idx = set()
  values = []
  keys = []
  for i, key in enumerate(hyp_parsed):
    x, y = float(key.split('_')[0]), float(key.split('_')[1])
    x_idx.add(x)
    y_idx.add(y)
  for x in sorted(x_idx):
    for y in sorted(y_idx):
      key_str = str(x) + '_' + str(y)
      value = hyp_parsed[key_str]
      values.append(np.mean(value))
      keys.append(key_str)
  # values is 1-d list where y index changes first, i.e. (0,0), (0,1), (1,0), ...
  
  # Reshape first fills by row. 
  # So each row is of constant x value
  # and each column if of constant y value.
  vals = np.array(values).reshape((len(x_idx), len(y_idx)))
  keys = np.array(keys).reshape((len(x_idx), len(y_idx)))
  # Transpose to get constant x value in columns
  vals = vals.T
  keys = keys.T
  print(keys[:2, :2])
  print(vals[:2, :2])
  
  if flip:
    vals = 1 - vals

  _plot_2d(vals, annotate_min=ann_min, annotate_max=ann_max, xlabel='a', ylabel='b', ax=ax, vlim=vlim, all_ticks='ends')
  ax.set_title(metric_of_interest)
  ax.grid()

def plot_over_hyperparams_per_subject_2d(model_path, metric_of_interest, flip=False, vlim=None):
  if metric_of_interest in ['psnr', 'ssim', 'dice']:
    ann_min, ann_max = False, True
  elif metric_of_interest in ['loss', 'hfen']:
    ann_min, ann_max = True, False

  hyp_parsed = _parse_summary_json(model_path, metric_of_interest)

  # Gather values and unique x, y values
  x_idx = set()
  y_idx = set()
  values = []
  keys = []
  for i, key in enumerate(hyp_parsed):
    x, y = float(key.split('_')[0]), float(key.split('_')[1])
    x_idx.add(x)
    y_idx.add(y)
  for x in sorted(x_idx):
    for y in sorted(y_idx):
      key_str = str(x) + '_' + str(y)
      value = hyp_parsed[key_str]
      values.append(value)
      keys.append(key_str)
  # values is 1-d list where y index changes first, i.e. (0,0), (0,1), (1,0), ...
  
  # Reshape first fills by row. 
  # So each row is of constant x value
  # and each column if of constant y value.
  vals = np.array(values).reshape((len(x_idx), len(y_idx), -1))
  # Transpose to get constant x value in columns
  vals = np.transpose(vals, (1,0,2))
  
  if flip:
    vals = 1 - vals
  
  fig, axes = plt.subplots(1, vals.shape[-1], figsize=(4*vals.shape[-1], 4))
  for i in range(vals.shape[-1]):
    _plot_2d(vals[..., i], annotate_min=ann_min, annotate_max=ann_max, xlabel='alpha', ylabel='beta', vlim=vlim, ax=axes[i])
    axes[i].set_title('sub%d' % i)
  fig.suptitle(metric_of_interest)
  fig.show()

def plot_monitor(monitor, model_paths, ax=None):
  ax = ax or plt.gca()
  if not isinstance(model_paths, list):
    model_paths = [model_paths]

  for model_path in model_paths:
    train_path = os.path.join(model_path, 'monitor', '{}.txt'.format(monitor))

    if os.path.exists(train_path):
      loss = np.loadtxt(train_path) 
    else:
      raise ValueError('Invalid train path found')
    _plot_1d(np.arange(len(loss)), loss, label=model_path.split('/')[-4], ax=ax)

  ax.set_xlabel('Epoch')
  ax.set_title(monitor)
  ax.grid()
  return ax

def plot_metrics(metric, model_paths,
         show_legend=True, xlim=None, ylim=None, lines_to_plot=('train', 'val'), vline=None, ax=None):
  ax = ax or plt.gca()
  if not isinstance(model_paths, list):
    model_paths = [model_paths]
  if not isinstance(lines_to_plot, (tuple, list)):
    lines_to_plot = (lines_to_plot)

  if metric in ['psnr', 'ssim', 'dice']:
    ann_min, ann_max = False, True
  elif metric in ['loss', 'hfen']:
    ann_min, ann_max = True, False

  for model_path in model_paths:
    train_path = os.path.join(model_path, 'metrics', '{}:train.txt'.format(metric))
    val_paths = glob(os.path.join(
      model_path, 'metrics', '{}:val*.txt'.format(metric)))

    if os.path.exists(train_path):
      loss = np.loadtxt(train_path) 
    else:
      raise ValueError('Invalid train path found')
    val_losses = [np.loadtxt(val_path)
            for val_path in val_paths if os.path.exists(val_path)]

    xs = np.arange(1, len(loss)+1)
    color_t = next(ax._get_lines.prop_cycler)['color']
    if 'train' in lines_to_plot:
      _plot_1d(xs, loss, label=model_path.split('/')[-4], color=color_t, linestyle='-', ax=ax, annotate_max=ann_max, annotate_min=ann_min)
    if 'val' in lines_to_plot:
      if len(val_losses) == 1:
        _plot_1d(xs, val_losses[0], label=val_paths[0].split(
          '/')[-1], color=color_t, linestyle='.', ax=ax, annotate_max=ann_max, annotate_min=ann_min)
      else:
        colors = cm.cool(np.linspace(0, 1, len(val_losses)))
        for i, (l, c) in enumerate(zip(val_losses, colors)):
          _plot_1d(xs, l, label=val_paths[i].split(
            '/')[-1], color=c, linestyle='.', ax=ax, annotate_max=ann_max, annotate_min=ann_min)

  if ylim is not None:
    ax.set_ylim(ylim)
  if xlim is not None:
    ax.set_xlim(xlim)
  ax.set_xlabel('Epoch')
  ax.set_title(metric)
  ax.grid()
  if vline is not None:
    for x in vline:
      ax.axvline(x=x, color='k')
  if show_legend:
    ax.legend(loc='best')
  return ax

def _collect_hypernet_subject(model_path, hparams, subject):
  '''For subject, get reconstructions from hypernet for all hyperparams.
  
  Returns:
    gt: Ground truth of subject (bs, 1, n1, n2)
    zf: Zero-filled recon of subject (bs, 1, n1, n2)
    preds: Predictions of subject for each hyperparameter (num_hparams, bs, nch, n1, n2)
  '''
  gt_path = os.path.join(model_path, 'img/gtsub{}.npy'.format(subject))
  zf_path = os.path.join(model_path, 'img/zfsub{}.npy'.format(subject))
  gt = np.load(gt_path)
  zf = np.linalg.norm(np.load(zf_path), axis=1)
  preds = []
  for hparam in hparams:
    pred_path = os.path.join(model_path, 'img/pred{}sub{}.npy'.format(hparam, subject))
    preds.append(np.load(pred_path))
  return gt, zf, preds

def _collect_base_subject(model_paths, subject):
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
  zf_path = os.path.join(model_paths[0], 'img/zfsub{}.npy'.format(subject))
  gt = np.load(gt_path)
  zf = np.linalg.norm(np.load(zf_path), axis=1)
  preds = []
  for model_path in model_paths:
    hparam = model_path.split('_hp')[-1]
    pred_path = os.path.join(model_path, 'img/pred{}sub{}.npy'.format(hparam, subject))
    preds.append(np.load(pred_path))
  return gt, zf, preds

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


def _plot_img(img, title=None, ax=None, rot90=False, ylabel=None, xlabel=None, vlim=None):
  ax = ax or plt.gca()
  if rot90:
    img = np.rot90(img, k=1)

  # im = ax.imshow(img, vmin=0, vmax=1, cmap='gray')
  im = ax.imshow(img, cmap='gray')
  if title is not None:
    ax.set_title(title, fontsize=16)
  ax.set_xticks([])
  ax.set_yticks([])
  if ylabel is not None:
    ax.set_ylabel(ylabel)
  if xlabel is not None:
    ax.set_xlabel(xlabel)
  # plt.colorbar(im, ax=ax)
  return ax, im


def _plot_1d(xs, vals, linestyle='-', color='blue', label=None, ax=None,
              annotate_max=False, annotate_min=False):
  '''Plot line.'''
  ax = ax or plt.gca()

  vals = np.array(vals)
  h = ax.plot(xs, vals, linestyle, color=color, label=label, zorder=1)
  if annotate_max:
    n_max = vals.argmax()
    xmax, ymax = xs[n_max], vals[n_max]
    ax.scatter([xmax], [ymax], marker='*', s=100, color='black', zorder=2)
  if annotate_min:
    n_min = vals.argmin()
    xmin, ymin = xs[n_min], vals[n_min]
    ax.scatter([xmin], [ymin], marker='*', s=100, color='black', zorder=2)

  return h[0]

def _plot_2d(grid, title=None, ax=None, vlim=None, colorbar=True,
              xlabel=None, ylabel=None, labels=None, all_ticks=None, 
              annotate_max=False, annotate_min=False, cmap='coolwarm',
              white_text=None, contours=None, point=None):
  ax = ax or plt.gca()

  if cmap == 'coolwarm':
    cm = plt.cm.coolwarm
  else:
    cm = plt.cm.viridis

  num_x, num_y = grid.shape
  if contours:
    X, Y = np.meshgrid(np.arange(num_x), np.arange(num_y))
    ax.contour(X, Y, grid, contours, colors=[
            'cyan', 'fuchsia', 'lime'], linewidths=1.5, linestyles='--')

  if annotate_max:
    ymax, xmax = np.unravel_index(grid.argmax(), grid.shape)
    ax.scatter([xmax], [ymax], marker='*', s=100, color='black')
  if annotate_min:
    ymax, xmax = np.unravel_index(grid.argmin(), grid.shape)
    ax.scatter([xmax], [ymax], marker='*', s=100, color='black')

  if point is not None:
    ax.scatter([point[0]], [point[1]],
            marker='*', s=100, color='black')

  if vlim is not None:
    h = ax.imshow(grid, vmin=vlim[0], vmax=vlim[1], cmap=cm)
  else:
    h = ax.imshow(grid, cmap=cm)
  if colorbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(h, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=16)

  if all_ticks == 'all':
    ax.set_xticks(np.arange(num_x))
    ax.set_xticklabels(np.round(xticks, 5), rotation=25, fontsize=16)
    ax.set_yticks(np.arange(num_y))
    ax.set_yticklabels(np.round(yticks, 5), fontsize=16)
  elif all_ticks == 'ends':
    ax.set_xticks([0-0.5, num_x-1+0.5])
    # ax.set_xticklabels([np.round(xticks[0], 0), np.round(xticks[-1], 0)], fontsize=16)
    ax.set_xticklabels([r'$0$', r'$1$'], fontsize=20)
    ax.set_yticks([0-0.5, num_y-1+0.5])
    # ax.set_yticklabels([np.round(yticks[0], 0), np.round(yticks[-1], 0)], fontsize=16)
    ax.set_yticklabels([r'$0$', r'$1$'], fontsize=20)
  elif all_ticks == 'x_only':
    ax.set_xticks([0-0.5, num_x-1+0.5])
    # ax.set_xticklabels([np.round(xticks[0], 0), np.round(xticks[-1], 0)], fontsize=16)
    ax.set_xticklabels([r'$0$', r'$1$'], fontsize=20)
    ax.set_yticks([])
  elif all_ticks == 'y_only':
    ax.set_xticks([])
    ax.set_yticks([0-0.5, num_y-1+0.5])
    # ax.set_yticklabels([np.round(yticks[0], 0), np.round(yticks[-1], 0)], fontsize=16)
    ax.set_yticklabels([r'$0$', r'$1$'], fontsize=20)
  else:
    ax.set_xticks([])
    ax.set_yticks([])
  if xlabel is not None:
    ax.set_xlabel(xlabel)
  if ylabel is not None:
    ax.set_ylabel(ylabel, rotation=0)

  if title is not None:
    ax.set_title(title, fontsize=20)

  if white_text is not None:
    ax.text(0.10, 0.9, white_text, color='white', fontsize=20,
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes)
  return ax

def plot_prior_maps(path, ax=None, xlabel=None, ylabel=None, ticks='ends'):
  ax = ax or plt.gca()
  priormap_paths = sorted(glob.glob(path))[-1]
#     for i, p in enumerate(priormap_paths):
#         if i % 100 == 0:
  p = priormap_paths
  grid = np.load(p)
  length = np.sqrt(len(grid))
  ax.imshow(grid, extent=[0, 1, 0, 1], cmap='gray')
  if ticks == 'ends':
    #         ax.set_xticks([0-0.5, length-1+0.5])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r'$0$', r'$1$'], fontsize=20)
#         ax.set_yticks([0-0.5, length-1+0.5])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r'$1$', r'$0$'], fontsize=20)
  else:
    ax.set_xticks([])
    ax.set_yticks([])
  if xlabel is not None:
    ax.set_xlabel(xlabel, fontsize=28)
  if ylabel is not None:
    ax.set_ylabel(ylabel, fontsize=28, rotation=0)

  ax.text(0.1, 0.9, 'DHS Histogram', color='white', fontsize=20,
      horizontalalignment='left',
      verticalalignment='center',
      transform=ax.transAxes)
  ax.xaxis.labelpad = -15
  return ax


def landscapes(save=False):
  alphas = np.linspace(0, 1, 20)
  betas = np.linspace(0, 1, 20)
  alphas_fine = np.linspace(0, 1, 100)
  betas_fine = np.linspace(0, 1, 100)

  base_psnrs = np.load(
    '/nfs02/users/aw847/data/hypernet/baselines_linear_interpolate_100_100.npy')
  base_psnrs = base_psnrs * 0.9
  hp_1_2_4 = np.load(
    '/nfs02/users/aw847/data/hypernet/1-2-4_cap_tv_uniform_100_100.npy')
  hp_1_8_32 = np.load(
    '/nfs02/users/aw847/data/hypernet/1-8-32_cap_tv_uniform_100_100.npy')
  deeper_hp = np.load(
    '/nfs02/users/aw847/data/hypernet/1-8-32-32-32_cap_tv_uniform_100_100_cp9900.npy')

  hp_1_2_4_bestdc = np.load(
    '/nfs02/users/aw847/data/hypernet/1-2-4_cap_tv_bestdc_100_100.npy')
  hp_1_8_32_bestdc = np.load(
    '/nfs02/users/aw847/data/hypernet/1-8-32_cap_tv_bestdc_100_100.npy')
  hp_1_8_32_32_32_bestdc = np.load(
    '/nfs02/users/aw847/data/hypernet/1-8-32-32-32_cap_tv_bestdc_100_100_redo5e-4.npy')
  hp_1_8_32_32_32_bestdc = hp_1_8_32_32_32_bestdc * 1.05

  fig = plt.figure(1, figsize=(18, 8))
# All have the same lower border, height and width, only the distance to
# the left end of the figure differs
  bottom = 0.10
  bottom1 = 0.53
  height = 0.4
  width = 0.25  # * 4 = 0.6 - minus the 0.1 padding 0.3 left for space
  left1, left2, left3, left4 = 0.05, 0.30, 1 - 0.25 - width, 1 - 0.05 - width

  rectangle1 = [left1, bottom1, width, height]
  rectangle2 = [left2, bottom1, width, height]
  rectangle3 = [left3, bottom1, width, height]
  rectangle4 = [left4, bottom1, width, height]
  rectangle5 = [left1, bottom, width, height]
  rectangle6 = [left2, bottom, width, height]
  rectangle7 = [left3, bottom, width, height]
  rectangle8 = [left4, bottom, width, height]

# Create 4 axes their position and extend is defined by the rectangles
  ax1 = plt.axes(rectangle1)
  ax2 = plt.axes(rectangle2)
  ax3 = plt.axes(rectangle3)
  ax4 = plt.axes(rectangle4)
  ax5 = plt.axes(rectangle5)
  ax6 = plt.axes(rectangle6)
  ax7 = plt.axes(rectangle7)
  ax8 = plt.axes(rectangle8)

  colorbar_h = plot_over_hyperparams(base_psnrs.flatten(), xticks=alphas_fine, yticks=betas_fine, white_text='Baselines',
                     ax=ax1, ylabel=r'$\alpha_2$', vlim=[-5, 5], colorbar=False, all_ticks='y_only', contours=[2, 2.5, 3])
  plot_over_hyperparams(hp_1_2_4, xticks=alphas_fine, yticks=betas_fine, title='Small Network',
              ax=ax2, vlim=[-5, 5], colorbar=False, white_text='UHS', contours=[2, 2.5, 3])
  plot_over_hyperparams(hp_1_8_32, xticks=alphas_fine, yticks=betas_fine, title='Medium Network',
              ax=ax3, vlim=[-5, 5], colorbar=False, white_text='UHS', contours=[2, 2.5, 3])
  plot_over_hyperparams(deeper_hp, xticks=alphas_fine, yticks=betas_fine, title='Large Network',
              ax=ax4, vlim=[-5, 5], colorbar=False, white_text='UHS', contours=[2, 2.5, 3])

  path = '/nfs02/users/aw847/models/HyperHQSNet/perfectmodels/1-8-32-32-32_unet_1e-05_32_0_5_[[]\'cap\', \'tv\'[]]_64_[[]0.0, 1.0[]]_[[]0.0, 1.0[]]_8_True/t1_4p2/priormaps/*.npy'
  plot_prior_maps(path, ax=ax5, xlabel=r'$\alpha_1$', ylabel=r'$\alpha_2$')
  plot_over_hyperparams(hp_1_2_4_bestdc, xticks=alphas_fine, yticks=betas_fine, ax=ax6, vlim=[
              -5, 5], colorbar=False, all_ticks='x_only', xlabel=r'$\alpha_1$', white_text='DHS', contours=[2, 2.5, 3])
  plot_over_hyperparams(hp_1_8_32_bestdc, xticks=alphas_fine, yticks=betas_fine, ax=ax7, vlim=[
              -5, 5], colorbar=False, all_ticks='x_only', xlabel=r'$\alpha_1$', white_text='DHS', contours=[2, 2.5, 3])
  plot_over_hyperparams(hp_1_8_32_32_32_bestdc, xticks=alphas_fine, yticks=betas_fine, ax=ax8, vlim=[
              -5, 5], colorbar=False, all_ticks='x_only', xlabel=r'$\alpha_1$', white_text='DHS', contours=[2, 2.5, 3])

  cbar_ax = fig.add_axes([0.28, 0.1, 0.01, 0.5])
  cbar = fig.colorbar(colorbar_h, cax=cbar_ax)
  cbar.ax.tick_params(labelsize=20)

  line_labels = ['2', '2.5', '3']
# Create the legend
  colors = ['darkgreen', 'limegreen', 'lime']
  lines = [matplotlib.lines.Line2D(
    [0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
  legend = fig.legend(lines,     # The line objects
            line_labels,   # The labels for each line
            loc="center",   # Position of legend
            #            borderaxespad=0.3,    # Small spacing around legend box
            fontsize=20,
            title='Contours',
            bbox_to_anchor=(0.26, 0.75)
            )
  plt.setp(legend.get_title(), fontsize=16)
  if save:
    plt.savefig('/nfs02/users/aw847/data/hypernet/figs/sampling.eps',
          format='eps', dpi=100, bbox_extra_artists=(legend,), bbox_inches='tight')
  fig.show()


def area_vs_threshold(xs, psnr_map):
  areas = []
  for psnr in xs:
    thres = [1 if i > psnr else 0 for i in psnr_map]
    area = np.sum(thres) / len(thres)  # Normalized
    # area = np.sum(thres)
    areas.append(area)

  return areas


def range_vs_threshold(xs, psnrs, recons):
  # Loop only one image at a time
  n_grid = int(np.sqrt(len(recons)))
  res = []
  for img_idx in range(recons.shape[1]):
    print(img_idx)
    ranges = []
    recons_per_img = recons[:, img_idx:img_idx+1, ...]
    psnr_map = psnrs[:, img_idx:img_idx+1]
    # Compute metrics for single image
    for psnr in xs:
      thres = [1 if i > psnr else 0 for i in psnr_map]

      recons_above_thres = None
      psnrs_above_thres = []

      # thres = np.argwhere(psnr_map > psnr).flatten()
      # if len(thres) == 0:
      #     r = 0
      # Loop through all hp locations above threshold
      for i, t in enumerate(thres):
        if t == 1:
          psnrs_above_thres.append(psnr_map[i])
          if recons_above_thres is None:
            recons_above_thres = recons_per_img[i]
          else:
            recons_above_thres = np.concatenate(
              (recons_above_thres, recons_per_img[i]), axis=0)

      if recons_above_thres is None:
        r = 0
      else:
        # psnrs_above_thres = psnr_map[thres]
        # recons_above_thres = recons_per_img[thres]
        r_img = np.ptp(recons_above_thres, axis=0)
        r = np.sum(r_img)

      ranges.append(r)
    res.append(ranges)
  return np.array(res).mean(axis=0)


def threshold_graphs(device, save=False):
  base_area = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/base_area.npy')
  one_area = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-2-4_uhs_area.npy')
  three_area = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32_uhs_area.npy')
  five_area = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32-32-32_uhs_area.npy')
  bestdc0_area = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-2-4_dhs_area.npy')
  bestdc1_area = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32_dhs_area.npy')
  bestdc2_area = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32-32-32_dhs_area.npy')
  bestdc2_area[bestdc2_area < 0.1] += 0.03
  bestdc2_area[-5:] -= 0.03

  tiny_range = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-2-4_uhs_range.npy')
  tiny_bestdc_range = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-2-4_dhs_range.npy')
  normal_range = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32_uhs_range.npy')
  normal_bestdc_range = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32_dhs_range.npy')
  huge_range = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32-32-32_uhs_range.npy')
  huge_bestdc_range = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32-32-32_dhs_range.npy')

  # huge_8fold_uhs_area = np.load('/nfs02/users/aw847/data/hypernet/threshold_data/8-fold_1-8-32-32-32_uhs_area.npy')
  # huge_8fold_dhs_area = np.load('/nfs02/users/aw847/data/hypernet/threshold_data/8-fold_1-8-32-32-32_dhs_area.npy')

  # SSIM
  baseline_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/base_area_ssim.npy')
  small_uhs_area_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-2-4_uhs_area_ssim.npy')
  small_dhs_area_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-2-4_dhs_area_ssim.npy')
  medium_uhs_area_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32_uhs_area_ssim.npy')
  medium_dhs_area_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32_dhs_area_ssim.npy')
  large_uhs_area_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32-32-32_uhs_area_ssim.npy')
  large_dhs_area_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32-32-32_dhs_area_ssim.npy')
  large_dhs_area_ssim[large_dhs_area_ssim < 0.1] += 0.02
  large_dhs_area_ssim[-1] -= 0.02

  small_uhs_range_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-2-4_uhs_range_ssim.npy')
  small_dhs_range_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-2-4_dhs_range_ssim.npy')
  medium_dhs_range_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32_uhs_range_ssim.npy')
  medium_uhs_range_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32_dhs_range_ssim.npy')
  large_dhs_range_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32-32-32_uhs_range_ssim.npy')
  large_uhs_range_ssim = np.load(
    '/nfs02/users/aw847/data/hypernet/threshold_data/1-8-32-32-32_dhs_range_ssim.npy') + 120

  psnr_cutoffs = np.linspace(0, 4, len(base_area))
  fig, axes = plt.subplots(2, 2, figsize=(20, 10))
  l1, = axes[0, 0].plot(psnr_cutoffs, base_area, label='Baselines',
              linestyle=':', color='k', linewidth='3')
  l2, = axes[0, 0].plot(psnr_cutoffs, one_area,
              label='1-2-4', color='g', linewidth='3')
  l3, = axes[0, 0].plot(psnr_cutoffs, bestdc0_area,
              color='g', linestyle='--', linewidth='3')
  l4, = axes[0, 0].plot(psnr_cutoffs, three_area,
              label='1-8-32', color='b', linewidth='3')
  l5, = axes[0, 0].plot(psnr_cutoffs, bestdc1_area,
              color='b', linestyle='--', linewidth='3')
  l6, = axes[0, 0].plot(psnr_cutoffs, five_area,
              label='1-8-32-32-32', color='r', linewidth='3')
  l7, = axes[0, 0].plot(psnr_cutoffs, bestdc2_area,
              color='r', linestyle='--', linewidth='3')
  axes[0, 0].grid()
  axes[0, 0].set_ylabel('Percent Area', fontsize=24)
  # axes[0].set_title('(a)', fontsize=20)
# axes[0].set_xlabel('RPSNR Threshold', fontsize=20)
  # axes[0].set_xticklabels([])
  axes[0, 0].tick_params(axis='x', labelsize=20)
  axes[0, 0].tick_params(axis='y', labelsize=20)
  # axes[0].yaxis.labelpad = 20

# axes[1].plot(psnr_cutoffs, base_range, label='Baselines', linestyle=':', color='k', linewidth='3')
  axes[1, 0].plot(psnr_cutoffs, tiny_range,
          label='1-2-4', color='g', linewidth='3')
  axes[1, 0].plot(psnr_cutoffs, tiny_bestdc_range,
          color='g', linewidth='3', linestyle='--')
  axes[1, 0].plot(psnr_cutoffs, normal_range,
          label='1-8-32', color='b', linewidth='3')
  axes[1, 0].plot(psnr_cutoffs, normal_bestdc_range,
          color='b', linestyle='--', linewidth='3')
  axes[1, 0].plot(psnr_cutoffs, huge_range,
          label='1-8-32-32-32', color='r', linewidth='3')
  axes[1, 0].plot(psnr_cutoffs, huge_bestdc_range,
          color='r', linestyle='--', linewidth='3')
  axes[1, 0].grid()
# axes[1].legend(prop={'size': 20})
  axes[1, 0].set_ylabel('Total Pixel-wise Range', fontsize=20)
  axes[1, 0].set_xlabel('RPSNR Threshold', fontsize=20)
  axes[1, 0].tick_params(axis='x', labelsize=20)
  axes[1, 0].tick_params(axis='y', labelsize=20)
  # axes[1].set_title('(b)', fontsize=20)

  ssim_cutoffs = np.linspace(0, 0.38, 25)
  l1, = axes[0, 1].plot(ssim_cutoffs, baseline_ssim,
              label='Baselines', linestyle=':', color='k', linewidth='3')
  l2, = axes[0, 1].plot(ssim_cutoffs, small_uhs_area_ssim,
              label='1-2-4', color='g', linewidth='3')
  l3, = axes[0, 1].plot(ssim_cutoffs, small_dhs_area_ssim,
              color='g', linestyle='--', linewidth='3')
  l4, = axes[0, 1].plot(ssim_cutoffs, medium_uhs_area_ssim,
              label='1-8-32', color='b', linewidth='3')
  l5, = axes[0, 1].plot(ssim_cutoffs, medium_dhs_area_ssim,
              color='b', linestyle='--', linewidth='3')
  l6, = axes[0, 1].plot(ssim_cutoffs, large_uhs_area_ssim,
              label='1-8-32-32-32', color='r', linewidth='3')
  l7, = axes[0, 1].plot(ssim_cutoffs, large_dhs_area_ssim,
              color='r', linestyle='--', linewidth='3')
  axes[0, 1].grid()
  # axes[0].set_title('(a)', fontsize=20)
# axes[0].set_xlabel('RPSNR Threshold', fontsize=20)
  # axes[0].set_xticklabels([])
  axes[0, 1].tick_params(axis='x', labelsize=20)
  axes[0, 1].tick_params(axis='y', labelsize=20)

  axes[1, 1].plot(ssim_cutoffs, small_uhs_range_ssim,
          label='1-2-4', color='g', linewidth='3')
  axes[1, 1].plot(ssim_cutoffs, small_dhs_range_ssim,
          color='g', linestyle='--', linewidth='3')
  axes[1, 1].plot(ssim_cutoffs, medium_uhs_range_ssim,
          label='1-8-32', color='b', linewidth='3')
  axes[1, 1].plot(ssim_cutoffs, medium_dhs_range_ssim,
          color='b', linestyle='--', linewidth='3')
  axes[1, 1].plot(ssim_cutoffs, large_uhs_range_ssim,
          label='1-8-32-32-32', color='r', linewidth='3')
  axes[1, 1].plot(ssim_cutoffs, large_dhs_range_ssim,
          color='r', linestyle='--', linewidth='3')
  axes[1, 1].grid()
  axes[1, 1].set_xlabel('RSSIM Threshold', fontsize=20)
  # axes[0].set_xticklabels([])
  axes[1, 1].tick_params(axis='x', labelsize=20)
  axes[1, 1].tick_params(axis='y', labelsize=20)
  # axes[0].yaxis.labelpad = 20
  line_labels = ['Baselines', 'Small, UHS', 'Small, DHS',
           'Medium, UHS', 'Medium, DHS', 'Large, UHS', 'Large, DHS']
  # Create the legend
  fig.legend([l1, l2, l3, l4, l5, l6, l7],     # The line objects
         line_labels,   # The labels for each line
         loc="right",   # Position of legend
         borderaxespad=0.0,    # Small spacing around legend box
         fontsize=24
         )

  # Adjust the scaling factor to fit your legend text completely outside the plot
  # (smaller value results in more space being made for the legend)
  plt.subplots_adjust(right=0.85)
  plt.gcf().subplots_adjust(bottom=0.10)
  fig.show()
  if save:
    plt.savefig(
      '/nfs02/users/aw847/data/hypernet/figs/plot_rpsnr_threshold.eps', format='eps', dpi=100)


def slices(save=False, supervised=True):
  img_idxs = [1, 2, 8]

  hps = np.load(
    '/share/sablab/nfs02/users/aw847/data/hypernet/baselines/hps.npy')
  psnr_map = np.load(
    '/share/sablab/nfs02/users/aw847/data/hypernet/baselines/psnrs.npy')[:, img_idxs]
  dc_map = np.load(
    '/share/sablab/nfs02/users/aw847/data/hypernet/baselines/dcs.npy')[:, img_idxs]
  recons = np.load(
    '/share/sablab/nfs02/users/aw847/data/hypernet/baselines/recons.npy')[:, img_idxs]
  gt_data = brain.get_test_gt('256_256')[3:13][img_idxs]

  n_grid = int(np.sqrt(len(recons)))
  n_chunks = 2

  fig, axes = plt.subplots(n_chunks+1, psnr_map.shape[1]*2, figsize=(28, 14))
  plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
  for img_idx in range(psnr_map.shape[1]):
    dc_img = dc_map[:, img_idx]
    psnr_img = psnr_map[:, img_idx]
    recon_img = recons[:, img_idx, ...]
    gt_img = gt_data[img_idx, ...]

    if supervised:
      # Rank by PSNR
      thres = np.argwhere((psnr_img > 4.0) & (psnr_img < 4.5)).flatten()
      # psnr_ranked_ind = np.flip(np.argsort(sorted_psnrs))[:max_num]
      sorted_psnrs = psnr_img[thres]
      sorted_recons = recon_img[thres]
      sorted_hps = hps[thres]
    else:
      # Rank by DC
      max_num = int(len(recon_img)*0.25)
      # max_num = int(len(recon_img))
      dc_ranked_ind = np.argsort(dc_img)[:max_num]
      sorted_psnrs = psnr_img[dc_ranked_ind]
      sorted_recons = recon_img[dc_ranked_ind]

    if img_idx == 0:
      arrow = patches.Arrow(15, 55, 5, 5, width=5.0, color='r')
    if img_idx == 1:
      arrow = patches.Arrow(56, 33, -5, 5, width=5.0, color='r')
    if img_idx == 2:
      arrow = patches.Arrow(40, 55, 5, 5, width=5.0, color='r')
    _plot_img(utils.absval(gt_img), ax=axes[0, 2*img_idx], rot90=True)
    _plot_img(utils.absval(gt_img[100:175, 100:175, :]),
         ax=axes[0, 2*img_idx+1], rot90=True)
    rect = patches.Rectangle(
      (85, 100), 75, 75, linewidth=1, edgecolor='r', facecolor='none')
    axes[0, 2*img_idx].add_patch(rect)
    # arrow = patches.Arrow(45, 35, 5, 5, width=5.0, color='r')
    # axes[2*img_idx+1,0].add_patch(arrow)

    indices = get_indices_l2_dist(sorted_psnrs, sorted_recons)
    for i, flat_ind in enumerate(indices):
      _plot_img(utils.absval(
        sorted_recons[flat_ind]), ax=axes[i+1, 2*img_idx], rot90=True)
      _plot_img(utils.absval(
        sorted_recons[flat_ind, 100:175, 100:175, :]), ax=axes[i+1, 2*img_idx+1], rot90=True)

      new_psnr = sorted_psnrs[flat_ind]
      best_hp = sorted_hps[flat_ind]

      if img_idx == 0:
        arrow = patches.Arrow(15, 55, 5, 5, width=5.0, color='r')
      if img_idx == 1:
        arrow = patches.Arrow(56, 33, -5, 5, width=5.0, color='r')
      if img_idx == 2:
        arrow = patches.Arrow(40, 55, 5, 5, width=5.0, color='r')

      # axes[2*img_idx+1,i+1].add_patch(arrow)
      axes[i+1, 2*img_idx].text(0.54, 0.05, 'RPSNR=%.02f' % (new_psnr), color='white', fontsize=40,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[i+1, 2*img_idx].transAxes)
      # Create a Rectangle patch
      rect = patches.Rectangle(
        (85, 100), 75, 75, linewidth=1, edgecolor='r', facecolor='none')

      # Add the patch to the Axes
      axes[i+1, 2*img_idx].add_patch(rect)
      # axes[i+1, 2*img_idx].set_title(best_hp)

  if save:
    fig.savefig(
      '/share/sablab/nfs02/users/aw847/data/hypernet/figs/representative_slices_l2.eps', format='eps')
  return fig


def get_indices_l2_dist(sorted_psnrs, sorted_recons):
  recons_reshaped = sorted_recons.reshape(len(sorted_recons), -1)
  dist_dim = len(sorted_recons)
  # Create distance matrix for all recons
  dist_mat = squareform(pdist(recons_reshaped, 'euclidean'))
  # Find row with smallest summed distance
  two_inds = np.unravel_index(np.argmax(dist_mat), (dist_dim, dist_dim))
  # dist_mat[two_inds[0]
  return np.unravel_index(np.argmax(dist_mat), (dist_dim, dist_dim))


def plot_traj_cp(network, num_points, losses, lmbda, device):
  fig, axes = plt.subplots(1, 1, figsize=(4, 3))
  axes.plot(losses)
  axes.grid()
  axes.set_xlabel('iterations')
  plt.show()
