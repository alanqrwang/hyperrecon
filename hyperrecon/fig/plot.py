import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from .util import _parse_summary_json, extract_kernel_layer
from matplotlib.pyplot import cm
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob

# global settings for plotting
matplotlib.rcParams['lines.linewidth'] = 2
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

def plot_over_hyperparams(path, metric_of_interest, label, flip=False, ax=None, ylim=None, base=False, color='blue'):
  ax = ax or plt.gca()
  if metric_of_interest in ['psnr', 'ssim', 'dice']:
    ann_min, ann_max = False, True
  elif metric_of_interest in ['loss', 'hfen']:
    ann_min, ann_max = True, False

  if base:
    xs, ys = [], []
    for base_path in path:
      base_parsed = _parse_summary_json(base_path, metric_of_interest)
      xs.append([float(n) for n in base_parsed.keys()][0])
      ys.append([np.mean(l) for l in base_parsed.values()][0])
    linestyle='.--'
  else:
    hyp_parsed = _parse_summary_json(path, metric_of_interest)
    xs = [float(n) for n in hyp_parsed.keys()]
    ys = np.array([np.mean(l) for l in hyp_parsed.values()])
    linestyle='-'

  if flip:
    ys = 1 - ys

  _plot_1d(xs, ys, color=color, label=label, linestyle=linestyle, annotate_min=ann_min, annotate_max=ann_max, ax=ax)
  ax.set_title(metric_of_interest)
  ax.set_xlabel('alpha')
  if ylim is not None:
    ax.set_ylim(ylim)
  ax.legend()
  ax.grid()

def plot_over_hyperparams_per_subject(model_path, base_paths, metric_of_interest, flip=False, ax=None, ylim=None):
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
  if ylim is not None:
    ax.set_ylim(ylim)
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

def plot_monitor(monitor, model_paths, ax=None, ylim=None, xlim=None, labels=None):
  ax = ax or plt.gca()
  if not isinstance(model_paths, list):
    model_paths = [model_paths]
  if not isinstance(labels, (tuple, list)):
    labels = [labels]
  
  if labels[0] is None:
    labels = ['Line %d' % n for n in range(len(model_paths))]

  for i, model_path in enumerate(model_paths):
    color_t = next(ax._get_lines.prop_cycler)['color']
    train_path = os.path.join(model_path, 'monitor', '{}.txt'.format(monitor))

    if os.path.exists(train_path):
      loss = np.loadtxt(train_path) 
    else:
      raise ValueError('Invalid train path found')
    _plot_1d(np.arange(len(loss)), loss, label=labels[i], ax=ax, color=color_t)

  if ylim is not None:
    ax.set_ylim(ylim)
  if xlim is not None:
    ax.set_xlim(xlim)
  ax.set_xlabel('Epoch')
  ax.set_title(monitor)
  ax.legend()
  ax.grid()
  return ax

def plot_metrics(metric, model_paths,
         show_legend=True, xlim=None, ylim=None, lines_to_plot=('train', 'val'), vline=None, ax=None, labels=None):
  ax = ax or plt.gca()
  if not isinstance(model_paths, list):
    model_paths = [model_paths]
  if not isinstance(lines_to_plot, (tuple, list)):
    lines_to_plot = (lines_to_plot)
  if not isinstance(labels, (tuple, list)):
    labels = [labels]
  
  if labels[0] is None:
    labels = ['Line %d' % n for n in range(len(model_paths))]
  assert len(labels) == len(model_paths), 'labels do not match model paths'

  if metric in ['psnr', 'ssim', 'dice']:
    ann_min, ann_max = False, True
  elif metric in ['loss', 'hfen']:
    ann_min, ann_max = True, False

  for i, model_path in enumerate(model_paths):
    train_path = os.path.join(model_path, 'metrics', '{}:train.txt'.format(metric))
    val_paths = glob(os.path.join(
      model_path, 'metrics', '{}:val*.txt'.format(metric)))

    if os.path.exists(train_path):
      loss = np.loadtxt(train_path) 
      print('num epochs:', len(loss))
    else:
      raise ValueError('Invalid train path found')
    val_losses = [np.loadtxt(val_path)
            for val_path in val_paths if os.path.exists(val_path)]

    xs = np.arange(1, len(loss)+1)
    color_t = next(ax._get_lines.prop_cycler)['color']
    if 'train' in lines_to_plot:
      _plot_1d(xs, loss, label=labels[i], color=color_t, linestyle='-', ax=ax, annotate_max=ann_max, annotate_min=ann_min)
    if 'val' in lines_to_plot:
      if len(val_losses) == 0:
        continue
      if len(val_losses) == 1:
        print('last loss:', val_losses[0][-1])
        _plot_1d(xs, val_losses[0], label=val_paths[0].split(
          '/')[-1], color=color_t, linestyle='--', ax=ax, annotate_max=ann_max, annotate_min=ann_min)
      else:
        colors = cm.cool(np.linspace(0, 1, len(val_losses)))
        for i, (l, c) in enumerate(zip(val_losses, colors)):
          _plot_1d(xs, l, label=val_paths[i].split(
            '/')[-1], color=c, linestyle='--', ax=ax, annotate_max=ann_max, annotate_min=ann_min)

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
  print()
  return ax

def _plot_img(img, title=None, ax=None, rot90=False, ylabel=None, xlabel=None, vlim=None, colorbar=False):
  ax = ax or plt.gca()
  if rot90:
    img = np.rot90(img, k=1)

  if vlim is not None:
    im = ax.imshow(img, vmin=vlim[0], vmax=vlim[1], cmap='gray')
  else:
    im = ax.imshow(img, cmap='gray')
  if title is not None:
    ax.set_title(title, fontsize=16)
  ax.set_xticks([])
  ax.set_yticks([])
  if ylabel is not None:
    ax.set_ylabel(ylabel)
  if xlabel is not None:
    ax.set_xlabel(xlabel)
  if colorbar:
    plt.colorbar(im, ax=ax)
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

  
  if all_ticks == 'ends':
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
  priormap_paths = sorted(glob(path))[-1]
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

def plot_mae_over_layer_idx(path, base=False, dim=64):
  maes = []
  layer_idxs = np.arange(14)
  for layer_idx in layer_idxs:
    if base:
      kernels0 = extract_kernel_layer(path[0], None, layer_idx, arch='baseline')
      kernels1 = extract_kernel_layer(path[1], None, layer_idx, arch='baseline')
    else:
      kernels0 = extract_kernel_layer(path, [1., 0.], layer_idx, dim, arch='hyperunet')
      kernels1 = extract_kernel_layer(path, [0., 1.], layer_idx, dim, arch='hyperunet')

    mae = [(k0-k1).abs() for k0, k1 in zip(kernels0, kernels1)]
    mae_avg = [m.mean().cpu().detach().numpy() for m in mae]
    maes.append(np.array(mae_avg).mean())
  
  return maes