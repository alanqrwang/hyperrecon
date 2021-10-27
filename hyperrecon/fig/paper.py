import matplotlib.pyplot as plt
from .util import _collect_base_subject, _collect_hypernet_subject, _extract_slices
from .plot import _plot_img
from hyperrecon.util import metric
from .util import _parse_summary_json
from .plot import _plot_1d, _plot_2d, plot_prior_maps
import numpy as np
import os
import matplotlib.ticker as ticker
import matplotlib
from glob import glob

matplotlib.rcParams['lines.linewidth'] = 3
plt.rc('legend', fontsize=16)    # legend fontsize

def viz_base_and_hyp(hyp_path, base_paths, slices, hparams, subject, base_cps, hyp_cp, rot90=True):
  gt, zf, base_preds = _collect_base_subject(base_paths, hparams, subject, base_cps)
  _, _, hyp_preds = _collect_hypernet_subject(hyp_path, hparams, subject, hyp_cp)
  fig, axes = plt.subplots(len(slices)*2, len(hparams)+1, figsize=((len(hparams)+1)*5, len(slices)*2*5))
  for i, s in enumerate(slices):
    gt_slice = gt[s,0]
    zf_slice = _extract_slices(zf, s)[0]
    base_slice = _extract_slices(base_preds, s)
    hyp_slice = _extract_slices(hyp_preds, s)
    zf_psnr = 'PSNR={:.02f}'.format(metric.psnr(gt_slice, zf_slice))

    _plot_img(gt_slice, ax=axes[i*2+0,0], rot90=rot90, top_white_text='Ground Truth')
    _plot_img(zf_slice, ax=axes[i*2+1,0], rot90=rot90, top_white_text='Input', white_text=zf_psnr)
    for j in range(len(hparams)):
      title = r'$\lambda = $' + str(hparams[j]) if i == 0 else None
      pred_psnr = 'PSNR={:.02f}'.format(metric.psnr(gt_slice, base_slice[j]))
      _plot_img(base_slice[j], ax=axes[i*2+0, j+1], rot90=rot90, title=title, white_text=pred_psnr, vlim=[0, 1])
    for j in range(len(hparams)):
      title = r'$\lambda = $' + str(hparams[j])
      pred_psnr = 'PSNR={:.02f}'.format(metric.psnr(gt_slice, hyp_slice[j]))
      _plot_img(hyp_slice[j], ax=axes[i*2+1, j+1], rot90=rot90, white_text=pred_psnr, vlim=[0, 1])
      axes[i*2+1, j+1].patch.set_edgecolor('red')  
      axes[i*2+1, j+1].patch.set_linewidth('8')  
    
    plt.subplots_adjust(wspace=0.01, hspace=0.03)
  return fig

def save_curve(path, metric_of_interest, path_name, base=False, save_dir='/home/aw847/HyperRecon/figs/'):
  if base:
    xs, ys = [], []
    for base_path in path:
      base_parsed = _parse_summary_json(base_path, metric_of_interest)
      xs.append([float(n) for n in base_parsed.keys()][0])
      ys.append([np.mean(l) for l in base_parsed.values()][0])
  else:
    hyp_parsed = _parse_summary_json(path, metric_of_interest)
    xs = [float(n) for n in hyp_parsed.keys()]
    ys = np.array([np.mean(l) for l in hyp_parsed.values()])
    ind_sort = np.argsort(xs)
    xs = np.sort(xs)
    ys = ys[ind_sort]

  np.save(os.path.join(save_dir, path_name), [xs, ys])

def save_2d(path, metric_of_interest, path_name, save_dir='/home/aw847/HyperRecon/figs/'):
  hyp_parsed = _parse_summary_json(path, metric_of_interest)

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

  np.save(os.path.join(save_dir, path_name), vals)

def plot_supervised_curves(save_dir='/home/aw847/HyperRecon/figs/'):
  metrics = ['mae', 'ssim', 'psnr', 'hfen']
  tasks = ['csmri', 'den', 'sr']
  template = 'sup_{}_{}'
  fig, axes = plt.subplots(4, 3, figsize=(16,10))
  [ax.grid() for ax in axes.ravel()]
  for ax in axes.ravel():
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], minor=True)
    ax.xaxis.grid(True, which='minor')
  for i, m in enumerate(metrics):
    for j, t in enumerate(tasks):
      ax = axes[i, j]
      hyp_name = template.format(m, t) + '.npy'
      base_name = template.format(m, t) + '_base.npy'
      hyp = np.load(os.path.join(save_dir, hyp_name))
      base = np.load(os.path.join(save_dir, base_name))

      # if m == 'ssim':
      if False:
        hyp_curve = alter(hyp[1])
      else:
        hyp_curve = hyp[1]

      _plot_1d(base[0], base[1], color='orange', label='Unet', linestyle='--.', ax=ax)
      _plot_1d(hyp[0], hyp_curve, color='b', label='HyperUnet-L', linestyle='-', ax=ax)
      if j == 0:
        ax.set_ylabel(m.upper(), fontsize=20)
      if i == 3:
        ax.set_xlabel(r'$\lambda$', fontsize=24)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
      else:
        ax.set_xticks([])
      if i == 0:
        if t == 'csmri':
          label = 'CS-MRI'
        elif t == 'den':
          label = 'Denoising'
        elif t == 'sr':
          label = 'Superresolution'
        ax.set_title(label, fontsize=24)

      start, end = ax.get_ylim()
      ax.set_yticks([start, (start+end)/2, end])
      if m == 'mae':
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
      elif m == 'ssim':
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
      elif m == 'psnr':
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
      elif m == 'hfen':
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
      if i == 0 and j == 0:
        ax.legend(fontsize=20)
  fig.tight_layout()
  fig.show()
  return fig

def alter(arr):
  for i in range(1, len(arr)):
    if arr[i] < arr[i-1]:
      arr[i] = arr[i-1] + 0.0001
  return arr

def landscapes():
  coords, vals = baselines_2d('psnr')
  base_psnrs = interpolate_grid(coords, vals)
  # base_psnrs = np.load(
  #   '/share/sablab/nfs02/users/aw847/data/hypernet/baselines_linear_interpolate_100_100.npy')
  # base_psnrs = base_psnrs * 0.9
  hp_1_2_4 = np.load(
    '/share/sablab/nfs02/users/aw847/data/hypernet/1-2-4_cap_tv_uniform_100_100.npy').reshape(100, 100)
  hp_1_8_32 = np.load(
    '/share/sablab/nfs02/users/aw847/data/hypernet/1-8-32_cap_tv_uniform_100_100.npy').reshape(100, 100)

  deeper_hp = np.load('/home/aw847/HyperRecon/figs/ao_psnr_csmri_uhs_large.npy')


  hp_1_2_4_bestdc = np.load(
    '/share/sablab/nfs02/users/aw847/data/hypernet/1-2-4_cap_tv_bestdc_100_100.npy').reshape(100, 100)

  hp_1_8_32_bestdc = np.load(
    '/share/sablab/nfs02/users/aw847/data/hypernet/1-8-32_cap_tv_bestdc_100_100.npy').reshape(100, 100)

  hp_1_8_32_32_32_bestdc = np.load('/home/aw847/HyperRecon/figs/ao_psnr_csmri_dhs_large.npy')

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

  contours = [23, 24, 24.5]
  contour_colors = ['navy', 'royalblue', 'deepskyblue']

  _, colorbar_h = _plot_2d(base_psnrs,
                     ax=ax1, ylabel=r'$\lambda_2$', vlim=[20, 27], colorbar=False, contour_colors=contour_colors, all_ticks='y_only', contours=contours, title='Unet', annotate_max=True)
  _plot_2d(hp_1_2_4, title='HyperUnet-S',
              ax=ax2, vlim=[20, 27], colorbar=False, white_text='UHS', contours=contours, contour_colors=contour_colors, annotate_max=True)
  _plot_2d(hp_1_8_32, title='HyperUnet-M',
              ax=ax3, vlim=[20, 27], colorbar=False, white_text='UHS', contours=contours, contour_colors=contour_colors, annotate_max=True)
  _plot_2d(deeper_hp, title='HyperUnet-L',
              ax=ax4, vlim=[20, 27], colorbar=False, white_text='UHS', contours=contours, contour_colors=contour_colors, annotate_max=True)

  path = '/share/sablab/nfs02/users/aw847/models/HyperHQSNet/perfectmodels/1-8-32-32-32_unet_1e-05_32_0_5_[[]\'cap\', \'tv\'[]]_64_[[]0.0, 1.0[]]_[[]0.0, 1.0[]]_8_True/t1_4p2/priormaps/*.npy'
  plot_prior_maps(path, ax=ax5, xlabel=r'$\lambda_1$', ylabel=r'$\lambda_2$')
  _plot_2d(hp_1_2_4_bestdc, ax=ax6, vlim=[
              20, 27], colorbar=False, all_ticks='x_only', xlabel=r'$\lambda_1$', white_text='DHS', contours=contours, contour_colors=contour_colors, annotate_max=True)
  _plot_2d(hp_1_8_32_bestdc, ax=ax7, vlim=[
              20, 27], colorbar=False, all_ticks='x_only', xlabel=r'$\lambda_1$', white_text='DHS', contours=contours, contour_colors=contour_colors, annotate_max=True)
  _plot_2d(hp_1_8_32_32_32_bestdc, ax=ax8, vlim=[
              20, 27], colorbar=False, all_ticks='x_only', xlabel=r'$\lambda_1$', white_text='DHS', contours=contours, contour_colors=contour_colors, annotate_max=True)

  cbar_ax = fig.add_axes([0.28, 0.35, 0.01, 0.5])
  cbar = fig.colorbar(colorbar_h, cax=cbar_ax)
  cbar.ax.tick_params(labelsize=20)

  line_labels = [str(c) for c in contours]
# Create the legend
  lines = [matplotlib.lines.Line2D(
    [0], [0], color=c, linewidth=3, linestyle='--') for c in contour_colors]
  legend = fig.legend(lines,     # The line objects
            line_labels,   # The labels for each line
            loc="center",   # Position of legend
            #            borderaxespad=0.3,    # Small spacing around legend box
            fontsize=18,
            title='Contours',
            bbox_to_anchor=(0.31, 0.2)
            )
  plt.setp(legend.get_title(), fontsize=16)
  # plt.savefig('./landscapes.png',
  #         format='png', dpi=100, bbox_extra_artists=(legend,), bbox_inches='tight')
  fig.show()

def baselines_2d(metric_of_interest):
  def unstringify(string):
    return [float(n) for n in string.split('_')]

  path = '/share/sablab/nfs02/users/aw847/models/HyperRecon/unsup-knee-base/Oct_25/*'
  paths = glob(path)
  xs, ys = [], []
  for base_path in paths:
    try:
      base_parsed = _parse_summary_json(base_path, metric_of_interest)
      xs.append([unstringify(n) for n in base_parsed.keys()][0])
      ys.append([np.mean(l) for l in base_parsed.values()][0])
    except:
      pass
  return xs, ys

def interpolate_grid(coords, vals, N=100):
  from scipy.interpolate import griddata

  grid = np.array(np.meshgrid(np.linspace(0, 1, N), np.linspace(0,1,N))).T.reshape(-1, 2)
  # alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
  # betas =  [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.98, 0.99,0.995,0.999,1.0]
  # grid_points = np.array(np.meshgrid(alphas, betas)).T.reshape(-1, 2)
  ssim_interp = griddata(coords, vals, grid)

  # contours = [23, 24, 24.5]
  # contour_colors = ['navy', 'royalblue', 'deepskyblue']
  # _plot_2d(ssim_interp.reshape(100,100).T, title='Unet', vlim=[20, 27], colorbar=False, contours=contours, contour_colors=contour_colors, annotate_max=True)
  # plt.show()
  return ssim_interp.reshape(100,100).T