import matplotlib.pyplot as plt
from .util import _collect_base_subject, _collect_hypernet_subject, _extract_slices
from .plot import _plot_img
from hyperrecon.util import metric
from .util import _parse_summary_json
from .plot import _plot_1d
import numpy as np
import os
import matplotlib.ticker as ticker

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

def plot_supervised_curves(save_dir='/home/aw847/HyperRecon/figs/'):
  metrics = ['mae', 'ssim', 'psnr', 'hfen']
  tasks = ['csmri', 'den', 'sr']
  template = 'sup_{}_{}'
  fig, axes = plt.subplots(3, 4, figsize=(16,10))
  # [ax.grid() for ax in axes.ravel()]
  for i, t in enumerate(tasks):
    for j, m in enumerate(metrics):
      ax = axes[i, j]
      hyp_name = template.format(m, t) + '.npy'
      base_name = template.format(m, t) + '_base.npy'
      hyp = np.load(os.path.join(save_dir, hyp_name))
      base = np.load(os.path.join(save_dir, base_name))

      if m in ['psnr', 'ssim', 'dice']:
        ann_min, ann_max = False, True
      elif m in ['loss', 'hfen', 'mae']:
        ann_min, ann_max = True, False
      ax.plot(hyp[0], hyp[1])
      _plot_1d(base[0], base[1], color='orange', label='Unet', linestyle='--.', annotate_min=ann_min, annotate_max=ann_max, ax=ax)
      _plot_1d(hyp[0], hyp[1], color='b', label='HyperUnet', linestyle='-', annotate_min=ann_min, annotate_max=ann_max, ax=ax)
      if i == 0:
        ax.set_title(m.upper())
      if i == 2:
        ax.set_xlabel(r'$\lambda$', fontsize=20)
      else:
        ax.set_xticks([])
      if j == 0:
        if t == 'csmri':
          label = 'CS-MRI'
        elif t == 'den':
          label = 'Denoising'
        elif t == 'sr':
          label = 'Superresolution'
        ax.set_ylabel(label, fontsize=18)

      start, end = ax.get_ylim()
      ax.set_yticks([start, (start+end)/2, end])
      ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
      if i == 0 and j == 0:
        ax.legend()
  fig.tight_layout()
  fig.show()