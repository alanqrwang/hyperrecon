"""
Test/inference functions for HyperRecon
For more details, please read:
  Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
  "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 

"""
from glob import glob
import os
import torch

from hyperrecon.util.train import BaseTrain
from hyperrecon.util import utils

class Predict(BaseTrain):
  '''Predict.
  
  Kind of hacky. Overrides the BaseTrain class,
  doesn't perform any training steps but simply
  loads from the most recent model and performs
  evaluation.
  '''
  def __init__(self, args):
    super(Predict, self).__init__(args=args)
    model_paths = sorted(glob(os.path.join(self.ckpt_dir, '*')))
    if len(model_paths) == 0:
      raise ValueError('No models found for prediction.')
    else:
      self.load = model_paths[-1]

  def train(self):
    print('Predict')
    self.train_begin()
    self.train_epoch_end(is_eval=True, is_save=False)
    self.train_end(verbose=True)
    
# def test(network, dataloader, args, hps, normalized, out_shape, criterion=None, \
#     give_recons=False, give_losses=False):
#   """Testing for a fixed set of hyperparameter setting.

#   Returns recons, losses, and metrics (if specified)
#   For every sample in the dataloader, evaluates with all hyperparameters in hps.
#   Batch size must match size of dataset (TODO change this)

#   If take_avg is True, then returns [len(hps)]
#   """
#   network.eval()

#   res = {}
#   if give_recons:
#     res['recons'] = np.full((len(hps), *out_shape), np.nan)
#     res['gts'] = np.full((len(hps), *out_shape), np.nan)
#     res['gt_segs'] = np.full((len(hps), *out_shape), np.nan)
#     res['recon_segs'] = np.full((len(hps), *out_shape), np.nan)
#     res['pred_gt_segs'] = np.full((len(hps), *out_shape), np.nan)
#   if give_losses:
#     res['losses'] = np.full((len(hps), out_shape[0]), np.nan)
#     # res['dcs'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
#     # res['cap_regs'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
#     # res['ws'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
#     # res['tvs'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
#     res['psnrs'] = np.full((len(hps), out_shape[0]), np.nan)
#     # all_rpsnrs = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
#     res['mses'] = np.full((len(hps), out_shape[0]), np.nan)
#     res['ssims'] = np.full((len(hps), out_shape[0]), np.nan)
#     res['l1s'] = np.full((len(hps), out_shape[0]), np.nan)
#     # res['percs'] = np.full((len(hps), total_subjects, vol_shape[0]), np.nan)
#     res['dices'] = np.full((len(hps), out_shape[0]), np.nan)
#     res['dices_gt'] = np.full((len(hps), out_shape[0]), np.nan)

#   for h, hp in enumerate(hps):
#     print(hp)
#     for i, batch in enumerate(dataloader): 
#       zf, gt, y, seg = utils.prepare_batch(batch, args, split='test')
#       batch_size = len(zf)

#       hyperparams = hp.expand(batch_size, -1)
#       with torch.set_grad_enabled(False):

#         if args['hyperparameters'] is None: # Hypernet
#           preds = network(zf, hyperparams)
#         else:
#           preds = network(zf) # Baselines
#         loss = criterion(preds, y, hyperparams, None, target=gt)


#       if give_losses:
#         assert criterion is not None, 'loss must be provided'
#         res['losses'][h, i*batch_size:i*batch_size+len(preds)] = loss.cpu().detach().numpy()
#         # dcs.append(regs['dc'].cpu().detach().numpy())
#         # cap_regs.append(regs['cap'].cpu().detach().numpy())
#         # tvs.append(regs['tv'].cpu().detach().numpy())
#         res['psnrs'][h, i*batch_size:i*batch_size+len(preds)] = utils.get_metrics(gt, preds, zf, metric_type='psnr', normalized=normalized)
#         # rpsnrs = utils.get_metrics(gt, preds, zf, metric_type='relative psnr', normalized=normalized)
#         res['mses'][h, i*batch_size:i*batch_size+len(preds)] = criterion.get_mse(gt, preds).detach().cpu().numpy()
#         res['ssims'][h, i*batch_size:i*batch_size+len(preds)] = criterion.get_ssim(gt, preds).detach().cpu().numpy()
#         res['l1s'][h, i*batch_size:i*batch_size+len(preds)] = criterion.get_l1(gt, preds).detach().cpu().numpy()
#         # percs = criterion.get_watson_dft(gt, preds).detach().cpu().numpy()
#         dices, dices_gt, pred_segs, pred_gt_segs, target_segs = criterion.get_dice(preds, gt, seg)#.detach().cpu().numpy()
#         res['dices'][h, i*batch_size:i*batch_size+len(preds)] = dices
#         res['dices_gt'][h, i*batch_size:i*batch_size+len(preds)] = dices_gt

#       if give_recons:
#         res['recons'][h, i*batch_size:i*batch_size+len(preds)] = preds.cpu().detach().numpy()
#         res['gts'][h, i*batch_size:i*batch_size+len(preds)] = gt.cpu().detach().numpy()
#         res['recon_segs'][h, i*batch_size:i*batch_size+len(preds)] = pred_segs#.cpu().detach().numpy()
#         res['gt_segs'][h, i*batch_size:i*batch_size+len(preds)] = target_segs
#         res['pred_gt_segs'][h, i*batch_size:i*batch_size+len(preds)] = pred_gt_segs#.permute(0, 2, 3, 1).cpu().detach().numpy()


#   if give_recons:
#     assert np.isnan(np.sum(res['recons'])) == False, 'Missed some predictions'
#     assert np.isnan(np.sum(res['gts'])) == False, 'Missed some gts'
#   if give_losses:
#     assert np.isnan(np.sum(res['losses'])) == False, 'Missed some predictions'
#     assert np.isnan(np.sum(res['psnrs'])) == False, 'Missed some gts'
#     assert np.isnan(np.sum(res['mses'])) == False, 'Missed some gts'
#     assert np.isnan(np.sum(res['ssims'])) == False, 'Missed some gts'
#     assert np.isnan(np.sum(res['l1s'])) == False, 'Missed some gts'
#     assert np.isnan(np.sum(res['dices'])) == False, 'Missed some gts'

#   return res
