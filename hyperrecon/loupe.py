import torch
import torch.nn as nn
from hyperrecon.util.train import BaseTrain
from hyperrecon.util.metric import bpsnr
from hyperrecon.util.metric import bpsnr, bssim, bhfen, dice, bmae, bwatson
import os
import numpy as np

class Loupe(BaseTrain):
  """Loupe."""

  def __init__(self, args):
    super(Loupe, self).__init__(args=args)

  def sample_hparams(self, num_samples):
    hyperparams = torch.ones((num_samples, self.num_hparams)) * 1/float(self.undersampling_rate)
    return hyperparams
  
  def set_eval_hparams(self):
    self.val_hparams = torch.tensor(1/float(self.undersampling_rate)).view(-1, 1)
    self.test_hparams = torch.tensor(1/float(self.undersampling_rate)).view(-1, 1)
  
  def compute_loss(self, pred, gt, *args, **kwargs):
    del args, kwargs
    return nn.MSELoss()(pred, gt)
  
  def inference(self, zf, *args):
    del args
    return self.network(zf)

  def train_step(self, targets, segs):
    '''Train for one step.'''
    batch_size = len(targets)

    targets, segs = targets.float().to(self.device), segs.float().to(self.device)
    self.optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      pred, _, _ = self.inference(targets)
      loss = self.compute_loss(pred, targets, None, None, None)
      loss.backward()
      self.optimizer.step()
    psnr = bpsnr(targets, pred)
    return loss.cpu().detach().numpy(), psnr, batch_size
  
  def eval_step(self, targets, segs, hparams):
    '''Eval for one step.
    
    Args:
      batch: Single batch from dataloader
      hparams: Single hyperparameter vector (1, num_hyperparams)
    '''
    targets, segs = targets.float().to(self.device), segs.float().to(self.device)
    targets = targets.view(-1, 1, *targets.shape[-2:])
    segs = segs.view(-1, 1, *targets.shape[-2:])
    with torch.set_grad_enabled(False):
      pred, measurement, measurement_ft = self.inference(targets)

    return measurement, measurement_ft, targets, pred, segs, hparams
  
class LoupeAgnostic(BaseTrain):
  """RateAgnostic."""

  def __init__(self, args):
    super(LoupeAgnostic, self).__init__(args=args)

  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    return torch.FloatTensor(num_samples, self.num_hparams).uniform_(0, 1)

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 0.25, 0.5, 0.75, 1.]).view(-1, 1)
  
  def compute_loss(self, pred, gt, *args, **kwargs):
    del args, kwargs
    return nn.MSELoss()(pred, gt)
  
  def train_step(self, targets, segs):
    '''Train for one step.'''
    batch_size = len(targets)
    hparams = self.sample_hparams(batch_size).to(self.device)

    targets, segs = targets.float().to(self.device), segs.float().to(self.device)
    self.optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      pred, _, _ = self.inference(targets, hparams)
      loss = self.compute_loss(pred, targets, None, None, None)
      loss.backward()
      self.optimizer.step()
    psnr = bpsnr(targets, pred)
    return loss.cpu().detach().numpy(), psnr, batch_size
  
  def eval_step(self, targets, segs, hparams):
    '''Eval for one step.
    
    Args:
      batch: Single batch from dataloader
      hparams: Single hyperparameter vector (1, num_hyperparams)
    '''
    targets, segs = targets.float().to(self.device), segs.float().to(self.device)
    targets = targets.view(-1, 1, *targets.shape[-2:])
    segs = segs.view(-1, 1, *targets.shape[-2:])
    batch_size = len(targets)
    hparams = hparams.repeat(batch_size, 1).to(self.device)
    with torch.set_grad_enabled(False):
      pred, measurement, measurement_ft = self.inference(targets, hparams)

    return measurement, measurement_ft, targets, pred, segs, hparams
  
  def test(self, save_preds=False):
    for hparam in self.test_hparams:
      hparam_str = self.stringify_list(hparam.tolist())
      print('Testing with hparam', hparam_str)
      zf, gt, y, pred, seg, coeffs = self.get_predictions(hparam, by_subject=True)
      for i in range(len(zf)):
        # Save predictions to disk
        if save_preds:
          gt_path = os.path.join(self.img_dir, 'gt'+'sub{}'.format(i) + '.npy')
          zf_path = os.path.join(self.img_dir, 'zf'+hparam_str+'sub{}'.format(i) + '.npy')
          pred_path = os.path.join(self.img_dir, 'pred'+hparam_str+'sub{}'.format(i)+'cp{:04d}'.format(self.epoch-1) + '.npy')
          np.save(pred_path, pred[i].cpu().detach().numpy())
          if not os.path.exists(gt_path):
            np.save(gt_path, gt[i].cpu().detach().numpy())
          if not os.path.exists(zf_path):
            np.save(zf_path, zf[i].cpu().detach().numpy())
        for key in self.test_metrics:
          if 'loss' in key and hparam_str in key and 'sub{}'.format(i) in key:
            loss = self.compute_loss(pred[i], gt[i], y[i], seg[i], coeffs[i], is_training=False)
            loss = self.process_loss(loss).item()
            self.test_metrics[key].append(loss)
          elif 'psnr' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(bpsnr(gt[i], pred[i]))
          elif 'ssim' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(bssim(gt[i], pred[i]))
          elif 'hfen' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(bhfen(gt[i], pred[i]))
          elif 'watson' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(bwatson(gt[i], pred[i]))
          elif 'mae' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(bmae(gt[i], pred[i]))
          elif 'dice' in key and hparam_str in key and 'sub{}'.format(i) in key:
            loss_roi, _,_,_,_ = dice(pred[i], gt[i], seg[i])
            self.test_metrics[key].append(float(loss_roi.mean()))