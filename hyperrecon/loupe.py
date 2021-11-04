import torch
import torch.nn as nn
from hyperrecon.util.train import BaseTrain
from hyperrecon.loss import loss_ops
from hyperrecon.util.metric import bhfen
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
  
  def prepare_batch(self, batch):
    targets = batch.view(-1, 1, *batch.shape[-2:]).float().to(self.device)
    bs = len(targets)
    return targets, bs

  def inference(self, zf, *args):
    del args
    return self.network(zf)

  def train_step(self, batch):
    '''Train for one step.'''
    targets, batch_size = self.prepare_batch(batch)
    self.optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      pred, _ = self.inference(targets)
      loss = self.compute_loss(pred, targets)
      loss.backward()
      self.optimizer.step()
    psnr = loss_ops.PSNR()(targets, pred)
    return loss.cpu().detach().numpy(), psnr, batch_size
  
  def eval_step(self, batch, hparams):
    '''Eval for one step.
    
    Args:
      batch: Single batch from dataloader
      hparams: Single hyperparameter vector (1, num_hyperparams)
    '''
    targets, _ = self.prepare_batch(batch)
    with torch.set_grad_enabled(False):
      pred, inputs = self.inference(targets)

    return inputs, targets, pred, hparams
  
class LoupeAgnostic(BaseTrain):
  """RateAgnostic."""

  def __init__(self, args):
    super(LoupeAgnostic, self).__init__(args=args)

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor(np.linspace(0.0, 1.0, 20)).float().view(-1, 1)
  
  def compute_loss(self, pred, gt, *args, **kwargs):
    del args, kwargs
    return nn.MSELoss()(pred, gt)

  def prepare_batch(self, batch):
    targets = batch.view(-1, 1, *batch.shape[-2:]).float().to(self.device)
    bs = len(targets)
    return targets, bs
  
  def train_step(self, batch):
    '''Train for one step.'''
    targets, batch_size = self.prepare_batch(batch)
    hparams = self.sample_hparams(batch_size).to(self.device)
    self.optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      pred, _ = self.inference(targets, hparams)
      loss = self.compute_loss(pred, targets)
      loss.backward()
      self.optimizer.step()
    psnr = loss_ops.PSNR()(targets, pred)
    return loss.cpu().detach().numpy(), psnr, batch_size
  
  def eval_step(self, batch, hparams):
    '''Eval for one step.
    
    Args:
      batch: Single batch from dataloader
      hparams: Single hyperparameter vector (1, num_hyperparams)
    '''
    targets, batch_size = self.prepare_batch(batch)
    hparams = hparams.repeat(batch_size, 1).to(self.device)
    with torch.set_grad_enabled(False):
      pred, inputs = self.inference(targets, hparams)

    return inputs, targets, pred, hparams
  
  def test(self, save_preds=False):
    for hparam in self.test_hparams:
      hparam_str = self.stringify_list(hparam.tolist())
      print('Testing with hparam', hparam_str)
      zf, gt, pred, _ = self.get_predictions(hparam, by_subject=True)
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
            loss = self.compute_loss(pred[i], gt[i])
            loss = self.process_loss(loss).item()
            self.test_metrics[key].append(loss)
          elif 'psnr' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(loss_ops.PSNR()(gt[i], pred[i]).mean().item())
          elif 'ssim' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(1-loss_ops.SSIM()(gt[i], pred[i]).mean().item())
          elif 'hfen' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(bhfen(gt[i], pred[i]))
          elif 'watson' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(loss_ops.WatsonDFT()(gt[i], pred[i]))
          elif 'mae' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(loss_ops.L1()(gt[i], pred[i]))
          # elif 'dice' in key and hparam_str in key and 'sub{}'.format(i) in key:
          #   loss_roi, _,_,_,_ = dice(pred[i], gt[i], seg[i])
          #   self.test_metrics[key].append(float(loss_roi.mean()))