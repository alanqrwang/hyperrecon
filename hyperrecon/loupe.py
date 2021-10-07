import torch
import torch.nn as nn
from hyperrecon.util.train import BaseTrain
from hyperrecon.util.metric import bpsnr

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
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)
  
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
    batch_size = len(targets)
    hparams = hparams.repeat(batch_size, 1).to(self.device)

    targets, segs = targets.float().to(self.device), segs.float().to(self.device)
    targets = targets.view(-1, 1, *targets.shape[-2:])
    segs = segs.view(-1, 1, *targets.shape[-2:])
    with torch.set_grad_enabled(False):
      pred, measurement, measurement_ft = self.inference(targets, hparams)

    return measurement, measurement_ft, targets, pred, segs, hparams