import torch
import torch.nn as nn
from hyperrecon.util.train import BaseTrain
from hyperrecon.util.metric import bpsnr

class RateAgnostic(BaseTrain):
  """RateAgnostic."""

  def __init__(self, args):
    super(RateAgnostic, self).__init__(args=args)

  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    return torch.FloatTensor(num_samples, self.num_hparams).uniform_(0, 1)

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)
  
  def compute_loss(self, pred, gt, y, seg, coeffs, is_training=False):
      return nn.MSELoss()(pred, gt)
  
  def train_step(self, targets, segs):
    '''Train for one step.'''
    batch_size = len(targets)
    hparams = self.sample_hparams(batch_size)
    coeffs = self.generate_coefficients(hparams)

    targets, segs = targets.float().to(self.device), segs.float().to(self.device)
    self.optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      undersample_mask = self.mask_module(batch_size, hparams).to(self.device)
      measurement, measurement_ksp = self.forward_model.generate_measurement(targets, undersample_mask)
      pred = self.inference(measurement, coeffs)
      loss = self.compute_loss(pred, targets, measurement_ksp, segs, coeffs, is_training=True)
      loss = self.process_loss(loss)
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
    targets, segs = targets.float().to(self.device), segs.float().to(self.device)
    targets = targets.view(-1, 1, *targets.shape[-2:])
    segs = segs.view(-1, 1, *targets.shape[-2:])

    undersample_mask = self.mask_module(batch_size, hparams).to(self.device)
    measurement, measurement_ksp = self.forward_model.generate_measurement(targets, undersample_mask)
    hparams = hparams.repeat(batch_size, 1)

    with torch.set_grad_enabled(False):
      coeffs = self.generate_coefficients(hparams)
      pred = self.inference(measurement, coeffs)

    return measurement, measurement_ksp, targets, pred, segs, coeffs