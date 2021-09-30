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
  
  def mask_inference(self, x):
    return self.mask_module(x)
  
  def compute_loss(self, pred, gt, y, seg, coeffs, is_training=False):
      return nn.MSELoss()(pred, gt)
  
  def prepare_batch(self, batch, hparams, is_training=True):
    targets, segs = batch[0].float().to(self.device), batch[1].float().to(self.device)
    if not is_training:
      targets = targets.view(-1, 1, *targets.shape[-2:])
      segs = segs.view(-1, 1, *targets.shape[-2:])

    undersample_mask = self.mask_inference(hparams).to(self.device)
    measurement, measurement_ksp = self.forward_model.generate_measurement(targets, undersample_mask)
    return measurement, targets, measurement_ksp, segs
  
  def train_step(self, batch):
    '''Train for one step.'''
    batch_size = len(batch)
    hparams = self.sample_hparams(batch_size)
    coeffs = self.generate_coefficients(hparams)
    zf, gt, y, seg = self.prepare_batch(batch)

    self.optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      pred = self.inference(zf, coeffs)
      loss = self.compute_loss(pred, gt)
      loss = self.process_loss(loss)
      loss.backward()
      self.optimizer.step()
    psnr = bpsnr(gt, pred)
    return loss.cpu().detach().numpy(), psnr, batch_size