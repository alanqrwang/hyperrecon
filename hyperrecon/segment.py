import torch
import numpy as np

from hyperrecon.loss.loss_ops import DiceLoss
from hyperrecon.util.train import BaseTrain
from hyperrecon.model.unet import Unet
from hyperrecon.util import utils

class Segment(BaseTrain):
  def __init__(self, args):
    super().__init__(args)
    self.criterion = DiceLoss()

  def set_metrics(self):
    self.list_of_metrics = [
      'loss:train',
      'psnr:train'
    ]
    self.list_of_val_metrics = [
      'loss:val:' + self.stringify_list(l.tolist()) for l in self.val_hparams
    ] 
    self.list_of_test_metrics = [
      'loss:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ] 

  def get_model(self):
    n_classes = 5
    self.network = Unet(
                      in_ch=1,
                      out_ch=n_classes,
                      h_ch=self.unet_hdim,
                      residual=self.unet_residual,
                      use_batchnorm=self.use_batchnorm
                   ).to(self.device)
    return self.network
  
  def compute_loss(self, gt, pred, seg, coeffs, scales, is_training=False):
    '''Compute loss.

    Args:
      pred: Predictions (bs, nch, n1, n2)
      gt: Ground truths (bs, nch, n1, n2)
      y: Under-sampled k-space (bs, nch, n1, n2)
      coeffs: Loss coefficients (bs, num_losses)

    Returns:
      loss: Per-sample loss (bs)
    '''
    loss = self.criterion(pred, gt)
    return loss, None

  def prepare_batch(self, batch):
    inputs, targets = batch[0], batch[1]
    inputs = inputs.view(-1, 1, *inputs.shape[-2:]).float().cuda()
    targets = targets.view(-1, 1, *targets.shape[-2:]).float().cuda()

    targets_onehot = utils.get_onehot(targets).cuda()
    batch_size = len(inputs)
    return inputs, targets_onehot, targets, batch_size

  def train_step(self, batch):
    '''Train for one step.'''
    inputs, targets, segs, batch_size = self.prepare_batch(batch)
    hparams = self.sample_hparams(batch_size)
    coeffs = self.generate_coefficients(hparams)

    self.optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      pred = self.inference(inputs, coeffs)
      loss, loss_dict = self.compute_loss(targets, pred, segs, coeffs, scales=self.per_loss_scale_constants, is_training=True)
      loss = self.process_loss(loss, loss_dict)
      loss.backward()
      self.optimizer.step()
    return loss.cpu().detach().numpy(), 0, batch_size