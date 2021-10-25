import time
import numpy as np
from tqdm import tqdm
import torch
from hyperrecon.util.train import BaseTrain

class AdaptiveScaling(BaseTrain):

  def __init__(self, args):
    super(AdaptiveScaling, self).__init__(args=args)
  
  def set_monitor(self):
    self.list_of_monitor = [
      'learning_rate', 
      'time:train',
      'scale0',
      'scale1'
    ]
  
  def set_eval_hparams(self):
    # hparams must be list of tensors, each of shape (num_hyperparams)
    if self.distribution == 'constant':
      self.val_hparams = torch.tensor(self.hyperparameters).view(-1, 1)
      self.test_hparams = torch.tensor(self.hyperparameters).view(-1, 1)
    else:
      self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
      self.test_hparams = torch.tensor(np.linspace(0, 1.00, 20)).float().view(-1, 1)

  def train_epoch(self):
    """Train for one epoch."""
    self.network.eval()
    hparams = torch.tensor([0., 1.]).view(-1, 1)
    _, gt, pred, coeffs = self.get_predictions(hparams[0])
    loss0 = self.compute_loss(pred, gt, coeffs, [1, 1], is_training=False)
    loss0 = self.process_loss(loss0).item()
    _, gt, pred, coeffs = self.get_predictions(hparams[1])
    loss1 = self.compute_loss(pred, gt, coeffs, [1, 1], is_training=False)
    loss1 = self.process_loss(loss1).item()
    self.per_loss_scale_constants = [loss0, loss1]
    print('new loss scales:', self.per_loss_scale_constants)
    self.monitor['scale0'].append(loss0)
    self.monitor['scale1'].append(loss1)

    self.network.train()
    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    start_time = time.time()
    for i, batch in tqdm(enumerate(self.train_loader), 
      total=min(len(self.train_loader), self.num_steps_per_epoch)):
      loss, psnr, batch_size = self.train_step(batch)
      epoch_loss += loss * batch_size
      epoch_psnr += psnr * batch_size
      epoch_samples += batch_size
      if i == self.num_steps_per_epoch:
        break
    self.scheduler.step()

    epoch_time = time.time() - start_time
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    self.metrics['loss:train'].append(epoch_loss)
    self.metrics['psnr:train'].append(epoch_psnr)
    self.monitor['learning_rate'].append(self.scheduler.get_last_lr()[0])
    self.monitor['time:train'].append(epoch_time)
    print("train loss={:.6f}, train psnr={:.6f}, train time={:.6f}".format(
      epoch_loss, epoch_psnr, epoch_time))