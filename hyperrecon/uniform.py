import torch
import random
from hyperrecon.util.train import BaseTrain
from hyperrecon.util.metric import bpsnr
import time
from tqdm import tqdm

class Uniform(BaseTrain):
  """Uniform."""

  def __init__(self, args):
    super(Uniform, self).__init__(args=args)

  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    return torch.FloatTensor(num_samples, self.num_hparams).uniform_(0, 1)

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)
    # self.val_hparams = torch.tensor([[0.,0.], [1.,1.]])
    # hparams = []
    # for i in np.linspace(0, 1, 50):
    #   for j in np.linspace(0, 1, 50):
    #     hparams.append([i, j])
    # self.test_hparams = torch.tensor(hparams).float()

class UniformConstant(BaseTrain):
  """UniformConstant."""

  def __init__(self, args):
    super(UniformConstant, self).__init__(args=args)
  
  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    cat = random.random()
    return torch.ones(num_samples, self.num_hparams) * cat

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)

class UniformDiversityPrior(BaseTrain):
  """UniformDiversityPrior."""

  def __init__(self, args):
    super(UniformDiversityPrior, self).__init__(args=args)
  
  def set_monitor(self):
    self.list_of_monitor = [
      'learning_rate', 
      'time:train',
      'diversity_loss',
      'recon_loss'
    ]
  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    return torch.FloatTensor(num_samples, self.num_hparams).uniform_(0, 1)
  
  def train_step(self, batch):
    '''Train for one step.'''
    zf, gt, y, _ = self.prepare_batch(batch)
    zf = torch.cat((zf, zf), dim=0)
    gt = torch.cat((gt, gt), dim=0)
    y = torch.cat((y, y), dim=0)
    batch_size = len(zf)

    self.optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      hparams = self.sample_hparams(batch_size)
      coeffs = self.generate_coefficients(hparams)
      pred = self.inference(zf, coeffs)

      loss, recon_loss, div_loss = self.compute_loss(pred, gt, y, coeffs, is_training=True)
      loss = self.process_loss(loss)
      loss.backward()
      self.optimizer.step()
    psnr = bpsnr(gt, pred)
    return loss.cpu().detach().numpy(), psnr, batch_size // 2, recon_loss.mean(), div_loss.mean()

  def train_epoch(self):
    """Train for one epoch."""
    self.network.train()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0
    epoch_div_loss = 0
    epoch_recon_loss = 0

    start_time = time.time()
    for i, batch in tqdm(enumerate(self.train_loader), total=self.num_steps_per_epoch):
      loss, psnr, batch_size, recon_loss, div_loss = self.train_step(batch)
      epoch_loss += loss * batch_size
      epoch_psnr += psnr * batch_size
      epoch_samples += batch_size
      epoch_recon_loss += recon_loss * batch_size
      epoch_div_loss += div_loss * batch_size
      if i == self.num_steps_per_epoch:
        break
    self.scheduler.step()

    epoch_time = time.time() - start_time
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    epoch_recon_loss /= epoch_samples
    epoch_div_loss /= epoch_samples
    self.metrics['loss:train'].append(epoch_loss)
    self.metrics['psnr:train'].append(epoch_psnr)
    self.monitor['learning_rate'].append(self.scheduler.get_last_lr()[0])
    self.monitor['time:train'].append(epoch_time)
    self.monitor['diversity_loss'].append(epoch_div_loss)
    self.monitor['recon_loss'].append(epoch_recon_loss)

    print("train loss={:.6f}, train psnr={:.6f}, train time={:.6f}".format(
      epoch_loss, epoch_psnr, epoch_time))

  def compute_loss(self, pred, gt, y, coeffs, is_training=False):
    '''Compute loss with diversity prior. 
    Batch size should be 2 * self.batch_size

    Args:
      pred: Predictions (2*bs, nch, n1, n2)
      gt: Ground truths (2*bs, nch, n1, n2)
      y: Under-sampled k-space (2*bs, nch, n1, n2)
      coeffs: Loss coefficients (2*bs, num_losses)

    Returns:
      loss: Per-sample loss (bs)
    '''
    bs, n_ch, n1, n2 = pred.shape
    assert len(self.losses) == coeffs.shape[1], 'loss and coeff mismatch'
    recon_loss = 0
    for i in range(len(self.losses)):
      c = coeffs[:self.batch_size, i]
      l = self.losses[i]
      recon_loss += c * l(pred[:self.batch_size], gt[:self.batch_size], y[:self.batch_size])
    
    if is_training:
      # TODO: generalize to higher-order coefficients
      hparams = coeffs[:, 1]
      lmbda = torch.abs(hparams[:self.batch_size] - hparams[self.batch_size:])
      pred_vec = pred.view(len(pred), -1)
      diversity_loss = 1/(n_ch*n1*n2) * (pred_vec[:self.batch_size] - pred_vec[self.batch_size:]).norm(p=2, dim=1)
      return recon_loss - lmbda*diversity_loss, recon_loss, diversity_loss
    else:
      return recon_loss

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)
