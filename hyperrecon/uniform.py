import torch
import random
from hyperrecon.util.train import BaseTrain
from hyperrecon.util.metric import bpsnr

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
    self.distance_metric = torch.nn.MSELoss()
  
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

      loss = self.compute_loss(pred, gt, y, coeffs)
      loss = self.process_loss(loss)
      loss.backward()
      self.optimizer.step()
    psnr = bpsnr(gt, pred)
    return loss.cpu().detach().numpy(), psnr, batch_size

  def compute_loss(self, pred, gt, y, coeffs):
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
    assert len(self.losses) == coeffs.shape[1], 'loss and coeff mismatch'
    assert len(pred) == len(coeffs), 'img and coeff mismatch'
    recon_loss = 0
    for i in range(len(self.losses)):
      c = coeffs[:self.batch_size, i]
      l = self.losses[i]
      recon_loss += c * l(pred[:self.batch_size], gt[:self.batch_size], y[:self.batch_size])
    
    # TODO: generalize to higher-order coefficients
    hparams = coeffs[:, 1]
    lmbda = torch.abs(hparams[:self.batch_size] - hparams[self.batch_size:])
    diversity_loss = (pred[:self.batch_size] - pred[self.batch_size:]).norm(p=2, dim=(1, 2, 3))
    return recon_loss - lmbda*diversity_loss

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)
