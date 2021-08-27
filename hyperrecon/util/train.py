from operator import is_
import torch
from torchvision import transforms
import numpy as np
import os
import time
from tqdm import tqdm
from pprint import pprint

from hyperrecon import sampler
from hyperrecon.loss.losses import compose_loss_seq
from hyperrecon.util.metric import bpsnr
from hyperrecon.util import utils
from hyperrecon.model.unet import HyperUnet
from hyperrecon.model.layers import ClipByPercentile
from hyperrecon.data.mask import get_mask
from hyperrecon.data.brain import ArrDataset, SliceDataset, get_train_data, get_train_gt

class BaseTrain(object):
  def __init__(self, args):
    # HyperRecon
    self.mask = get_mask(
      '160_224', args.undersampling_rate).to(args.device)
    self.undersampling_rate = args.undersampling_rate
    self.topK = args.topK
    self.method = args.method
    self.anneal = args.anneal
    self.range_restrict = args.range_restrict
    # ML
    self.num_epochs = args.num_epochs
    self.lr = args.lr
    self.force_lr = args.force_lr
    self.batch_size = args.batch_size
    self.num_steps_per_epoch = args.num_steps_per_epoch
    self.legacy_dataset = args.legacy_dataset
    self.loss_list = args.loss_list
    self.hyperparameters = args.hyperparameters
    self.hnet_hdim = args.hnet_hdim
    self.unet_hdim = args.unet_hdim
    self.n_ch_in = 2
    self.n_ch_out = args.n_ch_out
    # I/O
    self.load = args.load
    self.cont = args.cont
    self.device = args.device
    self.run_dir = args.run_dir
    self.ckpt_dir = args.ckpt_dir
    self.data_path = args.data_path
    self.log_interval = args.log_interval

    # Logging
    self.logger = {}
    self.logger['loss.train'] = []
    self.logger['loss.val'] = []
    self.logger['psnr.train'] = []
    self.logger['psnr.val'] = []
    self.logger['time.train'] = []
    self.logger['time.val'] = []

  def config(self):
    # Data
    self.get_dataloader()

    # Model, Optimizer, Sampler, Loss
    self.num_hyperparams = len(
      self.loss_list)-1 if self.range_restrict else len(self.loss_list)

    self.network = self.get_model()
    self.optimizer = self.get_optimizer()
    self.sampler = self.get_sampler()
    self.losses = compose_loss_seq(self.loss_list)

    if self.force_lr is not None:
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = self.force_lr

    # Checkpoint Loading
    if self.load:  # Load from path
      load_path = self.load
    elif self.cont > 0:  # Load from previous checkpoint
      load_path = os.path.join(
        self.run_dir, 'checkpoints', 'model.{epoch:04d}.h5'.format(epoch=self.cont))
      self.logger['loss.train'] = list(np.loadtxt(
        os.path.join(self.run_dir, 'loss.train.txt'))[:self.cont])
      self.logger['loss.val'] = list(np.loadtxt(
        os.path.join(self.run_dir, 'loss.val.txt'))[:self.cont])
      self.logger['psnr.train'] = list(np.loadtxt(
        os.path.join(self.run_dir, 'psnr.train.txt'))[:self.cont])
      self.logger['psnr.val'] = list(np.loadtxt(
        os.path.join(self.run_dir, 'psnr.val.txt'))[:self.cont])
      self.logger['time.train'] = list(np.loadtxt(
        os.path.join(self.run_dir, 'time.train.txt'))[:self.cont])
      self.logger['time.val'] = list(np.loadtxt(
        os.path.join(self.run_dir, 'time.val.txt'))[:self.cont])
    else:
      load_path = None

    if load_path is not None:
      self.network, self.optimizer = utils.load_checkpoint(
        self.network, load_path, self.optimizer)

  def get_dataloader(self):
    if self.legacy_dataset:
      xdata = get_train_data(maskname=self.undersampling_rate)
      gt_data = get_train_gt()
      trainset = ArrDataset(
        xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
      valset = ArrDataset(
        xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])
    else:
      transform = transforms.Compose([ClipByPercentile()])
      trainset = SliceDataset(
        self.data_path, 'train', total_subjects=50, transform=transform)
      valset = SliceDataset(
        self.data_path, 'validate', total_subjects=5, transform=transform)

    params = {'batch_size': self.batch_size,
          'shuffle': True,
          'num_workers': 0,
          'pin_memory': True}

    self.train_loader = torch.utils.data.DataLoader(trainset, **params)
    self.val_loader = torch.utils.data.DataLoader(valset, **params)

  def get_model(self):
    self.network = HyperUnet(
      self.num_hyperparams,
      self.hnet_hdim,
      hnet_norm=not self.range_restrict,
      in_ch_main=self.n_ch_in,
      out_ch_main=self.n_ch_out,
      h_ch_main=self.unet_hdim,
    ).to(self.device)
    utils.summary(self.network)
    return self.network

  def get_optimizer(self):
    return torch.optim.Adam(self.network.parameters(), lr=self.lr)

  def get_sampler(self):
    return sampler.Sampler(self.num_hyperparams)

  def train(self):
    for epoch in range(self.cont+1, self.num_epochs+1):
      pprint(self.logger)
      self.epoch = epoch

      self.train_epoch_begin()
      self.train_epoch()
      self.train_epoch_end(is_eval=True, is_save=(self.epoch % self.log_interval == 0))

  def train_epoch_begin(self):
    self.r1 = 0
    self.r2 = 1

    print('\nEpoch %d/%d' % (self.epoch, self.num_epochs))
    print('Learning rate:',
        self.lr if self.force_lr is None else self.force_lr)
    print('Sampling bounds [%.2f, %.2f]' % (self.r1, self.r2))

  def train_epoch_end(self, is_eval=False, is_save=False):
    '''Save loss and checkpoints. Evaluate if necessary.'''
    if is_eval:
      self.eval_epoch()

    utils.save_loss(self.run_dir, self.logger, 
                  'loss.train', 'loss.val', 'psnr.train', 
                  'psnr.val', 'time.train', 'time.val')
    if is_save:
      utils.save_checkpoint(self.epoch, self.network.state_dict(), self.optimizer.state_dict(),
          self.ckpt_dir)

  def compute_loss(self, pred, gt, y, coeffs):
    '''Compute loss.

    Args:
      pred: Predictions (bs, nch, n1, n2)
      gt: Ground truths (bs, nch, n1, n2)
      y: Under-sampled k-space (bs, nch, n1, n2)
      coeffs: Loss coefficients (bs, num_losses)
    
    Returns:
      loss: Per-sample loss (bs)
    '''
    loss = 0
    for i in range(len(self.losses)):
      c = coeffs[:, i]
      l = self.losses[i]
      loss += c * l(pred, gt, y, self.mask)
    return loss
  
  def process_loss(loss):
    '''Process loss.
    
    Args:
      loss: Per-sample loss (bs)
    
    Returns:
      Scalar loss value
    '''
    pass

  def prepare_batch(self, batch):
    targets = batch.float().to(self.device)
    segs = None

    under_ksp = utils.generate_measurement(targets, self.mask)
    zf = utils.ifft(under_ksp)
    under_ksp, zf = utils.scale(under_ksp, zf)
    return zf, targets, under_ksp, segs

  def inference(self, zf, hyperparams):
    return self.network(zf, hyperparams)

  def sample_hparams(self, num_samples, is_training=True):
    if is_training:
      hyperparams = self.sampler.sample(
        num_samples, self.r1, self.r2)
    else:
      hyperparams = torch.ones(
        (num_samples, self.num_hyperparams))
    return hyperparams

  def generate_coefficients(self, samples, num_losses, range_restrict):
    '''Generates coefficients from samples.'''
    if range_restrict and num_losses == 2:
      assert samples.shape[1] == 1, 'num_hyperparams and loss mismatch'
      alpha = samples[:, 0]
      coeffs = torch.stack((1-alpha, alpha), dim=1)

    elif range_restrict and num_losses == 3:
      assert samples.shape[1] == 2, 'num_hyperparams and loss mismatch'
      alpha = samples[:, 0]
      beta = samples[:, 1]
      coeffs = torch.stack((alpha, (1-alpha)*beta, (1-alpha)*(1-beta)), dim=1)

    else:
      assert samples.shape[1] == num_losses, 'num_hyperparams and loss mismatch'
      coeffs = None
      for i in range(num_losses):
        coeffs = samples[:,i:i+1] if coeffs is None else torch.cat((coeffs, samples[:,i:i+1]), dim=1)
      coeffs = coeffs / torch.sum(samples, dim=1)

    return coeffs

  def train_epoch(self):
    """Train for one epoch."""
    self.network.train()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    start_time = time.time()
    for i, batch in tqdm(enumerate(self.train_loader), total=self.num_steps_per_epoch):
      loss, psnr, batch_size = self.train_step(batch)
      epoch_loss += loss.data.cpu().numpy() * batch_size
      epoch_psnr += psnr * batch_size
      epoch_samples += batch_size
      if i == self.num_steps_per_epoch:
        break

    epoch_time = time.time() - start_time
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    self.logger['loss.train'].append(epoch_loss)
    self.logger['psnr.train'].append(epoch_psnr)
    self.logger['time.train'].append(epoch_time)

    print("Epoch {}: train loss={:.6f}, train psnr={:.6f}, train time={:.6f}".format(
        self.epoch, epoch_loss, epoch_psnr, epoch_time))

  def eval_epoch(self):
    """Eval for one epoch."""
    self.network.eval()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    start_time = time.time()
    for _, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
      loss, psnr, batch_size = self.eval_step(batch)
      epoch_loss += loss.data.cpu().numpy() * batch_size
      epoch_psnr += psnr * batch_size
      epoch_samples += batch_size

    epoch_time = time.time() - start_time
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    self.logger['loss.val'].append(epoch_loss)
    self.logger['psnr.val'].append(epoch_psnr)
    self.logger['time.val'].append(epoch_time)

    print("Epoch {}: val loss={:.6f}, val psnr={:.6f}, val time={:.6f}".format(
      self.epoch, epoch_loss, epoch_psnr, epoch_time))

  def train_step(self, batch):
    '''Train for one step.'''
    zf, gt, y, _ = self.prepare_batch(batch)
    batch_size = len(zf)

    self.optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      hyperparams = self.sample_hparams(batch_size).to(self.device)
      pred = self.inference(zf, hyperparams)

      coeffs = self.generate_coefficients(
        hyperparams, len(self.losses), self.range_restrict)
      loss = self.compute_loss(pred, gt, y, coeffs)
      loss = self.process_loss(loss)
      loss.backward()
      self.optimizer.step()
    psnr = bpsnr(gt, pred)
    return loss, psnr, batch_size

  def eval_step(self, batch):
    '''Eval for one step.'''
    zf, gt, y, _ = self.prepare_batch(batch)
    batch_size = len(zf)

    with torch.set_grad_enabled(False):
      hyperparams = self.sample_hparams(batch_size, is_training=False).to(self.device)
      pred = self.inference(zf, hyperparams)

      coeffs = self.generate_coefficients(
        hyperparams, len(self.losses), self.range_restrict)
      loss = self.compute_loss(pred, gt, y, coeffs)
      loss = self.process_loss(loss)
    psnr = bpsnr(gt, pred)
    return loss, psnr, batch_size