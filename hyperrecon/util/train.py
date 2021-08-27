import torch
from torchvision import transforms
import numpy as np
from hyperrecon import dataset, model, sampler, layers
import os
import time
from tqdm import tqdm

from hyperrecon.loss.losses import compose_loss_seq
from hyperrecon.loss.coefficients import generate_coefficients
from hyperrecon.util.metric import bpsnr
from hyperrecon.util import utils

class BaseTrain(object):
  def __init__(self, args):
    # HyperRecon
    self.mask = dataset.get_mask(
      '160_224', args.undersampling_rate).to(args.device)
    self.undersampling_rate = args.undersampling_rate
    self.topK = args.topK
    self.sampling_method = args.sampling_method
    self.anneal = args.anneal
    self.range_restrict = args.range_restrict
    # ML
    self.epochs = args.epochs
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
    self.logger['loss_train'] = []
    self.logger['loss_val'] = []
    self.logger['loss_val2'] = []
    self.logger['epoch_train_time'] = []
    self.logger['psnr_train'] = []
    self.logger['psnr_val'] = []

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
      self.logger['loss_train'] = list(np.loadtxt(
        os.path.join(self.run_dir, 'loss_train.txt'))[:self.cont])
      self.logger['loss_val'] = list(np.loadtxt(
        os.path.join(self.run_dir, 'loss_val.txt'))[:self.cont])
      self.logger['epoch_train_time'] = list(np.loadtxt(
        os.path.join(self.run_dir, 'epoch_train_time.txt'))[:self.cont])
      self.logger['psnr_train'] = list(np.loadtxt(
        os.path.join(self.run_dir, 'psnr_train.txt'))[:self.cont])
      self.logger['psnr_val'] = list(np.loadtxt(
        os.path.join(self.run_dir, 'psnr_val.txt'))[:self.cont])
    else:
      load_path = None

    if load_path is not None:
      self.network, self.optimizer = utils.load_checkpoint(
        self.network, load_path, self.optimizer)

  def get_dataloader(self):
    if self.legacy_dataset:
      xdata = dataset.get_train_data(maskname=self.undersampling_rate)
      gt_data = dataset.get_train_gt()
      trainset = dataset.Dataset(
        xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
      valset = dataset.Dataset(
        xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])
    else:
      transform = transforms.Compose([layers.ClipByPercentile()])
      trainset = dataset.SliceDataset(
        self.data_path, 'train', total_subjects=50, transform=transform)
      valset = dataset.SliceDataset(
        self.data_path, 'validate', total_subjects=5, transform=transform)

    params = {'batch_size': self.batch_size,
          'shuffle': True,
          'num_workers': 0,
          'pin_memory': True}

    self.train_loader = torch.utils.data.DataLoader(trainset, **params)
    self.val_loader = torch.utils.data.DataLoader(valset, **params)

  def get_model(self):
    if self.hyperparameters is None:
      self.network = model.HyperUnet(
        self.num_hyperparams,
        self.hnet_hdim,
        hnet_norm=not self.range_restrict,
        in_ch_main=self.n_ch_in,
        out_ch_main=self.n_ch_out,
        h_ch_main=self.unet_hdim,
      ).to(self.device)
    else:
      self.network = model.Unet(in_ch=self.n_ch_in,
                    out_ch=self.n_ch_out,
                    h_ch=self.unet_hdim).to(self.device)
    print('Total parameters:', utils.count_parameters(self.network))
    return self.network

  def get_optimizer(self):
    return torch.optim.Adam(self.network.parameters(), lr=self.lr)

  def get_sampler(self):
    return sampler.Sampler(self.num_hyperparams)

  def train(self):
    for epoch in range(self.cont+1, self.epochs+1):

      if self.hyperparameters is not None:
        self.r1 = self.hyperparameters
        self.r2 = self.hyperparameters
      elif not self.anneal:
        self.r1 = 0
        self.r2 = 1
      elif epoch < 500 and self.anneal:
        self.r1 = 0.5
        self.r2 = 0.5
      elif epoch < 1000 and self.anneal:
        self.r1 = 0.4
        self.r2 = 0.6
      elif epoch < 1500 and self.anneal:
        self.r1 = 0.2
        self.r2 = 0.8
      elif epoch < 2000 and self.anneal:
        self.r1 = 0
        self.r2 = 1

      print('\nEpoch %d/%d' % (epoch, self.epochs))
      print('Learning rate:',
          self.lr if self.force_lr is None else self.force_lr)
      print('DHS sampling' if self.sampling_method ==
          'dhs' else 'UHS sampling')
      print('Sampling bounds [%.2f, %.2f]' % (self.r1, self.r2))

      # Train
      tic = time.time()
      train_epoch_loss, train_epoch_psnr = self.train_epoch()
      train_epoch_time = time.time() - tic
      # Validate
      tic = time.time()
      val_epoch_loss, val_epoch_psnr = self.eval_epoch()
      val_epoch_time = time.time() - tic


      # Save checkpoints
      self.logger['loss_train'].append(train_epoch_loss)
      self.logger['loss_val'].append(val_epoch_loss)
      self.logger['psnr_train'].append(train_epoch_psnr)
      self.logger['psnr_val'].append(val_epoch_psnr)
      self.logger['epoch_train_time'].append(train_epoch_time)

      utils.save_loss(self.run_dir, self.logger, 'loss_train', 'loss_val', 'epoch_train_time',
              'psnr_train', 'psnr_val')
      if epoch % self.log_interval == 0:
        utils.save_checkpoint(epoch, self.network.state_dict(), self.optimizer.state_dict(),
                    self.ckpt_dir)

      print("Epoch {}: train loss={:.6f}, train psnr={:.6f}, train time={:.6f}".format(
        epoch, train_epoch_loss, train_epoch_psnr, train_epoch_time))
      print("Epoch {}: val loss={:.6f}, val psnr={:.6f}, val time={:.6f}".format(
        epoch, val_epoch_loss, val_epoch_psnr, val_epoch_time))

  def compute_loss(self, pred, gt, y, coeffs):
    '''Compute loss.

    Args:
      coeffs:  (bs, num_losses)
      losses:  (num_losses)
    '''
    loss = 0
    for i in range(len(self.losses)):
      c = coeffs[:, i]
      l = self.losses[i]
      loss += c * l(pred, gt, y, self.mask)

    if self.sampling_method == 'uhs':
      loss = torch.mean(loss)
    else:
      assert self.topK is not None
      dc_losses = loss_dict['dc']
      _, perm = torch.sort(dc_losses) # Sort by DC loss, low to high
      sort_losses = self.losses[perm] # Reorder total losses by lowest to highest DC loss
      hyperparams = hyperparams[perm]
      loss = torch.mean(sort_losses[:self.topK]) # Take only the losses with lowest DC

    return loss

  def prepare_batch(self, batch):
    targets = batch.float().to(self.device)
    segs = None

    under_ksp = utils.generate_measurement(targets, self.mask)
    zf = utils.ifft(under_ksp)
    under_ksp, zf = utils.scale(under_ksp, zf)
    return zf, targets, under_ksp, segs

  def train_epoch(self):
    """Train for one epoch."""
    self.network.train()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    for i, batch in tqdm(enumerate(self.train_loader), total=self.num_steps_per_epoch):
      loss, psnr, batch_size = self.train_step(batch)
      epoch_loss += loss.data.cpu().numpy() * batch_size
      epoch_psnr += psnr * batch_size
      epoch_samples += batch_size
      if i == self.num_steps_per_epoch:
        break

    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return epoch_loss, epoch_psnr

  def train_step(self, batch):
    zf, gt, y, _ = self.prepare_batch(batch)
    batch_size = len(zf)

    self.optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      hyperparams = self.sampler.sample(
        batch_size, self.r1, self.r2).to(self.device)
      coeffs = generate_coefficients(
        hyperparams, len(self.losses), self.range_restrict)
      if self.hyperparameters is None:  # Hypernet
        pred = self.network(zf, hyperparams)
      else:
        pred = self.network(zf)  # Baselines

      loss = self.compute_loss(pred, gt, y, coeffs)
      loss.backward()
      self.optimizer.step()
    psnr = bpsnr(gt, pred)
    return loss, psnr, batch_size

  def eval_epoch(self):
    """Validate for one epoch."""
    self.network.eval()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    for _, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
      zf, gt, y, _ = self.prepare_batch(batch)
      batch_size = len(zf)

      with torch.set_grad_enabled(False):
        if self.hyperparameters is not None:
          hyperparams = torch.ones((batch_size, self.num_hyperparams)).to(
            self.device) * self.hyperparameters
        else:
          hyperparams = torch.ones(
            (batch_size, self.num_hyperparams)).to(self.device)

        coeffs = generate_coefficients(
          hyperparams, len(self.losses), self.range_restrict)
        if self.hyperparameters is None:  # Hypernet
          pred = self.network(zf, hyperparams)
        else:
          pred = self.network(zf)  # Baselines

        loss = self.compute_loss(pred, gt, y, coeffs)
        epoch_loss += loss.data.cpu().numpy() * batch_size
        epoch_psnr += bpsnr(gt, pred) * batch_size

      epoch_samples += batch_size
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return epoch_loss, epoch_psnr