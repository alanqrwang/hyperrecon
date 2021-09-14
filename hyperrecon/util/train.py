import torch
from torchvision import transforms
import numpy as np
import os
import time
from tqdm import tqdm
import json
from glob import glob
import random

from hyperrecon.util import utils
from hyperrecon.loss.losses import compose_loss_seq
from hyperrecon.util.metric import bpsnr, bssim, bhfen, dice, bmae, bwatson
from hyperrecon.model.unet import HyperUnet
from hyperrecon.model.layers import ClipByPercentile
from hyperrecon.data.mask import get_mask
from hyperrecon.data.brain import ArrDataset, SliceDataset, SliceVolDataset, get_train_data, get_train_gt


class BaseTrain(object):
  def __init__(self, args):
    self.device = args.device
    self.image_dims = args.image_dims
    # HyperRecon
    self.mask_type = args.mask_type
    self.undersampling_rate = args.undersampling_rate
    self.topK = args.topK
    self.method = args.method
    self.anneal = args.anneal
    self.range_restrict = args.range_restrict
    self.loss_list = args.loss_list
    self.num_hparams = len(self.loss_list) - 1 if self.range_restrict else len(self.loss_list)
    self.num_coeffs = len(self.loss_list)
    # ML
    self.num_epochs = args.num_epochs
    self.lr = args.lr
    self.force_lr = args.force_lr
    self.batch_size = args.batch_size
    self.num_steps_per_epoch = args.num_steps_per_epoch
    self.arr_dataset = args.arr_dataset
    self.hyperparameters = args.hyperparameters
    self.hnet_hdim = args.hnet_hdim
    self.unet_hdim = args.unet_hdim
    self.n_ch_in = 2
    self.n_ch_out = args.n_ch_out
    self.scheduler_step_size = args.scheduler_step_size
    self.scheduler_gamma = args.scheduler_gamma
    self.seed = args.seed
    self.use_batchnorm = args.use_batchnorm
    self.optimizer_type = args.optimizer_type
    # I/O
    self.load = args.load
    self.cont = args.cont
    self.epoch = self.cont + 1
    self.run_dir = args.run_dir
    self.data_path = args.data_path
    self.log_interval = args.log_interval
    self.num_train_subjects = args.num_train_subjects
    self.num_val_subjects = args.num_val_subjects

    self.set_eval_hparams()
    self.set_metrics()

  def set_eval_hparams(self):
    # hparams must be list of tensors, each of shape (num_hyperparams)
    pass

  def set_metrics(self):
    self.list_of_metrics = [
      'loss:train',
      'psnr:train',
    ]
    self.list_of_val_metrics = [
      'loss:val:' + self.stringify_list(l.tolist()) for l in self.val_hparams
    ] + [
      'psnr:val:' + self.stringify_list(l.tolist()) for l in self.val_hparams
    ]
    self.list_of_test_metrics = [
      'loss:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ] + [
      'psnr:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ] + [
      'ssim:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ] + [
      'hfen:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ] + [
      'watson:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ] + [
      'mae:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    # ] + [
    #   'dice:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ]

  def set_random_seed(self):
    seed = self.seed
    if seed > 0:
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)

  def config(self):
    self.set_random_seed()
    # Data
    self.get_dataloader()
    self.mask = get_mask(self.mask_type,
      self.image_dims, self.undersampling_rate).to(self.device)

    # Model, Optimizer, Sampler, Loss
    self.network = self.get_model()
    self.optimizer = self.get_optimizer()
    self.scheduler = self.get_scheduler()
    self.losses = compose_loss_seq(self.loss_list, self.mask, self.device)

    if self.force_lr is not None:
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = self.force_lr

  def get_dataloader(self):
    if self.arr_dataset:
      xdata = get_train_data(maskname=self.undersampling_rate)
      gt_data = get_train_gt()
      trainset = ArrDataset(
        xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
      valset = ArrDataset(
        xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])
    else:
      transform = transforms.Compose([ClipByPercentile()])
      trainset = SliceDataset(
        self.data_path, 'train', total_subjects=self.num_train_subjects, transform=transform)
      valset = SliceDataset(
        self.data_path, 'validate', total_subjects=self.num_val_subjects, transform=transform)
      testset = SliceVolDataset(
        self.data_path, 'validate', total_subjects=self.num_val_subjects, transform=transform,
        subsample=False)

    self.train_loader = torch.utils.data.DataLoader(trainset, 
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=0,
          pin_memory=True)
    self.val_loader = torch.utils.data.DataLoader(valset,
          batch_size=self.batch_size*2,
          shuffle=False,
          num_workers=0,
          pin_memory=True)
    self.test_loader = torch.utils.data.DataLoader(testset,
          batch_size=1,
          shuffle=False,
          num_workers=0,
          pin_memory=True)

  def get_model(self):
    self.network = HyperUnet(
      self.num_coeffs,
      self.hnet_hdim,
      in_ch_main=self.n_ch_in,
      out_ch_main=self.n_ch_out,
      h_ch_main=self.unet_hdim,
      use_batchnorm=self.use_batchnorm
    ).to(self.device)
    utils.summary(self.network)
    return self.network

  def get_optimizer(self):
    if self.optimizer_type == 'sgd':
      return torch.optim.SGD(self.network.parameters(), lr=self.lr)
    else:
      return torch.optim.Adam(self.network.parameters(), lr=self.lr)

  def get_scheduler(self):
    return torch.optim.lr_scheduler.StepLR(self.optimizer,
                         step_size=self.scheduler_step_size,
                         gamma=self.scheduler_gamma)

  def train(self):
    self.train_begin()
    if self.num_epochs == 0:
      self.train_epoch_begin()
      self.train_epoch_end(is_val=False, save_metrics=False, save_ckpt=False)
    else:
      for epoch in range(self.start_epoch, self.num_epochs+1):
        self.epoch = epoch

        self.train_epoch_begin()
        self.train_epoch()
        self.train_epoch_end(is_val=True, save_metrics=True, save_ckpt=(
          self.epoch % self.log_interval == 0))
      self.train_epoch_end(is_val=False, save_metrics=True, save_ckpt=True)
    self.train_end(verbose=True)

  def manage_checkpoint(self):
    '''Manage checkpoints.
    
    If self.load is set, loads from path specified by self.load.
    If self.cont is set, loads from current model directory at
      epoch specified by self.cont.
    Otherwise, loads from most recent saved model in current 
      model directory, if it exists.

    Assumes model path has the form:
      <ckpt_dir>/model.{epoch:04d}.h5
    '''
    load_path = None
    cont_epoch = None
    if self.load:  # Load from path
      load_path = self.load
    elif self.cont > 0:  # Load from previous checkpoint
      cont_epoch = self.cont
    else: # Try to load from latest checkpoint
      model_paths = sorted(glob(os.path.join(self.ckpt_dir, '*')))
      if len(model_paths) == 0 and self.num_epochs > 0:
        print('Randomly initialized model')
      elif len(model_paths) > 0:
        load_path = model_paths[-1]
        cont_epoch = int(load_path.split('.')[-2])
      else:
        raise ValueError('No model found for prediction', self.run_dir)

    if cont_epoch is not None:
      load_path = os.path.join(
        self.ckpt_dir, 'model.{epoch:04d}.h5'.format(epoch=cont_epoch))

      self.metrics.update({key: list(np.loadtxt(os.path.join(
        self.metric_dir, key + '.txt')))[:cont_epoch] for key in self.list_of_metrics})
      self.val_metrics.update({key: list(np.loadtxt(os.path.join(
        self.metric_dir, key + '.txt')))[:cont_epoch] for key in self.list_of_val_metrics})
      self.monitor.update({'learning_rate': list(np.loadtxt(os.path.join(
        self.monitor_dir, 'learning_rate.txt')))[:cont_epoch]})
      self.monitor.update({'time:train': list(np.loadtxt(os.path.join(
        self.monitor_dir, 'time:train.txt')))[:cont_epoch]})
    if load_path is not None:
      self.network, self.optimizer, self.scheduler = utils.load_checkpoint(
        self.network, load_path, self.optimizer, self.scheduler)

  def train_begin(self):
    self.start_epoch = self.cont + 1
    # Logging
    self.metrics = {}
    self.metrics.update({key: [] for key in self.list_of_metrics})

    self.val_metrics = {}
    self.val_metrics.update({key: []
                  for key in self.list_of_val_metrics})

    self.test_metrics = {}
    self.test_metrics.update({key: []
                  for key in self.list_of_test_metrics})
    self.monitor = {
      'learning_rate': [],
      'time:train': [],
    }

    # Directories to save information
    self.ckpt_dir = os.path.join(self.run_dir, 'checkpoints')
    if not os.path.exists(self.ckpt_dir):
      os.makedirs(self.ckpt_dir)
    self.metric_dir = os.path.join(self.run_dir, 'metrics')
    if not os.path.exists(self.metric_dir):
      os.makedirs(self.metric_dir)
    self.monitor_dir = os.path.join(self.run_dir, 'monitor')
    if not os.path.exists(self.monitor_dir):
      os.makedirs(self.monitor_dir)
    self.img_dir = os.path.join(self.run_dir, 'img')
    if not os.path.exists(self.img_dir):
      os.makedirs(self.img_dir)

    # Checkpoint Loading
    self.manage_checkpoint()

  def train_end(self, verbose=False):
    """Called at the end of training.

    Save summary statistics in json format
    Print in command line some basic statistics

    Args:
      verbose: Boolean. Print messages if True.
    """
    if verbose:
      summary_dict = {}
      summary_dict.update({key: self.val_metrics[key][-1]
                 for key in self.list_of_val_metrics})
      summary_dict.update({key: self.test_metrics[key][-1]
                 for key in self.list_of_test_metrics})
      
      with open(os.path.join(self.run_dir, 'summary_full.json'),
            'w') as outfile:
        json.dump(summary_dict, outfile, sort_keys=True, indent=4)

      # Print basic information
      print('')
      print('---------------------------------------------------------------')
      print('Train is done. Below are file path and basic test stats\n')
      print('File path:\n')
      print(self.run_dir)
      print('Eval stats:\n')
      print(json.dumps(summary_dict, sort_keys=True, indent=4))
      print('---------------------------------------------------------------')
      print()

  def train_epoch_begin(self):
    print('\nEpoch %d/%d' % (self.epoch, self.num_epochs))
    print('Learning rate:', self.scheduler.get_last_lr())

  def train_epoch_end(self, is_val=True, save_metrics=False, save_ckpt=False):
    '''Save loss and checkpoints. Evaluate if necessary.'''
    self.eval_epoch(is_val)

    if save_metrics:
      utils.save_metrics(self.metric_dir, self.metrics, *self.list_of_metrics)
      utils.save_metrics(self.metric_dir, self.val_metrics,
                *self.list_of_val_metrics)
      utils.save_metrics(self.monitor_dir, self.monitor, 'learning_rate', 'time:train')
    if save_ckpt:
      utils.save_checkpoint(self.epoch, self.network, self.optimizer,
                  self.ckpt_dir, self.scheduler)

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
    assert len(self.losses) == coeffs.shape[1], 'loss and coeff mismatch'
    loss = 0
    for i in range(len(self.losses)):
      c = coeffs[:, i]
      l = self.losses[i]
      loss += c * l(pred, gt, y)
    return loss

  def process_loss(self, loss):
    '''Process loss.

    Args:
      loss: Per-sample loss (bs)

    Returns:
      Scalar loss value
    '''
    return loss.mean()

  def prepare_batch(self, batch, is_training=True):
    targets, segs = batch[0].float().to(self.device), batch[1].float().to(self.device)
    if not is_training:
      targets = targets.view(-1, 1, *targets.shape[-2:])
      segs = segs.view(-1, 1, *targets.shape[-2:])

    under_ksp = utils.generate_measurement(targets, self.mask)
    zf = utils.ifft(under_ksp)
    under_ksp, zf = utils.scale(under_ksp, zf)
    return zf, targets, under_ksp, segs

  def inference(self, zf, coeffs):
    return self.network(zf, coeffs)

  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    pass

  def generate_coefficients(self, samples):
    '''Generates coefficients from samples.'''
    if self.range_restrict and len(self.losses) == 2:
      alpha = samples[:, 0]
      coeffs = torch.stack((1-alpha, alpha), dim=1)

    elif self.range_restrict and len(self.losses) == 3:
      alpha = samples[:, 0]
      beta = samples[:, 1]
      coeffs = torch.stack(
        (alpha, (1-alpha)*beta, (1-alpha)*(1-beta)), dim=1)

    else:
      coeffs = samples / torch.sum(samples, dim=1)

    return coeffs.to(self.device)

  def train_epoch(self):
    """Train for one epoch."""
    self.network.train()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    start_time = time.time()
    for i, batch in tqdm(enumerate(self.train_loader), total=self.num_steps_per_epoch):
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

  def train_step(self, batch):
    '''Train for one step.'''
    zf, gt, y, _ = self.prepare_batch(batch)
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

  def eval_epoch(self, is_val):
    '''Eval for one epoch.
    
    For each hyperparameter, computes all reconstructions for that 
    hyperparameter in the test set. 
    '''
    self.network.eval()

    if is_val:
      self.validate()
    else:
      self.test(save_preds=True)
  
  def validate(self):
    for hparam in self.val_hparams:
      hparam_str = self.stringify_list(hparam.tolist())
      print('Validating with hparam', hparam_str)
      zf, gt, y, pred, segs, coeffs = self.get_predictions(hparam)
      for key in self.val_metrics:
        if 'loss' in key and hparam_str in key:
          loss = self.compute_loss(pred, gt, y, coeffs)
          loss = self.process_loss(loss).item()
          self.val_metrics[key].append(loss)
        elif 'psnr' in key and hparam_str in key:
          self.val_metrics[key].append(bpsnr(gt, pred))
        elif 'ssim' in key and hparam_str in key:
          self.val_metrics[key].append(bssim(gt, pred))
        elif 'hfen' in key and hparam_str in key:
          self.val_metrics[key].append(bhfen(gt, pred))

  def test(self, save_preds=False):
    for hparam in self.test_hparams:
      hparam_str = self.stringify_list(hparam.tolist())
      print('Testing with hparam', hparam_str)
      zf, gt, y, pred, seg, coeffs = self.get_predictions(hparam, by_subject=True)
      for i in range(len(zf)):
        # Save predictions to disk
        if save_preds:
          gt_path = os.path.join(self.img_dir, 'gt' + 'sub{}'.format(i) + '.npy')
          zf_path = os.path.join(self.img_dir, 'zf' + 'sub{}'.format(i) + '.npy')
          pred_path = os.path.join(self.img_dir, 'pred'+hparam_str+'sub{}'.format(i) + '.npy')
          if not os.path.exists(pred_path):
            np.save(pred_path, pred[i].cpu().detach().numpy())
          if not os.path.exists(gt_path):
            np.save(gt_path, gt[i].cpu().detach().numpy())
          if not os.path.exists(zf_path):
            np.save(zf_path, zf[i].cpu().detach().numpy())
        for key in self.test_metrics:
          if 'loss' in key and hparam_str in key and 'sub{}'.format(i) in key:
            loss = self.compute_loss(pred[i], gt[i], y[i], coeffs[i])
            loss = self.process_loss(loss).item()
            self.test_metrics[key].append(loss)
          elif 'psnr' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(bpsnr(gt[i], pred[i]))
          elif 'ssim' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(bssim(gt[i], pred[i]))
          elif 'hfen' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(bhfen(gt[i], pred[i]))
          elif 'watson' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(bwatson(gt[i], pred[i]))
          elif 'mae' in key and hparam_str in key and 'sub{}'.format(i) in key:
            self.test_metrics[key].append(bmae(gt[i], pred[i]))
          elif 'dice' in key and hparam_str in key and 'sub{}'.format(i) in key:
            loss_roi, _,_,_,_ = dice(pred[i], gt[i], seg[i])
            self.test_metrics[key].append(float(loss_roi.mean()))

  def get_predictions(self, hparam, by_subject=False):
    '''Get predictions, optionally separated by subject'''
    zfs = []
    ys = []
    gts = []
    preds = []
    segs = []
    coeffs = []

    loader = self.test_loader if by_subject else self.val_loader
    for _, batch in tqdm(enumerate(loader), total=len(loader)):
      zf, y, gt, pred, seg, coeff = self.eval_step(batch, hparam)

      zfs.append(zf)
      ys.append(y)
      gts.append(gt)
      preds.append(pred)
      segs.append(seg)
      coeffs.append(coeff)

    if by_subject:
      return zfs, gts, ys, preds, segs, coeffs
    else:
      return torch.cat(zfs, dim=0), torch.cat(gts, dim=0),  \
             torch.cat(ys, dim=0), torch.cat(preds, dim=0), \
             torch.cat(segs, dim=0), torch.cat(coeffs, dim=0)


  def eval_step(self, batch, hparams):
    '''Eval for one step.
    
    Args:
      batch: Single batch from dataloader
      hparams: Single hyperparameter vector (1, num_hyperparams)
    '''
    zf, gt, y, seg = self.prepare_batch(batch, is_training=False)
    batch_size = len(zf)
    hparams = hparams.repeat(batch_size, 1)

    with torch.set_grad_enabled(False):
      coeffs = self.generate_coefficients(hparams)
      pred = self.inference(zf, coeffs)

    return zf, y, gt, pred, seg, coeffs

  @staticmethod
  def stringify_list(l):
    if not isinstance(l, (list, tuple)):
      l = [l]
    s = str(l[0])
    for i in range(1, len(l)):
      s += '_' + str(l[i])
    return s
