import torch
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
from hyperrecon.model.unet import Unet, HyperUnet
from hyperrecon.model.loupeunet import LoupeUnet, LoupeHyperUnet, ConditionalLoupeHyperUnet
from hyperrecon.model.image import SimpleImage
# from hyperrecon.model.unet_v2 import Unet, HyperUnet, LastLayerHyperUnet
from hyperrecon.util.forward import CSMRIForward, DenoisingForward, InpaintingForward, SuperresolutionForward
from hyperrecon.data.mask import EPIHorizontal, EPIVertical, VDSPoisson, FirstHalf, SecondHalf, CenterPatch, RandomBox
from hyperrecon.data.knee import FastMRI, KneeArr, KneeArrSingle
from hyperrecon.data.brain import Abide, BrainArr
from hyperrecon.data.cardiac import ACDC
from hyperrecon.util.sample import Uniform, UniformConstant, Constant, CategoricalConstant


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
    self.epoch_of_p_max = args.epoch_of_p_max
    self.p_min = args.p_min
    self.p_max = args.p_max
    self.additive_gauss_std = args.additive_gauss_std
    self.beta = args.beta
    self.unet_residual = args.unet_residual
    self.forward_type = args.forward_type
    self.distance_type = args.distance_type
    self.denoising_sigma = args.denoising_sigma
    self.distribution = args.distribution
    self.uniform_bounds = args.uniform_bounds
    # ML
    self.dataset = args.dataset
    self.num_epochs = args.num_epochs
    self.lr = args.lr
    self.batch_size = args.batch_size
    self.num_steps_per_epoch = args.num_steps_per_epoch
    self.hyperparameters = args.hyperparameters
    self.arch = args.arch
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
    self.run_dir = args.run_dir
    self.data_path = args.data_path
    self.log_interval = args.log_interval
    self.num_train_subjects = args.num_train_subjects
    self.num_val_subjects = args.num_val_subjects
    self.dc_scale = args.dc_scale

    self.set_eval_hparams()
    self.set_monitor()
    self.set_metrics()

  def set_eval_hparams(self):
    # hparams must be list of tensors, each of shape (num_hyperparams)
    if self.distribution == 'constant':
      self.val_hparams = torch.tensor(self.hyperparameters).view(-1, 1)
      self.test_hparams = torch.tensor(self.hyperparameters).view(-1, 1)
    else:
      self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
      # self.test_hparams = torch.tensor([0., 0.25, 0.5, 0.75, 1.]).view(-1, 1)
      self.test_hparams = torch.tensor(np.linspace(0, 1.00, 20)).float().view(-1, 1)
      # self.val_hparams = torch.tensor([[0.,0.], [1.,1.]])
      # hparams = []
      # for i in np.linspace(0, 1, 50):
      #   for j in np.linspace(0, 1, 50):
      #     hparams.append([i, j])
      # self.test_hparams = torch.tensor(hparams).float()

  def set_monitor(self):
    self.list_of_monitor = [
      'learning_rate', 
      'time:train',
    ]

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
    self.per_loss_scale_constants = self.get_per_loss_scale_constants()

    self.get_dataloader()
    self.mask_module = self.get_mask()
    self.forward_model = self.get_forward_model()
    self.sampler = self.get_sampler()

    self.network = self.get_model()
    self.optimizer = self.get_optimizer()
    self.scheduler = self.get_scheduler()
    self.losses = compose_loss_seq(self.loss_list, self.forward_model, self.mask_module, self.device)

  def get_per_loss_scale_constants(self):
    # Constants for mean losses on test sets.
    # TODO: handle this better
    # L1 + SSIM
    if self.stringify_list(self.loss_list) == 'l1_ssim':
      if self.mask_type == 'poisson' and self.undersampling_rate == '16p3' and self.dataset == 'abide' and self.forward_type == 'csmri':
        scales = [0.05797722685674671, 0.27206547738363346]
      elif self.mask_type == 'poisson' and self.undersampling_rate == '8p3' and self.dataset == 'abide' and self.forward_type == 'csmri':
        scales = [0.012755771, 0.0489692]
      elif self.mask_type == 'poisson' and self.undersampling_rate == '8p3' and self.dataset == 'knee_arr' and self.forward_type == 'csmri':
        scales = [0.04525472, 0.31786856]
      elif self.mask_type == 'epi_vertical' and self.undersampling_rate == '4' and self.dataset == 'knee_arr' and self.forward_type == 'csmri':
        scales = [0.041984833776950836, 0.2628784775733948]
      elif self.mask_type == 'epi_vertical' and self.undersampling_rate == '8' and self.dataset == 'knee_arr' and self.forward_type == 'csmri':
        scales = [1, 1]
      elif self.undersampling_rate == '4' and self.dataset == 'knee_arr' and self.forward_type == 'superresolution':
        scales = [0.037357181310653687, 0.3851676881313324]
      elif self.denoising_sigma == 0.1 and self.dataset == 'knee_arr' and self.forward_type == 'denoising':
        scales = [0.03143206238746643, 0.27412155270576477]
      else:
        scales = [1, 1]
        # scales = [0.05797722685674671, 0.27206547738363346]
    # DC + TV
    elif self.stringify_list(self.loss_list) == 'dc_tv':
      # scales = [self.dc_scale, (1-self.dc_scale)]
      # scales = [0.151795, 0.0616901] # knee_arr, poisson 8p3, worst-case
      scales = [1,1]
    # MinNormDC + TV
    elif self.stringify_list(self.loss_list) == 'mindc_tv':
      scales = [1, 1]
    else:
      raise ValueError('No loss scale constants found.')
    print('\nusing loss scales', scales)
    return scales

  def get_mask(self):
    if self.mask_type == 'poisson':
      mask = VDSPoisson(self.image_dims, self.undersampling_rate)
    elif self.mask_type == 'epi_horizontal':
      mask = EPIHorizontal(self.image_dims, self.undersampling_rate)
    elif self.mask_type == 'epi_vertical':
      mask = EPIVertical(self.image_dims, self.undersampling_rate)
    elif self.mask_type == 'first_half':
      mask = FirstHalf(self.image_dims, self.undersampling_rate)
    elif self.mask_type == 'second_half':
      mask = SecondHalf(self.image_dims, self.undersampling_rate)
    elif self.mask_type == 'center_patch':
      mask = CenterPatch(self.image_dims, self.undersampling_rate)
    elif self.mask_type == 'random_box':
      mask = RandomBox(self.image_dims)
    elif self.mask_type == 'loupe':
      mask = None
    return mask

  def get_sampler(self):
    if self.distribution == 'uniform':
      sampler = Uniform(*self.uniform_bounds)
    elif self.distribution == 'uniform_constant':
      sampler = UniformConstant(*self.uniform_bounds)
    elif self.distribution == 'constant':
      sampler = Constant(self.hyperparameters)
    elif self.distribution == 'categorical_constant':
      sampler = CategoricalConstant([0, 0.5, 1])
    return sampler

  def get_dataloader(self):
    if self.dataset == 'brain_arr':
      dataset = BrainArr(self.batch_size)
    elif self.dataset == 'abide':
      dataset = Abide(self.batch_size, self.num_train_subjects, self.num_val_subjects)
    elif self.dataset == 'knee_arr':
      dataset = KneeArr(self.batch_size)
    elif self.dataset == 'knee_arr_single':
      dataset = KneeArrSingle(self.batch_size)
    elif self.dataset == 'fastmri':
      dataset = FastMRI(self.batch_size, img_dims=self.image_dims)
    elif self.dataset == 'acdc':
      dataset = ACDC(self.batch_size, img_dims=self.image_dims)
    self.train_loader, self.val_loader, self.test_loader = dataset.load()

  def get_model(self):
    if self.arch == 'simple_img':
      data = next(iter(self.train_loader)).float()
      self.network = SimpleImage(self.image_dims, data).to(self.device)
    elif self.arch == 'unet':
      self.network = Unet(
                      in_ch=self.n_ch_in,
                      out_ch=self.n_ch_out,
                      h_ch=self.unet_hdim,
                      residual=self.unet_residual,
                      use_batchnorm=self.use_batchnorm
                   ).to(self.device)
    elif self.arch == 'hyperunet':
      self.network = HyperUnet(
                        self.num_coeffs,
                        self.hnet_hdim,
                        in_ch_main=self.n_ch_in,
                        out_ch_main=self.n_ch_out,
                        h_ch_main=self.unet_hdim,
                        residual=self.unet_residual,
                        use_batchnorm=self.use_batchnorm
                      ).to(self.device)
    elif self.arch == 'loupe_unet':
      self.network = LoupeUnet(
                        in_ch=self.n_ch_in,
                        out_ch=self.n_ch_out,
                        h_ch=self.unet_hdim,
                        image_dims=self.image_dims,
                        undersampling_rate=self.undersampling_rate,
                        residual=self.unet_residual,
                        use_batchnorm=self.use_batchnorm
                      ).to(self.device)
    elif self.arch == 'loupe_hyperunet':
      self.network = LoupeHyperUnet(
                        1,
                        self.hnet_hdim,
                        in_ch_main=self.n_ch_in,
                        out_ch_main=self.n_ch_out,
                        h_ch_main=self.unet_hdim,
                        image_dims=self.image_dims,
                        residual=self.unet_residual,
                        use_batchnorm=self.use_batchnorm
                      ).to(self.device)
    elif self.arch == 'condloupe_hyperunet':
      self.network = ConditionalLoupeHyperUnet(
                        1,
                        self.hnet_hdim,
                        in_ch_main=self.n_ch_in,
                        out_ch_main=self.n_ch_out,
                        h_ch_main=self.unet_hdim,
                        image_dims=self.image_dims,
                        residual=self.unet_residual,
                        use_batchnorm=self.use_batchnorm
                      ).to(self.device)
    else:
      raise ValueError('No architecture found')
    # elif self.arch == 'last_layer_hyperunet':
    #   self.network = LastLayerHyperUnet(
    #                     self.num_coeffs,
    #                     self.hnet_hdim,
    #                     in_ch_main=self.n_ch_in,
    #                     out_ch_main=self.n_ch_out,
    #                     h_ch_main=self.unet_hdim,
    #                     residual=self.unet_residual,
    #                     use_batchnorm=self.use_batchnorm
    #                   ).to(self.device)

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

  def get_forward_model(self):
    if self.forward_type == 'csmri':
      self.forward_model = CSMRIForward()
    elif self.forward_type == 'inpainting':
      self.forward_model = InpaintingForward()
    elif self.forward_type == 'superresolution':
      self.forward_model = SuperresolutionForward(self.undersampling_rate)
    elif self.forward_type == 'denoising':
      fixed_noise = True if self.num_epochs == 0 else False
      self.forward_model = DenoisingForward(self.denoising_sigma, self.image_dims, fixed_noise)
    return self.forward_model

  def train(self):
    self.train_begin()
    self.epoch = self.start_epoch
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
      self.start_epoch = 1
    elif self.cont > 0:  # Load from previous checkpoint
      cont_epoch = self.cont
      self.start_epoch = self.cont + 1
    else: # Try to load from latest checkpoint
      model_paths = sorted(glob(os.path.join(self.ckpt_dir, '*')))
      if len(model_paths) == 0 and self.num_epochs > 0:
        print('Randomly initialized model')
        self.start_epoch = 1
      elif len(model_paths) > 0:
        cont_epoch = int(model_paths[-1].split('.')[-2])
        self.start_epoch = cont_epoch + 1
      else:
        raise ValueError('No model found for prediction', self.run_dir)

    if cont_epoch is not None:
      load_path = os.path.join(
        self.ckpt_dir, 'model.{epoch:04d}.h5'.format(epoch=cont_epoch))

      self.metrics.update({key: list(np.loadtxt(os.path.join(
        self.metric_dir, key + '.txt')))[:cont_epoch] for key in self.list_of_metrics})
      self.val_metrics.update({key: list(np.loadtxt(os.path.join(
        self.metric_dir, key + '.txt')))[:cont_epoch] for key in self.list_of_val_metrics})
      self.monitor.update({key: list(np.loadtxt(os.path.join(
        self.monitor_dir, key + '.txt')))[:cont_epoch] for key in self.list_of_monitor})
    if load_path is not None:
      self.network, self.optimizer, self.scheduler = utils.load_checkpoint(
        self.network, load_path, self.optimizer, self.scheduler)
    
  def train_begin(self):
    # Logging
    self.metrics = {}
    self.metrics.update({key: [] for key in self.list_of_metrics})

    self.val_metrics = {}
    self.val_metrics.update({key: []
                  for key in self.list_of_val_metrics})

    self.test_metrics = {}
    self.test_metrics.update({key: []
                  for key in self.list_of_test_metrics})
    self.monitor = {}
    self.monitor.update({key: []
                  for key in self.list_of_monitor})

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
      utils.save_metrics(self.monitor_dir, self.monitor, *self.list_of_monitor)
    if save_ckpt:
      utils.save_checkpoint(self.epoch, self.network, self.optimizer,
                  self.ckpt_dir, self.scheduler)

  def compute_loss(self, pred, gt, coeffs, scales, is_training=False):
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
      per_loss_scale = scales[i]
      loss += c / per_loss_scale * l(pred, gt, network=self.network)
    return loss

  def process_loss(self, loss):
    '''Process loss.

    Args:
      loss: Per-sample loss (bs)

    Returns:
      Scalar loss value
    '''
    return loss.mean()

  def inference(self, zf, coeffs):
    return self.network(zf, coeffs)

  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    return self.sampler((num_samples, self.num_hparams))

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

  def prepare_batch(self, batch):
    targets = batch.view(-1, 1, *batch.shape[-2:]).float().to(self.device)
    bs = len(targets)

    undersample_mask = self.mask_module(bs).to(self.device)
    measurements = self.forward_model(targets, undersample_mask)
    if self.forward_type == 'csmri':
      inputs = utils.ifft(measurements)
    else:
      inputs = measurements
    return inputs, targets, bs

  def train_step(self, batch):
    '''Train for one step.'''
    inputs, targets, batch_size = self.prepare_batch(batch)
    hparams = self.sample_hparams(batch_size)
    coeffs = self.generate_coefficients(hparams)

    self.optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      pred = self.inference(inputs, coeffs)
      loss = self.compute_loss(pred, targets, coeffs, scales=self.per_loss_scale_constants, is_training=True)
      loss = self.process_loss(loss)
      loss.backward()
      self.optimizer.step()
    psnr = bpsnr(targets, pred)
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
      self.test(save_preds=False)
  
  def validate(self):
    for hparam in self.val_hparams:
      hparam_str = self.stringify_list(hparam.tolist())
      print('Validating with hparam', hparam_str)
      _, gt, pred, coeffs = self.get_predictions(hparam)
      for key in self.val_metrics:
        if 'loss' in key and hparam_str in key:
          loss = self.compute_loss(pred, gt, coeffs, [1, 1], is_training=False)
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
      input, gt, pred, coeffs = self.get_predictions(hparam, by_subject=True)
      for i in range(len(input)):
        # Save predictions to disk
        if save_preds:
          gt_path = os.path.join(self.img_dir, 'gt' + 'sub{}'.format(i) + '.npy')
          zf_path = os.path.join(self.img_dir, 'zf' + 'sub{}'.format(i) + '.npy')
          pred_path = os.path.join(self.img_dir, 'pred'+hparam_str+'sub{}'.format(i)+'cp{:04d}'.format(self.epoch-1) + '.npy')
          np.save(pred_path, pred[i].cpu().detach().numpy())
          if not os.path.exists(gt_path):
            np.save(gt_path, gt[i].cpu().detach().numpy())
          if not os.path.exists(zf_path):
            np.save(zf_path, input[i].cpu().detach().numpy())
        for key in self.test_metrics:
          if 'loss' in key and hparam_str in key and 'sub{}'.format(i) in key:
            loss = self.compute_loss(pred[i], gt[i], coeffs[i], [1, 1], is_training=False)
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
          # elif 'dice' in key and hparam_str in key and 'sub{}'.format(i) in key:
          #   loss_roi, _,_,_,_ = dice(pred[i], gt[i], seg[i])
          #   self.test_metrics[key].append(float(loss_roi.mean()))

  def get_predictions(self, hparam, by_subject=False):
    '''Get predictions, optionally separated by subject'''
    all_inputs = []
    all_gts = []
    all_preds = []
    all_coeffs = []

    loader = self.test_loader if by_subject else self.val_loader
    for batch in tqdm(loader, total=len(loader)):
      input, gt, pred, coeff = self.eval_step(batch, hparam)

      all_inputs.append(input)
      all_gts.append(gt)
      all_preds.append(pred)
      all_coeffs.append(coeff)

    if by_subject:
      return all_inputs, all_gts, all_preds, all_coeffs
    else:
      return torch.cat(all_inputs, dim=0), torch.cat(all_gts, dim=0),  \
             torch.cat(all_preds, dim=0), torch.cat(all_coeffs, dim=0)


  def eval_step(self, batch, hparams):
    '''Eval for one step.
    
    Args:
      batch: Single batch from dataloader
      hparams: Single hyperparameter vector (1, num_hyperparams)
    '''
    inputs, targets, batch_size = self.prepare_batch(batch)
    hparams = hparams.repeat(batch_size, 1)
    coeffs = self.generate_coefficients(hparams)
    with torch.set_grad_enabled(False):
      pred = self.inference(inputs, coeffs)
    return inputs, targets, pred, coeffs

  @staticmethod
  def stringify_list(l):
    if not isinstance(l, (list, tuple)):
      l = [l]
    s = str(l[0])
    for i in range(1, len(l)):
      s += '_' + str(l[i])
    return s
