import torch
from torchvision import transforms
import numpy as np
import os
import time
from tqdm import tqdm
import json

from hyperrecon import sampler
from hyperrecon.loss.losses import compose_loss_seq
from hyperrecon.util.metric import bpsnr, bssim
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
        self.loss_list = args.loss_list
        self.num_hyperparams = len(self.loss_list)
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
        # I/O
        self.load = args.load
        self.cont = args.cont
        self.epoch = self.cont + 1
        self.device = args.device
        self.run_dir = args.run_dir
        self.ckpt_dir = args.ckpt_dir
        self.data_path = args.data_path
        self.log_interval = args.log_interval

        self.set_val_hparams()
        self.set_metrics()

    def set_val_hparams(self):
        self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)

    def set_metrics(self):
        self.list_of_metrics = [
            'loss.train',
            'psnr.train',
            'time.train',
        ]
        self.list_of_eval_metrics = [
            'loss.val' + self.stringify_list(l.tolist()) for l in self.val_hparams
        ] + [
            'psnr.val' + self.stringify_list(l.tolist()) for l in self.val_hparams
        ] + [
            'ssim.val' + self.stringify_list(l.tolist()) for l in self.val_hparams
        ]
        # 'hfen.val',
        # ]

    def config(self):
        # Data
        self.get_dataloader()

        # Model, Optimizer, Sampler, Loss
        self.network = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.sampler = self.get_sampler()
        self.losses = compose_loss_seq(self.loss_list)

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
            in_ch_main=self.n_ch_in,
            out_ch_main=self.n_ch_out,
            h_ch_main=self.unet_hdim,
        ).to(self.device)
        utils.summary(self.network)
        return self.network

    def get_optimizer(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def get_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=64, gamma=0.5)

    def get_sampler(self):
        return sampler.Sampler(self.num_hyperparams)

    def train(self):
        self.train_begin()
        if self.num_epochs == 0:
            self.train_epoch_begin()
            self.train_epoch_end(is_eval=True, is_save=False)
        else:
            for epoch in range(self.start_epoch, self.num_epochs+1):
                self.epoch = epoch

                self.train_epoch_begin()
                self.train_epoch()
                self.train_epoch_end(is_eval=True, is_save=(
                    self.epoch % self.log_interval == 0))
            self.train_epoch_end(is_eval=True, is_save=True)
        self.train_end(verbose=True)

    def train_begin(self):
        self.start_epoch = self.cont + 1
        # Logging
        self.metrics = {}
        self.metrics.update({key: [] for key in self.list_of_metrics})

        self.eval_metrics = {}
        self.eval_metrics.update({key: []
                                 for key in self.list_of_eval_metrics})

        self.monitor = {
            'learning_rate': []
        }
        # Checkpoint Loading
        if self.load:  # Load from path
            load_path = self.load
        elif self.cont > 0:  # Load from previous checkpoint
            load_path = os.path.join(
                self.run_dir, 'checkpoints', 'model.{epoch:04d}.h5'.format(epoch=self.cont))
            self.metrics.update({key: list(np.loadtxt(os.path.join(
                self.run_dir, key + '.txt')))[:self.cont] for key in self.list_of_metrics})
            self.eval_metrics.update({key: list(np.loadtxt(os.path.join(
                self.run_dir, key + '.txt')))[:self.cont] for key in self.list_of_eval_metrics})
        else:
            load_path = None

        if load_path is not None:
            self.network, self.optimizer = utils.load_checkpoint(
                self.network, load_path, self.optimizer)

    def train_end(self, verbose=False):
        """Called at the end of training.

        Save summary statistics in json format
        Print in command line some basic statistics

        Args:
          verbose: Boolean. Print messages if True.
        """
        if verbose:
            summary_dict = {}
            summary_dict.update({key: self.eval_metrics[key][-1]
                                 for key in self.list_of_eval_metrics})
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
        self.r1 = 0
        self.r2 = 1

        print('\nEpoch %d/%d' % (self.epoch, self.num_epochs))
        print('Learning rate:', self.scheduler.get_last_lr())
        print('Sampling bounds [%.2f, %.2f]' % (self.r1, self.r2))

    def train_epoch_end(self, is_eval=False, is_save=False):
        '''Save loss and checkpoints. Evaluate if necessary.'''
        if is_eval:
            self.eval_epoch()

        utils.save_metrics(self.run_dir, self.metrics, *self.list_of_metrics)
        utils.save_metrics(self.run_dir, self.eval_metrics,
                           *self.list_of_eval_metrics)
        utils.save_metrics(self.run_dir, self.monitor, 'learning_rate')
        if is_save:
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
        loss = 0
        for i in range(len(self.losses)):
            c = coeffs[:, i]
            l = self.losses[i]
            loss += c * l(pred, gt, y, self.mask)
        return loss

    def process_loss(self, loss):
        '''Process loss.

        Args:
          loss: Per-sample loss (bs)

        Returns:
          Scalar loss value
        '''
        return loss.mean()

    def prepare_batch(self, batch):
        targets = batch.float().to(self.device)
        segs = None

        under_ksp = utils.generate_measurement(targets, self.mask)
        zf = utils.ifft(under_ksp)
        under_ksp, zf = utils.scale(under_ksp, zf)
        return zf, targets, under_ksp, segs

    def inference(self, zf, hyperparams):
        return self.network(zf, hyperparams)

    def sample_hparams(self, num_samples):
        '''Samples hyperparameters from distribution.'''
        hyperparams = self.sampler.sample(
            num_samples, self.r1, self.r2)
        return hyperparams

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
        self.metrics['loss.train'].append(epoch_loss)
        self.metrics['psnr.train'].append(epoch_psnr)
        self.metrics['time.train'].append(epoch_time)
        self.monitor['learning_rate'].append(self.scheduler.get_last_lr())

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

    def eval_epoch(self):
        '''Eval for one epoch.'''
        self.network.eval()

        for hparam in self.val_hparams:
            hparam_str = self.stringify_list(hparam.tolist())
            zf, gt, y, pred, coeffs = self.get_predictions(hparam)
            for key in self.eval_metrics:
                if 'loss' in key and hparam_str in key:
                    loss = self.compute_loss(pred, gt, y, coeffs)
                    loss = self.process_loss(loss).item()
                    self.eval_metrics[key].append(loss)
                elif 'psnr' in key and hparam_str in key:
                    self.eval_metrics[key].append(bpsnr(gt, pred))
                elif 'ssim' in key and hparam_str in key:
                    self.eval_metrics[key].append(bssim(gt, pred))

    def get_predictions(self, hparam):
        print('Evaluating with hparam', hparam)
        zfs = None
        ys = None
        gts = None
        preds = None
        coeffs = None

        for _, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
            zf, y, gt, pred, coeff = self.eval_step(batch, hparam)

            zfs = self.smart_concat(zfs, zf)
            ys = self.smart_concat(ys, y)
            gts = self.smart_concat(gts, gt)
            preds = self.smart_concat(preds, pred)
            coeffs = self.smart_concat(coeffs, coeff)

        return zfs, gts, ys, preds, coeffs

    def eval_step(self, batch, hparams):
        '''Eval for one step.'''
        zf, gt, y, _ = self.prepare_batch(batch)
        batch_size = len(zf)
        hparams = hparams.repeat(batch_size, 1)

        with torch.set_grad_enabled(False):
            coeffs = self.generate_coefficients(hparams)
            pred = self.inference(zf, coeffs)

        return zf, y, gt, pred, coeffs

    @staticmethod
    def stringify_list(l):
        if not isinstance(l, (list, tuple)):
            l = [l]
        s = str(l[0])
        for i in range(1, len(l)):
            s += '_' + str(l[i])
        return s

    @staticmethod
    def smart_concat(var1, var2):
        """Smart concat."""

        def _smart_concat(var1, var2):
            return var2 if var1 is None else torch.cat((var1, var2), dim=0)

        if isinstance(var2, list):
            if var1 is not None:
                assert isinstance(var1, list)
                return [_smart_concat(v1, v2) for v1, v2 in zip(var1, var2)]
            else:
                return var2
        else:
            if var1 is not None:
                assert not isinstance(var1, list)
            return _smart_concat(var1, var2)
