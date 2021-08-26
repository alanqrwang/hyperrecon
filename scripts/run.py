from loss.losses import compose_loss_seq
from loss.coefficients import generate_coefficients
import torch
import torchio as tio
from torchvision import transforms
import numpy as np
from hyperrecon import utils, dataset, model, sampler, layers
import argparse
import os
import json
from pprint import pprint
import time
from tqdm import tqdm

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='HyperRecon')
        # I/O parameters
        self.add_argument('-fp', '--filename_prefix', type=str, help='filename prefix', required=True)
        self.add_argument('--models_dir', default='/share/sablab/nfs02/users/aw847/models/HyperRecon/', type=str, help='directory to save models')
        self.add_argument('--data_path', default='/share/sablab/nfs02/users/aw847/data/brain/abide/',
                type=str, help='directory to load data')
        self.add_argument('--log_interval', type=int, default=25, help='Frequency of logs')
        self.add_argument('--load', type=str, default=None, help='Load checkpoint at .h5 path')
        self.add_argument('--cont', type=int, default=0, help='Load checkpoint at .h5 path')
        self.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
        self.add_argument('--date', type=str, default=None, help='Override date')
        self.add_bool_arg('legacy_dataset', default=False)
        
        # Machine learning parameters
        self.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
        self.add_argument('--force_lr', type=float, default=None, help='Learning rate')
        self.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.add_argument('--epochs', type=int, default=100, help='Total training epochs')
        self.add_argument('--unet_hdim', type=int, default=32)
        self.add_argument('--hnet_hdim', type=int, help='Hypernetwork architecture', default=64)
        self.add_argument('--n_ch_out', type=int, help='Number of output channels of main network', default=1)
        self.add_bool_arg('rescale_in', default=True)

        # Model parameters
        self.add_argument('--topK', type=int, default=None)
        self.add_argument('--undersampling_rate', type=str, default='4p2', choices=['4p2', '8p25', '8p3', '16p2', '16p3'])
        self.add_argument('--loss_list', choices=['dc', 'tv', 'cap', 'wave', 'shear', 'mse', 'l1', 'ssim', 'watson-dft'], \
                nargs='+', type=str, help='<Required> Set flag', required=True)
        self.add_argument('--sampling_method', choices=['uhs', 'dhs'], type=str, help='Sampling method', required=True)
        self.add_bool_arg('range_restrict')
        self.add_bool_arg('anneal', default=False)
        self.add_argument('--hyperparameters', type=float, default=None)

    def add_bool_arg(self, name, default=True):
        """Add boolean argument to argparse parser"""
        group = self.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no_' + name, dest=name, action='store_false')
        self.set_defaults(**{name:default})

    def parse(self):
        args = self.parse_args()
        if args.sampling_method == 'dhs':
            assert args.topK is not None, 'DHS sampling must set topK'
        if args.date is None:
            date = '{}'.format(time.strftime('%b_%d'))
        else:
            date = args.date

        args.run_dir = os.path.join(args.models_dir, args.filename_prefix, date, \
            '{lr}_{batch_size}_{losses}_{hnet_hdim}_{unet_hdim}_{topK}_{range_restrict}_{hps}'.format(
            lr=args.lr,
            batch_size=args.batch_size,
            losses=args.loss_list,
            hnet_hdim=args.hnet_hdim,
            unet_hdim=args.unet_hdim,
            range_restrict=args.range_restrict,
            topK=args.topK,
            hps=args.hyperparameters,
            ))

        args.ckpt_dir = os.path.join(args.run_dir, 'checkpoints')
        if not os.path.isdir(args.ckpt_dir):   
            os.makedirs(args.ckpt_dir)

        # Print args and save to file
        print('Arguments:')
        pprint(vars(args))
        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)
        return args

class BaseTrain(object):
    def __init__(self, args):
        # HyperRecon
        self.mask = dataset.get_mask('160_224', args.undersampling_rate).to(args.device)
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
        self.legacy_dataset = args.legacy_dataset
        self.loss_list = args.loss_list
        self.hyperparameters = args.hyperparameters
        self.rescale_in = args.rescale_in
        self.hnet_hdim = args.hnet_hdim
        self.unet_hdim = args.unet_hdim
        self.n_ch_in = 1 if self.rescale_in else 2
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
        #  Data
        self.get_dataloader()

        # Model, Optimizer, Sampler, Loss
        self.num_hyperparams = len(self.loss_list)-1 if self.range_restrict else len(self.loss_list)

        self.network = self.get_model()
        self.optimizer = self.get_optimizer() 
        self.sampler = self.get_sampler()
        self.losses = compose_loss_seq(self.loss_list)

        if self.force_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.force_lr

        # Checkpoint Loading 
        if self.load: # Load from path
            load_path = self.load
        elif self.cont > 0: # Load from previous checkpoint
            load_path = os.path.join(self.run_dir, 'checkpoints', 'model.{epoch:04d}.h5'.format(epoch=self.cont))
            self.logger['loss_train'] = list(np.loadtxt(os.path.join(self.run_dir, 'loss_train.txt'))[:self.cont])
            self.logger['loss_val'] = list(np.loadtxt(os.path.join(self.run_dir, 'loss_val.txt'))[:self.cont])
            self.logger['epoch_train_time'] = list(np.loadtxt(os.path.join(self.run_dir, 'epoch_train_time.txt'))[:self.cont])
            self.logger['psnr_train'] = list(np.loadtxt(os.path.join(self.run_dir, 'psnr_train.txt'))[:self.cont])
            self.logger['psnr_val'] = list(np.loadtxt(os.path.join(self.run_dir, 'psnr_val.txt'))[:self.cont])
        else:
            load_path = None
    
        if load_path is not None:
            self.network, self.optimizer = utils.load_checkpoint(self.network, load_path, self.optimizer)

    def get_dataloader(self):
        if self.legacy_dataset:
            xdata = dataset.get_train_data(maskname=self.undersampling_rate)
            gt_data = dataset.get_train_gt()
            trainset = dataset.Dataset(xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
            valset = dataset.Dataset(xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])
        else:
            transform = transforms.Compose([layers.ClipByPercentile()])
            trainset = dataset.SliceDataset(self.data_path, 'train', total_subjects=50, transform=transform)
            valset = dataset.SliceDataset(self.data_path, 'validate', total_subjects=5, transform=transform)

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
            tic = time.time()

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
            print('Learning rate:', self.lr if self.force_lr is None else self.force_lr)
            print('DHS sampling' if self.sampling_method == 'dhs' else 'UHS sampling')
            print('Sampling bounds [%.2f, %.2f]' % (self.r1, self.r2))

            # Train
            train_epoch_loss, train_epoch_psnr = self.train_epoch()
            # Validate
            eval_epoch_loss, eval_epoch_psnr = self.eval_epoch()
            
            epoch_train_time = time.time() - tic

            # Save checkpoints
            self.logger['loss_train'].append(train_epoch_loss)
            self.logger['loss_val'].append(eval_epoch_loss)
            self.logger['psnr_train'].append(train_epoch_psnr)
            self.logger['psnr_val'].append(eval_epoch_psnr)
            self.logger['epoch_train_time'].append(epoch_train_time)

            utils.save_loss(self.run_dir, self.logger, 'loss_train', 'loss_val', 'epoch_train_time', \
                    'psnr_train', 'psnr_val')
            if epoch % self.log_interval == 0:
                utils.save_checkpoint(epoch, self.network.state_dict(), self.optimizer.state_dict(), \
                        self.ckpt_dir)

            print("Epoch {}: train loss: {:.6f}, train psnr: {:.6f}, time: {:.6f}".format( \
                epoch, train_epoch_loss, train_epoch_psnr, epoch_train_time))
            print("Epoch {}: test loss: {:.6f}, test psnr: {:.6f}, time: {:.6f}".format( \
                epoch, eval_epoch_loss, eval_epoch_psnr, epoch_train_time))


    def compute_loss(self, pred, gt, y, coeffs):
        '''
        Args:
            coeffs:  (bs, num_losses)
            losses:  (num_losses)
        '''
        loss = 0
        for i in range(len(self.losses)):
            c = coeffs[:,i]
            l = self.losses[i]
            loss += c * l(pred, gt, y, self.mask)
        
        if self.topK is None:
            loss = torch.mean(loss)
        else:
            assert self.sampling_method == 'dhs'
            dc_losses = loss_dict['dc']
            _, perm = torch.sort(dc_losses) # Sort by DC loss, low to high
            sort_losses = self.losses[perm] # Reorder total losses by lowest to highest DC loss
            hyperparams = hyperparams[perm]
            loss = torch.mean(sort_losses[:self.topK]) # Take only the losses with lowest DC

        return loss

    def prepare_batch(self, batch):
        if type(batch) == list:
            targets = batch[0].to(self.device).float()
            segs = batch[1].to(self.device).float()

        else:
            targets = batch.float().to(self.device)
            segs = None

        under_ksp = utils.undersample(targets, self.mask)
        zf = utils.ifft(under_ksp)

        if self.rescale_in:
            zf = zf.norm(p=2, dim=-1, keepdim=True)
            zf = utils.rescale(zf)
        else:
            under_ksp, zf = utils.scale(under_ksp, zf)

        return zf, targets, under_ksp, segs

    def train_epoch(self):
        """Train for one epoch"""
        self.network.train()

        epoch_loss = 0
        epoch_samples = 0
        epoch_psnr = 0

        for _, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            zf, gt, y, _ = self.prepare_batch(batch)
            batch_size = len(zf)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):

                hyperparams = self.sampler.sample(batch_size, self.r1, self.r2).to(self.device)
                coeffs = generate_coefficients(hyperparams, len(self.losses), self.range_restrict)
                if args.hyperparameters is None: # Hypernet
                    pred = self.network(zf, hyperparams)
                else:
                    pred = self.network(zf) # Baselines

                loss = self.compute_loss(pred, gt, y, coeffs)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.data.cpu().numpy() * batch_size
                epoch_psnr += np.mean(utils.get_metrics(gt.permute(0, 2, 3, 1), \
                        pred.permute(0, 2, 3, 1), zf.permute(0, 2, 3, 1), \
                        'psnr', take_absval=False)) * batch_size
            epoch_samples += batch_size

        epoch_loss /= epoch_samples
        epoch_psnr /= epoch_samples
        return epoch_loss, epoch_psnr

    def eval_epoch(self):
        """Validate for one epoch"""
        self.network.eval()

        epoch_loss = 0
        epoch_samples = 0
        epoch_psnr = 0

        for _, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
            zf, gt, y, _ = self.prepare_batch(batch)
            batch_size = len(zf)

            with torch.set_grad_enabled(False):
                if self.hyperparameters is not None:
                    hyperparams = torch.ones((batch_size, self.num_hyperparams)).to(self.device) * self.hyperparameters
                else:
                    hyperparams = torch.ones((batch_size, self.num_hyperparams)).to(self.device)

                coeffs = generate_coefficients(hyperparams, len(self.losses), self.range_restrict)
                if self.hyperparameters is None: # Hypernet
                    pred = self.network(zf, hyperparams)
                else:
                    pred = self.network(zf) # Baselines

                loss = self.compute_loss(pred, gt, y, coeffs)
                epoch_loss += loss.data.cpu().numpy() * batch_size
                metrics = utils.get_metrics(gt.permute(0, 2, 3, 1), pred.permute(0, 2, 3, 1), \
                        zf.permute(0, 2, 3, 1), 'psnr', take_absval=False)
                epoch_psnr += np.mean(metrics) * batch_size

            epoch_samples += batch_size
        epoch_loss /= epoch_samples
        epoch_psnr /= epoch_samples
        return epoch_loss, epoch_psnr


if __name__ == "__main__":
    args = Parser().parse()
    
    # GPU Handling
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    trainer = BaseTrain(args)
    trainer.config()
    trainer.train()

    