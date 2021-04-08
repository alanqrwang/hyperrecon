import torch
import numpy as np
from hyperrecon import utils, dataset, networks, sampler, metrics, losses
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
        self.add_argument('--log_interval', type=int, default=100, help='Frequency of logs')
        self.add_argument('--load_path', type=str, default=None, help='Load checkpoint at .h5 path')
        self.add_argument('--cont', type=int, default=0, help='Load checkpoint at .h5 path')
        self.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
        self.add_argument('--date', type=str, default=None, help='Override date')
        
        # Machine learning parameters
        self.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
        self.add_argument('--force_lr', type=float, default=None, help='Learning rate')
        self.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.add_argument('--epochs', type=int, default=100, help='Total training epochs')
        self.add_argument('--unet_hdim', type=int, default=64)
        self.add_argument('--hnet_hdim', type=int, default=None, help='Hypernetwork architecture')

        # Model parameters
        self.add_argument('--topK', type=int, default=None)
        self.add_argument('--undersampling_rate', type=int, default=4, choices=[4, 8])
        self.add_argument('--reg_types', nargs='+', type=str, help='<Required> Set flag', required=True)
        self.add_argument('--sampling', choices=['uhs', 'dhs'], type=str, help='Sampling method', required=True)
        utils.add_bool_arg(self, 'range_restrict')
        utils.add_bool_arg(self, 'use_tanh')
        self.add_argument('--weights', nargs='+', type=float, default=None)
        
    def parse(self):
        args = self.parse_args()
        if args.sampling == 'dhs':
            assert args.topK is not None, 'DHS sampling must set topK'
        if args.range_restrict:
            assert len(args.reg_types) <= 2, 'Unsupported num regs and loss type'
        if args.date is None:
            date = '{}'.format(time.strftime('%b_%d'))
        else:
            date = args.date

        args.num_hyperparams = len(args.reg_types) if args.range_restrict else len(args.reg_types) + 1

        if args.weights is None:
            args.run_dir = os.path.join(args.models_dir, args.filename_prefix, date, \
                '{lr}_{batch_size}_{reg_types}_{hnet_hdim}_{unet_hdim}_{topK}_{range_restrict}_{use_tanh}'.format(
                lr=args.lr,
                batch_size=args.batch_size,
                reg_types=args.reg_types,
                hnet_hdim=args.hnet_hdim,
                unet_hdim=args.unet_hdim,
                range_restrict=args.range_restrict,
                topK=args.topK,
                use_tanh=args.use_tanh,
                ))
        else:
            args.run_dir = os.path.join(args.models_dir, args.filename_prefix, date, \
                '{lr}_{batch_size}_{reg_types}_{unet_hdim}_{range_restrict}_{weights}'.format(
                lr=args.lr,
                batch_size=args.batch_size,
                reg_types=args.reg_types,
                unet_hdim=args.unet_hdim,
                range_restrict=args.range_restrict,
                weights=args.weights,
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

args = Parser().parse()

# GPU Handling
if torch.cuda.is_available():
    args.device = torch.device('cuda:'+str(args.gpu_id))
else:
    args.device = torch.device('cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Dataset
xdata = dataset.get_train_data(args.undersampling_rate, old=True)
gt_data = dataset.get_train_gt(old=True)
trainset = dataset.Dataset(xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
valset = dataset.Dataset(xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])
params = {'batch_size': args.batch_size,
     'shuffle': True,
     'num_workers': 4, 
     'pin_memory': True}
dataloaders = {
    'train': torch.utils.data.DataLoader(trainset, **params),
    'val': torch.utils.data.DataLoader(valset, **params),
}

# Undersampling mask
args.mask = dataset.get_mask(args.undersampling_rate).to(args.device)

# Model
model = networks.HyperUnet(args.num_hyperparams, \
        not args.range_restrict, \
        args.unet_hdim, \
        hh=args.hnet_hdim, \
        use_tanh=args.use_tanh).to(args.device)
print('Total parameters:', utils.count_parameters(model))

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Hyperparameter Sampler
hpsampler = sampler.HpSampler(args.num_hyperparams, args.device, args.range_restrict, args.weights)

# Loss
criterion = losses.AmortizedLoss(args.reg_types, args.range_restrict, args.sampling, \
    args.topK, args.device, args.mask)

if args.force_lr is not None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.force_lr


logger = {}
logger['loss_train'] = []
logger['loss_val'] = []
logger['epoch_train_time'] = []
logger['psnr_train'] = []
logger['psnr_val'] = []

# Load from path
if args.load_path:
    model, optimizer = utils.load_checkpoint(model, args.path, optimizer)
# Load from previous checkpoint
if args.cont > 0:
    load_file = os.path.join(args.run_dir, 'checkpoints', 'model.{epoch:04d}.h5'.format(epoch=args.cont))
    model, optimizer = utils.load_checkpoint(model, load_file, optimizer)
    logger['loss_train'] = list(np.loadtxt(os.path.join(args.run_dir, 'loss_train.txt'))[:args.cont])
    logger['loss_val'] = list(np.loadtxt(os.path.join(args.run_dir, 'loss_val.txt'))[:args.cont])
    logger['epoch_train_time'] = list(np.loadtxt(os.path.join(args.run_dir, 'epoch_train_time.txt'))[:args.cont])
    logger['psnr_train'] = list(np.loadtxt(os.path.join(args.run_dir, 'psnr_train.txt'))[:args.cont])
    logger['psnr_val'] = list(np.loadtxt(os.path.join(args.run_dir, 'psnr_val.txt'))[:args.cont])

############## Training loop #####################
for epoch in range(args.cont+1, args.epochs+1):
    print('\nEpoch %d/%d' % (epoch, args.epochs))
    print('Learning rate:', args.lr if args.force_lr is None else args.force_lr)
    print('DHS sampling' if args.sampling == 'dhs' else 'UHS sampling')

    tic = time.time()
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        epoch_loss = 0
        epoch_samples = 0
        epoch_psnr = 0

        for batch_idx, (y, gt) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
            batch_size = len(y)
            y, gt = y.float().to(args.device), gt.float().to(args.device)
            zf = utils.ifft(y)

            hyperparams = hpsampler.sample(batch_size, phase).to(args.device)
            recon = model(y, hyperparams)
            cap_reg = model.get_l1_weight_penalty().to(args.device)
            print(y.shape, gt.shape, recon.shape)
            
            loss, _, _ = criterion(recon, y, hyperparams, cap_reg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO: fix zf 2 channel for computing rpsnr
            epoch_loss += loss.data.cpu().numpy()
            epoch_psnr += metrics.get_metrics(gt.permute(0, 2, 3, 1), \
                        recon.permute(0, 2, 3, 1), zf.permute(0, 2, 3, 1), \
                        'psnr', normalized=True, reduction='sum')
            epoch_samples += batch_size

        epoch_loss /= epoch_samples
        epoch_psnr /= epoch_samples
        if phase == 'train':
            train_epoch_loss = epoch_loss
            train_epoch_psnr = epoch_psnr
        else:
            val_epoch_loss = epoch_loss
            val_epoch_psnr = epoch_psnr

    epoch_time = time.time() - tic

    # Save checkpoints
    logger['loss_train'].append(train_epoch_loss)
    logger['loss_val'].append(val_epoch_loss)
    logger['psnr_train'].append(train_epoch_psnr)
    logger['psnr_val'].append(val_epoch_psnr)
    logger['epoch_train_time'].append(epoch_time)

    utils.save_loss(args.run_dir, logger, 'loss_train', 'loss_val', 'epoch_train_time', \
            'psnr_train', 'psnr_val')
    if epoch % args.log_interval == 0:
        utils.save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), \
                args.ckpt_dir)

    print("Epoch {}: test loss: {:.6f}, test psnr: {:.6f}, time: {:.6f}".format( \
        epoch, val_epoch_loss, val_epoch_psnr, epoch_time))
