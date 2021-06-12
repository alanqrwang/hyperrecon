import torch
import torchio as tio
from torchvision import transforms
import numpy as np
from hyperrecon import utils, dataset, model, sampler, layers
from hyperrecon import loss as hyperloss
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
        utils.add_bool_arg(self, 'legacy_dataset')
        
        # Machine learning parameters
        self.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.add_argument('--force_lr', type=float, default=None, help='Learning rate')
        self.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.add_argument('--epochs', type=int, default=100, help='Total training epochs')
        self.add_argument('--unet_hdim', type=int, default=32)
        self.add_argument('--hnet_hdim', type=int, help='Hypernetwork architecture', required=True)
        self.add_argument('--n_ch_out', type=int, help='Number of output channels of main network', default=1)
        utils.add_bool_arg(self, 'rescale_in', default=True)

        # Model parameters
        self.add_argument('--topK', type=int, default=None)
        self.add_argument('--undersampling_rate', type=str, default='4p2', choices=['4p2', '8p25', '8p3', '16p2', '16p3'])
        self.add_argument('--losses', choices=['dc', 'tv', 'cap', 'w', 'sh', 'mse', 'l1', 'ssim', 'perc'], \
                nargs='+', type=str, help='<Required> Set flag', required=True)
        self.add_argument('--sampling', choices=['uhs', 'dhs'], type=str, help='Sampling method', required=True)
        utils.add_bool_arg(self, 'range_restrict')
        utils.add_bool_arg(self, 'anneal', default=False)
        self.add_argument('--hyperparameters', type=float, default=None)
        
    def parse(self):
        args = self.parse_args()
        if args.sampling == 'dhs':
            assert args.topK is not None, 'DHS sampling must set topK'
        if args.date is None:
            date = '{}'.format(time.strftime('%b_%d'))
        else:
            date = args.date

        args.run_dir = os.path.join(args.models_dir, args.filename_prefix, date, \
            '{lr}_{batch_size}_{losses}_{hnet_hdim}_{unet_hdim}_{topK}_{range_restrict}_{hps}'.format(
            lr=args.lr,
            batch_size=args.batch_size,
            losses=args.losses,
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

def train(network, dataloader, criterion, optimizer, hpsampler, args):
    """Train for one epoch"""
    network.train()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        zf, gt, y, _ = utils.prepare_batch(batch, vars(args))
        batch_size = len(zf)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):

            hyperparams = hpsampler.sample(batch_size, args.r1, args.r2).to(args.device)
            print('hyperparams', hyperparams)
            if args.hyperparameters is None: # Hypernet
                preds = network(zf, hyperparams)
            else:
                preds = network(zf) # Baselines

            loss = criterion(preds, y, hyperparams, None, target=gt)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.cpu().numpy() * batch_size
            epoch_psnr += np.mean(utils.get_metrics(gt.permute(0, 2, 3, 1), \
                    preds.permute(0, 2, 3, 1), zf.permute(0, 2, 3, 1), \
                    'psnr', take_absval=False)) * batch_size
        epoch_samples += batch_size

    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return network, optimizer, epoch_loss, epoch_psnr

def validate(network, dataloader, criterion, hpsampler, args):
    """Validate for one epoch"""
    network.eval()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        zf, gt, y, _ = utils.prepare_batch(batch, vars(args), split='test')
        batch_size = len(zf)

        with torch.set_grad_enabled(False):
            if args.hyperparameters is not None:
                hyperparams = torch.ones((batch_size, args.num_hyperparams)).to(args.device) * args.hyperparameters
            else:
                hyperparams = torch.ones((batch_size, args.num_hyperparams)).to(args.device)

            if args.hyperparameters is None: # Hypernet
                preds = network(zf, hyperparams)
            else:
                preds = network(zf) # Baselines
            loss = criterion(preds, y, hyperparams, None, target=gt)
                
            epoch_loss += loss.data.cpu().numpy() * batch_size
            metrics = utils.get_metrics(gt.permute(0, 2, 3, 1), preds.permute(0, 2, 3, 1), \
                    zf.permute(0, 2, 3, 1), 'psnr', take_absval=False)
            epoch_psnr += np.mean(metrics) * batch_size

        epoch_samples += batch_size
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return network, epoch_loss, epoch_psnr


if __name__ == "__main__":
    args = Parser().parse()
    
    # GPU Handling
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Undersampling mask
    args.mask = dataset.get_mask('160_224', args.undersampling_rate).to(args.device)

    # Train
    logger = {}
    logger['loss_train'] = []
    logger['loss_val'] = []
    logger['loss_val2'] = []
    logger['epoch_train_time'] = []
    logger['psnr_train'] = []
    logger['psnr_val'] = []

    ###############  Dataset ########################
    if args.legacy_dataset:
        xdata = dataset.get_train_data(maskname=args.undersampling_rate)
        gt_data = dataset.get_train_gt()
        trainset = dataset.Dataset(xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
        valset = dataset.Dataset(xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])
        params = {'batch_size': args.batch_size,
             'shuffle': True,
             'num_workers': 0, 
             'pin_memory': True}

    else:
        transform = transforms.Compose([layers.ClipByPercentile()])
        trainset = dataset.SliceDataset(args.data_path, 'train', total_subjects=50, transform=transform)
        valset = dataset.SliceDataset(args.data_path, 'validate', total_subjects=5, transform=transform)
        params = {'batch_size': args.batch_size,
             'shuffle': True,
             'num_workers': 0, 
             'pin_memory': True}

    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params),
    }
    ##################################################

    ##### Model, Optimizer, Sampler, Loss ############
    args.num_hyperparams = len(args.losses)-1 if args.range_restrict else len(args.losses)
    if args.hyperparameters is None:
        network = model.HyperUnet(
                         args.num_hyperparams,
                         args.hnet_hdim,
                         hnet_norm=not args.range_restrict,
                         in_ch_main=1 if args.rescale_in else 2,
                         out_ch_main=args.n_ch_out,
                         h_ch_main=args.unet_hdim, 
                         ).to(args.device)
    else:
        network = model.Unet(in_ch=1 if args.rescale_in else 2,
                             out_ch=args.n_ch_out, 
                             h_ch=args.unet_hdim).to(args.device)
    print('Total parameters:', utils.count_parameters(network))

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    hpsampler = sampler.Sampler(args.num_hyperparams)
    criterion = hyperloss.AmortizedLoss(args.losses, args.range_restrict, args.sampling, \
            args.topK, args.device, args.mask)
    ##################################################

    if args.force_lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.force_lr

    ############ Checkpoint Loading ##################
    # Load from path
    if args.load:
        load_path = args.load
    # Load from previous checkpoint
    elif args.cont > 0:
        load_path = os.path.join(args.run_dir, 'checkpoints', 'model.{epoch:04d}.h5'.format(epoch=args.cont))
        logger['loss_train'] = list(np.loadtxt(os.path.join(args.run_dir, 'loss_train.txt'))[:args.cont])
        logger['loss_val'] = list(np.loadtxt(os.path.join(args.run_dir, 'loss_val.txt'))[:args.cont])
        logger['epoch_train_time'] = list(np.loadtxt(os.path.join(args.run_dir, 'epoch_train_time.txt'))[:args.cont])
        logger['psnr_train'] = list(np.loadtxt(os.path.join(args.run_dir, 'psnr_train.txt'))[:args.cont])
        logger['psnr_val'] = list(np.loadtxt(os.path.join(args.run_dir, 'psnr_val.txt'))[:args.cont])
    else:
        load_path = None
 
    if load_path is not None:
        network, optimizer = utils.load_checkpoint(network, load_path, optimizer)
    ##################################################

    ############## Training loop #####################
    for epoch in range(args.cont+1, args.epochs+1):
        tic = time.time()

        if args.hyperparameters is not None:
            args.r1 = args.hyperparameters
            args.r2 = args.hyperparameters
        elif not args.anneal:
            args.r1 = 0
            args.r2 = 1
        elif epoch < 500 and args.anneal:
            args.r1 = 0.5
            args.r2 = 0.5
        elif epoch < 1000 and args.anneal:
            args.r1 = 0.4
            args.r2 = 0.6
        elif epoch < 1500 and args.anneal:
            args.r1 = 0.2
            args.r2 = 0.8
        elif epoch < 2000 and args.anneal:
            args.r1 = 0
            args.r2 = 1

        print('\nEpoch %d/%d' % (epoch, args.epochs))
        print('Learning rate:', args.lr if args.force_lr is None else args.force_lr)
        print('DHS sampling' if args.sampling == 'dhs' else 'UHS sampling')
        print('Sampling bounds [%.2f, %.2f]' % (args.r1, args.r2))

        # Train
        network, optimizer, train_epoch_loss, train_epoch_psnr = train(network, dataloaders['train'], \
                criterion, optimizer, hpsampler, args)
        # Validate
        network, val_epoch_loss, val_epoch_psnr = validate(network, dataloaders['val'], criterion, hpsampler, args)
        
        epoch_train_time = time.time() - tic

        # Save checkpoints
        logger['loss_train'].append(train_epoch_loss)
        logger['loss_val'].append(val_epoch_loss)
        logger['psnr_train'].append(train_epoch_psnr)
        logger['psnr_val'].append(val_epoch_psnr)
        logger['epoch_train_time'].append(epoch_train_time)

        utils.save_loss(args.run_dir, logger, 'loss_train', 'loss_val', 'epoch_train_time', \
                'psnr_train', 'psnr_val')
        if epoch % args.log_interval == 0:
            utils.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), \
                    args.ckpt_dir)

        print("Epoch {}: test loss: {:.6f}, test psnr: {:.6f}, time: {:.6f}".format( \
            epoch, train_epoch_loss, train_epoch_psnr, epoch_train_time))
