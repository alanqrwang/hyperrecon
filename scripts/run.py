import torch
import numpy as np
from hyperrecon import utils, train, dataset, model, sampler
from hyperrecon import loss as hyperloss
import argparse
import os
import json
from pprint import pprint
import time


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='HyperRecon')
        # I/O parameters
        self.add_argument('-fp', '--filename_prefix', type=str, help='filename prefix', required=True)
        self.add_argument('--models_dir', default='/nfs02/users/aw847/models/HyperRecon/', type=str, help='directory to save models')
        self.add_argument('--log_interval', type=int, default=100, help='Frequency of logs')
        self.add_argument('--load', type=str, default=None, help='Load checkpoint at .h5 path')
        self.add_argument('--cont', type=int, default=None, help='Load checkpoint at .h5 path')
        self.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
        
        # Machine learning parameters
        self.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.add_argument('--epochs', type=int, default=100, help='Total training epochs')
        self.add_argument('--unet_hdim', type=int, default=64)
        self.add_argument('--hnet_hdim', type=int, help='Hypernetwork architecture', required=True)

        # Model parameters
        self.add_argument('--topK', type=int, default=None)
        self.add_argument('--loss_schedule', type=int, default=None)
        self.add_argument('--undersampling_rate', type=int, default=4, choices=[4, 8])
        self.add_argument('--reg_types', nargs='+', type=str, help='<Required> Set flag', required=True)
        self.add_argument('--sampling', choices=['uhs', 'dhs'], type=str, help='Sampling method', required=True)
        utils.add_bool_arg(self, 'range_restrict')
        
    def parse(self):
        args = self.parse_args()
        if args.sampling == 'dhs':
            assert args.topK is not None, 'DHS sampling must set topK'
        date = '{}'.format(time.strftime('%b_%d'))

        args.run_dir = os.path.join(args.models_dir, args.filename_prefix, date, \
            '{lr}_{batch_size}_{reg_types}_{hnet_hdim}_{unet_hdim}_{topK}_{range_restrict}_{schedule}'.format(
            lr=args.lr,
            batch_size=args.batch_size,
            reg_types=args.reg_types,
            hnet_hdim=args.hnet_hdim,
            unet_hdim=args.unet_hdim,
            range_restrict=args.range_restrict,
            topK=args.topK,
            schedule=args.loss_schedule
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

def trainer(xdata, gt_data, args):
    """Training loop. 

    Parameters
    ----------
    xdata : Dataset of under-sampled measurements
    gt_data : Dataset of fully-sampled images
    args : Miscellaneous parameters

    Returns
    ----------
    network : Main network and hypernetwork
    optimizer : Adam optimizer
    epoch_loss : Loss for this epoch
    """
    logger = {}
    logger['loss_train'] = []
    logger['loss_val'] = []
    logger['epoch_train_time'] = []
    logger['psnr_train'] = []
    logger['psnr_val'] = []

    ###############  Dataset ########################
    trainset = dataset.Dataset(xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
    valset = dataset.Dataset(xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])

    params = {'batch_size': args.batch_size,
         'shuffle': True,
         'num_workers': 4}
    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params),
    }
    ##################################################

    ##### Model, Optimizer, Sampler, Loss ############
    num_hyperparams = len(args.reg_types) if args.range_restrict else len(args.reg_types) + 1
    network = model.Unet(args.device, num_hyperparams, args.hnet_hdim, \
                args.unet_hdim).to(args.device)

    print('Total parameters:', utils.count_parameters(network))
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    hpsampler = sampler.HpSampler(num_hyperparams, args.device, args.range_restrict)
    criterion = hyperloss.AmortizedLoss(args.reg_types, args.range_restrict, args.sampling, \
        args.topK, args.device, args.mask)
    ##################################################

    ############ Checkpoint Loading ##################
    # Load from path
    if args.load:
        network, optimizer = utils.load_checkpoint(network, args.load, optimizer)
    # Load from previous checkpoint
    if args.cont:
        load_file = os.path.join(args.run_dir, 'checkpoints', 'model.{epoch:04d}.h5'.format(epoch=args.cont))
        network, optimizer = utils.load_checkpoint(network, load_file, optimizer)
        logger['loss_train'] = list(np.loadtxt(os.path.join(args.run_dir, 'loss_train.txt'))[:args.cont])
        logger['loss_val'] = list(np.loadtxt(os.path.join(args.run_dir, 'loss_val.txt'))[:args.cont])
        logger['epoch_train_time'] = list(np.loadtxt(os.path.join(args.run_dir, 'epoch_train_time.txt'))[:args.cont])
        logger['psnr_train'] = list(np.loadtxt(os.path.join(args.run_dir, 'psnr_train.txt'))[:args.cont])
        logger['psnr_val'] = list(np.loadtxt(os.path.join(args.run_dir, 'psnr_val.txt'))[:args.cont])

    ##################################################

    ############## Training loop #####################
    for epoch in range(1, args.epochs+1):
        tic = time.time()

        print('\nEpoch %d/%d' % (epoch, args.epochs))
        print('Learning rate:', args.lr)
        print('DHS sampling' if args.sampling == 'dhs' else 'UHS sampling')

        # Loss scheduling for >2 hyperparameters. 
        if args.loss_schedule is None:
            schedule = len(args.reg_types)
            print('%d losses' % schedule)
        else:
            schedule = min(epoch//args.loss_schedule, len(args.reg_types))
            print('%d losses' % schedule)

        # Train
        network, optimizer, train_epoch_loss, train_epoch_psnr = train.train(network, dataloaders['train'], \
                criterion, optimizer, hpsampler, args.device, schedule)
        # Validate
        network, val_epoch_loss, val_epoch_psnr = train.validate(network, dataloaders['val'], criterion, hpsampler, args.device, \
                schedule)
        
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
            epoch, val_epoch_loss, val_epoch_psnr, epoch_train_time))

if __name__ == "__main__":
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
    if gt_data.shape[-1] == 1:
        print('Appending complex dimension into gt...')
        gt_data = np.concatenate((gt_data, np.zeros(gt_data.shape)), axis=3)

    # Undersampling mask
    args.mask = dataset.get_mask(args.undersampling_rate).to(args.device)

    # Train
    trainer(xdata, gt_data, args)
