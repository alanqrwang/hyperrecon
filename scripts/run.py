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
        self.add_argument('-fp', '--filename_prefix', type=str, help='filename prefix', required=True)
        # parser.add_argument('--models_dir', default='out/', type=str, help='directory to save models')
        self.add_argument('--models_dir', default='/nfs02/users/aw847/models/HyperRecon/', type=str, help='directory to save models')
        
        self.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.add_argument('--force_lr', type=float, default=None, help='Force learning rate')
        self.add_argument('--force_date', type=str, default=None, help='Force date')
        self.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.add_argument('--num_epochs', type=int, default=100, help='Total training epochs')
        self.add_argument('--load_checkpoint', type=int, default=0, help='Load checkpoint at specificed epoch')
        self.add_argument('--log_interval', type=int, default=100, help='Frequency of logs')
        self.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
        self.add_argument('--unet_hidden', type=int, default=64)
        self.add_argument('--topK', type=int, default=None)
        self.add_argument('--loss_schedule', type=int, default=0)
        self.add_argument('--sample_schedule', type=int, default=0)
        utils.add_bool_arg(self, 'range_restrict')
        self.add_argument('--undersampling_rate', type=int, default=4, choices=[4, 8])

        self.add_argument('--reg_types', nargs='+', type=str, help='<Required> Set flag', required=True)
        self.add_argument('--sampling', choices=['uhs', 'dhs'], type=str, help='Sampling method', required=True)
        self.add_argument('--hyparch', choices=['small', 'medium', 'large'], type=str, help='Hypernetwork architecture', required=True)
        
    def parse(self):
        args = self.parse_args()
        date = '{}'.format(time.strftime('%b_%d'))
        if args.force_date:
            date = args.force_date

        args.run_dir = os.path.join(args.models_dir, args.filename_prefix, date, \
            '{lr}_{batch_size}_{reg_types}_{unet_hidden}_{topK}_{range_restrict}'.format(
            lr=args.lr,
            batch_size=args.batch_size,
            reg_types=args.reg_types,
            unet_hidden=args.unet_hidden,
            range_restrict=args.range_restrict,
            topK=args.topK
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

    params = {'batch_size': args['batch_size'],
         'shuffle': True,
         'num_workers': 4}
    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params),
    }
    ##################################################

    ##### Model, Optimizer, Sampler, Loss ############
    num_hyperparams = len(args['reg_types']) if args['range_restrict'] else len(args['reg_types']) + 1
    network = model.Unet(args['device'], num_hyperparams=num_hyperparams, hyparch=args['hyparch'], \
                nh=args['unet_hidden']).to(args['device'])

    print(utils.count_parameters(network))
    sys.exit()
    optimizer = torch.optim.Adam(network.parameters(), lr=args['lr'])
    if args['force_lr'] is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args['force_lr']

    hpsampler = sampler.HpSampler(num_hyperparams)
    criterion = hyperloss.AmortizedLoss(args['reg_types'], args['range_restrict'], args['sampling'], args['device'], args['mask'])
    ##################################################

    ############ Checkpoint Loading ##################
    if args['load_checkpoint'] != 0:
        pretrain_path = os.path.join(args['ckpt_dir'], 'model.{epoch:04d}.h5'.format(epoch=args['load_checkpoint']))
        print(pretrain_path)
        network, optimizer = utils.load_checkpoint(network, pretrain_path, optimizer)
        # pretrain_path = '/nfs02/users/aw847/models/HyperHQSNet/8fold_1e-05_42_[\'cap\', \'tv\']_64_None_True/model.5000.h5'
        # network = utils.load_checkpoint(network, pretrain_path)
    ##################################################

    ############## Training loop #####################
    for epoch in range(args['load_checkpoint']+1, args['num_epochs']+1):
        tic = time.time()

        print('\nEpoch %d/%d' % (epoch, args['num_epochs']))
        if args['force_lr'] is not None:
            print('Force learning rate:', args['force_lr'])
        else:
            print('Learning rate:', args['lr'])

        # Setting hyperparameter sampling parameters.
        # topK is number in mini-batch to backprop. If None, then uniform
        if args['sampling'] == 'dhs' and epoch > args['sample_schedule']:
            assert args['topK'] is not None
            topK = args['topK']
            print('DHS sampling')
        else:
            topK = None
            print('UHS sampling')

        # Loss scheduling. If activated, then first 100 epochs is trained 
        # as single reg function
        if len(args['reg_types']) <= 2 and args['range_restrict']:
            if epoch > args['loss_schedule']:
                schedule = True
                print('Loss schedule: 2 regs')
            else:
                schedule = False
                print('Loss schedule: 1 reg')
        else:
            if args['loss_schedule'] > 0:
                div = epoch // args['loss_schedule']
                schedule = min(div, len(args['reg_types']))
                print('%d losses' % schedule)
            else:
                schedule = len(args['reg_types'])
                print('%d losses' % schedule)


        # Train
        network, optimizer, train_epoch_loss, train_epoch_psnr = train.train(network, dataloaders['train'], \
                criterion, optimizer, hpsampler, args, topK, schedule)
        # Validate
        network, val_epoch_loss, val_epoch_psnr = train.validate(network, dataloaders['val'], criterion, hpsampler, args, \
                topK, schedule)
	
        epoch_train_time = time.time() - tic

        # Save checkpoints
        logger['loss_train'].append(train_epoch_loss)
        logger['loss_val'].append(val_epoch_loss)
        logger['psnr_train'].append(train_epoch_psnr)
        logger['psnr_val'].append(val_epoch_psnr)
        logger['epoch_train_time'].append(epoch_train_time)
        utils.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), \
                logger, args['ckpt_dir'], args['log_interval'])
        utils.save_loss(args['run_dir'], logger, 'loss_train', 'loss_val', 'epoch_train_time', \
                'psnr_train', 'psnr_val')

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
    mask = dataset.get_mask(args.undersampling_rate)
    args.mask = torch.tensor(mask, requires_grad=False).float().to(args.device)

    # Train
    trainer(xdata, gt_data, vars(args))
