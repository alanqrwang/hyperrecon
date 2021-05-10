import torch
import torchio as tio
import numpy as np
from hyperrecon import utils, dataset, model, sampler
from hyperrecon import loss as hyperloss
import argparse
import os
import json
from pprint import pprint
import time
from tqdm import tqdm
import multiprocessing


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='HyperRecon')
        # I/O parameters
        self.add_argument('-fp', '--filename_prefix', type=str, help='filename prefix', required=True)
        self.add_argument('--models_dir', default='/share/sablab/nfs02/users/aw847/models/HyperRecon/', type=str, help='directory to save models')
        self.add_argument('--data_path', default='/share/sablab/nfs02/users/gid-dalcaav/projects/neuron/data/t1_mix/proc/resize256-crop_x32/', 
                type=str, help='directory to load data')
        self.add_argument('--log_interval', type=int, default=100, help='Frequency of logs')
        self.add_argument('--load', type=str, default=None, help='Load checkpoint at .h5 path')
        self.add_argument('--cont', type=int, default=0, help='Load checkpoint at .h5 path')
        self.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
        self.add_argument('--date', type=str, default=None, help='Override date')
        
        # Machine learning parameters
        self.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.add_argument('--force_lr', type=float, default=None, help='Learning rate')
        self.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.add_argument('--epochs', type=int, default=100, help='Total training epochs')
        self.add_argument('--unet_hdim', type=int, default=32)
        self.add_argument('--hnet_hdim', type=int, help='Hypernetwork architecture', required=True)
        self.add_argument('--n_ch_out', type=int, help='Number of output channels of main network', default=1)
        self.add_argument('--task', type=str, choices=['recon', 'seg'], default='recon')

        # Model parameters
        self.add_argument('--topK', type=int, default=None)
        self.add_argument('--undersampling_rate', type=str, default='8p3', choices=['4p2', '8p25', '8p3'])
        self.add_argument('--losses', choices=['dc', 'tv', 'cap', 'w', 'sh', 'mse', 'l1', 'ssim', 'perc', 'dice'], \
                nargs='+', type=str, help='<Required> Set flag', required=True)
        self.add_argument('--sampling', choices=['uhs', 'dhs'], type=str, help='Sampling method', required=True)
        utils.add_bool_arg(self, 'range_restrict')
        utils.add_bool_arg(self, 'scheduler', default=False)
        utils.add_bool_arg(self, 'use_tanh', default=False)
        self.add_argument('--hyperparameters', type=float, default=None)
        self.add_argument('--organ', choices=['brain', 'knee'], type=str, default='knee')
        
    def parse(self):
        args = self.parse_args()
        if args.sampling == 'dhs':
            assert args.topK is not None, 'DHS sampling must set topK'
        if args.date is None:
            date = '{}'.format(time.strftime('%b_%d'))
        else:
            date = args.date

        args.run_dir = os.path.join(args.models_dir, args.filename_prefix, date, \
            '{lr}_{batch_size}_{losses}_{hnet_hdim}_{unet_hdim}_{topK}_{range_restrict}_{hps}_{use_tanh}'.format(
            lr=args.lr,
            batch_size=args.batch_size,
            losses=args.losses,
            hnet_hdim=args.hnet_hdim,
            unet_hdim=args.unet_hdim,
            range_restrict=args.range_restrict,
            topK=args.topK,
            hps=args.hyperparameters,
            use_tanh=args.use_tanh
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

def prepare_batch(batch, args):
    if args.task == 'seg':
        batch_input = batch['mri'][tio.DATA].squeeze(2).to(args.device)
        batch_target = batch['seg'][tio.DATA].squeeze(2).to(args.device)
        batch_y = None
    else:
        batch_target = batch['mri'][tio.DATA].squeeze(2).to(args.device)

        batch_y = utils.undersample(batch_target, args.mask)
        batch_input = utils.ifft(y)
        batch_y, batch_input = utils.scale(batch_y, batch_input)

    return batch_input, batch_target, batch_y

def train(network, dataloader, criterion, optimizer, hpsampler, args):
    """Train for one epoch

        Parameters
        ----------
        network : Main network and hypernetwork
        dataloader : Training set dataloader
        optimizer : Optimizer to use
        hpsampler : Hyperparameter sampler
    """
    network.train()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    for batch_idx, batch in enumerate(dataloader):#tqdm(enumerate(dataloader), total=len(dataloader)):
        input, target, y = prepare_batch(batch, args)
        batch_size = len(input)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):

            hyperparams = hpsampler.sample(batch_size).to(args.device)
            print(hyperparams)

            pred, cap_reg = network(input, hyperparams)
            loss, _, sort_hyperparams = criterion(pred, y, hyperparams, cap_reg, target=target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.cpu().numpy() * batch_size
            # epoch_psnr += np.mean(utils.get_metrics(gt.permute(0, 2, 3, 1), \
            #         recon.permute(0, 2, 3, 1), zf.permute(0, 2, 3, 1), \
            #         'psnr', take_absval=False)) * batch_size
        epoch_samples += batch_size

    # np.save('', np.array(backproped_hps))
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return network, optimizer, epoch_loss, epoch_psnr

def validate(network, dataloader, criterion, hpsampler, args):
    network.eval()

    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_samples = 0
    epoch_psnr = 0

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input, target, y = prepare_batch(batch)
        batch_size = len(input)

        with torch.set_grad_enabled(False):
            hyperparams = hpsampler.sample(batch_size, val='one').to(args.device)
            pred, cap_reg = network(input, hyperparams)
            loss1, _, _ = criterion(pred, y, hyperparams, cap_reg, target=target)
                
            hyperparams = hpsampler.sample(batch_size, val='zero').to(args.device)
            pred, cap_reg = network(input, hyperparams)
            loss2, _, _ = criterion(pred, y, hyperparams, cap_reg, target=target)

            epoch_loss1 += loss1.data.cpu().numpy() * batch_size
            epoch_loss2 += loss2.data.cpu().numpy() * batch_size
            # metrics = utils.get_metrics(target.permute(0, 2, 3, 1), pred.permute(0, 2, 3, 1), \
            #         zf.permute(0, 2, 3, 1), 'psnr', take_absval=False)
            # epoch_psnr += np.mean(metrics) * batch_size
        epoch_samples += batch_size
    epoch_loss1 /= epoch_samples
    epoch_loss2 /= epoch_samples
    epoch_psnr /= epoch_samples
    return network, epoch_loss1, epoch_loss2, epoch_psnr


if __name__ == "__main__":
    args = Parser().parse()
    
    # GPU Handling
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    num_workers = multiprocessing.cpu_count()
    # Dataset
    trainset = dataset.SubjectDataset(args.data_path, 'train').get_tio_dataset()
    valset = dataset.SubjectDataset(args.data_path, 'validate').get_tio_dataset()

    patch_sampler = tio.data.UniformSampler((1, 160, 224))
    trainset = tio.Queue(
        subjects_dataset=trainset,
        max_length=8,
        samples_per_volume=4,
        sampler=patch_sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True)
    valset = tio.Queue(
        subjects_dataset=valset,
        max_length=8,
        samples_per_volume=4, 
        sampler=patch_sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True)
    # else:
    #     trainset = dataset.ReconDataset(args.data_path, 'train')
    #     valset = dataset.ReconDataset(args.data_path, 'validate')

    # xdata = dataset.get_train_data(args.undersampling_rate, organ=args.organ)
    # gt_data = dataset.get_train_gt(organ=args.organ)
    # xdata = dataset.get_test_data('small')
    # gt_data = dataset.get_test_gt('small')
    # if args.n_ch_out == 2:
    #     print('Appending complex dimension into gt...')
    #     gt_data = np.concatenate((gt_data, np.zeros(gt_data.shape)), axis=3)

    # Undersampling mask
    args.mask = dataset.get_mask(args.undersampling_rate).to(args.device)

    # Train
    logger = {}
    logger['loss_train'] = []
    logger['loss_val'] = []
    logger['loss_val2'] = []
    logger['epoch_train_time'] = []
    logger['psnr_train'] = []
    logger['psnr_val'] = []

    ###############  Dataset ########################
    # trainset = dataset.Dataset(xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
    # valset = dataset.Dataset(xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    validation_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=2*args.batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    ##################################################

    ##### Model, Optimizer, Sampler, Loss ############
    num_hyperparams = len(args.losses)-1 if args.range_restrict else len(args.losses)
    n_ch_in = 1 if args.task == 'seg' else 2
    if args.hyperparameters is None:
        network = model.HyperUnet(args.device,
                         num_hyperparams,
                         args.hnet_hdim,
                         args.unet_hdim, 
                         hnet_norm=not args.range_restrict,
                         n_ch_in=n_ch_in,
                         n_ch_out=args.n_ch_out,
                         use_tanh=args.use_tanh).to(args.device)
    else:
        network = model.Unet(n_ch_in=n_ch_in, 
                             n_ch_out=args.n_ch_out, 
                             unet_hdim=args.unet_hdim).to(args.device)

    print('Total parameters:', utils.count_parameters(network))
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    hpsampler = sampler.HpSampler(num_hyperparams, args.range_restrict, debug=False, hps=args.hyperparameters)
    criterion = hyperloss.AmortizedLoss(args.losses, args.range_restrict, args.sampling, \
            args.topK, args.device, args.mask)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, min_lr=1e-7)
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
        if args.scheduler:
            network, optimizer, scheduler = utils.load_checkpoint(network, load_path, optimizer, scheduler)
        else:
            network, optimizer = utils.load_checkpoint(network, load_path, optimizer)
    ##################################################

    ############## Training loop #####################
    for epoch in range(args.cont+1, args.epochs+1):
        tic = time.time()

        print('\nEpoch %d/%d' % (epoch, args.epochs))
        print('Learning rate:', args.lr if args.force_lr is None else args.force_lr)
        print('DHS sampling' if args.sampling == 'dhs' else 'UHS sampling')

        # Train
        network, optimizer, train_epoch_loss, train_epoch_psnr = train(network, train_loader, 
                criterion, optimizer, hpsampler, args)
        # Validate
        network, val_epoch_ssim, val_epoch_l1, val_epoch_psnr = validate(network, validation_loader, 
                criterion, hpsampler, args)
        # val_epoch_loss, val_epoch_psnr = 0, 0
        if args.scheduler:
            scheduler.step(val_epoch_loss)
        
        epoch_train_time = time.time() - tic

        # Save checkpoints
        logger['loss_train'].append(train_epoch_loss)
        logger['loss_val'].append(val_epoch_ssim)
        logger['loss_val2'].append(val_epoch_l1)
        logger['psnr_train'].append(train_epoch_psnr)
        logger['psnr_val'].append(val_epoch_psnr)
        logger['epoch_train_time'].append(epoch_train_time)

        utils.save_loss(args.run_dir, logger, 'loss_train', 'loss_val', 'loss_val2', 'epoch_train_time', \
                'psnr_train', 'psnr_val')
        if epoch % args.log_interval == 0:
            utils.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), \
                    args.ckpt_dir)

        print("Epoch {}: test loss: {:.6f}, test psnr: {:.6f}, time: {:.6f}".format( \
            epoch, train_epoch_loss, train_epoch_psnr, epoch_train_time))
