from . import loss as losslayer
from . import utils, model, dataset
import myutils
import torch
import torch.nn as nn
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os

def trainer(xdata, gt_data, conf):
    ###############  Dataset ########################
    trainset = dataset.Dataset(xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
    valset = dataset.Dataset(xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])

    params = {'batch_size': conf['batch_size'],
         'shuffle': True,
         'num_workers': 4}
    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params)
    }
    ##################################################

    ############### Model and Optimizer ##############
    if conf['recon_type'] == 'unroll':
        network = model.HQSNet(conf['K'], conf['mask'], conf['lmbda'], conf['learn_reg_coeff'], conf['device'], n_hidden=conf['num_hidden']).to(conf['device'])
    elif conf['recon_type'] == 'unet':
        network = model.Unet(conf['device'], num_hyperparams=len(conf['reg_types']), n_hyp_layers=conf['n_hyp_layers'], \
                alpha_bound=conf['alpha_bound'], beta_bound=conf['beta_bound'], nh=conf['num_hidden']).to(conf['device'])

    optimizer = torch.optim.Adam(network.parameters(), lr=conf['lr'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100])
    # scheduler = None
    ##################################################

    if conf['load_checkpoint'] != 0:
        pretrain_path = os.path.join(conf['filename'], 'model.{epoch:04d}.h5'.format(epoch=conf['load_checkpoint']))

        pretrain_temp = os.path.join(conf['filename'], 'model.*.h5')
        pretrain_temp = pretrain_temp.replace("[", "(") 
        pretrain_temp = pretrain_temp.replace("]", "[]]")
        pretrain_temp = pretrain_temp.replace("(", "[[]")

        pretrained_models = glob.glob(pretrain_temp)
        print('total models in %s:\n ' % pretrain_temp, len(pretrained_models))
        print('loading', pretrain_path)
        cont = input('continue?')
        if cont == 'y':
            pass
        else:
            sys.exit()

        pretrain_path = '/nfs02/users/aw847/models/HyperHQSNet/deeper-new-loss-fn_unet_5e-05_32_0_5_[\'cap\', \'tv\']_64_[0.0, 1.0]_[0.0, 1.0]_None_True/t1_4p2/model.5000.h5' 
                
        print('loading from checkpoint', pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location=torch.device('cpu'))
        network.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        #     scheduler.load_state_dict(checkpoint['scheduler'])

    loaded_map = np.load('/nfs02/users/aw847/data/hypernet/deep_cap_tv_t1_single_100_100.npy')
    loaded_map = np.load('/nfs02/users/aw847/data/hypernet/deep_cap_tv_knee_single_100_100.npy')
    # loaded_map = None

    # Training loop
    for epoch in range(conf['load_checkpoint']+1, conf['num_epochs']+1):
        if conf['force_lr'] is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = conf['force_lr']

        psnr_map = loaded_map if epoch > -1 else None
        topK = conf['topK'] if epoch > 200 else None

        network, optimizer, train_epoch_loss, prior_map = train(network, dataloaders['train'], optimizer, conf['mask'], \
                conf['device'], conf['alpha_bound'], conf['beta_bound'], topK, conf['reg_types'], conf['range_restrict'], psnr_map, epoch=None)

        network, val_epoch_loss1 = validate(network, dataloaders['val'], conf['mask'], \
                conf['device'], conf['alpha_bound'], conf['beta_bound'], topK, conf['reg_types'], conf['range_restrict'], psnr_map, epoch=None)

        # scheduler.step()
            
        # Save checkpoints 
        myutils.io.save_losses(epoch, train_epoch_loss, val_epoch_loss1, conf['filename'])
        myutils.io.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), train_epoch_loss, val_epoch_loss1, conf['filename'], conf['log_interval'], scheduler)

        # Save prior maps
        priormaps_dir = os.path.join(conf['filename'], 'priormaps')
        if not os.path.isdir(priormaps_dir):   
            os.makedirs(priormaps_dir)
        np.save(os.path.join(priormaps_dir, '{epoch:04d}'.format(epoch=epoch)), prior_map)


def train(network, dataloader, optimizer, mask, device, alpha_bound, beta_bound, topK, reg_types, range_restrict, psnr_map, epoch):
    samples = 100
    grid = np.zeros((samples+1, samples+1))
    grid_offset = np.array([alpha_bound[0], beta_bound[0]])
    grid_spacing = np.array([(alpha_bound[1]-alpha_bound[0])/samples, (beta_bound[1]-beta_bound[0])/samples])

    network.train()

    epoch_loss = 0
    epoch_samples = 0

    for batch_idx, (y, gt) in tqdm(enumerate(dataloader), total=len(dataloader)):
        y = y.float().to(device)
        gt = gt.float().to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            if psnr_map is not None:
                print('sampling from map')
                hyperparams = utils.map_based_sampling(psnr_map, len(y)).float().to(device)
            else:
                print('sampling vanilla')
                hyperparams = utils.sample_hyperparams(len(y), len(reg_types), alpha_bound, beta_bound).to(device)

            x_hat, cap_reg = network(zf, y, hyperparams)
            unsup_losses, dc_losses,_,_ = losslayer.unsup_loss(x_hat, y, mask, hyperparams, device, reg_types, cap_reg, range_restrict, epoch)
                
            if topK is not None:
                print('doing topK')
                _, perm = torch.sort(dc_losses) # Sort by DC loss, low to high
                sort_losses = unsup_losses[perm] # Reorder total losses by lowest to highest DC loss
                sort_hyperparams = hyperparams[perm]

                loss = torch.sum(sort_losses[:topK]) # Take only the losses with lowest DC
            else:
                print('not topK')
                loss = torch.sum(unsup_losses)
                sort_hyperparams = hyperparams

            loss.backward()
            optimizer.step()

            # Find nearest discrete grid points
            if topK is not None:
                gpoints = np.rint((sort_hyperparams[:topK].cpu().detach().numpy() - grid_offset) / grid_spacing)
            else:
                gpoints = np.rint((sort_hyperparams.cpu().detach().numpy() - grid_offset) / grid_spacing)
            for gp in gpoints:
                gp = gp.astype(int)
                grid[tuple(gp)] += 1


            epoch_loss += loss.data.cpu().numpy()
        epoch_samples += len(y)
    epoch_loss /= epoch_samples
    return network, optimizer, epoch_loss, grid.T

def validate(network, dataloader, mask, device, alpha_bound, beta_bound, topK, reg_types, range_restrict, psnr_map, epoch, hparams_val=None):
    network.eval()

    epoch_loss = 0
    epoch_samples = 0

    for batch_idx, (y, gt) in tqdm(enumerate(dataloader), total=len(dataloader)):
        y = y.float().to(device)
        gt = gt.float().to(device)

        with torch.set_grad_enabled(False):
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            if hparams_val is not None:
                print('in val', hparams_val.shape)
                hyperparams = hparams_val.expand(len(y), -1)
            elif psnr_map is not None:
                hyperparams = utils.map_based_sampling(psnr_map, len(y)).float().to(device)
            else:
                hyperparams = utils.sample_hyperparams(len(y), len(reg_types), alpha_bound, beta_bound).to(device)

            x_hat, cap_reg = network(zf, y, hyperparams)
            unsup_losses, dc_losses,_, tv_losses = losslayer.unsup_loss(x_hat, y, mask, hyperparams, device, reg_types, cap_reg, range_restrict, epoch)
                
            if topK is not None:
                print('doing topK')
                _, perm = torch.sort(dc_losses) # Sort by DC loss, low to high
                sort_losses = unsup_losses[perm] # Reorder total losses by lowest to highest DC loss

                loss = torch.sum(sort_losses[:topK]) # Take only the losses with lowest DC
            else:
                print('doing vanilla')
                loss = torch.sum(unsup_losses)

            epoch_loss += loss.data.cpu().numpy()
        epoch_samples += len(y)
    epoch_loss /= epoch_samples
    return network, epoch_loss

def test(network, dataloader, mask, device, alpha_bound, beta_bound, topK, reg_types, range_restrict, hyperparams):
    network.eval()

    recons = []
    losses = []
    dcs = []
    cap_regs = []
    tvs = []
    for batch_idx, (y, gt) in tqdm(enumerate(dataloader), total=len(dataloader)):
        y = y.float().to(device)
        gt = gt.float().to(device)
        with torch.set_grad_enabled(False):
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            pred, cap_reg = network(zf, y, hyperparams)
            loss, dc, tv = losslayer.unsup_loss(pred, y, mask, hyperparams, device, reg_types, cap_reg, range_restrict)
            recons.append(pred.cpu().detach().numpy())
            losses.append(loss.item())
            cap_regs.append(cap_reg.item())
            tvs.append(tv.item())

    preds = np.array(recons)[:,0,...]
    return preds, losses, dcs, cap_regs, tvs
