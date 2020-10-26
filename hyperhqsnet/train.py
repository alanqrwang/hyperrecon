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
        network = model.Unet(conf['device'], num_hyperparams=len(conf['reg_types']), nh=conf['num_hidden']).to(conf['device'])

    optimizer = torch.optim.Adam(network.parameters(), lr=conf['lr'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50])
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
        print('loading from checkpoint', pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location=torch.device('cpu'))
        network.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])

    # Training loop
    for epoch in range(conf['load_checkpoint']+1, conf['num_epochs']+1):
        topK = conf['topK'] if epoch > 500 else None
        print(type(conf['range_restrict']))

        network, optimizer, train_epoch_loss, prior_map = train(network, dataloaders['train'], optimizer, conf['mask'], \
                conf['filename'], conf['device'], conf['alpha_bound'], conf['beta_bound'], topK, conf['reg_types'], conf['range_restrict'])
        network, val_epoch_loss = test(network, dataloaders['val'], conf['mask'], \
                conf['filename'], conf['device'], conf['alpha_bound'], conf['beta_bound'], topK, conf['reg_types'], conf['range_restrict'])
        scheduler.step()
            
        # Optionally save checkpoints here, e.g.:
        myutils.io.save_losses(epoch, train_epoch_loss, val_epoch_loss, conf['filename'])
        myutils.io.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), train_epoch_loss, val_epoch_loss, conf['filename'], conf['log_interval'], scheduler.state_dict())

        # Save prior maps
        priormaps_dir = os.path.join(conf['filename'], 'priormaps')
        if not os.path.isdir(priormaps_dir):   
            os.makedirs(priormaps_dir)
        np.save(os.path.join(priormaps_dir, '{epoch:04d}'.format(epoch=epoch)), prior_map)


def train(network, dataloader, optimizer, mask, filename, device, alpha_bound, beta_bound, topK, reg_types, range_restrict):
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

            hyperparams = utils.sample_hyperparams(len(y), len(reg_types), alpha_bound, beta_bound).to(device)

            x_hat, cap_reg = network(zf, y, hyperparams)
            unsup_losses, dc_losses = losslayer.unsup_loss(x_hat, y, mask, hyperparams, device, reg_types, cap_reg, range_restrict)
                
            if topK is not None:
                _, perm = torch.sort(dc_losses) # Sort by DC loss, low to high
                sort_losses = unsup_losses[perm] # Reorder total losses by lowest to highest DC loss
                sort_hyperparams = hyperparams[perm]

                loss = torch.sum(sort_losses[:topK]) # Take only the losses with lowest DC
            else:
                loss = torch.sum(unsup_losses)
                sort_hyperparams = hyperparams

            loss.backward()
            optimizer.step()

            # Find nearest discrete grid points
            # if topK is not None:
            #     gpoints = np.rint((sort_hyperparams[:topK].cpu().detach().numpy() - grid_offset) / grid_spacing)
            # else:
            #     gpoints = np.rint((sort_hyperparams.cpu().detach().numpy() - grid_offset) / grid_spacing)
            # for gp in gpoints:
            #     gp = gp.astype(int)
            #     grid[tuple(gp)] += 1


            epoch_loss += loss.data.cpu().numpy()
        epoch_samples += len(y)
    epoch_loss /= epoch_samples
    return network, optimizer, epoch_loss, grid.T

def test(network, dataloader, mask, filename, device, alpha_bound, beta_bound, topK, reg_types, range_restrict):
    network.eval()

    epoch_loss = 0
    epoch_samples = 0

    for batch_idx, (y, gt) in tqdm(enumerate(dataloader), total=len(dataloader)):
        y = y.float().to(device)
        gt = gt.float().to(device)

        with torch.set_grad_enabled(False):
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            hyperparams = utils.sample_hyperparams(len(y), len(reg_types), alpha_bound, beta_bound).to(device)

            x_hat, cap_reg = network(zf, y, hyperparams)
            unsup_losses, dc_losses = losslayer.unsup_loss(x_hat, y, mask, hyperparams, device, reg_types, cap_reg, range_restrict)
                
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
