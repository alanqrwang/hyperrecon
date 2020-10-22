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
        network = model.Unet(conf['device'], num_hyperparams=conf['num_hyperparams'], nh=conf['num_hidden']).to(conf['device'])

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
        if checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])

    for epoch in range(conf['load_checkpoint']+1, conf['num_epochs']+1):
        topK = conf['topK'] if epoch > 500 else None

        network, optimizer, train_epoch_loss, val_epoch_loss, prior_map = train(network, dataloaders, optimizer, conf['mask'], conf['num_hyperparams'], \
                conf['filename'], conf['device'], conf['alpha_bound'], conf['beta_bound'], topK)
        scheduler.step()
            
        # Optionally save checkpoints here, e.g.:
        myutils.io.save_losses(epoch, train_epoch_loss, val_epoch_loss, conf['filename'])
        myutils.io.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), train_epoch_loss, val_epoch_loss, conf['filename'], conf['log_interval'], scheduler.state_dict())

        # Save prior maps
        priormaps_dir = os.path.join(conf['filename'], 'priormaps')
        if not os.path.isdir(priormaps_dir):   
            os.makedirs(priormaps_dir)
        np.save(os.path.join(priormaps_dir, '{epoch:04d}'.format(epoch=epoch)), prior_map)


def train(network, dataloaders, optimizer, mask, num_hyperparams, filename, device, alpha_bound, beta_bound, topK):
    samples = 100
    grid = np.zeros((samples+1, samples+1))
    grid_offset = np.array([alpha_bound[0], beta_bound[0]])
    grid_spacing = np.array([(alpha_bound[1]-alpha_bound[0])/samples, (beta_bound[1]-beta_bound[0])/samples])

    for phase in ['train', 'val']:
        if phase == 'train':
            network.train()
        elif phase == 'val':
            network.eval()

        epoch_loss = 0
        epoch_samples = 0

        for batch_idx, (y, gt) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
            y = y.float().to(device)
            gt = gt.float().to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                zf = utils.ifft(y)
                y, zf = utils.scale(y, zf)

                if num_hyperparams == 2:
                    r1 = float(alpha_bound[0])
                    r2 = float(alpha_bound[1])
                    alpha = utils.sample_alpha(len(y), r1, r2, fixed=False).to(device)
                    r1 = float(beta_bound[0])
                    r2 = float(beta_bound[1])
                    beta = utils.sample_alpha(len(y), r1, r2, fixed=False).to(device)

                    hyperparams = torch.cat([alpha, beta], dim=1)
                else:
                    alpha = utils.sample_alpha().to(device).view(1)
                    hyperparams = torch.cat([alpha], dim=0)

                # (b, l, w, 2), (b, 2)
                x_hat, cap_reg = network(zf, y, hyperparams)
                unsup_losses, dc_losses = losslayer.unsup_loss(x_hat, y, mask, hyperparams, device, cap_reg)
                    
                if topK is not None:
                    print('doing topK')
                    _, perm = torch.sort(dc_losses) # Sort by DC loss, low to high
                    sort_losses = unsup_losses[perm] # Reorder total losses by lowest to highest DC loss
                    sort_hyperparams = hyperparams[perm]

                    loss = torch.sum(sort_losses[:topK]) # Take only the losses with lowest DC
                else:
                    print('doing vanilla')
                    loss = torch.sum(unsup_losses)
                    sort_hyperparams = hyperparams

                if phase == 'train':
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
        if phase == 'train':
            train_epoch_loss = epoch_loss
        if phase == 'val':
            val_epoch_loss = epoch_loss
    return network, optimizer, train_epoch_loss, val_epoch_loss, grid.T
