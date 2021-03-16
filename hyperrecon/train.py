"""
Training loop for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
from . import loss as losslayer
from . import utils, model, dataset, sampler, plot
import torch
from tqdm import tqdm
import numpy as np
import sys
import glob
import os


def train(network, dataloader, criterion, optimizer, hpsampler, args, topK, epoch):
    """Train for one epoch

        Parameters
        ----------
        network : hyperrrecon.UNet
            Main network and hypernetwork
        dataloader : torch.utils.data.DataLoader
            Training set dataloader
        optimizer : torch.optim.Adam
            Adam optimizer
        hpsampler : hyperrecon.HpSampler
            Hyperparameter sampler
        args : dict
            Miscellaneous parameters
        topK : int or None
            K for DHS sampling
        epoch : int
            Current training epoch

        Returns
        ----------
        network : hyperrecon.UNet
            Main network and hypernetwork
        optimizer : torch.optim.Adam
            Adam optimizer
        epoch_loss : float
            Loss for this epoch
    """
    network.train()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    for batch_idx, (y, gt) in tqdm(enumerate(dataloader), total=len(dataloader)):
        y = y.float().to(args['device'])
        gt = gt.float().to(args['device'])

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            hyperparams = hpsampler.sample(len(y)).to(args['device'])

            recon = network(zf, y, hyperparams)
            cap_reg = network.get_l1_weight_penalty(len(y))
            loss, _, sort_hyperparams = criterion(recon, y, hyperparams, cap_reg, topK, epoch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.cpu().numpy()
            epoch_psnr += np.sum(utils.get_metrics(gt.permute(0, 2, 3, 1), recon.permute(0, 2, 3, 1), 'psnr', False, normalized=True))
        epoch_samples += len(y)
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return network, optimizer, epoch_loss, epoch_psnr

def validate(network, dataloader, criterion, hpsampler, args, topK, epoch):
    """Validate for one epoch

        Parameters
        ----------
        network : hyperrecon.UNet
            Main network and hypernetwork
        dataloader : torch.utils.data.DataLoader
            Training set dataloader
        hpsampler : hyperrecon.HpSampler
            Hyperparameter sampler
        args : dict
            Miscellaneous parameters
        topK : int or None
            K for DHS sampling
        epoch : int
            Current training epoch

        Returns
        ----------
        network : hyperrecon.UNet
            Main network and hypernetwork
        epoch_loss : float
            Loss for this epoch
    """
    network.eval()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    for batch_idx, (y, gt) in tqdm(enumerate(dataloader), total=len(dataloader)):
        y = y.float().to(args['device'])
        gt = gt.float().to(args['device'])

        with torch.set_grad_enabled(False):
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            hyperparams = hpsampler.sample(len(y)).to(args['device'])

            recon = network(zf, y, hyperparams)
            cap_reg = network.get_l1_weight_penalty(len(y))
            loss, _, _ = criterion(recon, y, hyperparams, cap_reg, topK, epoch)
                

            epoch_loss += loss.data.cpu().numpy()
            epoch_psnr += np.sum(utils.get_metrics(gt.permute(0, 2, 3, 1), recon.permute(0, 2, 3, 1), 'psnr', False, normalized=True))
        epoch_samples += len(y)
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return network, epoch_loss, epoch_psnr

def trajtrain(network, dataloader, trained_reconnet, criterion, optimizer, args, lmbda, psnr_map=None, dc_map=None):
    losses = []

    for epoch in range(1, args['num_epochs']+1):
        for batch_idx, (y, gt) in tqdm(enumerate(dataloader), total=len(dataloader)):
            if batch_idx > 100:
                break
            y = y.float().to(args['device'])
            gt = gt.float().to(args['device'])
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            # Forward through trajectory net
            traj = torch.rand(args['num_points']*args['batch_size']).float().to(args['device']).unsqueeze(1)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out = network(traj)

            # Forward through recon net
            zf = torch.repeat_interleave(zf, args['num_points'], dim=0)
            y = torch.repeat_interleave(y, args['num_points'], dim=0)
            recons, cap_reg = trained_reconnet(zf, y, out)

            # Evaluate loss
            _, loss_dict, _ = criterion(recons, y, out, cap_reg, None, True)
            dc_losses = loss_dict['dc']
            recons = recons.view(args['batch_size'], args['num_points'], *recons.shape[1:])
            dc_losses = dc_losses.view(args['batch_size'], args['num_points'])
            loss = losslayer.trajloss(recons, dc_losses, lmbda, args['device'], args['loss_type'])
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            plot.plot_traj_cp(network, args['num_points'], losses, lmbda, args['device'], psnr_map, dc_map, None)
            
            utils.save_loss(epoch, loss, 0, args['save_path'])
            utils.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), \
                    loss, 0, args['save_path'], args['log_interval'])
        
    return network
