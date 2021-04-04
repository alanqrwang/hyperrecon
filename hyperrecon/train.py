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


def train(network, dataloader, criterion, optimizer, hpsampler, device, epoch):
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
        batch_size = len(y)
        y = y.float().to(device)
        gt = gt.float().to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            hyperparams = hpsampler.sample(batch_size).to(device)

            recon, cap_reg = network(zf, hyperparams)
            loss, _, sort_hyperparams = criterion(recon, y, hyperparams, cap_reg, epoch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.cpu().numpy()
            epoch_psnr += np.sum(utils.get_metrics(gt.permute(0, 2, 3, 1), \
                    recon.permute(0, 2, 3, 1), zf.permute(0, 2, 3, 1), \
                    'psnr', take_avg=False, take_absval=False))
        epoch_samples += batch_size
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return network, optimizer, epoch_loss, epoch_psnr

def validate(network, dataloader, criterion, hpsampler, device, epoch):
    """Validate for one epoch

        Parameters
        ----------
        network : hyperrecon.UNet
            Main network and hypernetwork
        dataloader : torch.utils.data.DataLoader
            Training set dataloader
        hpsampler : hyperrecon.HpSampler
            Hyperparameter sampler
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
        batch_size = len(y)
        y = y.float().to(device)
        gt = gt.float().to(device)

        with torch.set_grad_enabled(False):
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            hyperparams = hpsampler.sample(batch_size, val=True).to(device)

            recon, cap_reg = network(zf, hyperparams)
            loss, _, _ = criterion(recon, y, hyperparams, cap_reg, epoch)
                

            epoch_loss += loss.data.cpu().numpy()
            metrics = utils.get_metrics(gt.permute(0, 2, 3, 1), recon.permute(0, 2, 3, 1), \
                    zf.permute(0, 2, 3, 1), 'psnr', False, take_absval=False)
            epoch_psnr += np.sum(metrics)
        epoch_samples += batch_size
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return network, epoch_loss, epoch_psnr

def trajtrain(network, dataloader, trained_reconnet, optimizer, args):
    logger = {}
    logger['loss_train'] = []
    logger['loss_val'] = []

    for epoch in range(1, args.epochs+1):
        for batch_idx, (y, gt) in enumerate(dataloader):
            print(batch_idx)
            y = y.float().to(args.device)
            gt = gt.float().to(args.device)
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            # Forward through trajectory net
            traj = torch.rand(args.num_points*args.batch_size).float().to(args.device).unsqueeze(1)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out = network(traj)

                # Forward through recon net
                gt = gt.repeat_interleave(args.num_points, dim=0)
                y = y.repeat_interleave(args.num_points, dim=0)
                zf = zf.repeat_interleave(args.num_points, dim=0)
                recons = trained_reconnet(zf, out)

                # Evaluate loss
                dc_losses = losslayer.get_dc_loss(recons, y, args.mask)
                mse = torch.nn.MSELoss()(gt, recons)

                recons = recons.view(args.batch_size, args.num_points, *recons.shape[1:])
                dc_losses = dc_losses.view(args.batch_size, args.num_points)
                loss = losslayer.trajloss(recons, dc_losses, args.lmbda, args.device, args.loss_type, mse)
                
                loss.backward()
                optimizer.step()

            logger['loss_train'].append(loss.item())
            # plot.plot_traj_cp(network, args.num_points, logger['loss_train'], args.lmbda, args.device)
            
            utils.save_loss(args.run_dir, logger, 'loss_train')

        utils.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), \
            logger, args.ckpt_dir, args.log_interval)
        
    return network
