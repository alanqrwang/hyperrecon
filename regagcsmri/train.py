"""
Training loop for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
from . import loss as losslayer
from . import utils, model, dataset, sampler
import torch
from tqdm import tqdm
import numpy as np
import sys
import glob
import os

def trainer(xdata, gt_data, conf):
    """Training loop. 

    Handles model, optimizer, loss, and sampler generation.
    Handles data loading. Handles i/o and checkpoint loading.
        

    Parameters
    ----------
    xdata : numpy.array (N, img_height, img_width, 2)
        Dataset of under-sampled measurements
    gt_data : numpy.array (N, img_height, img_width, 2)
        Dataset of fully-sampled images
    conf : dict
        Miscellaneous parameters

    Returns
    ----------
    network : regagcsmri.UNet
        Main network and hypernetwork
    optimizer : torch.optim.Adam
        Adam optimizer
    epoch_loss : float
        Loss for this epoch
    """
    ###############  Dataset ########################
    trainset = dataset.Dataset(xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
    valset = dataset.Dataset(xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])

    params = {'batch_size': conf['batch_size'],
         'shuffle': True,
         'num_workers': 4}
    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params),
    }
    ##################################################

    ##### Model, Optimizer, Sampler, Loss ############
    network = model.Unet(conf['device'], num_hyperparams=len(conf['reg_types']), hyparch=conf['hyparch'], \
                nh=conf['unet_hidden']).to(conf['device'])

    optimizer = torch.optim.Adam(network.parameters(), lr=conf['lr'])
    if conf['force_lr'] is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = conf['force_lr']

    bounds = np.array(conf['bounds']).reshape(-1, 2)
    hpsampler = sampler.HpSampler(len(conf['reg_types']), bounds)
    criterion = losslayer.AmortizedLoss(conf['reg_types'], conf['range_restrict'], conf['mask'], conf['sampling'])
    ##################################################

    ############ Checkpoint Loading ##################
    if conf['load_checkpoint'] != 0:
        pretrain_path = os.path.join(conf['filename'], 'model.{epoch:04d}.h5'.format(epoch=conf['load_checkpoint']))
        network, optimizer = load_checkpoint(network, optimizer, pretrain_path)
    ##################################################

    ############## Training loop #####################
    for epoch in range(conf['load_checkpoint']+1, conf['num_epochs']+1):
        print('\nEpoch %d/%d' % (epoch, conf['num_epochs']))

        # Setting hyperparameter sampling parameters.
        # topK is number in mini-batch to backprop. If None, then uniform
        if conf['sampling'] == 'dhs' and epoch > conf['sample_schedule']:
            topK = conf['topK']
            print('DHS sampling')
        else:
            topK = None
            print('UHS sampling')

        # Loss scheduling. If activated, then first 100 epochs is trained 
        # as single reg function
        if epoch > conf['loss_schedule']:
            schedule = True
            print('Loss schedule: 2 reg')
        else:
            schedule = False
            print('Loss schedule: 1 reg')


        # Train
        network, optimizer, train_epoch_loss = train(network, dataloaders['train'], \
                criterion, optimizer, hpsampler, conf, topK, schedule)
        # Validate
        network, val_epoch_loss = validate(network, dataloaders['val'], criterion, hpsampler, conf, \
                topK, schedule)
        # Save checkpoints
        utils.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), \
                train_epoch_loss, val_epoch_loss, conf['filename'], conf['log_interval'])
        utils.save_loss(epoch, train_epoch_loss, val_epoch_loss, conf['filename'])

def train(network, dataloader, criterion, optimizer, hpsampler, conf, topK, epoch):
    """Train for one epoch

        Parameters
        ----------
        network : regagcsmri.UNet
            Main network and hypernetwork
        dataloader : torch.utils.data.DataLoader
            Training set dataloader
        optimizer : torch.optim.Adam
            Adam optimizer
        hpsampler : regagcsmri.HpSampler
            Hyperparameter sampler
        conf : dict
            Miscellaneous parameters
        topK : int or None
            K for DHS sampling
        epoch : int
            Current training epoch

        Returns
        ----------
        network : regagcsmri.UNet
            Main network and hypernetwork
        optimizer : torch.optim.Adam
            Adam optimizer
        epoch_loss : float
            Loss for this epoch
    """
    network.train()

    epoch_loss = 0
    epoch_samples = 0

    for batch_idx, (y, gt) in tqdm(enumerate(dataloader), total=len(dataloader)):
        y = y.float().to(conf['device'])
        gt = gt.float().to(conf['device'])

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            hyperparams = hpsampler.sample(len(y)).to(conf['device'])

            recon, cap_reg = network(zf, y, hyperparams)
            loss, _, sort_hyperparams = criterion(recon, y, hyperparams, cap_reg, topK, epoch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.cpu().numpy()
        epoch_samples += len(y)
    epoch_loss /= epoch_samples
    return network, optimizer, epoch_loss

def validate(network, dataloader, criterion, hpsampler, conf, topK, epoch):
    """Validate for one epoch

        Parameters
        ----------
        network : regagcsmri.UNet
            Main network and hypernetwork
        dataloader : torch.utils.data.DataLoader
            Training set dataloader
        hpsampler : regagcsmri.HpSampler
            Hyperparameter sampler
        conf : dict
            Miscellaneous parameters
        topK : int or None
            K for DHS sampling
        epoch : int
            Current training epoch

        Returns
        ----------
        network : regagcsmri.UNet
            Main network and hypernetwork
        epoch_loss : float
            Loss for this epoch
    """
    network.eval()

    epoch_loss = 0
    epoch_samples = 0

    for batch_idx, (y, gt) in tqdm(enumerate(dataloader), total=len(dataloader)):
        y = y.float().to(conf['device'])
        gt = gt.float().to(conf['device'])

        with torch.set_grad_enabled(False):
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            hyperparams = hpsampler.sample(len(y)).to(conf['device'])

            recon, cap_reg = network(zf, y, hyperparams)
            loss, _, _ = criterion(recon, y, hyperparams, cap_reg, topK, epoch)
                

            epoch_loss += loss.data.cpu().numpy()
        epoch_samples += len(y)
    epoch_loss /= epoch_samples
    return network, epoch_loss

