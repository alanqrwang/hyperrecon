from . import loss as losslayer
from . import utils, model, dataset, sampler, test
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

def trainer(xdata, gt_data, test_data, test_gt_data, conf):
    ###############  Dataset ########################
    trainset = dataset.Dataset(xdata[:int(len(xdata)*0.8)], gt_data[:int(len(gt_data)*0.8)])
    valset = dataset.Dataset(xdata[int(len(xdata)*0.8):], gt_data[int(len(gt_data)*0.8):])
    testset = dataset.Dataset(test_data, test_gt_data)

    params = {'batch_size': conf['batch_size'],
         'shuffle': True,
         'num_workers': 4}
    test_params = {'batch_size': len(test_data),
         'shuffle': False,
         'num_workers': 0}
    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, **params),
        'val': torch.utils.data.DataLoader(valset, **params),
        'test': torch.utils.data.DataLoader(testset, **test_params)
    }
    ##################################################

    ############### Model and Optimizer ##############
    if conf['recon_type'] == 'unroll':
        network = model.HQSNet(conf['K'], conf['mask'], conf['lmbda'], conf['learn_reg_coeff'], conf['device'], n_hidden=conf['num_hidden']).to(conf['device'])
    elif conf['recon_type'] == 'unet':
        network = model.Unet(conf['device'], num_hyperparams=len(conf['reg_types']), n_hyp_layers=conf['n_hyp_layers'], \
                alpha_bound=conf['alpha_bound'], beta_bound=conf['beta_bound'], nh=conf['num_hidden']).to(conf['device'])

    optimizer = torch.optim.Adam(network.parameters(), lr=conf['lr'])
    if conf['force_lr'] is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = conf['force_lr']
    ##################################################

    ############ Hyperparameter Sampling #############
    hpsampler = sampler.HpSampler(conf['sampling'], len(conf['reg_types']), conf['alpha_bound'], conf['beta_bound'])
    criterion = losslayer.AmortizedLoss(conf['reg_types'], conf['range_restrict'], conf['mask'], conf['sampling'])
    ##################################################

    ############ Checkpoint Loading ##################
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
        # pretrain_path = '/nfs02/users/aw847/models/HyperHQSNet/deeper-new-loss-fn_unet_5e-05_32_0_5_[\'cap\', \'tv\']_64_[0.0, 1.0]_[0.0, 1.0]_None_True/t1_4p2/model.5000.h5' 
                
        print('loading from checkpoint', pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location=torch.device('cpu'))
        network.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    ##################################################

    ############## Training loop #####################
    for epoch in range(conf['load_checkpoint']+1, conf['num_epochs']+1):
        print('\nEpoch %d/%d' % (epoch, conf['num_epochs']))

        # Setting hyperparameter sampling parameters.
        # topK is number in mini-batch to backprop. If None, then uniform
        # psnr_map is supervised map. If none, then uniform
        if conf['sampling'] == 'bestsup' and epoch > conf['sample_schedule']:
            topK = None

            print('Generating sup map')
            n_samples = 20
            alpha_bound = conf['alpha_bound']
            beta_bound = conf['beta_bound']
            alphas = np.linspace(alpha_bound[0], alpha_bound[1], n_samples)
            betas = np.linspace(beta_bound[0], beta_bound[1], n_samples)
            hps = np.stack(np.meshgrid(alphas, betas), -1).reshape(-1,2)
            print(hps.shape)
            test_results = test.test(network, dataloaders['test'], conf, hps, topK, give_metrics=True)
            psnr_map = test_results['rpsnr']
            print(psnr_map.shape)
            priormaps_dir = os.path.join(conf['filename'], 'priormaps')
            if not os.path.isdir(priormaps_dir):   
                os.makedirs(priormaps_dir)
            np.save(os.path.join(priormaps_dir, 'supmap{epoch:04d}'.format(epoch=epoch)), psnr_map.reshape((n_samples, n_samples)))
            print('Supervised metric sampling')
        elif conf['sampling'] == 'bestdc' and epoch > conf['sample_schedule']:
            topK = conf['topK']
            psnr_map = None
            print('Best DC sampling')
        else:
            topK = None
            psnr_map = None
            print('Uniform sampling')

        # Loss scheduling. If activated, then first 100 epochs is trained 
        # as single reg function
        if epoch > conf['loss_schedule']:
            schedule = True
            print('Loss schedule: 2 reg')
        else:
            schedule = False
            print('Loss schedule: 1 reg')


        network, optimizer, train_epoch_loss, prior_map = train(network, dataloaders['train'], \
                criterion, optimizer, hpsampler, conf, topK, psnr_map, schedule)

        network, val_epoch_loss = validate(network, dataloaders['val'], criterion, hpsampler, conf, \
                topK, psnr_map, schedule)

            
        # Save checkpoints 
        myutils.io.save_losses(epoch, train_epoch_loss, val_epoch_loss, conf['filename'])
        myutils.io.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), train_epoch_loss, val_epoch_loss, conf['filename'], conf['log_interval'])

        # Save prior maps
        if epoch % 1 == 0:
            priormaps_dir = os.path.join(conf['filename'], 'priormaps')
            if not os.path.isdir(priormaps_dir):   
                os.makedirs(priormaps_dir)
            np.save(os.path.join(priormaps_dir, 'samplemap{epoch:04d}'.format(epoch=epoch)), prior_map)
    ##################################################


def train(network, dataloader, criterion, optimizer, hpsampler, conf, topK, psnr_map, epoch):
    alpha_bound = conf['alpha_bound']
    beta_bound = conf['beta_bound']
    samples = 100
    grid = np.zeros((samples+1, samples+1))
    grid_offset = np.array([alpha_bound[0], beta_bound[0]])
    grid_spacing = np.array([(alpha_bound[1]-alpha_bound[0])/samples, (beta_bound[1]-beta_bound[0])/samples])

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

            hyperparams = hpsampler.sample(len(y), psnr_map).float().to(conf['device'])

            recon, cap_reg = network(zf, y, hyperparams)
            loss, _, sort_hyperparams = criterion(recon, y, hyperparams, cap_reg, topK, epoch)

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

def validate(network, dataloader, criterion, hpsampler, conf, topK, psnr_map, epoch):
    network.eval()

    epoch_loss = 0
    epoch_samples = 0

    for batch_idx, (y, gt) in tqdm(enumerate(dataloader), total=len(dataloader)):
        y = y.float().to(conf['device'])
        gt = gt.float().to(conf['device'])

        with torch.set_grad_enabled(False):
            zf = utils.ifft(y)
            y, zf = utils.scale(y, zf)

            hyperparams = hpsampler.sample(len(y), psnr_map).float().to(conf['device'])

            recon, cap_reg = network(zf, y, hyperparams)
            loss, _, _ = criterion(recon, y, hyperparams, cap_reg, topK, epoch)
                

            epoch_loss += loss.data.cpu().numpy()
        epoch_samples += len(y)
    epoch_loss /= epoch_samples
    return network, epoch_loss

