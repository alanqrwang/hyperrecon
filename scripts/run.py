import torch
import torch.nn as nn
from hyperhqsnet import utils, train, dataset, model
import numpy as np
import argparse
import os
import pprint
import sys
import glob

if __name__ == "__main__":

    ############### Argument Parsing #################
    parser = argparse.ArgumentParser(description='Half-Quadratic Minimization for CS-MRI in Pytorch')
    parser.add_argument('-fp', '--filename_prefix', type=str, help='filename prefix', required=True)
    parser.add_argument('--models_dir', default='/nfs02/users/aw847/models/HyperHQSNet/', type=str, help='directory to save models')
    
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lmbda', type=float, default=0, help='gpu id to train on')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_hyperparams', type=int, default=2)
    parser.add_argument('--load_checkpoint', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
    parser.add_argument('--undersample_rate', choices=['4p1', '4p2', '8p25', '8p3'], type=str, help='undersample rate', required=True)
    parser.add_argument('--dataset', choices=['t1', 't2', 'knee', 'brats'], type=str, help='dataset', required=True)
    parser.add_argument('--recon_type', choices=['unroll', 'unet'], type=str, help='dataset', required=True)

    parser.add_argument('--alpha_bound', nargs='+', type=float, help='<Required> Set flag', required=True)
    parser.add_argument('--beta_bound', nargs='+', type=float, help='<Required> Set flag', required=True)
    utils.add_bool_arg(parser, 'learn_reg_coeff')

    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--topK', type=int, default=1)
    
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    pprint.pprint(vars(args))
    ##################################################

    ############### Undersampling Mask ###############
    mask = utils.get_mask(args.undersample_rate)
    args.mask = torch.tensor(mask, requires_grad=False).float().to(args.device)
    ##################################################

    ############### Dataset ##########################
    if args.dataset == 't1':
        data_path = '/nfs02/users/aw847/data/brain/adrian/brain_train_normalized_{maskname}.npy'
        gt_path = '/nfs02/users/aw847/data/brain/adrian/brain_train_normalized.npy'
    elif args.dataset == 't2':
        data_path = '/nfs02/users/aw847/data/brain/IXI-T2/IXI-T2_train_normalized_{maskname}.npy'
        gt_path = '/nfs02/users/aw847/data/brain/IXI-T2/IXI-T2_train_normalized.npy'
    elif args.dataset == 'knee':
        data_path = '/nfs02/users/aw847/data/knee/knee_train_normalized_{maskname}.npy'
        gt_path = '/nfs02/users/aw847/data/knee/knee_train_normalized.npy'

    xdata = utils.get_data(data_path.format(maskname=args.undersample_rate))
    gt_data = utils.get_data(gt_path)
    if gt_data.shape[-1] == 1:
        print('Appending complex dimension into gt...')
        gt_data = np.concatenate((gt_data, np.zeros(gt_data.shape)), axis=3)


    ################### Filename #####################
    local_name = '{prefix}_{recon_type}_{lr}_{lmbda}_{K}_{learn_reg_coeff}_{num_hidden}_{alpha_bound}_{beta_bound}_{topK}/{dataset}_{undersample_rate}'.format(
        prefix=args.filename_prefix,
        dataset=args.dataset,
        undersample_rate=args.undersample_rate,
        recon_type=args.recon_type,
        lr=args.lr,
        lmbda=args.lmbda,
        K=args.K,
        learn_reg_coeff=args.learn_reg_coeff,
        num_hidden=args.num_hidden,
        alpha_bound=args.alpha_bound,
        beta_bound=args.beta_bound,
        topK=args.topK,
        )
    model_folder = os.path.join(args.models_dir, local_name)
    if not os.path.isdir(model_folder):   
        os.makedirs(model_folder)
    args.filename = model_folder

    train.train(xdata, gt_data, vars(args))
