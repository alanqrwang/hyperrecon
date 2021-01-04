import torch
import torch.nn as nn
from regagcsmri import utils, train, dataset, model
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
    parser.add_argument('--force_lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--load_checkpoint', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')

    parser.add_argument('--bounds', nargs='+', type=float, help='<Required> Set flag', required=True)
    parser.add_argument('--reg_types', nargs='+', type=str, help='<Required> Set flag', required=True)
    parser.add_argument('--sampling', choices=['uhs', 'dhs'], type=str, help='Sampling method', required=True)
    parser.add_argument('--loss_schedule', type=int, default=100)
    parser.add_argument('--sample_schedule', type=int, default=200)
    utils.add_bool_arg(parser, 'range_restrict')

    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--topK', type=int, default=None)
    parser.add_argument('--n_hyp_layers', type=int, default=2)
    
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    pprint.pprint(vars(args))
    ##################################################

    ############### Undersampling Mask ###############
    mask = utils.get_mask()
    args.mask = torch.tensor(mask, requires_grad=False).float().to(args.device)
    ##################################################

    ############### Dataset ##########################
    xdata = utils.get_train_data()
    gt_data = utils.get_train_gt()
    if gt_data.shape[-1] == 1:
        print('Appending complex dimension into gt...')
        gt_data = np.concatenate((gt_data, np.zeros(gt_data.shape)), axis=3)

    train.trainer(xdata, gt_data, vars(args))
