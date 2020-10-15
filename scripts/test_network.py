import torch
import torch.nn as nn
from hqsnet import utils, train, dataset, model
import numpy as np
import argparse
import os
import myutils

if __name__ == "__main__":

    ############### Argument Parsing #################
    parser = argparse.ArgumentParser(description='Half-Quadratic Minimization for CS-MRI in Pytorch')
    parser.add_argument('--model_dir', type=str, help='filename prefix', required=True)
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
    parser.add_argument('--noise_std', type=float, default=0, help='gpu id to train on')
    parser.add_argument('--undersample_rate', choices=['4p1', '4p2', '8p25', '8p3'], type=str, help='undersample rate', required=True)
    parser.add_argument('--dataset', choices=['t1', 't2', 'knee', 'brats'], type=str, help='dataset', required=True)
    parser.add_argument('--strategy', choices=['sup', 'unsup', 'refine'], type=str, help='training strategy', required=True)
    parser.add_argument('--alpha', type=float, default=0, help='gpu id to train on')
    utils.add_bool_arg(parser, 'learn_reg_coeff', default=False)

    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--K', type=int, default=5)

    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(args.device)
    ##################################################

    ############### Undersampling Mask ###############
    maskname = args.undersample_rate
    mask = utils.get_mask(maskname)
    mask = torch.tensor(mask, requires_grad=False).float().to(args.device)
    ##################################################

    ############### Dataset ##########################
    if args.dataset == 't1':
        data_path = '/nfs02/users/aw847/data/brain/adrian/brain_test_normalized_{maskname}.npy'
    elif args.dataset == 't2':
        data_path = '/nfs02/users/aw847/data/brain/IXI-T2/IXI-T2_test_normalized_{maskname}.npy'
    elif args.dataset == 'knee':
        data_path = '/nfs02/users/aw847/data/knee/knee_test_normalized_{maskname}.npy'

    xdata = utils.get_data(data_path.format(maskname=maskname))
    xdata = xdata + np.random.normal(size=xdata.shape, loc=0, scale=args.noise_std)
    ##################################################

    ############### Hyper-parameters #################
    lmbda = utils.get_lmbda()
    K = args.K
    learn_reg_coeff = args.learn_reg_coeff
    print(learn_reg_coeff)
    ##################################################

    ############### Model and Optimizer ##############
    network = model.HQSNet(K, mask, lmbda, learn_reg_coeff, args.device, n_hidden=args.num_hidden).to(args.device)
    network = myutils.io.load_checkpoint(network, args.model_dir)
    ##################################################
    
    preds = train.test_hqsnet(network, xdata, args.strategy, args.device, args.alpha)

    if args.noise_std == 0:
        save_path = os.path.join(os.path.dirname(args.model_dir), 'recons_%.5f.npy' % args.alpha)
    else:
        save_path = os.path.join(os.path.dirname(args.model_dir), 'recons_awgn_%.2f.npy' % args.noise_std)
    print('saving recons to', save_path)
    np.save(save_path, preds)
