import os
import torch
from hyperrecon import model, utils, dataset, test, train
import hyperrecon.loss as losslayer
import numpy as np
import argparse
import sys
import json
from pprint import pprint

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='TrajectoryNet')

        self.add_argument('--model_dir', default='/nfs02/users/aw847/models/HyperRecon/hyperbiaskernels_tanh/Mar_16/', type=str)
        self.add_argument('--model_name', required=True, type=str)
        self.add_argument('--model_num', type=int, required=True, help='Model checkpoint number')
        
        self.add_argument('--lr', type=float, default=1, help='Learning rate')
        self.add_argument('--batch_size', type=int, default=2, help='Batch size')
        self.add_argument('--num_epochs', type=int, default=1, help='Total training epochs')
        self.add_argument('--log_interval', type=int, default=1, help='Frequency of logs')
        self.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')

        self.add_argument('--num_points', type=int, default=12, help='Number of reconstructions (i.e. hyperparameters) for each slice')
        self.add_argument('--lmbda', type=float, default=None, help='Total training epochs')
        self.add_argument('--loss_type', required=True, type=str, choices=['l2', 'perceptual'], help='Total training epochs')
        self.add_argument('--undersampling_rate', type=int, default=4)
        self.add_argument('-fp', '--prefix', type=str, required=True)

    def parse(self):
        args = self.parse_args()
        args.run_dir = os.path.join(args.model_dir, args.model_name, 'trajnet',\
            '{fp}_{model_num}_{lr}_{batch_size}_{num_points}_{lmbda}_{loss_type}_{undersampling_rate}'.format(
            fp=args.prefix,
            model_num=args.model_num,
            lr=args.lr,
            batch_size=args.batch_size,
            num_points=args.num_points,
            lmbda=args.lmbda,
            loss_type=args.loss_type,
            undersampling_rate=args.undersampling_rate
            ))

        args.ckpt_dir = os.path.join(args.run_dir, 'checkpoints')
        if not os.path.isdir(args.ckpt_dir):   
            os.makedirs(args.ckpt_dir)

        # Print args and save to file
        print('Arguments:')
        pprint(vars(args))
        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)
        return args

if __name__ == "__main__":
    args = Parser().parse()
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        sys.exit('No GPU found')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    model_path = os.path.join(args.model_dir, args.model_name, 'checkpoints/model.%d.h5' % args.model_num)
    args_txtfile = os.path.join(args.model_dir, args.model_name, 'args.txt')
    with open(args_txtfile) as json_file:
        model_args = json.load(json_file)
    reg_types = model_args['reg_types']
    num_hyperparams = len(reg_types) if model_args['range_restrict'] else len(reg_types) + 1
    hyparch = model_args['hyparch']


    #### Load trained recon net ####
    trained_reconnet = model.Unet(args.device, num_hyperparams, hyparch=hyparch, nh=64).to(args.device)
    trained_reconnet = utils.load_checkpoint(trained_reconnet, model_path)
    trained_reconnet.eval()
    for param in trained_reconnet.parameters():
        param.requires_grad = False

    mask = dataset.get_mask(args.undersampling_rate)
    args.mask = torch.tensor(mask, requires_grad=False).float().to(args.device)

    #### Load data ####
    xdata = dataset.get_test_data(old=True)
    gt_data = dataset.get_test_gt(old=True)
    testset = dataset.Dataset(xdata, gt_data)
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 4}
    dataloader = torch.utils.data.DataLoader(testset, **params)

    # Train with fixed lambda if provided
    network = model.TrajNet(out_dim=num_hyperparams).to(args.device)
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    network = train.trajtrain(network, dataloader, trained_reconnet, \
                              optimizer, args)
