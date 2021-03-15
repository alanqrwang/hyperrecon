import os
import torch
from hyperrecon import model, utils, dataset, test, train
import hyperrecon.loss as losslayer
import numpy as np
import argparse
import sys

def compute_dc_thres(xdata, gt_data, trained_reconnet, criterion, num_hyperparams, device, top_percentage=25):
    batch_size = 1
    conf = {}
    conf['device'] = device
    valset = dataset.Dataset(xdata[:10], gt_data[:10])
    params = {'batch_size': batch_size,
         'shuffle': False,
         'num_workers': 0}
    dataloader = torch.utils.data.DataLoader(valset, **params)
    
    hps = torch.rand(1000, num_hyperparams).float().to(device)
    
    res_dict = test.test(trained_reconnet, dataloader, conf, hps, None, take_avg=True, criterion=criterion, give_loss=True, give_metrics=True)
    print(res_dict['dc'].shape)
    
    for i, p in enumerate(res_dict['rpsnr']):
        print(p)

    return np.percentile(res_dict['dc'], top_percentage)

if __name__ == "__main__":
    ############### Argument Parsing #################
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('--models_dir', required=True, type=str, help='directory to save models')
    # # parser.add_argument('--model_num', type=int, required=True, help='Model checkpoint number')
    
    
    # parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    # parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    # parser.add_argument('--num_epochs', type=int, default=100, help='Total training epochs')
    # parser.add_argument('--load_checkpoint', type=int, default=0, help='Load checkpoint at specificed epoch')
    # parser.add_argument('--log_interval', type=int, default=1, help='Frequency of logs')
    # parser.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')

    # parser.add_argument('--num_points', type=int, default=12, help='Number of reconstructions (i.e. hyperparameters) for each slice')
    # parser.add_argument('--lmbda', type=int, default=None, help='Total training epochs')
    # # parser.add_argument('--loss_type', required=True, type=str, choices=['l2', 'perceptual'], help='Total training epochs')

    
    # args = parser.parse_args()
    # if torch.cuda.is_available():
    #     args.device = torch.device('cuda:'+str(args.gpu_id))
    # else:
    #     args.device = torch.device('cpu')

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    models_dir = sys.argv[1]
    print(sys.argv[0])
    print(sys.argv[1])
    print(sys.argv[2])
    model_num = int(sys.argv[2])
    gpu_id = int(sys.argv[3])
    loss_type = sys.argv[4]
    print(models_dir, model_num, gpu_id, loss_type)
    ##################################################

    model_path = os.path.join(models_dir, 'model.%d.h5' % model_num)
    args.num_hyperparams = 3
    hyparch = 'gigantic'
    reg_types = ['cap', 'tv']


    #### Load trained recon net ####
    trained_reconnet = model.Unet(args.device, args.num_hyperparams, hyparch=hyparch, nh=64).to(args.device)
    trained_reconnet = utils.load_checkpoint(trained_reconnet, model_path)
    trained_reconnet.eval()
    for param in trained_reconnet.parameters():
        param.requires_grad = False

    # Shell loss that only is needed for DC, so arguments don't matter
    criterion = losslayer.AmortizedLoss(reg_types, False, 'uhs', args.device, evaluate=True)
	

    #### Load data ####
    xdata = dataset.get_test_data(old=True)
    gt_data = dataset.get_test_gt(old=True)
    testset = dataset.Dataset(xdata, gt_data)
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 4}
    dataloader = torch.utils.data.DataLoader(testset, **params)

    # Compute DC threshold if saved cache doesn't exist
    if os.path.exists(os.path.join(args.models_dir, 'dc_thres.npy')):
        dc_thres = np.load(os.path.join(args.models_dir, 'dc_thres.npy'))
        print('Loaded DC Threshold from cache: ', dc_thres)
    else:
        dc_thres = compute_dc_thres(xdata, gt_data, trained_reconnet, criterion, args.num_hyperparams, args.device)
        np.save(os.path.join(args.models_dir, 'dc_thres.npy'), dc_thres)

    if args.lmbda:
        args.save_path = os.path.join(conf['models_dir'], 'trajnet_%d_%.02f/' % (args.model_num, args.lmbda))
        if not os.path.isdir(args.save_path):   
            os.makedirs(args.save_path)
        network = train.trajtrain(network, dataloader, trained_reconnet, criterion, optimizer, vars(args), args.lmbda)
    else:
        # Schedule lmbda
        lmbda = 0.99
        while True:
            print(lmbda)
            args.save_path = os.path.join(args.models_dir, 'trajnet_%d_%.02f/' % (args.model_num, lmbda))
            if not os.path.isdir(args.save_path):   
                os.makedirs(args.save_path)

            network = model.TrajNet(out_dim=args.num_hyperparams).to(args.device)
            network.train()
            optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
            network = train.trajtrain(network, dataloader, trained_reconnet, criterion, optimizer, vars(args), lmbda)

	    # Evaluate on set points, linearly spaced in [0, 1]
            traj = torch.linspace(0, 1, args.num_points).float().to(device).unsqueeze(1)
            hyperparams = network(traj)
            recons, cap_reg = trained_reconnet(zf, y, hyperparams)
            _, loss_dict, _ = criterion(recons, y, out, cap_reg, None, True)

            # If biggest loss is over DC threshold, then give more weight to the DC loss term
            if torch.max(loss_dict['dc']) > conf['dc_thres']:
                lmbda -= 0.05
            else:
                break
