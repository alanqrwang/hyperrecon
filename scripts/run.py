import torch
import numpy as np
from regagcsmri import utils, train, dataset, model
import argparse
import os

if __name__ == "__main__":

    ############### Argument Parsing #################
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', '--filename_prefix', type=str, help='filename prefix', required=True)
    parser.add_argument('--models_dir', default='out/', type=str, help='directory to save models')
    
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--force_lr', type=float, default=None, help='Force learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--load_checkpoint', type=int, default=0, help='Load checkpoint at specificed epoch')
    parser.add_argument('--log_interval', type=int, default=1, help='Frequency of logs')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
    parser.add_argument('--unet_hidden', type=int, default=64)
    parser.add_argument('--topK', type=int, default=None)
    parser.add_argument('--loss_schedule', type=int, default=100)
    parser.add_argument('--sample_schedule', type=int, default=200)
    utils.add_bool_arg(parser, 'range_restrict')

    parser.add_argument('--bounds', nargs='+', type=float, help='<Required> Set flag', required=True)
    parser.add_argument('--reg_types', nargs='+', type=str, help='<Required> Set flag', required=True)
    parser.add_argument('--sampling', choices=['uhs', 'dhs'], type=str, help='Sampling method', required=True)
    parser.add_argument('--hyparch', choices=['small', 'medium', 'large'], type=str, help='Hypernetwork architecture', required=True)

    
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
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

    ################### Filename #####################
    local_name = '{prefix}_{lr}_{batch_size}_{reg_types}_{unet_hidden}_{bounds}_{topK}_{range_restrict}'.format(
        prefix=args.filename_prefix,
        lr=args.lr,
        batch_size=args.batch_size,
        reg_types=args.reg_types,
        unet_hidden=args.unet_hidden,
        bounds=args.bounds,
        range_restrict=args.range_restrict,
        topK=args.topK,
        )
    model_folder = os.path.join(args.models_dir, local_name)
    if not os.path.isdir(model_folder):   
        os.makedirs(model_folder)
    args.filename = model_folder

    train.trainer(xdata, gt_data, vars(args))
