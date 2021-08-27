import argparse
import json
import time
import os
from pprint import pprint


class Parser(argparse.ArgumentParser):
  def __init__(self):
    super(Parser, self).__init__(description='HyperRecon')
    # I/O parameters
    self.add_argument('-fp', '--filename_prefix', type=str,
              help='filename prefix', required=True)
    self.add_argument('--models_dir', default='/share/sablab/nfs02/users/aw847/models/HyperRecon/',
              type=str, help='directory to save models')
    self.add_argument('--data_path', default='/share/sablab/nfs02/users/aw847/data/brain/abide/',
              type=str, help='directory to load data')
    self.add_argument('--log_interval', type=int,
              default=25, help='Frequency of logs')
    self.add_argument('--load', type=str, default=None,
              help='Load checkpoint at .h5 path')
    self.add_argument('--cont', type=int, default=0,
              help='Load checkpoint at .h5 path')
    self.add_argument('--gpu_id', type=int, default=0,
              help='gpu id to train on')
    self.add_argument('--date', type=str, default=None,
              help='Override date')
    self.add_bool_arg('legacy_dataset', default=False)

    # Machine learning parameters
    self.add_argument('--lr', type=float, default=1e-3,
              help='Learning rate')
    self.add_argument('--force_lr', type=float,
              default=None, help='Learning rate')
    self.add_argument('--batch_size', type=int,
              default=32, help='Batch size')
    self.add_argument('--num_steps_per_epoch', type=int,
              default=256, help='Batch size')
    self.add_argument('--epochs', type=int, default=100,
              help='Total training epochs')
    self.add_argument('--unet_hdim', type=int, default=32)
    self.add_argument('--hnet_hdim', type=int,
              help='Hypernetwork architecture', default=64)
    self.add_argument('--n_ch_out', type=int,
              help='Number of output channels of main network', default=1)

    # Model parameters
    self.add_argument('--topK', type=int, default=None)
    self.add_argument('--undersampling_rate', type=str, default='4p2',
              choices=['4p2', '8p25', '8p3', '16p2', '16p3'])
    self.add_argument('--loss_list', choices=['dc', 'tv', 'cap', 'wave', 'shear', 'mse', 'l1', 'ssim', 'watson-dft'],
              nargs='+', type=str, help='<Required> Set flag', required=True)
    self.add_argument(
      '--method', choices=['uhs', 'dhs', 'baseline'], type=str, help='Training method', required=True)
    self.add_bool_arg('range_restrict')
    self.add_bool_arg('anneal', default=False)
    self.add_argument('--hyperparameters', type=float, default=None)

  def add_bool_arg(self, name, default=True):
    """Add boolean argument to argparse parser"""
    group = self.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    self.set_defaults(**{name: default})

  def parse(self):
    args = self.parse_args()
    if args.method == 'dhs':
      assert args.topK is not None, 'DHS sampling must set topK'
    if args.date is None:
      date = '{}'.format(time.strftime('%b_%d'))
    else:
      date = args.date

    args.run_dir = os.path.join(args.models_dir, args.filename_prefix, date,
                  '{lr}_{batch_size}_{losses}_{hnet_hdim}_{unet_hdim}_{topK}_{range_restrict}_{hps}'.format(
                    lr=args.lr,
                    batch_size=args.batch_size,
                    losses=args.loss_list,
                    hnet_hdim=args.hnet_hdim,
                    unet_hdim=args.unet_hdim,
                    range_restrict=args.range_restrict,
                    topK=args.topK,
                    hps=args.hyperparameters,
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
