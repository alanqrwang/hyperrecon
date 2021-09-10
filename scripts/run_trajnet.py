import os
import torch
from hyperrecon.model.unet import HyperUnet
from hyperrecon.util import utils
from hyperrecon.data.mask import get_mask
from hyperrecon.traj.model import TrajNet
from hyperrecon.traj.loss import compute_loss
from hyperrecon.data.brain import SliceDataset
from hyperrecon.model.layers import ClipByPercentile
from torchvision import transforms
import hyperrecon.loss as losslayer
from tqdm import tqdm
import argparse
import sys
import json
from pprint import pprint
import numpy as np


class Parser(argparse.ArgumentParser):
  def __init__(self):
    super(Parser, self).__init__(description='TrajectoryNet')

    self.add_argument(
      '--model_dir', default='/nfs02/users/aw847/models/HyperRecon/hyperbiaskernels_tanh/Mar_16/', type=str)
    self.add_argument('--model_name', type=str,
              required=True, help='Model checkpoint number')
    self.add_argument('--model_num', type=int,
              required=True, help='Model checkpoint number')

    self.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    self.add_argument('--batch_size', type=int,
              default=2, help='Batch size')
    self.add_argument('--num_epochs', type=int, default=1,
              help='Total training epochs')
    self.add_argument('--log_interval', type=int,
              default=1, help='Frequency of logs')
    self.add_argument('--gpu_id', type=int, default=0,
              help='gpu id to train on')
    self.add_argument('--num_steps_per_epoch', type=int, default=256,
              help='gpu id to train on')

    self.add_argument('--num_points', type=int, default=12,
              help='Number of reconstructions (i.e. hyperparameters) for each slice')
    self.add_argument('--lmbda', type=float, default=None,
              help='Total training epochs')
    self.add_argument('--loss_type', required=True, type=str,
              choices=['l2', 'perceptual'], help='Total training epochs')
    self.add_argument('--num_val_subjects', type=int, default=5,
              help='Number of subjects to validate on')

  def parse(self):
    args = self.parse_args()
    args.run_dir = os.path.join(args.model_dir, 'traj', args.model_name, 
                  '{model_num}_{lr}_{batch_size}_{num_points}_{lmbda}_{loss_type}'.format(
                    model_num=args.model_num,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    num_points=args.num_points,
                    lmbda=args.lmbda,
                    loss_type=args.loss_type,
                  ))

    # Create save directories
    args.ckpt_dir = os.path.join(args.run_dir, 'checkpoints')
    if not os.path.isdir(args.ckpt_dir):
      os.makedirs(args.ckpt_dir)
    args.monitor_dir = os.path.join(args.run_dir, 'monitor')
    if not os.path.exists(args.monitor_dir):
      os.makedirs(args.monitor_dir)
    args.img_dir = os.path.join(args.run_dir, 'img')
    if not os.path.exists(args.img_dir):
      os.makedirs(args.img_dir)

    # Print args and save to file
    print('Arguments:')
    pprint(vars(args))
    with open(args.run_dir + "/args.txt", 'w') as args_file:
      json.dump(vars(args), args_file, indent=4)
    return args

def prepare_batch(batch, device, mask, num_points):
  targets, segs = batch[0].float().to(device), batch[1].float().to(device)
  targets = targets.repeat_interleave(num_points, dim=0)

  under_ksp = utils.generate_measurement(targets, mask)
  zf = utils.ifft(under_ksp)
  under_ksp, zf = utils.scale(under_ksp, zf)
  return zf, targets, under_ksp, segs

def generate_coefficients(device, samples):
  '''Generates coefficients from samples.'''
  alpha = samples[:, 0]
  beta = samples[:, 1]
  coeffs = torch.stack(
    (alpha, (1-alpha)*beta, (1-alpha)*(1-beta)), dim=1)
  return coeffs.to(device)

def train(network, train_loader, val_loader, trained_reconnet, optimizer, scheduler, args):
  logger = {}
  logger['loss:train'] = []
  logger['learning_rate'] = []

  for _ in range(1, args.num_epochs+1):
    for i, batch in tqdm(enumerate(train_loader), total=args.num_steps_per_epoch):
      zf, gt, y, _ = prepare_batch(batch, args.device, args.mask, args.num_points)

      # Forward through trajectory net
      traj = torch.rand(
        args.num_points*args.batch_size).float().to(args.device).unsqueeze(1)

      optimizer.zero_grad()
      with torch.set_grad_enabled(True):
        out = network(traj)
        coeffs = generate_coefficients(args.device, out)

        # Forward through recon-net
        recons = trained_reconnet(zf, coeffs)
        _, nch, n1, n2 = recons.shape
        recons = recons.view(
          args.batch_size, args.num_points, nch, n1, n2)

        # Evaluate loss
        loss = compute_loss(recons, args.loss_type)
        loss.backward()
        optimizer.step()
        scheduler.step()

      logger['loss:train'].append(loss.item())
      logger['learning_rate'].append(scheduler.get_last_lr())
      utils.save_metrics(args.monitor_dir, logger, 'loss:train')
      utils.save_metrics(args.monitor_dir, logger, 'learning_rate')

      if i == args.num_steps_per_epoch:
        break

    final_out = network(torch.linspace(0, 1, 12).view(-1, 1).to(args.device))
    batch = next(iter(val_loader))
    zf, _, _, _ = prepare_batch(batch, args.device, args.mask, args.num_points)
    recons = trained_reconnet(zf, final_out)
    print(final_out)
    np.save(os.path.join(args.img_dir, 'recons'), recons.cpu().detach().numpy())

  return network


if __name__ == "__main__":
  args = Parser().parse()
  if torch.cuda.is_available():
    args.device = torch.device('cuda:'+str(args.gpu_id))
  else:
    sys.exit('No GPU found')

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

  model_path = os.path.join(
    args.model_dir, 'checkpoints/model.%d.h5' % args.model_num)
  args_txtfile = os.path.join(args.model_dir, 'args.txt')
  with open(args_txtfile) as json_file:
    model_args = json.load(json_file)

  # Load trained recon net
  num_hyperparams = len(model_args['loss_list'])
  trained_reconnet = HyperUnet(
        num_hyperparams,
        model_args['hnet_hdim'],
        in_ch_main=2,
        out_ch_main=model_args['n_ch_out'],
        h_ch_main=model_args['unet_hdim'],
        use_batchnorm=model_args['use_batchnorm']
      ).to(args.device)
  trained_reconnet = utils.load_checkpoint(trained_reconnet, model_path)
  trained_reconnet.eval()
  for param in trained_reconnet.parameters():
    param.requires_grad = False

  args.mask = get_mask(
      model_args['mask_type'],
      model_args['image_dims'], 
      model_args['undersampling_rate']).to(args.device)

  # Load data 
  transform = transforms.Compose([ClipByPercentile()])
  dataset = SliceDataset(
        model_args['data_path'], 'validate', total_subjects=args.num_val_subjects, transform=transform)
  train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True)
  val_loader = torch.utils.data.DataLoader(dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True)

  # Train 
  network = TrajNet(out_dim=num_hyperparams).to(args.device)
  network.train()
  optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                         step_size=50,
                         gamma=0.1)
  network = train(network, train_loader, val_loader, trained_reconnet,
            optimizer, scheduler, args)
