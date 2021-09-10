import torch
import torch.nn.functional as F
from hyperrecon.util import utils

def pperc(recons):
  num_points = len(recons)
  for i in range(num_points-1):
    for j in range(i+1, num_points):
      img1 = utils.absval(recons[i:i+1, ...]).unsqueeze(1)
      img2 = utils.absval(recons[j:j+1, ...]).unsqueeze(1)
      dist_loss = dist_loss + loss_function(img1, img2)
  return dist_loss

def pl2(recons):
  num_points = len(recons)
  return torch.sum(F.pdist(recons.reshape(num_points, -1)))

def compute_loss(recons, loss_type):
  '''Trajectory Net Loss.

  recons: (batch_size, num_points, n1, n2, 2)
  Loss = (1-lmbda)*dist_loss + lmbda*dc_loss
  '''
  # provider = LossProvider()
  # loss_function = provider.get_loss_function('Watson-DFT', colorspace='grey', pretrained=True, reduction='sum').to(device)

  batch_size = len(recons)
  loss = 0
  for b in range(batch_size):
    if loss_type == 'perceptual':
      loss = loss + pperc(recons[b])
    else:
      loss = loss - pl2(recons[b])
  return loss
