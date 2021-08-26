import torch
from . import utils

def trajloss(recons, dc_losses, lmbda, device, loss_type, mse=None):
  '''Trajectory Net Loss

  recons: (batch_size, num_points, n1, n2, 2)
  recons: (batch_size, num_points, n1, n2, 2)
  Loss = (1-lmbda)*dist_loss + lmbda*dc_loss
  '''
  # provider = LossProvider()
  # loss_function = provider.get_loss_function('Watson-DFT', colorspace='grey', pretrained=True, reduction='sum').to(device)

  batch_size, num_points = recons.shape[0], recons.shape[1]
  dist_loss = 0
  reg_loss = 0
  for b in range(batch_size):
    if loss_type == 'perceptual':
      for i in range(num_points-1):
        for j in range(i+1, num_points):
          img1 = utils.absval(recons[b, i:i+1, ...]).unsqueeze(1)
          img2 = utils.absval(recons[b, j:j+1, ...]).unsqueeze(1)
          dist_loss = dist_loss + loss_function(img1, img2)

    elif loss_type == 'l2':
      dist_loss = dist_loss - torch.sum(F.pdist(recons[b].reshape(num_points, -1)))
    else:
      raise Exception()

    reg_loss = reg_loss + torch.sum(dc_losses[b])

  if mse is None:
    print(lmbda)
    loss = (1-lmbda)*dist_loss + lmbda*reg_loss
  else:
    print(lmbda)
    loss = (1-lmbda)*dist_loss + lmbda*mse
  
  return loss
