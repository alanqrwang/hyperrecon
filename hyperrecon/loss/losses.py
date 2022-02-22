from . import loss_ops
import functools

REGISTERED_SUP_LOSSES = [
                          'ssim',
                          'watson-dft',
                          'l1',
                          'mse',
                          'dice',
                          'lpips',
                        ]
REGISTERED_UNSUP_LOSSES = [
                              'dc',
                              'mindc',
                              'tv',
                              'wave',
                              'shear',
                              'l1pen'
                            ]


def compose_loss_seq(loss_list, forward_model, mask, device):
  """Compose loss list.

  Args:
    aug_list: List of tuples (aug_type, kwargs)
    mask: Under-sampling mask
    device: Cuda device
  """
  return [
    generate_loss_ops(loss_type, forward_model, mask, device)
    for loss_type in loss_list
  ]

def generate_loss_ops(loss_type, forward_model, mask, device):
  """Generate Loss Operators."""
  assert loss_type.lower() in REGISTERED_SUP_LOSSES + REGISTERED_UNSUP_LOSSES

  if loss_type.lower() == 'tv':
    tx_op = loss_ops.TotalVariation()
  elif loss_type.lower() == 'wave':
    tx_op = loss_ops.L1Wavelets(device)
  elif loss_type.lower() == 'ssim':
    tx_op = loss_ops.SSIM()
  elif loss_type.lower() == 'dc':
    tx_op = loss_ops.DataConsistency(forward_model, mask)
  elif loss_type.lower() == 'l1':
    tx_op = loss_ops.L1()
  elif loss_type.lower() == 'mse':
    tx_op = loss_ops.MSE()
  elif loss_type.lower() == 'l1pen':
    tx_op = loss_ops.L1PenaltyWeights()
  else:
    raise NotImplementedError

  return functools.partial(tx_op)