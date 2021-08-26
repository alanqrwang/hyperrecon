from . import loss_ops
import functools

def get_registered_sup_losses():
  return [
      'ssim',
      'watson-dft',
      'l1',
      'mse',
      'dice',
  ]
def get_registered_unsup_losses():
  return [
      'dc',
      'tv',
      'wave',
      'shear',
  ]


def compose_loss_seq(loss_list, is_training=False):
  """Compose Augmentation Sequence.

  Args:
    aug_list: List of tuples (aug_type, kwargs)
    is_training: Boolean

  Returns:
    sequence of augmentation ops
  """
  return [
      generate_loss_ops(loss_type)
      for loss_type in loss_list
  ]

def generate_loss_ops(loss_type):
  """Generate Augmentation Operators.

  Args:
    loss_type: Augmentation type
    is_training: Boolean
    **kwargs: for backward compatibility.

  Returns:
    augmentation ops
  """
  assert loss_type.lower() in get_registered_sup_losses() + get_registered_unsup_losses()

  if loss_type.lower() == 'tv':
    tx_op = loss_ops.Total_Variation()
  elif loss_type.lower() == 'wave':
    tx_op = loss_ops.L1_Wavelets()
  elif loss_type.lower() == 'shear':
    tx_op = loss_ops.L1_Shearlets()
  elif loss_type.lower() == 'ssim':
    tx_op = loss_ops.SSIM()
  elif loss_type.lower() == 'watson-dft':
    tx_op = loss_ops.Watson_DFT()
  elif loss_type.lower() == 'dc':
    tx_op = loss_ops.Data_Consistency()
  elif loss_type.lower() == 'l1':
    tx_op = loss_ops.L1()
  elif loss_type.lower() == 'mse':
    tx_op = loss_ops.MSE()
  elif loss_type.lower() == 'dice':
    tx_op = loss_ops.DICE()
  else:
    raise NotImplementedError

  return functools.partial(tx_op)