"""Utilities to run model training and evaluation."""

from hyperrecon.data_driven import DataDriven
from hyperrecon.util.train import BaseTrain

def get_trainer(args):
  """Get trainer."""
  if args.method.lower() == 'base_train':
    trainer = BaseTrain(args)
  elif args.method.lower() == 'dhs':
    trainer = DataDriven(args)
  else:
    raise NotImplementedError
  return trainer