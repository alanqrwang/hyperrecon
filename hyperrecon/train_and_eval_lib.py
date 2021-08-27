"""Utilities to run model training and evaluation."""

from hyperrecon.uhs import UHS
from hyperrecon.dhs import DHS

def get_trainer(args):
  """Get trainer."""
  if args.method.lower() == 'uhs':
    trainer = UHS(args)
  elif args.method.lower() == 'dhs':
    trainer = DHS(args)
  else:
    raise NotImplementedError
  return trainer