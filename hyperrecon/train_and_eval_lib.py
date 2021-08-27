"""Utilities to run model training and evaluation."""

from hyperrecon.uhs import UHS

def get_trainer(args):
  """Get trainer."""
  if args.method.lower() == 'uhs':
    trainer = UHS(args)
  else:
    raise NotImplementedError
  return trainer