"""Utilities to run model training and evaluation."""

from hyperrecon.uhs import UHS
from hyperrecon.dhs import DHS
from hyperrecon.uhs_anneal import UHSAnneal
from hyperrecon.baseline import Baseline

def get_trainer(args):
  """Get trainer."""
  if args.method.lower() == 'uhs':
    trainer = UHS(args)
  elif args.method.lower() == 'dhs':
    trainer = DHS(args)
  elif args.method.lower() == 'uhs_anneal':
    trainer = UHSAnneal(args)
  elif args.method.lower() == 'baseline':
    trainer = Baseline(args)
  else:
    raise NotImplementedError
  return trainer