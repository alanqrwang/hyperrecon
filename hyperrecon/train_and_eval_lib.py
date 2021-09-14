"""Utilities to run model training and evaluation."""

from hyperrecon.uhs import UHS
from hyperrecon.dhs import DHS
from hyperrecon.uhs_anneal import UHSAnneal
from hyperrecon.baseline import Baseline
from hyperrecon.constant import Constant
from hyperrecon.binary import Binary
from hyperrecon.hypernet_baseline_fit import HypernetBaselineFit
from hyperrecon.hypernet_baseline_fit_layer import HypernetBaselineFitLayer

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
  elif args.method.lower() == 'constant':
    trainer = Constant(args)
  elif args.method.lower() == 'binary':
    trainer = Binary(args)
  elif args.method.lower() == 'hypernet_baseline_fit':
    trainer = HypernetBaselineFit(args)
  elif args.method.lower() == 'hypernet_baseline_fit_layer':
    trainer = HypernetBaselineFitLayer(args)
  else:
    raise NotImplementedError
  return trainer