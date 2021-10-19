"""Utilities to run model training and evaluation."""

from hyperrecon.uniform import UniformDiversityPrior
from hyperrecon.data_driven import DataDriven
from hyperrecon.binary import BinaryAnneal
from hyperrecon.hypernet_baseline_fit import HypernetBaselineFit
from hyperrecon.hypernet_baseline_fit_layer import HypernetBaselineFitLayer
from hyperrecon.loupe import Loupe, LoupeAgnostic
from hyperrecon.util.train import BaseTrain

def get_trainer(args):
  """Get trainer."""
  if args.method.lower() == 'base_train':
    trainer = BaseTrain(args)
  elif args.method.lower() == 'binary_anneal':
    trainer = BinaryAnneal(args)
  elif args.method.lower() == 'dhs':
    trainer = DataDriven(args)
  elif args.method.lower() == 'hypernet_baseline_fit':
    trainer = HypernetBaselineFit(args)
  elif args.method.lower() == 'hypernet_baseline_fit_layer':
    trainer = HypernetBaselineFitLayer(args)
  elif args.method.lower() == 'uniform_diversity_prior':
    trainer = UniformDiversityPrior(args)
  elif args.method.lower() == 'loupe':
    trainer = Loupe(args)
  elif args.method.lower() == 'loupe_agnostic':
    trainer = LoupeAgnostic(args)
  else:
    raise NotImplementedError
  return trainer