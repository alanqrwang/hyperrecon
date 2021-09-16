"""Utilities to run model training and evaluation."""

from hyperrecon.uniform import Uniform
from hyperrecon.uniform import UniformConstant
from hyperrecon.dhs import DHS
from hyperrecon.baseline import Baseline
from hyperrecon.categorical import CategoricalConstant
from hyperrecon.constant import Constant
from hyperrecon.binary import Binary
from hyperrecon.binary import BinaryConstantBatch
from hyperrecon.binary import BinaryAnneal
from hyperrecon.hypernet_baseline_fit import HypernetBaselineFit
from hyperrecon.hypernet_baseline_fit_layer import HypernetBaselineFitLayer

def get_trainer(args):
  """Get trainer."""
  if args.method.lower() == 'baseline':
    trainer = Baseline(args)
  elif args.method.lower() == 'binary':
    trainer = Binary(args)
  elif args.method.lower() == 'binary_constant_batch':
    trainer = BinaryConstantBatch(args)
  elif args.method.lower() == 'binary_anneal':
    trainer = BinaryAnneal(args)
  elif args.method.lower() == 'categorical_constant':
    trainer = CategoricalConstant(args)
  elif args.method.lower() == 'constant':
    trainer = Constant(args)
  elif args.method.lower() == 'dhs':
    trainer = DHS(args)
  elif args.method.lower() == 'hypernet_baseline_fit':
    trainer = HypernetBaselineFit(args)
  elif args.method.lower() == 'hypernet_baseline_fit_layer':
    trainer = HypernetBaselineFitLayer(args)
  elif args.method.lower() == 'uniform':
    trainer = Uniform(args)
  elif args.method.lower() == 'uniform_constant':
    trainer = UniformConstant(args)
  else:
    raise NotImplementedError
  return trainer