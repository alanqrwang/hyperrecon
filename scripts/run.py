import torch
import os
from hyperrecon.argparser import Parser
from hyperrecon import train_and_eval_lib


if __name__ == "__main__":
  args = Parser().parse()

  # GPU Handling
  if torch.cuda.is_available():
    args.device = torch.device('cuda:0')
  else:
    args.device = torch.device('cpu')
    print('WARNING: No GPU detected!')
  os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

  trainer = train_and_eval_lib.get_trainer(args)
  trainer.config()
  trainer.train()
