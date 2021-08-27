import torch
import os
from hyperrecon.argparser import Parser
from hyperrecon.util.train import BaseTrain


if __name__ == "__main__":
  args = Parser().parse()

  # GPU Handling
  if torch.cuda.is_available():
    args.device = torch.device('cuda:'+str(args.gpu_id))
  else:
    args.device = torch.device('cpu')
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

  trainer = BaseTrain(args)
  trainer.config()
  trainer.train()
