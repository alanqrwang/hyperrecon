from torch.utils import data

class ArrDataset(data.Dataset):
  def __init__(self, xs, ys):
    self.xs = xs
    self.ys = ys

  def __len__(self):
    return len(self.xs)

  def __getitem__(self, index):
    # Load data and get label
    x = self.xs[index]
    y = self.ys[index]
    return x, y