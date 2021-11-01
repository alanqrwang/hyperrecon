from torch.utils import data

class ArrDataset(data.Dataset):
  def __init__(self, x):
    self.x = x

  def __len__(self):
    return len(self.x)

  def __getitem__(self, index):
    # Load data and get label
    return self.x[index], self.x[index]