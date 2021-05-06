"""
Dataset wrapper class for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data and get label
        x = self.data[index]
        y = self.labels[index]
        return x, y

######### Loading data #####################
def get_mask(undersampling_rate, as_tensor=True, centered=False):
    # mask = np.load('data/mask.npy')
    if undersampling_rate == '4p2':
        mask = np.load('/share/sablab/nfs02/users/aw847/data/masks/poisson_disk_4p2_256_256.npy')
    if undersampling_rate == '8p25':
        mask = np.load('/share/sablab/nfs02/users/aw847/data/masks/poisson_disk_8p25_256_256.npy')
    if undersampling_rate == '8p3':
        mask = np.load('/share/sablab/nfs02/users/aw847/data/masks/poisson_disk_8p3_256_256.npy')

    if not centered:
        mask = np.fft.fftshift(mask)

    if as_tensor:
        return torch.tensor(mask, requires_grad=False).float()
    else:
        return mask

def get_data(data_path):
    print('Loading from', data_path)
    xdata = np.load(data_path)
    assert len(xdata.shape) == 4
    print('Shape:', xdata.shape)
    return xdata

def get_train_gt(organ='brain', size='med'):
    # gt_path = 'data/example_x.npy'
    if size == 'med':
        if organ == 'brain':
            gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_train_normalized.npy'
        else:
            gt_path = '/share/sablab/nfs02/users/aw847/data/knee/knee_train_normalized.npy'
    else:
        gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/20000splits/brain_train_normalized.npy'
    gt = get_data(gt_path)
    return gt

def get_train_data(maskname, organ='brain', size='med'):
    # data_path = 'data/example_y.npy'
    if size == 'med':
        if organ == 'brain':
            data_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_train_normalized_{maskname}.npy'.format(maskname=maskname)
        elif organ == 'knee':
            data_path = '/share/sablab/nfs02/users/aw847/data/knee/knee_train_normalized_{maskname}.npy'.format(maskname=maskname)
    else:
        data_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/20000splits/brain_train_normalized_4p2.npy'
    data = get_data(data_path)
    return data


def get_test_gt(organ='brain', size='med'):
    if size=='small':
        gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_test_normalized_10slices.npy'
    elif size=='med' and organ=='brain':
        gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_test_normalized.npy'
    elif size=='med' and organ=='knee':
        gt_path = '/share/sablab/nfs02/users/aw847/data/knee/knee_test_normalized.npy'
    elif size=='large':
        gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/20000splits/brain_test_normalized.npy'
    gt = get_data(gt_path)
    return gt

def get_test_data(maskname, organ='brain', size='med'):
    if size=='small':
        data_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_test_normalized_4p2_10slices.npy'
    elif size=='med':
        if organ == 'brain':
            data_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_test_normalized_{maskname}.npy'.format(maskname=maskname)
        elif organ == 'knee':
            data_path = '/share/sablab/nfs02/users/aw847/data/knee/knee_test_normalized_{maskname}.npy'.format(maskname=maskname)
    elif size=='large':
        data_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/20000splits/brain_test_normalized_4p2.npy'
    data = get_data(data_path)
    return data
