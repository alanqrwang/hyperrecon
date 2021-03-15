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
def get_mask(undersampling_rate, centered=False):
    # mask = np.load('data/mask.npy')
    if undersampling_rate == 4:
        mask = np.load('/nfs02/users/aw847/data/masks/poisson_disk_4p2_256_256.npy')
    if undersampling_rate == 8:
        mask = np.load('/nfs02/users/aw847/data/masks/poisson_disk_8p3_256_256.npy')
    if not centered:
        return np.fft.fftshift(mask)
    else:
        return mask

def get_data(data_path):
    print('Loading from', data_path)
    xdata = np.load(data_path)
    assert len(xdata.shape) == 4
    print('Shape:', xdata.shape)
    return xdata

def get_train_gt(old=False):
    # gt_path = 'data/example_x.npy'
    if old:
        gt_path = '/nfs02/users/aw847/data/brain/adrian/brain_train_normalized.npy'
    else:
        gt_path = '/nfs02/users/aw847/data/brain/adrian/20000splits/brain_train_normalized.npy'
    gt = get_data(gt_path)
    return gt

def get_train_data(undersampling_rate, old=False):
    # data_path = 'data/example_y.npy'
    if old:
        if undersampling_rate == 4:
            data_path = '/nfs02/users/aw847/data/brain/adrian/brain_train_normalized_4p2.npy'
        else:
            data_path = '/nfs02/users/aw847/data/brain/adrian/brain_train_normalized_8p3.npy'
    else:
        data_path = '/nfs02/users/aw847/data/brain/adrian/20000splits/brain_train_normalized_4p2.npy'
    data = get_data(data_path)
    return data


def get_test_gt(old=False):
    if old:
        gt_path = '/nfs02/users/aw847/data/brain/adrian/brain_test_normalized.npy'
    else:
        gt_path = '/nfs02/users/aw847/data/brain/adrian/20000splits/brain_test_normalized.npy'
    gt = get_data(gt_path)
    return gt

def get_test_data(old=False):
    if old:
        data_path = '/nfs02/users/aw847/data/brain/adrian/brain_test_normalized_4p2.npy'
    else:
        data_path = '/nfs02/users/aw847/data/brain/adrian/20000splits/brain_test_normalized_4p2.npy'
    data = get_data(data_path)
    return data
