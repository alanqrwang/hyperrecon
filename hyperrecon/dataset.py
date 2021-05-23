"""
Dataset wrapper class for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
from torch.utils import data
import numpy as np
import os
import torchio as tio
import matplotlib.pyplot as plt

class Dataset(data.Dataset):
    def __init__(self, data, gts, segs=None):
        self.data = data
        self.gts = gts
        self.segs = segs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data and get label
        x = self.data[index]
        y = self.gts[index]
        if self.segs is not None:
            z = self.segs[index]
            return x, y, z

        return x, y

class VolumeDataset(data.Dataset):
    def __init__(self, data_path, split, total_subjects=None, transform=None, include_seg=False):
        super(VolumeDataset, self).__init__()
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.include_seg = include_seg
        self.total_subjects = total_subjects

        print('gather subjects')
        self.subjects = self._gather_subjects()
        print('done')
        self.total_subjects = len(self.subjects) if total_subjects is None else total_subjects

    def _gather_subjects(self):
        vol_base_path = os.path.join(self.data_path, self.split, 'origs/')
        seg_base_path = os.path.join(self.data_path, self.split, 'asegs/')
        vol_names = os.listdir(vol_base_path)[:self.total_subjects]
        num_subjects = len(vol_names) if self.total_subjects is None else self.total_subjects
        vol_names = vol_names[:num_subjects]

        subjects = []
        for vol_name in vol_names:
            if vol_name.endswith('.npz'):
                name = vol_name[:-8]
                vol_path = os.path.join(vol_base_path, name + 'orig.npz')
                seg_path = os.path.join(seg_base_path, name + 'aseg.npz')
                if self.include_seg:
                    subject = tio.Subject(
                                 mri=tio.ScalarImage(vol_path, reader=self._reader),
                                 seg=tio.ScalarImage(seg_path, reader=self._reader),
                             )
                else:
                    subject = tio.Subject(
                                 mri=tio.ScalarImage(vol_path, reader=self._reader),
                             )
                subjects.append(subject)
        return subjects

    def _reader(self, path):
        img = np.load(path)
        img = img['vol_data']
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.permute(0, 2, 1, 3).float()
        return img, np.eye(4)

    def get_tio_dataset(self):
        return tio.SubjectsDataset(self.subjects, transform=self.transform)

######### Loading data #####################
def get_mask(mask_dims, undersampling_rate, as_tensor=True, centered=False):
    # mask = np.load('data/mask.npy')
    path = '/share/sablab/nfs02/users/aw847/data/masks/poisson_disk_{maskname}_{mask_dims}.npy'.format(maskname=undersampling_rate, mask_dims=mask_dims)
    print('Loading mask:', path)
    mask = np.load(path)
    print('done')

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

def get_train_gt(img_dims='160_224', organ='brain'):
    if organ == 'brain':
        # gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_train_normalized.npy'
        gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/segs/train_mri_normalized_{img_dims}.npy'.format(img_dims=img_dims)
        
    else:
        gt_path = '/share/sablab/nfs02/users/aw847/data/knee/knee_train_normalized.npy'
    gt = get_data(gt_path)
    return gt

def get_train_data(img_dims, maskname, organ='brain', size='med'):
    if organ == 'brain':
        # data_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_train_normalized_{maskname}.npy'.format(maskname=maskname)
        data_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/segs/train_mri_normalized_{img_dims}_{maskname}.npy'.format(img_dims=img_dims, maskname=maskname)
    elif organ == 'knee':
        data_path = '/share/sablab/nfs02/users/aw847/data/knee/knee_train_normalized_{maskname}.npy'.format(maskname=maskname)
    data = get_data(data_path)
    return data


def get_test_gt(img_dims, organ='brain', size='med'):
    if size=='small':
        gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_test_normalized_10slices.npy'
    elif size=='med' and organ=='brain':
        # gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_test_normalized.npy'
        gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/segs/test_mri_normalized_{img_dims}.npy'.format(img_dims=img_dims)
    elif size=='med' and organ=='knee':
        gt_path = '/share/sablab/nfs02/users/aw847/data/knee/knee_test_normalized.npy'
    gt = get_data(gt_path)
    return gt

def get_test_data(img_dims, maskname, organ='brain', size='med'):
    if size=='small':
        data_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_test_normalized_4p2_10slices.npy'
    elif size=='med':
        if organ == 'brain':
            # data_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_test_normalized_{maskname}.npy'.format(maskname=maskname)
            data_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/segs/test_mri_normalized_{img_dims}_{maskname}.npy'.format(img_dims=img_dims, maskname=maskname)
        elif organ == 'knee':
            data_path = '/share/sablab/nfs02/users/aw847/data/knee/knee_test_normalized_{maskname}.npy'.format(maskname=maskname)
    data = get_data(data_path)
    return data

def get_seg_data(split, img_dims):
    seg_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/segs/{split}_seg_{img_dims}.npy'.format(split=split, img_dims=img_dims)
    seg = get_data(seg_path)
    return seg
