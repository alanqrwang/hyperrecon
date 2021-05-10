"""
Dataset wrapper class for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
from torch.utils import data
import torchio as tio
import numpy as np
import os

class SubjectDataset(data.Dataset):
    def __init__(self, data_path, split, transform=None):
        super(SubjectDataset, self).__init__()
        self.data_path = data_path
        self.split = split
        self.transform = transform

        self.subjects = self._gather_subjects()

    def _gather_subjects(self):
        vol_base_path = os.path.join(self.data_path, self.split, 'origs/')
        seg_base_path = os.path.join(self.data_path, self.split, 'asegs/')
        vol_names = os.listdir(vol_base_path)

        subjects = []
        for vol_name in vol_names:
            if vol_name.endswith('.npz'):
                name = vol_name[:-8]
                vol_path = os.path.join(vol_base_path, name + 'orig.npz')
                seg_path = os.path.join(seg_base_path, name + 'aseg.npz')
                subject = tio.Subject(
                             mri=tio.ScalarImage(vol_path, reader=self._reader),
                             seg=tio.LabelMap(seg_path, reader=self._reader),
                         )
                # print((subject.spatial_shape[2]))
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
