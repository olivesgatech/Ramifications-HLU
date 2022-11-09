#!/usr/bin/env python
# coding: utf-8
## The script for Data Preprocessing. Currently contains functions to generate CIFAR-10 rotations ##
import os
from torchvision import transforms, datasets
import numpy as np
import scipy.ndimage as ndim


# get the validation set of CIFAR10 to be converted to rotation version.
valset = datasets.CIFAR10(root='/media/cmhung/MySSD/dataset/', train=False, download=False)
data = valset.data
# data.shape

steps = 16
data_rotated = np.zeros((steps-1,data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
# data_rotated.shape

s_rot = 0
end_rot = 180
rotations = (np.linspace(s_rot, end_rot, steps)).astype(int)

for r in range(1,steps):
    angle = rotations[r]

    data_rotated[r-1] = ndim.interpolation.rotate(data,angle, axes=(1,2),reshape=False ,mode='nearest')
    print(f'generated {angle} degree rotated data!')

data_rotated = data_rotated.astype(np.uint8)
save_path = '/media/cmhung/MySSD/dataset/CIFAR-10-R' 
os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path, "rotation.npy"),data_rotated)
