import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import PIL.Image as Image
import numpy as np


test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_twoaug_transform = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class DatasetObject(Dataset):
    def __init__(self, data, targets, transform=None, twoaug=False):
        # initialize data
        self._X = data
        self._Y = targets
        self.transform = transform
        self.twoaug = twoaug

    def __getitem__(self, index):
        x, y = self._X[index], self._Y[index]
        im = Image.fromarray(x)
        if self.twoaug:
            return self.transform(im), self.transform(im), y
        else:
            if self.transform is not None:
                x = self.transform(im)
            return x, y

    def __len__(self):
        return len(self._X)


class CIFAR10R:
    """`CIFAR10R rotation`_ Dataset.

    Args:
        
    """
    def __init__(self, cfg, ):
        assert cfg.testset.name == 'cifar10r'
        self.datapath = os.path.join(cfg.testset.path, "rotation.npy") # path to the dataset file
        self.angles = (np.linspace(0, 180, 16)).astype(int)
        # load labels from the clean test set of CIFAR10
        valset = datasets.CIFAR10(root='/content/dataset/', train=False, download=False, transform=test_cifar10_transform)
        # list of classes
        self.labels = torch.Tensor(valset.targets).long()
        self.batch_size = cfg.test.batch_size
        self.twoaug = cfg.testset.twoaug
        # a list of dataloaders. [degree1, ..., degree15]
        self.loaders = []
        self._get_loaders()

    def _get_loaders(self):    
        chal_data = np.load(self.datapath)
        for j in range(len(chal_data)):  # enumerate N degrees of challenges.
            print(f'==> Loading {self.angles[j+1]} degree rotation of CIFAR10R')
            chal_temp_data = chal_data[j] 
            assert chal_temp_data.shape == (10000, 32, 32, 3)
            if self.twoaug:
                transform = test_cifar10_twoaug_transform
            else:
                transform = test_cifar10_transform
            chal_dataset = DatasetObject(chal_temp_data, self.labels, transform=transform, twoaug=self.twoaug)
            chal_loader = DataLoader(chal_dataset, batch_size=self.batch_size, shuffle=False)

            self.loaders.append(chal_loader)
