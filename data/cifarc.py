"""The scripts to construct Test dataloader(s)"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
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


class CIFAR10C:
    """`CIFAR10C <https://zenodo.org/record/2535967>`_ Dataset.

    Args:
        
    """
    def __init__(self, cfg, ):
        assert cfg.testset.name == 'cifar10c'
        self.chalPath = cfg.testset.path  # path to the dataset files
        self.chals = sorted(os.listdir(self.chalPath))  # list of challenges (*.npy) files
        self.chals.remove('labels.npy')
        # load labels from the clean test set of CIFAR10
        valset = torchvision.datasets.CIFAR10(root='/media/cmhung/MySSD/dataset/', train=False, download=False, transform=test_cifar10_transform)
        # list of classes
        self.chal_labels = torch.Tensor(valset.targets).long()
        self.batch_size = cfg.test.batch_size
        self.twoaug = cfg.testset.twoaug
        # a dict with list of dataloaders. {<challenge type>: [level1, ..., level5]}
        self.loaders = dict()
        self._get_loaders()

    def _get_loaders(self):
        for challenge in range(len(self.chals)):
            chal_data = np.load(os.path.join(self.chalPath, self.chals[challenge]))
            self.loaders[self.chals[challenge]] = []
            for j in range(5):  # enumerate 5 levels of challenges.
                print(f'==> Loading level {j} {self.chals[challenge]} of CIFAR10C')
                chal_temp_data = chal_data[j * 10000:(j + 1) * 10000] 
                # chal_temp_data = preprocess_test(chal_temp_data)
                if self.twoaug:
                    transform = test_cifar10_twoaug_transform
                else:
                    transform = test_cifar10_transform
                chal_dataset = DatasetObject(chal_temp_data, self.chal_labels, transform=transform, twoaug=self.twoaug)
                chal_loader = DataLoader(chal_dataset, batch_size=self.batch_size, shuffle=False)

                self.loaders[self.chals[challenge]].append(chal_loader)