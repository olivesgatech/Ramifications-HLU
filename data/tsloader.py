"""The script to construct dataloader from torch.tensor"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class TensorSet(Dataset):
    def __init__(self, data, targets, transform=None, twoaug=False):
        # initialize data
        self._X = data
        self._Y = targets
        self.transform = transform
        self.twoaug = twoaug

        if torch.is_tensor(self._X): 
            self._X = torch.permute(self._X, (0, 2, 3, 1)).numpy()
  
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
    
    
def get_tensorloader(cfg, data, targets, transform=None):
    tsdataset = TensorSet(data, targets, transform = transform)
    tsloader = DataLoader(dataset=tsdataset,
                          batch_size = cfg.test.batch_size,
                          num_workers=cfg.dataloader.num_workers,
                          shuffle=False)

    return tsloader
