"""load class-conditional random label noise, with some kind of randomisation """
import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from .utils import noisify, noisify_instance, check_integrity
import pdb

train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR10-LabelNoise dataset #
class CIFAR10N(Dataset):
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, cfg, subset='train',
                 transform=None, target_transform=None):
        # load all three random settings.
        self.root = os.path.expanduser(cfg.trainset.root)
        self.transform = transform
        self.target_transform = target_transform
        self.subset = subset  # training set or val set
        self.dataset='cifar10'
        self.nb_classes=10
        self.noise_file = cfg.trainset.noise_path
        idx_each_class_noisy = [[] for i in range(10)]

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                            ' You can use download=True to download it')
        # now load the picked numpy arrays
        self.train_data = []
        self.train_labels = []
        for fentry in self.train_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.train_data.append(entry['data'])
            if 'labels' in entry:
                self.train_labels += entry['labels']
            else:
                self.train_labels += entry['fine_labels']
            fo.close()

        # the original training data (w/o multiple copies)
        self.train_data = np.concatenate(self.train_data)
        self.train_data = self.train_data.reshape((50000, 3, 32, 32))
        self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        # case 0: generate synthetic class-conditioned label noise (noisify train labels)
        if cfg.trainset.nlbl.tf not in ['instance', 'manual']:  # 'pairflip' | 'symmetric'
            self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
            self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.train_labels, noise_type=cfg.trainset.nlbl.tf, noise_rate=cfg.trainset.nlbl.rate, random_state=cfg.seed, nb_classes=self.nb_classes)
            self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
            _train_labels=[i[0] for i in self.train_labels]
            for i in range(len(_train_labels)):
                idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
            class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
            self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
            print(f'The noisy data ratio in each class is {self.noise_prior}')
            self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(_train_labels)
        # case 1: generate synthetic instance-dependent asymmetric noise generate on the fly
        elif cfg.trainset.nlbl.tf == 'instance': # instance-dependent label noise 'symmetric' | 'asymmetric'
            self.train_noisy_labels, self.actual_noise_rate = noisify_instance(self.train_data, self.train_labels,noise_rate=cfg.trainset.nlbl.rate)
            print('over all noise rate is ', self.actual_noise_rate)
            #self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
            #self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
            #_train_labels=[i[0] for i in self.train_labels]
            for i in range(len(self.train_labels)):
                idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
            class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
            self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
            print(f'The noisy data ratio in each class is {self.noise_prior}')
            self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)
        # case 2: synthetic asymmetric noise loaded from a local file
        elif cfg.trainset.nlbl.tf == 'manual':
            # load noise label
            train_noisy_labels = self.load_label()
            self.train_noisy_labels = train_noisy_labels.numpy().tolist()
            print(f'noisy labels loaded from {self.noise_file}')

            for i in range(len(self.train_noisy_labels)):
                idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
            class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
            self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
            print(f'The noisy data ratio in each class is {self.noise_prior}')
            self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)        

        num_training_samples = 50000

        if self.subset=='train':
            self._ids = list(range(num_training_samples))[ int(num_training_samples*cfg.trainset.val_ratio): ] 
        elif self.subset == 'val':
            self._ids = list(range(num_training_samples))[ :int(num_training_samples*cfg.trainset.val_ratio) ]
        else:
            raise Exception(f'invalid {self.subset} subset')
        
        # print(f'The noisy data ratio in each class is {self.noise_prior}')
        # self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)
        self.actual_noise_rate = np.sum(self.noise_or_not)/num_training_samples
        print('over all noise rate is ', self.actual_noise_rate)        

    def load_label(self):
        #NOTE presently only use for load manual training label
        # noise_label = torch.load(self.noise_type)   # f'../../{self.noise_type}'
        # noise_label = torch.load(f'noise_label/cifar-10/{self.noise_type}')
        assert self.noise_file != 'None'
        noise_label = torch.load(self.noise_file)
        if isinstance(noise_label, dict):
            if "clean_label_train" in noise_label.keys():
                clean_label = noise_label['clean_label_train']
                assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0  # commented for noise identification (NID) since we need to replace labels
            if "clean_label" in noise_label.keys() and 'raw_index' in noise_label.keys():
                assert torch.sum(torch.tensor(self.train_labels)[noise_label['raw_index']] != noise_label['clean_label']) == 0
                noise_level = torch.sum(noise_label['clean_label'] == noise_label['noisy_label'])*1.0/(noise_label['clean_label'].shape[0])
                print(f'the overall noise level is {noise_level}')
                self.train_data = self.train_data[noise_label['raw_index']]
            return noise_label['noise_label_train'].view(-1).long() if 'noise_label_train' in noise_label.keys() else noise_label['noisy_label'].view(-1).long()  # % 10
            
        else:
            return noise_label.view(-1).long()  # % 10        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[self._ids[index]], self.train_noisy_labels[self._ids[index]]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self._ids[index]

    def __len__(self): 
        return len(self._ids)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.subset #'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    
def getCCNloader(cfg):
    """only supports the label noise setting"""
    assert cfg.trainset.setting == 'NoisyLabel'
    if cfg.trainset.name == 'cifar10':
        num_classes = 10
        train_dataset = CIFAR10N(cfg, 'train', transform = train_cifar10_transform)
        val_dataset = CIFAR10N(cfg, 'val', transform = train_cifar10_transform)
        return train_dataset, val_dataset, num_classes
