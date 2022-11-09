"""The scripts to construct Uncertain training dataloader(s)"""
from PIL import Image
import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pdb


# dict of noise type in CIFAR-N dataset
noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}

class CIFAR10U(Dataset):
    """Construct uncertain training data from cifar-10-N with label noise <http://noisylabels.com>
    """
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    noise_type_list = ['rand1', 'rand2', 'rand3']

    def __init__(self, cfg, subset='train',
                 transform=None, target_transform=None):
        # load all three random settings.
        self.root = os.path.expanduser(cfg.trainset.root)
        self.transform = transform
        self.target_transform = target_transform
        self.subset = subset  # training set or val set
        self.dataset='cifar10'
        self.nb_classes=10
        self.noise_path = cfg.trainset.noise_path
        idx_each_class_noisy = [[] for i in range(10)]
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
        
        self._all_train_noisy_labels = dict() # stores all the noisy labels from all annotators
        self._all_noise_or_not = dict()
        self._all_actual_noise_rate = dict()
        
        #
        self.concat_train_noisy_ids = [] # contains (duplicate) ids for multiple instances
        #
        self.concat_train_noisy_labels = []
        self.concat_data_uncertain_flag = []
        
        # iterate all the Random (noise_type) label sets
        for noise_type in self.noise_type_list:
            # Load human noisy labels
            train_noisy_labels = self.load_label(noise_type_map[noise_type])
            self._all_train_noisy_labels[noise_type_map[noise_type]] = train_noisy_labels.tolist()
            print(f'noisy labels loaded from {self.noise_path}')



            for i in range(len(train_noisy_labels)):
                idx_each_class_noisy[train_noisy_labels[i]].append(i)
            class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
            noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
            print(f'The noisy data ratio in each class is {noise_prior}')
            noise_or_not = np.transpose(train_noisy_labels)!=np.transpose(self.train_labels)
            self._all_noise_or_not[noise_type_map[noise_type]] = noise_or_not
            actual_noise_rate = np.sum(noise_or_not)/50000
            self._all_actual_noise_rate[noise_type_map[noise_type]] = actual_noise_rate
            print('over all noise rate is ', actual_noise_rate)

        for i, (l1, l2, l3) in enumerate(zip(*[self._all_noise_or_not[noise_type_map[noise_type]] for noise_type in self.noise_type_list])): 
            # print(f'{i}-th sample noise_or_not: {(l1, l2, l3)}', f'clean?: {l1==l2==l3==False}')
            self.concat_data_uncertain_flag.append((not (l1==l2==l3==False)))
            if not (l1==l2==l3==False):  # at least one of the labels is incorrect
                _collect_noisy_labels = [self._all_train_noisy_labels[noise_type_map[noise_type]][i] for noise_type in self.noise_type_list]
                assert len(_collect_noisy_labels) == len(self.noise_type_list)
                _collect_noisy_labels = np.unique(_collect_noisy_labels)
                for _lbl in _collect_noisy_labels:
                    self.concat_train_noisy_ids.append(i)
                    self.concat_train_noisy_labels.append(_lbl)

        # sanity check by loading the worst label set.
        self._worst_noise_or_not = (np.transpose(self.load_label(noise_type_map['worst']))!=np.transpose(self.train_labels))
        assert (self._worst_noise_or_not.tolist()==self.concat_data_uncertain_flag)
        
        self._train_certain_ids = (np.where(self._worst_noise_or_not == False)[0]).tolist() # no duplicate, no overlapping w/ noisy ids..
        # self._train_uncertain_ids = (np.where(self._worst_noise_or_not == True)[0]).tolist() # the unique list of `concat_train_noisy_ids`
        
        num_training_samples = 50000

        if self.subset=='train':
            self.concat_train_correct_ids = self._train_certain_ids[ int(num_training_samples*cfg.trainset.val_ratio): ] # NOTE: the uncertain ids should not be hheeerrree

            self.concat_train_correct_labels = [self.train_labels[i] for i in self.concat_train_correct_ids]
            ## concatenate correct&noisy labels
            self._labels = self.concat_train_correct_labels + self.concat_train_noisy_labels
            ## concatenate correct&noisy ids...
            self._ids = self.concat_train_correct_ids + self.concat_train_noisy_ids

        elif self.subset == 'val':
            self._ids = self._train_certain_ids[ :int(num_training_samples*cfg.trainset.val_ratio) ]
            self._labels = [self.train_labels[i] for i in self._ids]
        else:
            raise Exception(f'invalid {self.subset} subset')
        assert len(self._labels)==len(self._ids)
        
    def load_label(self, noise_type):
        #NOTE only load manual training label
        noise_label = torch.load(self.noise_path)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0  
                print(f'Loaded {noise_type} from {self.noise_path}.')
                print(f'The overall noise rate is {1-np.mean(clean_label.numpy() == noise_label[noise_type])}')
            return noise_label[noise_type].reshape(-1)  
        else:
            raise Exception('Input Error')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[self._ids[index]], self._labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self._labels)
        
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
