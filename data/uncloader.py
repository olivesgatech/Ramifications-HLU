"""The script loads samples with uncertain labels."""
from PIL import Image
import os
import sys
sys.path.append('../image-clustering/')
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import pdb


# dict of noise type in CIFAR-N dataset
noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}


train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def get_uncertain_labels(X, y, unc_indices, unc_aggre_indices, unc_lbl_assign, unc_lblers, dilute_noise_type, nb_classes):
    """ Flip classes of a subset with symmetric probability.
    It expects a number (other than the original classes) between 0 and the number of classes - 1.
    """
    assert set(unc_aggre_indices).issubset(set(unc_indices))
    _all_train_noisy_labels = dict()
    _all_noise_or_not = dict()
    for lbler in unc_lblers:
        _all_train_noisy_labels[lbler] = []
        _all_noise_or_not[lbler] = []

    assert len(np.unique(unc_indices)) == len(unc_indices)
    conf_indices = np.setdiff1d(list(range(len(y))), unc_indices)

    if unc_lbl_assign.name == 'kmeans':     # load k-means model
        with open(f'../image-clustering/kmeans_{unc_lbl_assign.noise_type}.pkl', 'rb') as f:
            kmeans = pickle.load(f) 
        kmeans.centroids = np.asarray(kmeans.centroids)
        print(f'Loaded k-means model from ../image-clustering/kmeans_{unc_lbl_assign.noise_type}.pkl')
    for idx in range(len(y)):
        # if it is (assumed) a confident sample
        if idx in conf_indices:
            for lbler in unc_lblers:
                _all_noise_or_not[lbler].append(False) # all labels are (assummed no disagreement) correct!
                _all_train_noisy_labels[lbler].append(y[idx])
        # if it is an uncertain sample
        elif idx in unc_indices:
            if unc_lbl_assign.name =='symmetric': # random (symmetric) label assignment #
                _all_incor_cls = list(range(nb_classes))
                _all_incor_cls.remove(y[idx])  # all classes except for the correct one.
            
            elif unc_lbl_assign.name == 'kmeans':     ## assign diluted labels according to its distance to the k-means centroids
                kmeans_distances = np.linalg.norm(np.subtract(X[idx], kmeans.centroids), axis=1)
                _all_clusters_labels =  np.argsort(kmeans_distances)
                # convert cluster labels to (majority voted k-means) classes; note that majority vote can be misleading in the worst-case
                _all_incor_cls = [kmeans.clusters_labels[k] for k in _all_clusters_labels]  
                
            if idx in unc_aggre_indices: # the most uncertain samples, all of labels are incorrect but can be sampled with replacement
                if unc_lbl_assign.name =='symmetric':
                    _incor_cls = np.random.choice(_all_incor_cls, size=len(unc_lblers), replace=True,)
                elif unc_lbl_assign.name == 'kmeans':     
                    _incor_cls = set()  ## instead of list we make sets to ensure the disagreement
                    for _cls in _all_incor_cls:
                        if _cls != y[idx]:
                            _incor_cls.add(_cls)
                        if len(_incor_cls) == len(unc_lblers)-1:  # dilute with two additional unique labels
                            break
                    _incor_cls.add(y[idx])
                    if _incor_cls.__class__==set:
                        _incor_cls = list(_incor_cls)

                for i, lbler in enumerate(unc_lblers):
                    _all_noise_or_not[lbler].append(True) # two more diluted labels are different than the original one!
                    _all_train_noisy_labels[lbler].append(_incor_cls[i])
            else:
                if unc_lbl_assign.name =='symmetric':
                    _incor_cls = np.random.choice(_all_incor_cls, size=1, replace=True,).item()
                elif unc_lbl_assign.name == 'kmeans':     
                    for _cls in _all_incor_cls:
                        if _cls != y[idx]:
                            _incor_cls = _cls
                            break

                for i, lbler in enumerate(unc_lblers):
                    if i==0:
                        _all_noise_or_not[lbler].append(True) # one label is incorrect
                        _all_train_noisy_labels[lbler].append(_incor_cls)
                    else:
                        _all_noise_or_not[lbler].append(False) # all labels are correct!
                        _all_train_noisy_labels[lbler].append(y[idx])

        else:
            raise ValueError

    return _all_train_noisy_labels, _all_noise_or_not


class CIFAR10UNC(Dataset):
    """
    Construct uncertain training labels for CIFAR10, 
    specifically, natrual scene statistics are used
    to guide the label generation.
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

        self.train_data = np.concatenate(self.train_data)
        self.train_data_flatten = self.train_data.copy()
        self.train_data = self.train_data.reshape((50000, 3, 32, 32))
        self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        #if noise_type is not None:
        if cfg.trainset.unclbl.dilute.noise_type !='clean':
            self.noise_type = noise_type_map[cfg.trainset.unclbl.dilute.noise_type]
            # Load human noisy labels
            train_noisy_labels = self.load_label()
            self.train_noisy_labels = train_noisy_labels.tolist()
            print(f'noisy labels loaded from {self.noise_path}')



            for i in range(len(self.train_noisy_labels)):
                idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
            class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
            self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
            print(f'The noisy data ratio in each class is {self.noise_prior}')
            self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)
            self.actual_noise_rate = np.sum(self.noise_or_not)/50000
            print('over all noise rate is ', self.actual_noise_rate)

        self._all_train_noisy_labels = dict() # stores all the noisy labels from all annotators
        self._all_noise_or_not = dict()
        
        self.concat_train_noisy_ids = [] # contains (duplicate) ids for multiple instances
        self.concat_train_noisy_labels = []
        self.concat_data_uncertain_flag = []
        assert 0 < cfg.trainset.nlbl.rate < 1
        if cfg.trainset.nlbl.type == 'nss':
            if cfg.trainset.unclbl.nss == 'brisque': 
                nss_path = f'./misc/brisque_{cfg.trainset.name}_train.npy'
                if os.path.exists(nss_path):
                    with open(nss_path, 'rb') as f:
                        nss_scores = np.load(f)
                    print(f'loaded {cfg.trainset.unclbl.nss} scores from {nss_path}')
                else: 
                    print(f'compute {cfg.trainset.unclbl.nss} scores ...')
                    nss_scores = []
                    for _, img in enumerate(tqdm(self.train_data)):
                        nss_scores.append(cv2.quality.QualityBRISQUE_compute(img,  './misc/brisque_model_live.yml', './misc/brisque_range_live.yml')[0])
                # sort in descending order of all BRISQUE ( A smaller score indicates better perceptual quality.)
                uncIndices = np.argsort(nss_scores)[::-1][: int(cfg.trainset.nlbl.rate*len(self.train_data))]
                uncAggreIndices = np.argsort(nss_scores)[::-1][: int(cfg.trainset.nlbl.rate_aggre*len(self.train_data))]

            else:
                raise NotImplementedError
        elif cfg.trainset.nlbl.type == 'mss':
            if (cfg.trainset.unclbl.mss).startswith('cc'):  # cc_pixel or cc_embedding
                # load cluster labels
                with open(f'./misc/{cfg.trainset.unclbl.mss}_{cfg.trainset.unclbl.dilute.noise_type}_{cfg.trainset.name}_labels.npy', 'rb') as f:
                    cc_labels = np.load(f) 
                assert len(np.unique(cc_labels))==3
                # load sample 
                with open(f'./misc/{cfg.trainset.unclbl.mss}_{cfg.trainset.unclbl.dilute.noise_type}_{cfg.trainset.name}_densities.npy', 'rb') as f:
                    cc_densities = np.load(f)
 
                print(f'Loaded ./misc/{cfg.trainset.unclbl.mss}_{cfg.trainset.unclbl.dilute.noise_type}_{cfg.trainset.name}_labels.npy!')
                # indices of difficult cluster 
                ids_dfclt = np.where(cc_labels==2)[0]
                # get difficult densities
                densities_dfclt = cc_densities[ids_dfclt]
                # indices of moderate cluster
                ids_mod = np.where(cc_labels==1)[0]
                # get moderate densities
                densities_mod = cc_densities[ids_mod]
                # indices of sorted densities (from smaller to larger)
                ids_mss = np.asarray( ids_dfclt[np.argsort(densities_dfclt)].tolist() + ids_mod[np.argsort(densities_mod)].tolist() )

                uncIndices = ids_mss[: int(cfg.trainset.nlbl.rate*len(self.train_data))]
                uncAggreIndices = ids_mss[: int(cfg.trainset.nlbl.rate_aggre*len(self.train_data))]

            else:
                raise NotImplementedError

        elif cfg.trainset.nlbl.type == 'random':
            # the (broad) set of indices that corresponds to uncertain samples
            uncIndices = np.random.choice(len(self.train_data), size=int(cfg.trainset.nlbl.rate*len(self.train_data)), replace=False)
            # the (finer-grained) set of indices that corresponds to samples with two correct labels
            uncAggreIndices = np.random.choice(uncIndices, size=int(cfg.trainset.nlbl.rate_aggre*len(self.train_data)), replace=False)
            
        elif cfg.trainset.nlbl.type == 'annotator':
            # find the uncertain indices from loaded human noisy annotations. 
            _noise_labels = torch.load(self.noise_path)
            # the (finer-grained) set of indices that corresponds to samples with two correct labels
            uncAggreIndices = np.where(_noise_labels['clean_label'] != _noise_labels['aggre_label'])[0]
            uncIndices = np.where(_noise_labels['clean_label'] != _noise_labels['worse_label'])[0]
            print(f'\t loaded uncertain indices from {self.noise_path}!')

        elif cfg.trainset.nlbl.type == 'all':
            # all samples are diluted.
            uncAggreIndices = np.arange(len(self.train_data))
            uncIndices = np.arange(len(self.train_data))
        else:
            raise NotImplementedError

        assert set(uncAggreIndices).issubset(set(uncIndices))
        if cfg.trainset.unclbl.dilute.noise_type !='clean':
            y = self.train_noisy_labels
        else:
            y = self.train_labels  

        # generate uncertain label assignment
        self._all_train_noisy_labels, self._all_noise_or_not = get_uncertain_labels(self.train_data_flatten, y, uncIndices, uncAggreIndices, unc_lbl_assign=cfg.trainset.unclbl.assign, unc_lblers=self.noise_type_list, dilute_noise_type=cfg.trainset.unclbl.dilute.noise_type, nb_classes=10)


        for i, _noise_or_not in enumerate(zip(*[self._all_noise_or_not[noise_type] for noise_type in self.noise_type_list])): 
            self.concat_data_uncertain_flag.append(any(_noise_or_not))
            if any(_noise_or_not):  # at least one of the labels is incorrect
                _collect_noisy_labels = [self._all_train_noisy_labels[noise_type][i] for noise_type in self.noise_type_list]
                assert len(_collect_noisy_labels) == len(self.noise_type_list)
                _collect_noisy_labels = np.unique(_collect_noisy_labels)
                for _lbl in _collect_noisy_labels:
                    self.concat_train_noisy_ids.append(i)
                    self.concat_train_noisy_labels.append(_lbl)

        self._train_certain_ids = (np.where(np.asarray(self.concat_data_uncertain_flag) == False)[0]).tolist() # samples that are assumed no label uncertainty

        num_training_samples = 50000 

        if self.subset=='train':
            self.concat_train_correct_ids = self._train_certain_ids[ int(num_training_samples*cfg.trainset.val_ratio): ] # NOTE: the uncertain ids should not be here
            if cfg.trainset.unclbl.dilute.noise_type =='clean':
                self.concat_train_correct_labels = [self.train_labels[i] for i in self.concat_train_correct_ids]
            else:
                self.concat_train_correct_labels = [self.train_noisy_labels[i] for i in self.concat_train_correct_ids] 
            ## concatenate correct&noisy labels
            self._labels = self.concat_train_correct_labels + self.concat_train_noisy_labels
            ## concatenate correct&noisy ids...
            self._ids = self.concat_train_correct_ids + self.concat_train_noisy_ids

        elif self.subset == 'val':
            if len(self._train_certain_ids)==0:
                self._ids = np.arange(int(num_training_samples*cfg.trainset.val_ratio))
            else:
                self._ids = self._train_certain_ids[ :int(num_training_samples*cfg.trainset.val_ratio) ]
            if cfg.trainset.unclbl.dilute.noise_type =='clean':
                self._labels = [self.train_labels[i] for i in self._ids]
            else:
                self._labels = [self.train_noisy_labels[i] for i in self._ids] 
        else:
            raise Exception(f'invalid {self.subset} subset')
        assert len(self._labels)==len(self._ids)
        
    def load_label(self):
        #NOTE only load manual training label
        noise_label = torch.load(self.noise_path)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0  
                print(f'Loaded {self.noise_type} from {self.noise_path}.')
                print(f'The overall noise rate is {1-np.mean(clean_label.numpy() == noise_label[self.noise_type])}')
            return noise_label[self.noise_type].reshape(-1)  
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


def getUNCloader(cfg):
    """only supports the multi-label setting"""
    assert cfg.trainset.setting == 'MultiLabel'
    if cfg.trainset.name == 'cifar10':
        train_dataset = CIFAR10UNC(cfg, 'train', transform = train_cifar10_transform)
        val_dataset = CIFAR10UNC(cfg, 'val', transform = train_cifar10_transform)
    return train_dataset, val_dataset
