import numpy as np
import copy
import torchvision.transforms as transforms
try:
    from .cifar import CIFAR10, CIFAR100
    from .cifarc import CIFAR10C
    from .cifarr import CIFAR10R
    from .cifaru import CIFAR10U
    from .uncloader import getUNCloader
except:
    from cifar import CIFAR10, CIFAR100
    from cifarc import CIFAR10C
    from cifarr import CIFAR10R
    from cifaru import CIFAR10U
    from uncloader import getUNCloader

train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

def input_dataset(root, dataset, noise_type, noise_path, is_human, val_ratio = 0.1):
    num_training_samples = 50000
    idx_full = np.arange(num_training_samples)
    np.random.shuffle(idx_full)
    if dataset == 'cifar10':
        num_classes = 10

        train_dataset_full = CIFAR10(root,
                                download=True,  
                                train=True, 
                                transform = train_cifar10_transform,
                                noise_type = noise_type,
                                noise_path = noise_path, 
                                is_human=is_human,
                           )

        test_dataset = CIFAR10(root,
                                download=False,  
                                train=False, 
                                transform = test_cifar10_transform,
                                noise_type=noise_type,
                          )
        
        
    elif dataset == 'cifar100':
        num_classes = 100
        train_dataset_full = CIFAR100(root,
                                download=True,  
                                train=True, 
                                transform=train_cifar100_transform,
                                noise_type=noise_type,
                                noise_path = noise_path, is_human=is_human
                            )



        test_dataset = CIFAR100(root,
                                download=False,  
                                train=False, 
                                transform=test_cifar100_transform,
                                noise_type=noise_type
                            )
        
    train_dataset = copy.copy(train_dataset_full)
    train_dataset.train_data = train_dataset.train_data[idx_full[int(num_training_samples*val_ratio):]]
    train_dataset.train_noisy_labels = (np.array(train_dataset.train_noisy_labels)[idx_full[int(num_training_samples*val_ratio):]]).tolist()
    train_dataset.train_labels = (np.array(train_dataset.train_labels)[idx_full[int(num_training_samples*val_ratio):]]).tolist()
    print(f'Train with {len(train_dataset.train_noisy_labels)} noisy instances.')
    
    val_dataset = copy.copy(train_dataset_full)
    val_dataset.transform = test_cifar10_transform
    val_dataset.train_data = val_dataset.train_data[idx_full[:int(num_training_samples*val_ratio)]]
    val_dataset.train_noisy_labels = (np.array(val_dataset.train_noisy_labels)[idx_full[:int(num_training_samples*val_ratio)]]).tolist()
    val_dataset.train_labels = (np.array(val_dataset.train_labels)[idx_full[:int(num_training_samples*val_ratio)]]).tolist()
    print(f'Validate with {len(val_dataset.train_noisy_labels)} noisy instances.')
    return train_dataset, val_dataset, test_dataset, num_classes, num_training_samples


def get_testloader(cfg):
    if cfg.testset.name == 'cifar10c':
        return CIFAR10C(cfg)
    elif cfg.testset.name == 'cifar10r':
        return CIFAR10R(cfg)

def get_trainloader(cfg):
    if cfg.trainset.setting == 'MultiLabel':
        if cfg.trainset.name == 'cifar10':
            num_classes = 10
            if cfg.trainset.nlbl.type == 'human': # default in the .yml
                train_dataset = CIFAR10U(cfg, 'train', transform = train_cifar10_transform)
                val_dataset = CIFAR10U(cfg, 'val', transform = train_cifar10_transform)
            else:  # nss or random or mss strategies
                train_dataset, val_dataset = getUNCloader(cfg)
            return train_dataset, val_dataset, num_classes
