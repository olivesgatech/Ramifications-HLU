import torch
from data.cifar import CIFAR10
from data.datasets import input_dataset, get_testloader, get_trainloader
from data.tsloader import get_tensorloader
# from data.styloader import get_stylizedloader
from data.ccnloader import getCCNloader
import torchvision.transforms as transforms
import pdb


# dict of noise type in CIFAR-N dataset
noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}

def make_dataset(cfg):
    cfg.trainset.noise_type = noise_type_map[cfg.trainset.noise_type]
    # path to the noisy labels (CIFAR-N)
    if cfg.trainset.noise_path is None:
        if cfg.trainset.name == 'cifar10':
            cfg.trainset.noise_path = './data/CIFAR-10_human.pt'
        elif cfg.trainset.name == 'cifar100':
            cfg.trainset.noise_path = './data/CIFAR-100_human.pt'
        else: 
            raise NameError(f'Undefined trainset {cfg.trainset.name}')
    test_dataset = None
    # specify the method to generate label noise!
    if cfg.trainset.setting == 'MultiLabel': # uncertainty setting where some training instances are associated with multiple ground-truths from different independent annotators.
        train_dataset, val_dataset, num_classes = get_trainloader(cfg)

    elif cfg.trainset.setting == 'NoisyLabel': # no uncertainty setting where every training instance is associated with a single ground-truth (either correct or wrong)
        if cfg.trainset.nlbl.type == 'human': # Load train/val dataset (instance-dependent label noise, CIFAR-N Only). code credit: https://github.com/UCSC-REAL/cifar-10-100n/blob/ijcai-lmnl-2022/ce_baseline.py#L107 #
            train_dataset, val_dataset, _, num_classes, num_training_samples = input_dataset(cfg.trainset.root, cfg.trainset.name, cfg.trainset.noise_type, cfg.trainset.noise_path, is_human = cfg.trainset.noise_human, val_ratio = cfg.trainset.val_ratio)
        # load class-conditional random label noise, with randomisation 
        elif cfg.trainset.nlbl.type == 'synthetic':
            train_dataset, val_dataset, num_classes = getCCNloader(cfg)

    elif cfg.trainset.setting == 'RMNoisyLabel': # no uncertainty setting where the training instances associated with multiple ground-truths are removed from the dataset
        pass
    else:
        raise Exception(f'{cfg.trainset.setting} is an invalid training setting!')
    
    if test_dataset is None and cfg.testset.name == 'cifar10':
        if cfg.testset.twoaug:  # two augmented transforms
            transform = transforms.Compose([
                                            transforms.RandomApply([
                                                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                            ], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])
        else:
            transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])

        test_dataset = CIFAR10(root=cfg.trainset.root,
                                download=False,  
                                train=False, 
                                transform = transform,
                                twoaug=cfg.testset.twoaug,
                                noise_type=cfg.trainset.noise_type,
                              )
    
    cfg.trainset.num_classes = num_classes
    # return train/val test.
    return train_dataset, val_dataset, test_dataset

def make_dataloader(cfg, logger=None):
    # create ALL dataloaders 
    train_dataset, val_dataset, test_dataset = make_dataset(cfg)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = cfg.solver.batch_size,
                                   num_workers=cfg.dataloader.num_workers,
                                   shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                   batch_size = cfg.solver.batch_size,
                                   num_workers=cfg.dataloader.num_workers,
                                   shuffle=False)

    if cfg.testset.name in ['cifar10c', 'cifar10r']:
        test_loader = get_testloader(cfg)

    else:
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = cfg.test.batch_size,
                                  num_workers=cfg.dataloader.num_workers,
                                  shuffle=False)

    if cfg.trainset.setting == 'MultiLabel':     
        if cfg.trainset.nlbl.type == cfg.trainset.unclbl.assign.name == '':
            cfg.trainset.noise_type = 'multi_uniqlbl'
        else:
            if cfg.trainset.nlbl.type == 'nss':
                noise_type = cfg.trainset.unclbl.nss 
            elif cfg.trainset.nlbl.type == 'mss':
                noise_type = cfg.trainset.unclbl.mss
            else:
                noise_type = cfg.trainset.nlbl.type
            cfg.trainset.noise_type = f'dilute_{cfg.trainset.unclbl.dilute.noise_type}_{noise_type}_{cfg.trainset.unclbl.assign.name}_{cfg.trainset.unclbl.assign.noise_type}'
    elif cfg.trainset.setting == 'NoisyLabel' and cfg.trainset.nlbl.type == 'synthetic':
        cfg.trainset.noise_type = cfg.trainset.nlbl.tf

    return train_loader, val_loader, test_loader
