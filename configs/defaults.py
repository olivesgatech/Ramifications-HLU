import os
from yacs.config import CfgNode as cn
"""contains default nested structure in configurations and hyperparameters"""
_c = cn()

_c.project = ''
_c.ckpt_dir = ''
_c.device = 'cuda'
# _c.gpu = '0'
_c.seed = None
_c.print_freq = 50

# ------ dataloader configs ----
_c.dataloader = cn()
_c.dataloader.num_workers = 4 # of subprocesses to use for data loading

# ------ train dataset configs ----
_c.trainset = cn()
_c.trainset.setting = ''  # 'MultiLabel' | 'NoisyLabel' | 'RMNoisyLabel'
_c.trainset.nlbl = cn()   # label noise
_c.trainset.nlbl.type = '' # 'human' | 'synthetic' | 'nss'
_c.trainset.nlbl.tf = '' # 'symmetric' | 'asymmetric'
_c.trainset.nlbl.rate = 0.2  # corruption rate (worst case), should be less than 1
_c.trainset.nlbl.rate_aggre = 0. # corruption rate after aggregation (majority vote)
_c.trainset.unclbl = cn() # label uncertainty
_c.trainset.unclbl.assign = cn() # label assignment method
_c.trainset.unclbl.assign.name = '' # name of the label assignment method
_c.trainset.unclbl.assign.noise_type = ''  # default is the same as `unclbl.dilute.noise_type`
_c.trainset.unclbl.nss = None  # the estimator that uses natrual scene statistics
_c.trainset.unclbl.mss = None  # the method that uses machine scene statistics
_c.trainset.unclbl.dilute = cn() # label dilution configs
_c.trainset.unclbl.dilute.noise_type = ''
_c.trainset.root = ''
_c.trainset.name = ''
_c.trainset.num_classes = 0
_c.trainset.noise_type = '' # opts for cifar-N trainset: 'clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100'
_c.trainset.noise_path = None # path of CIFAR-X_human.pt
_c.trainset.noise_human = True
_c.trainset.val_ratio = 0.1

# ------ test dataset configs ----
_c.testset = cn()
_c.testset.name = ''
_c.testset.path = ''
_c.testset.twoaug = False

# ------ uncertainty quantification configs ------
_c.uncertain_quant = cn()
_c.uncertain_quant.method = '' # 'Ensemble', 'MCD', 'DUQ'
_c.uncertain_quant.backbone = cn()
_c.uncertain_quant.backbone.arch = 'resnet'
_c.uncertain_quant.backbone.depth = 18 # (resnet)-18, 34, etc.
_c.uncertain_quant.backbone.pdrop = 0.5  # probability of drpoout
_c.uncertain_quant.bayesian = cn()
_c.uncertain_quant.bayesian.train_ens = 1
_c.uncertain_quant.bayesian.val_ens = 1
_c.uncertain_quant.bayesian.beta_type = None  # 'Blundell', 'Standard', etc. Use float for const value
_c.uncertain_quant.n_samples = 0  # the number of samples (forward passes) per data instance (during inference). 1 for deterministic methods
_c.uncertain_quant.anlys = cn()
_c.uncertain_quant.anlys.eval = False
_c.uncertain_quant.anlys.eval_human = False # whether to evaluate the empirical human uncertainty
_c.uncertain_quant.anlys.metrics = None
_c.uncertain_quant.anlys.normalize_preds = False # whether to normalise the saved prediction output from models
_c.uncertain_quant.anlys.testlabel_uncertainty = False # whether to use human uncertainn label for test
_c.uncertain_quant.anlys.tta_flag = False # whether do test-time-augmentation (TTA) to study the relation between human versus model
_c.uncertain_quant.anlys.tta_type = ''
_c.uncertain_quant.anlys.tta_beta = 0.
_c.uncertain_quant.anlys.cal_discr = False
# ------ style transfer configs (texture versus shape experiments)----
_c.uncertain_quant.anlys.tta_cont_size = 0
_c.uncertain_quant.anlys.tta_sty_size = 0
_c.uncertain_quant.anlys.tta_crop = False
_c.uncertain_quant.anlys.tta_pres_color = False
_c.uncertain_quant.anlys.tta_stytf_vgg  = ''
_c.uncertain_quant.anlys.tta_stytf_decoder = ''
_c.uncertain_quant.anlys.tta_stytf_alpha = 1.0

# ------ training configs ----
_c.solver = cn()
_c.solver.num_epoch = 0
_c.solver.lr = 0.1
_c.solver.batch_size = 128
_c.solver.optimizer = ''
_c.solver.momentum = 0.9
_c.solver.weight_decay = 5e-4

# ------ test configs ----
_c.test = cn()
_c.test.inference = False
_c.test.batch_size = 128