"""The script to evaluate uncertainty methods"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from metrics import * #Metrics
from sklearn.preprocessing import normalize
import pickle
import pdb


## the following experiment is just a check on the empirical human uncertainty.
def eval_uncertainty_human(cfg):
    assert cfg.testset.name == 'cifar10', 'only have access to cifar10 human uncertain labels!!!'
    testset_name_map = {'cifar10c': 'CIFAR-10-C', 'cifar10':'CIFAR-10', 'cifar10r':'CIFAR-10-R'}
    
    metrics_eval = cfg.uncertain_quant.anlys.metrics.split('-')
    metrics = np.zeros((1, len(metrics_eval)))

    targets = None
    preds = np.load('../cifar-10h/data/cifar10h-probs.npy')
    # generate uncertainty metrics
    for j, metric_eval in enumerate(metrics_eval):
        if metric_eval == 'uniform':
            metric = get_uniform(preds)
        # elif metric_eval == 'align':
        #     metric = get_align(preds_aug1, preds_aug2)
            # print(f'alignment between two {methods[idx]} predictions: {metric}')
        elif metric_eval.startswith('distance'):
            _, dist_met, topk = metric_eval.split('_')
            metric = get_distance(preds, dist_met, topk=int(topk))
        else: 
            metric = globals()[f'get_{metric_eval}'](preds, targets)        
        metrics[0, j] = metric 
        
    # create a dataFrame to store evaluation metrics generated from the ground-truths and predictions of all methods
    df_metrics = pd.DataFrame(metrics, columns=metrics_eval, index=['human'])

    # save_path = os.path.join(cfg.ckpt_dir, 'anlys/metrics', cfg.trainset.name, cfg.trainset.noise_type, f'seed{cfg.seed}')
    # os.makedirs(save_path, exist_ok=True)
    # df_metrics.to_csv(os.path.join(save_path, f'metrics_{testset_name_map[cfg.testset.name]}.csv')) # save the dataFrame to csv file
    print(f'=======> Uncertainty Metrics on {testset_name_map[cfg.testset.name]}: \n', df_metrics)


def cal_discr_uncertainty(cfg):
    """
    calculate and rank the discrepancy (brier scores) between model and human response
    """
    testset_name_map = {'cifar10c': 'CIFAR-10-C', 'cifar10':'CIFAR-10', 'cifar10r':'CIFAR-10-R'}
    brier_list = []  # contains the brier scores between model and human response.
    agree_flag = []  # collects the top-1 prediction consensus between model and human
 
    save_path = os.path.join(cfg.ckpt_dir, f'{cfg.uncertain_quant.method}',  cfg.trainset.name, cfg.trainset.noise_type, f'seed{cfg.seed}') 
    preds = np.load(os.path.join(save_path, f'preds_{testset_name_map[cfg.testset.name]}.npy'))
    print(f'loaded the saved predictions from {save_path} !')
    if cfg.testset.name == 'cifar10':
        cifar10h_probs = np.load('../cifar-10h/data/cifar10h-probs.npy')
        N = len(cifar10h_probs)
    else:
        raise NotImplementedError

    ## calculate the entropy 
    entropy_human= []
    entropy_model = []

    # iterate all test samples, default original cifar10 testset.
    for i in range(N):
        agree_flag.append( preds[i].argmax() == cifar10h_probs[i].argmax() )
        brier_list.append( get_brier_h(np.expand_dims(preds[i],0), np.expand_dims(cifar10h_probs[i],0)) )

        entropy_model.append(get_entropy(np.expand_dims(preds[i], 0)))
        entropy_human.append(get_entropy(np.expand_dims(cifar10h_probs[i], 0)))
    
    os.makedirs(os.path.join(save_path, 'uncertain_discr'), exist_ok=True)
    if not os.path.isfile(os.path.join(save_path, 'uncertain_discr', 'brier.pkl')):
        with open(os.path.join(save_path, 'uncertain_discr', 'brier.pkl'), 'wb') as f:
            pickle.dump(brier_list, f)
    if not os.path.isfile(os.path.join(save_path, 'uncertain_discr', 'agree.pkl')):
        with open(os.path.join(save_path, 'uncertain_discr', 'agree.pkl'), 'wb') as f:
            pickle.dump(agree_flag, f)
    if not os.path.isfile(os.path.join(save_path, 'uncertain_discr', 'entropy_model.pkl')):
        with open(os.path.join(save_path, 'uncertain_discr', 'entropy_model.pkl'), 'wb') as f:
            pickle.dump(entropy_model, f)
    if not os.path.isfile(os.path.join(save_path, 'uncertain_discr', 'entropy_human.pkl')):
        with open(os.path.join(save_path, 'uncertain_discr', 'entropy_human.pkl'), 'wb') as f:
            pickle.dump(entropy_human, f)
    return  

def eval_uncertainty(cfg, testloader):
    testset_name_map = {'cifar10c': 'CIFAR-10-C', 'cifar10':'CIFAR-10', 'cifar10r':'CIFAR-10-R'}
    # the evaluation metrics
    metrics_eval = cfg.uncertain_quant.anlys.metrics.split('-')
    methods = cfg.uncertain_quant.method.split('-')  # list of methods to evaluate 
    preds_path_list = [os.path.join(cfg.ckpt_dir, f'{method}',  cfg.trainset.name, cfg.trainset.noise_type, f'seed{cfg.seed}') for method in methods] 


    metrics = np.zeros((len(preds_path_list), len(metrics_eval)))

    if cfg.testset.name == 'cifar10c':
        N = 5*19
        targets = np.tile(testloader.chal_labels, N) # replicate the test labels (# challenges)*(# levels) times
    elif cfg.testset.name == 'cifar10r':
        N = 15
        targets = np.tile(testloader.labels, N)
    elif cfg.testset.name == 'cifar10':
        N = 1
        targets = np.tile(testloader.dataset.test_labels, N)
    else:
        raise NotImplementedError

    # load test uncertain labels
    if cfg.uncertain_quant.anlys.testlabel_uncertainty: # evaluate with human uncertainn test labels
        tgt = np.load('../cifar-10h/data/cifar10h-probs.npy')
        targets_ll_h = np.tile(tgt.argmax(1), N)  # targets for log-likelihood
        targets_brier_h = np.tile(tgt, (N,1)) # targets for brier score
    
    for idx, save_path in enumerate(tqdm(preds_path_list)):
        # load the saved prediction (per test sample)
        preds = np.load(os.path.join(save_path, f'preds_{testset_name_map[cfg.testset.name]}.npy'))
        print(f'loaded predictions from {save_path}, the prediction array has shape: {preds.shape}')
        if cfg.testset.twoaug: # load two agument predictions 
            preds_aug1 = np.load(os.path.join(save_path, f'preds_aug1_{testset_name_map[cfg.testset.name]}.npy'))
            preds_aug2 = np.load(os.path.join(save_path, f'preds_aug2_{testset_name_map[cfg.testset.name]}.npy')) 
            print(f'loaded (augmented 1) predictions from {save_path}, the prediction array has shape: {preds_aug1.shape}')
            print(f'loaded (augmented 2) predictions from {save_path}, the prediction array has shape: {preds_aug2.shape}')

        # generate uncertainty metrics
        for j, metric_eval in enumerate(metrics_eval):
            if not ((metric_eval == 'uniform' or metric_eval.startswith('distance')) and cfg.testset.name in ['cifar10c', 'cifar10r']):
                if metric_eval == 'uniform':
                    metric = get_uniform(preds)
                elif metric_eval == 'align':
                    metric = get_align(preds_aug1, preds_aug2)
                    # print(f'alignment between two {methods[idx]} predictions: {metric}')
                elif metric_eval.startswith('distance'):
                    _, dist_met, topk = metric_eval.split('_')
                    metric = get_distance(preds, dist_met, topk=int(topk))
                
                elif metric_eval == 'll_h':
                    if 'DUQ' in save_path and cfg.uncertain_quant.anlys.normalize_preds: 
                        metric = get_ll_h(normalize(preds, axis=1, norm='l1'), targets_ll_h)
                    else:
                        metric = get_ll_h(preds, targets_ll_h)
                elif metric_eval == 'brier_h':
                    if 'DUQ' in save_path and cfg.uncertain_quant.anlys.normalize_preds: 
                        metric = get_brier_h(normalize(preds, axis=1, norm='l1'), targets_brier_h)
                    else:
                        metric = get_brier_h(preds, targets_brier_h)

                else: 
                    if 'DUQ' in save_path and cfg.uncertain_quant.anlys.normalize_preds: 
                        metric = globals()[f'get_{metric_eval}'](normalize(preds, axis=1, norm='l1'), targets)
                    else:
                        metric = globals()[f'get_{metric_eval}'](preds, targets)
            else: # set-wise uniformity scores
                if cfg.testset.name == 'cifar10c':
                    len_labels = len(testloader.chal_labels)
                elif cfg.testset.name == 'cifar10r':
                    len_labels = len(testloader.labels)
                else:
                    raise Exception(f'{cfg.testset.name} does not contain multiple sets!')
                metric = 0 # will be averaged by #challenges * #challenging levels.
                l_i = 0
                for cnt, r_i in enumerate(range(len_labels, preds.__len__()+len_labels, len_labels), 1):
                    if metric_eval == 'uniform':
                        metric += get_uniform(preds[l_i:r_i])
                    elif metric_eval.startswith('distance'):
                        _, dist_met, topk = metric_eval.split('_')
                        metric += get_distance(preds[l_i:r_i], dist_met, topk=int(topk))
                    l_i = r_i
                metric /= cnt  
 
            metrics[idx, j] = metric #ll,brier,acc
    # create a dataFrame to store evaluation metrics generated from the ground-truths and predictions of all methods
    df_metrics = pd.DataFrame(metrics, columns=metrics_eval, index=methods)  # "Log-likelihood", "Brier score", "Accuracy

    save_path = os.path.join(cfg.ckpt_dir, 'anlys/metrics', cfg.trainset.name, cfg.trainset.noise_type, f'seed{cfg.seed}')
    os.makedirs(save_path, exist_ok=True)
    df_metrics.to_csv(os.path.join(save_path, f'metrics_{testset_name_map[cfg.testset.name]}.csv')) # save the dataFrame to csv file
    print(f'=======> Uncertainty Metrics on {testset_name_map[cfg.testset.name]}: \n', df_metrics)
