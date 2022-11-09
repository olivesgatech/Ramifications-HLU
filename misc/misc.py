import os
import sys
sys.path.append(os.path.realpath('.'))
import numpy as np
import pandas as pd
import argparse
from configs import cfg
from data import noise_type_map
from pyarrow import feather as pf
from tqdm import tqdm
# import pdb

def dump_stats(cfg):
    """Given the same training setting, either uncertainty training or certainty training, generate excel sheet."""
    if cfg.trainset.setting == 'MultiLabel':     
        if cfg.trainset.nlbl.type == cfg.trainset.unclbl.assign == '':
            cfg.trainset.noise_type = 'multi_uniqlbl'
        else:
            noise_type = cfg.trainset.unclbl.nss if cfg.trainset.nlbl.type == 'nss' else cfg.trainset.nlbl.type
            cfg.trainset.noise_type = f'unc_{noise_type}_{cfg.trainset.unclbl.assign}'
    elif cfg.trainset.setting == 'NoisyLabel':
        if cfg.trainset.nlbl.type == 'synthetic':
            cfg.trainset.noise_type = cfg.trainset.nlbl.tf
        else:
            cfg.trainset.noise_type = noise_type_map[cfg.trainset.noise_type]
    else:
        raise Exception(f'{cfg.trainset.setting} is an invalid training setting!')

    testset_name_list = ['CIFAR-10-R', 'CIFAR-10-C','CIFAR-10']
    
    df_concat_map = {'avg':[], 'std':[]}
    # df_avg_concat_map = {k:[] for k in testset_name_list}
    # df_std_concat_map = {k:[] for k in testset_name_list}

    for testset_name in testset_name_list:
        df_list = []
        for seed in range(5):
            save_path = os.path.join(cfg.ckpt_dir, f'anlys/metrics/{cfg.trainset.name}/{cfg.trainset.noise_type}/seed{seed}/metrics_{testset_name}.csv')
            print('loading results from: ', save_path)
            df = pd.read_csv(save_path)

            df_list.append(df)
        
        # 1. calculate the mean and standard deviation
        p = pd.concat(df_list)
        df_avg = p.groupby('Unnamed: 0').mean()
        df_std = p.groupby('Unnamed: 0').std()

        # print('mean over 5 seeds: \n', df_avg)
        # print('std over 5 seeds: \n', df_std)
        
        df_concat_map['avg'].append(df_avg)
        df_concat_map['std'].append(df_std)
        
    # 2. dump these stats 
    df_concat_avg = pd.concat( df_concat_map['avg'], axis=1)
    df_concat_std = pd.concat( df_concat_map['std'], axis=1)

    df_concat_avg.to_excel(os.path.join(cfg.ckpt_dir, f'anlys/metrics/{cfg.trainset.name}/{cfg.trainset.noise_type}', f'metrics_{cfg.trainset.noise_type}.xlsx'))  
    # 
    print(df_concat_avg)
    print(df_concat_std)


def dump_ttaInput(cfg):
    """Dump the test-time-augmented samples"""
    if cfg.trainset.setting == 'MultiLabel':     
        cfg.trainset.noise_type = 'multi_uniqlbl'
    elif cfg.trainset.setting == 'NoisyLabel':
        cfg.trainset.noise_type = noise_type_map[cfg.trainset.noise_type]
    else:
        raise Exception(f'{cfg.trainset.setting} is an invalid training setting!')
    save_path = os.path.join(cfg.ckpt_dir, 'anlys', f'{cfg.uncertain_quant.anlys.tta_type}/{cfg.testset.name}/{cfg.trainset.noise_type}/seed{cfg.seed}')
    tta_df = pf.read_feather(os.path.join(save_path, 'tta_df.ftr'))
    print(f'loaded dataframe from {save_path} !')
    # load human prob labels
    if cfg.testset.name == 'cifar10':
        h_probs = np.load('../cifar-10h/data/cifar10h-probs.npy')
 
    df_concat_list = []
    # group by original class.
    for _, dfgb in tqdm(tta_df.groupby(['origCls', 'origIdx'])):
        # pdb.set_trace()   # check if the oidx is single.
        for idx in np.unique(dfgb['origIdx'].to_numpy()):
            h_prob = h_probs[idx] # the human response to this sample
            for hcls in reversed(np.argsort(h_prob)):   # sort the human probabilities.
                if h_prob[hcls] < 0.2: continue
                # sort the 'respDiff' in ascending order
                dfgb_hcls = dfgb.query('ttaCls==@hcls & respDiff<0').sort_values(by=['respDiff'])
                df_concat_list.append(dfgb_hcls) 

    df_uncertain = pd.concat(df_concat_list, ignore_index=True)
    # columns = df_uncertain.columns.values.tolist()
    # columns.remove('ttaInput')
    # df_uncertain.drop_duplicates(subset=columns, inplace=True)
    # save the dataframe
    pf.write_feather(df_uncertain, os.path.join(save_path, 'tta_uncertain_df.ftr'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--function", help="function to call", type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
                         
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args() 
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    locals()[args.function](cfg)

    # dump_ttaInput(cfg)
