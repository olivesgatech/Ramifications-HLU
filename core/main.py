import os
import sys
sys.path.append(os.path.realpath('.'))
from data import make_dataloader
from models import make_model
from engine import train, validate, train_bayesian, validate_bayesian, inference, inference_tta, eval_uncertainty, eval_uncertainty_human, cal_discr_uncertainty
from utils import set_global_seeds, build_optimizer, learning_rate, adjust_learning_rate
import metrics
import torch
import argparse
from configs import cfg
from contextlib2 import redirect_stdout
import time
import pdb


def main():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of Uncertainty Quantification")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Seed
    set_global_seeds(cfg.seed)
    ## ============ construct dataloaders ============ ##
    train_dataloader, val_dataloader, test_dataloader = make_dataloader(cfg)
    print('Dataloader(s) built!')
    ## ======= evaluate uncertainty quantification approaches ======= ##
    if cfg.uncertain_quant.anlys.eval:
        eval_uncertainty(cfg, test_dataloader)
        exit()
    ## ======= evaluate human uncertainty ======= ##
    if cfg.uncertain_quant.anlys.eval_human:
        eval_uncertainty_human(cfg)
        exit()
    ## ======= calculate the discrepancy human uncertainty ======= ##
    elif cfg.uncertain_quant.anlys.cal_discr:
        cal_discr_uncertainty(cfg)
        exit()
    ## ========= build model, optimizer and scheduler ========= ##
    # print('building model...')
    model = make_model(cfg)
    model.to(cfg.device)
    print('Model built!')
    optimizer = build_optimizer(cfg, model)
    print('optimizer built!')
    ## =============== train/val and test =============== ##
    if cfg.test.inference:  # test mode
        # data from cifar10-rotations/ cifar-10-C
        save_path= os.path.join(cfg.ckpt_dir, cfg.uncertain_quant.method,  cfg.trainset.name, cfg.trainset.noise_type, f'seed{cfg.seed}', 'best.pth.tar')
        checkpoint = torch.load(save_path)
        # load state dict.
        model.load_state_dict(checkpoint['state_dict'])
        print(f'loaded model checkpoint from {save_path}!')
        ## =============== test-time-augmentation =============== ##
        if cfg.uncertain_quant.anlys.tta_flag:
            inference_tta(cfg, checkpoint['epoch'], model, test_dataloader)
        else:## inference engine. need to account for different datasets
            test_acc = inference(cfg, checkpoint['epoch'], model, test_dataloader)
            print('test acc is ', test_acc)
    else:  ## train/val mode
        _ckpt_root = os.path.join(cfg.ckpt_dir, cfg.uncertain_quant.method,)  # method root
        _ckpt_dir = os.path.join(_ckpt_root, cfg.trainset.name, cfg.trainset.noise_type, f'seed{cfg.seed}')  
        os.makedirs(_ckpt_dir, exist_ok=True)        
        with open(os.path.join(_ckpt_dir, 'configs.yml'), 'w') as f:
            with redirect_stdout(f): print(cfg.dump()) # (where cfg is a CfgNode)
        # _best_val_loss = float('inf')
        best_acc = 0.0
        time_start = time.time()
        # training
        for epoch in range(cfg.solver.num_epoch):
            # adjust learning rate
            adjust_learning_rate(optimizer, learning_rate(cfg.solver.lr, epoch))
            print("Training Epoch:{},\t lr:{:6},\t".format(epoch, optimizer.param_groups[0]['lr']))
            
            if cfg.uncertain_quant.method in ['BbBLRT', 'BbB']: # train models w/ ELBO
                criterion = metrics.ELBO(len(train_dataloader.dataset)).to(cfg.device)
                train_loss, train_acc, train_kl = train_bayesian(cfg, epoch, train_dataloader, model, optimizer, criterion)
            else: # train models w/o ELBO
                train_acc = train(cfg, epoch, train_dataloader, model, optimizer)
            print('train acc is ', train_acc)

            # validate models
            print('previous best (val) ', best_acc)
            if cfg.trainset.val_ratio > 0.0:
                # save results
                if cfg.uncertain_quant.method in ['BbBLRT', 'BbB']:
                    criterion = metrics.ELBO(len(train_dataloader.dataset)).to(cfg.device)
                    val_loss, val_acc = validate_bayesian(cfg, epoch, val_dataloader, model, criterion, save = True, best_acc = best_acc)
                else:
                    val_acc = validate(cfg, epoch, loader=val_dataloader, model=model, save = True, best_acc = best_acc)
                if val_acc > best_acc:
                    best_acc = val_acc
                print('validation acc is ', val_acc)

            time_curr = time.time()
            time_elapsed = time_curr - time_start
            print(f'[Epoch {epoch}] Time elapsed {time_elapsed//3600:.0f}h {(time_elapsed%3600)//60:.0f}m {(time_elapsed%3600)%60:.0f}s', flush=True)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
