import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import pickle
from data import get_tensorloader#, get_stylizedloader
from itertools import repeat
from pyarrow import feather as pf
from torchvision.utils import save_image
import pdb


def slide_bbox(size, cut_rat=1/3, stride=1/6):
    assert type(size)==torch.Size
    W = size[2]
    H = size[3]

    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    bbox_list = []
    # slide (can also be uniformly sampled)
    for i in range(0, H-cut_h, int(H*stride)): ## slide the height (the first dimension)
        for j in range(0, W-cut_w, int(W*stride)):
            bbox_list.append([j, i, j+cut_w, i+cut_h])
    
    return bbox_list

def inference_tta(cfg, epoch, model, dataloader):
    """inference with test-time-augmentation"""
    model.eval()
    save_path = os.path.join(cfg.ckpt_dir, f'{cfg.uncertain_quant.method}',  cfg.trainset.name, cfg.trainset.noise_type, f'seed{cfg.seed}') 

    if cfg.testset.name == 'cifar10':
        ## load the discrepancy file(s) 
        with open(os.path.join(save_path, 'uncertain_discr', 'brier.pkl'), 'rb') as f:
            brier_list = pickle.load(f)
        with open(os.path.join(save_path, 'uncertain_discr', 'agree.pkl'), 'rb') as f:
            agree_flag = pickle.load(f)
        ## load entropy files
        with open(os.path.join(save_path, 'uncertain_discr', 'entropy_model.pkl'), 'rb') as f:
            entropy_model = pickle.load(f)
        with open(os.path.join(save_path, 'uncertain_discr', 'entropy_human.pkl'), 'rb') as f:
            entropy_human = pickle.load(f)
        
        #test_cifar10_transform 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        #load and group the classes
        test_labels = dataloader.dataset.test_labels
        
        idx_brier_sorted = np.asanyarray(brier_list).argsort()  ## the larger indices, the larger discrepancy between model and human response

        gb_cls = dict()
        for idx in idx_brier_sorted:
            cls_cur = test_labels[idx]
            if cls_cur not in gb_cls:
                gb_cls[cls_cur] = []
            
            if agree_flag[idx]: 
                gb_cls[cls_cur].append(idx)

        # for k in (gb_cls).keys(): print(gb_cls[k].__len__()) 
        hu_mc_dict = {k:[] for k in gb_cls.keys()}
        hc_mu_dict = {k:[] for k in gb_cls.keys()}
        hc_mc_dict = {k:[] for k in gb_cls.keys()}
        # the following separation has large margin
        for cls, idx_cls_list in gb_cls.items(): # iterate each class
            for idx in reversed(idx_cls_list):  # iterate inversely (from larger discrepancy) 
                if cfg.uncertain_quant.anlys.tta_type == 'cutmix': # calculate the discrepancy between model and human entropy;
                    if entropy_human[idx] - entropy_model[idx] > 1: # human uncertain but model confident
                        hu_mc_dict[cls].append(idx)
                    elif entropy_model[idx] - entropy_human[idx] > 1: # human confident but model uncertain,
                        hc_mu_dict[cls].append(idx)
                # in the shape versus texture experiments, we only need modle confident samples! we do not force large margin
                elif cfg.uncertain_quant.anlys.tta_type == 'texshape': 
                    # only consider model is confident cases
                    if entropy_human[idx] > entropy_model[idx] and entropy_model[idx] < .5: # human is more uncertain than model.. probably the shape cues are less salient
                        hu_mc_dict[cls].append(idx)
                    elif 0.5 > entropy_model[idx] > entropy_human[idx] : # human is more confident than model.. probably the texture cues (from model perspective) are less salient
                        hc_mc_dict[cls].append(idx)  # this is unclear in terms of usage
    else:
        raise NotImplementedError

    if cfg.uncertain_quant.anlys.tta_type == 'cutmix':
        # generate mixed sample
        response_diff_list = []
        orig_cls_list = []
        tta_cls_list = []
        tta_input_list = []
        orig_index_list = []
        tta_index_list = []
        bbox_coord_list = []
        slide_list = []

        for src_cls in gb_cls.keys():
            # create tensors
            src_tensor = torch.from_numpy( dataloader.dataset.test_data[hu_mc_dict[src_cls]] )
            tgt_tensor = []
            tgt_cls_list = []
            tta_index = []
            for tgt_cls in hc_mu_dict.keys():
                if tgt_cls == src_cls: continue
                tgt_tensor.append( torch.from_numpy( dataloader.dataset.test_data[hc_mu_dict[tgt_cls]] ) )
                tgt_cls_list.append( torch.from_numpy( np.asarray(dataloader.dataset.test_labels)[hc_mu_dict[tgt_cls]] ) )
                tta_index.extend( hc_mu_dict[tgt_cls] )
            tgt_tensor = torch.vstack(tgt_tensor)
            tgt_cls_list = torch.cat(tgt_cls_list)

            src_tensor = torch.permute(src_tensor, (0, 3, 1, 2))
            tgt_tensor = torch.permute(tgt_tensor, (0, 3, 1, 2))
            # apply augmentation (we can do three different sets)
            for ratio, stride in zip([1/3,1/2], [1/6,1/4]): # iterate multi-scale cutmix
                print(f'cutmix ratio: {ratio}, stride: {stride}')
                bbox_list = slide_bbox(tgt_tensor.size())
                for (bbx1, bby1, bbx2, bby2) in bbox_list: # sliding the bbox patches
                    src_dup_tensor_list = []
                    for i in range(len(src_tensor)):  # iterate each source sample
                        src_dup_tensor = src_tensor[i].unsqueeze(0).repeat(len(tgt_tensor), 1, 1, 1)  # duplicate current input sample
                        ## cut and mix the patches from all other classes
                        src_dup_tensor[:, :, bbx1:bbx2, bby1:bby2] = tgt_tensor[:, :, bbx1:bbx2, bby1:bby2]
                        ## collect i-the source duplicated tensor!
                        src_dup_tensor_list.append(src_dup_tensor)
                    
                    src_dup_tensor_list = torch.cat(src_dup_tensor_list)

                    # create a test-time-augmented dataloader
                    tsloader_tta = get_tensorloader(cfg, src_dup_tensor_list, [src_cls]*len(src_dup_tensor_list), transform=transform)
                    #  inference
                    preds_tta = inference_batch(cfg, epoch, model, tsloader_tta)
                    tgt_dup_cls_list = torch.tile(tgt_cls_list, (len(src_tensor),))
                    # response w.r.t. the target classes
                    response_tgt_cls = preds_tta[range(len(tgt_dup_cls_list)), tgt_dup_cls_list]
                    
                    # create original test dataloader and inference.
                    tsloader_src = get_tensorloader(cfg, src_tensor, [src_cls]*len(src_tensor), transform=transform)
                    preds_src = inference_batch(cfg, epoch, model, tsloader_src) 
                    # response w.r.t. the original classes
                    response_src_cls = np.concatenate([resp.repeat(len(tgt_tensor),) for resp in preds_src[:, src_cls]])
                    # collect margin stats.
                    response_diff = response_src_cls - response_tgt_cls
                    ## append several lists
                    response_diff_list.extend(response_diff)
                    orig_cls_list.extend([src_cls]*len(response_diff))
                    tta_cls_list.extend(tgt_dup_cls_list.tolist())
                    tta_input_list.extend(src_dup_tensor_list.cpu().detach().view(src_dup_tensor_list.shape[0], src_dup_tensor_list.shape[1],-1).numpy().tolist())
                    orig_index_list.extend( [x for item in hu_mc_dict[src_cls] for x in repeat(item, len(tgt_tensor))] )
                    tta_index_list.extend(np.tile( tta_index, (len(src_tensor),) ))
                    bbox_coord_list.extend([[bbx1, bby1, bbx2, bby2]]*len(response_diff))
                    slide_list.extend([[ratio, stride]]*len(response_diff))
                    print(f'source samples in class {src_cls} done with patch: {(bbx1, bby1, bbx2, bby2)}')
                    # Generate a Dataframe
                    tta_df = pd.DataFrame(list(zip(orig_index_list, tta_index_list, orig_cls_list, tta_cls_list, tta_input_list, bbox_coord_list, slide_list, response_diff_list)),
                                          columns = ['origIdx', 'ttaIdx', 'origCls', 'ttaCls', 'ttaInput', 'bboxCoord', 'slideParam', 'respDiff']) 

        save_path = os.path.join(cfg.ckpt_dir, 'anlys', f'{cfg.uncertain_quant.anlys.tta_type}/{cfg.testset.name}/{cfg.trainset.noise_type}/seed{cfg.seed}')
        os.makedirs(save_path, exist_ok = True)
        pf.write_feather(tta_df, os.path.join(save_path, 'tta_df.ftr'))

    elif cfg.uncertain_quant.anlys.tta_type == 'texshape': # test texture versus shape hypothesis
        ## TODO: to load the texture representations; keep track of the targeted classes!
        print('testing texture versus shape hypothesis!')
        _tensor_list = []
        _cls_list = []
        _idx_list = []
        for _cls in gb_cls.keys(): # iterate each class
            # create a test-time-augmented dataloader
            _tensor = torch.from_numpy(dataloader.dataset.test_data[hu_mc_dict[_cls]])  # torch.Size([N, H, W, C])
            _tensor_list.append(_tensor)
            _cls_list.extend([_cls]*len(_tensor))
            ## also append the sample indices
            _idx_list.extend(hu_mc_dict[_cls])

        _tensor_list = torch.cat(_tensor_list) 
        
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # transorm only consists of normalisation
        tsloader_tta = get_stylizedloader(cfg, _tensor_list.numpy(), _cls_list, _idx_list, transform=normalize)
        pdb.set_trace()
        ###Ensure that the focus is what? Given a texture image, test the output (from model perspective)!# 
        # img = Image.open('./data/texture/truck.png').convert('RGB').resize((32, 32))
        # out = model(transform(img).unsqueeze(0).to(cfg.device)) # pil.image
        # print(out)
        # single image test
        # from torchvision.datasets import CIFAR10
        # trainset = CIFAR10(root='/media/cmhung/MySSD/dataset/', train=True, download=False, )
        # out_orig=model(transform(trainset[7][0]).unsqueeze(0).to(cfg.device)) # pil.image
        # x = transform(transforms.Resize(32)(Image.open('../pytorch-AdaIN/output/cifar10/7_stylized_elephant1.jpg')))
        # out = model(x.unsqueeze(0).to(cfg.device))
        # # uncertain texture as the style iamge, transfer to model confident iamge!) 
        # visualize 10 differnt stylized samples< given the same texture input
        # for i in range(10):
        #     im_stytf, im_orig = tsloader_tta.dataset.__getitem__(i, mode='ldim')
        #     save_image(im_stytf, f'misc/figs/stytf/im_stytf_{i}.png')
        #     save_image(im_orig, f'misc/figs/stytf/im_orig_{i}.png')
    else:
        raise NotImplementedError

        
def inference_batch(cfg, epoch, model, dataloader):
    preds_list= []
    test_error = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            if cfg.testset.twoaug:  
                x1, x2, _y = batch
                x, y = torch.cat([x1, x2]), torch.cat([_y, _y])
            else:
                x, y = batch
            # cost: NLL, probs: softmax
            cost, err, probs = model.sample_eval(x, y, Nsamples=cfg.uncertain_quant.n_samples, logits=False)

            if cfg.testset.twoaug: 
                probs1, probs2 = probs.chunk(2)
                preds_aug1_list.append(probs1)
                preds_aug2_list.append(probs2)
            else:
                preds_list.append(probs)
            test_error += err.cpu().numpy()

    avg = 1 - (test_error/len(dataloader.dataset))

    print(f'TEST acc of {cfg.testset.name}: {avg}')

    return np.vstack(preds_list)

def inference(cfg, epoch, model, dataloader):
    preds_list= []
    model.eval()
    avg_list = []
    chal_acc_list = []
    if cfg.testset.twoaug: 
        preds_aug1_list, preds_aug2_list = [], [] # define two sets of output lists

    if cfg.testset.name == 'cifar10c':
        chals = dataloader.chals
        for challenge in range(len(chals)):
            avg = 0
            for j in range(5):

                chal_loader = dataloader.loaders[chals[challenge]][j]
                chal_error = 0

                with torch.no_grad():
                    for idx, batch in enumerate(tqdm(chal_loader)):
                        if cfg.testset.twoaug:
                            x1, x2, _y = batch
                            x, y = torch.cat([x1, x2]), torch.cat([_y, _y])
                        else:
                            x, y = batch

                        # cost: NLL, probs: softmax
                        cost, err, probs = model.sample_eval(x, y, Nsamples=cfg.uncertain_quant.n_samples, logits=False)
                        if cfg.testset.twoaug: 
                            probs1, probs2 = probs.chunk(2)
                            preds_aug1_list.append(probs1)
                            preds_aug2_list.append(probs2)
                        else:
                            preds_list.append(probs)

                        chal_error += err.cpu().numpy()
                        # print(err)
                chal_acc = 1 - (chal_error/len(chal_loader.dataset))
                avg += chal_acc
                print(f'TEST acc of level {j} {chals[challenge]}: {chal_acc}')
                chal_acc_list.append(chal_acc)

            avg /= 5
            avg_list.append(avg)
            print("Average all levels:",avg," ", chals[challenge])

        print("OVERALL Mean Acc: ", np.mean(avg_list))

        ## save files
        save_path= os.path.join(cfg.ckpt_dir, cfg.uncertain_quant.method,  cfg.trainset.name, cfg.trainset.noise_type, f'seed{cfg.seed}')
        print(f'saving to {save_path}')
        if cfg.testset.twoaug: 
            preds_aug1_list = np.vstack(preds_aug1_list)
            preds_aug2_list = np.vstack(preds_aug2_list)
            np.save(os.path.join(save_path, 'preds_aug1_CIFAR-10-C.npy'), preds_aug1_list)
            np.save(os.path.join(save_path, 'preds_aug2_CIFAR-10-C.npy'), preds_aug2_list)
            np.save(os.path.join(save_path, 'avg_list_aug_CIFAR-10-C.npy'), avg_list)
            np.save(os.path.join(save_path, 'chal_acc_list_aug_CIFAR-10-C.npy'), chal_acc_list)
        else:
            preds_list = np.vstack(preds_list)
            np.save(os.path.join(save_path, 'preds_CIFAR-10-C.npy'), preds_list)
            np.save(os.path.join(save_path, 'avg_list_CIFAR-10-C.npy'), avg_list)
            np.save(os.path.join(save_path, 'chal_acc_list_CIFAR-10-C.npy'), chal_acc_list)
    
    elif cfg.testset.name == 'cifar10r':
        avg = 0
        for j, chal_loader in enumerate(dataloader.loaders):

            chal_error = 0

            with torch.no_grad():
                for idx, batch in enumerate(tqdm(chal_loader)):
                    if cfg.testset.twoaug:
                        x1, x2, _y = batch
                        x, y = torch.cat([x1, x2]), torch.cat([_y, _y])
                    else:
                        x, y = batch

                    # cost: NLL, probs: softmax
                    cost, err, probs = model.sample_eval(x, y, Nsamples=cfg.uncertain_quant.n_samples, logits=False)
                    if cfg.testset.twoaug: 
                        probs1, probs2 = probs.chunk(2)
                        preds_aug1_list.append(probs1)
                        preds_aug2_list.append(probs2)
                    else:
                        preds_list.append(probs)

                    chal_error += err.cpu().numpy()
                    # print(err)
            chal_acc = 1 - (chal_error/len(chal_loader.dataset))
            avg += chal_acc
            print(f'TEST acc of {dataloader.angles[j+1]} degree rotation: {chal_acc}')
            chal_acc_list.append(chal_acc)

        avg /= len(dataloader.loaders)
        print("OVERALL Mean Acc: ", avg)

        ## save files
        save_path= os.path.join(cfg.ckpt_dir, cfg.uncertain_quant.method,  cfg.trainset.name, cfg.trainset.noise_type, f'seed{cfg.seed}')
        print(f'saving to {save_path}')
        if cfg.testset.twoaug: 
            preds_aug1_list = np.vstack(preds_aug1_list)
            preds_aug2_list = np.vstack(preds_aug2_list)
            np.save(os.path.join(save_path, 'preds_aug1_CIFAR-10-R.npy'), preds_aug1_list)
            np.save(os.path.join(save_path, 'preds_aug2_CIFAR-10-R.npy'), preds_aug2_list)
            np.save(os.path.join(save_path, 'chal_acc_list_aug_CIFAR-10-R.npy'), chal_acc_list)
        else:
            preds_list = np.vstack(preds_list)
            np.save(os.path.join(save_path, 'preds_CIFAR-10-R.npy'), preds_list)
            np.save(os.path.join(save_path, 'chal_acc_list_CIFAR-10-R.npy'), chal_acc_list)

    elif cfg.testset.name == 'cifar10':

        test_error = 0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader)):
                if cfg.testset.twoaug:  
                    x1, x2, _y, _ = batch
                    x, y = torch.cat([x1, x2]), torch.cat([_y, _y])
                else:
                    x, y, _ = batch
                # cost: NLL, probs: softmax
                cost, err, probs = model.sample_eval(x, y, Nsamples=cfg.uncertain_quant.n_samples, logits=False)

                if cfg.testset.twoaug: 
                    probs1, probs2 = probs.chunk(2)
                    preds_aug1_list.append(probs1)
                    preds_aug2_list.append(probs2)
                else:
                    preds_list.append(probs)
                test_error += err.cpu().numpy()
        
        avg = 1 - (test_error/len(dataloader.dataset))
        
        print(f'TEST acc of {cfg.testset.name}: {avg}')

        ## save files
        save_path= os.path.join(cfg.ckpt_dir, cfg.uncertain_quant.method,  cfg.trainset.name, cfg.trainset.noise_type, f'seed{cfg.seed}')
        print(f'saving to {save_path}')

        if cfg.testset.twoaug: 
            preds_aug1_list = np.vstack(preds_aug1_list)
            preds_aug2_list = np.vstack(preds_aug2_list)
            np.save(os.path.join(save_path, 'preds_aug1_CIFAR-10.npy'), preds_aug1_list)
            np.save(os.path.join(save_path, 'preds_aug2_CIFAR-10.npy'), preds_aug2_list)
        else:
            preds_list = np.vstack(preds_list)
            np.save(os.path.join(save_path, 'preds_CIFAR-10.npy'), preds_list)

    return avg_list
