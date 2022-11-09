"""The script of stylized dataloader"""
import sys
sys.path.append('../pytorch-AdaIN/')
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from function import adaptive_instance_normalization, coral
import net


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


class StylizedSet(Dataset):
    def __init__(self, cfg, data, targets, indices, transform=None):
        # initialize data, currently assuming cifar10
        self._X = data
        self._Y = targets
        self._indices = indices
        self.transform = transform
        
        if cfg.testset.name == 'cifar10':
            self._class_map = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6:'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
            # load the uncertain test (human) labels
            self._Y_h = np.load('../cifar-10h/data/cifar10h-probs.npy')
        if torch.is_tensor(self._X): 
            self._X = torch.permute(self._X, (0, 2, 3, 1)).numpy()
        # transforms for content and style # 
        self.content_tf = test_transform(cfg.uncertain_quant.anlys.tta_cont_size, cfg.uncertain_quant.anlys.tta_crop)
        self.style_tf = test_transform(cfg.uncertain_quant.anlys.tta_sty_size, cfg.uncertain_quant.anlys.tta_crop)
        self.preserve_color = cfg.uncertain_quant.anlys.tta_pres_color
        self.device = cfg.device # "cpu" #
        self.stytf_vgg_path = cfg.uncertain_quant.anlys.tta_stytf_vgg
        self.stytf_decoder_path = cfg.uncertain_quant.anlys.tta_stytf_decoder
        self._vgg, self._decoder = None, None
        self.stytf_alpha = cfg.uncertain_quant.anlys.tta_stytf_alpha
        self._load_stytf_models()
 
    def _load_stytf_models(self):
        self._decoder = net.decoder
        self._vgg = net.vgg

        self._decoder.eval()
        self._vgg.eval()

        self._decoder.load_state_dict(torch.load(self.stytf_decoder_path))
        self._vgg.load_state_dict(torch.load(self.stytf_vgg_path))
        self._vgg = nn.Sequential(*list(self._vgg.children())[:31])

        self._vgg.to(self.device)
        self._decoder.to(self.device)

    def __getitem__(self, index, mode=''):
        x, y = self._X[index], self._Y[index]
        im = Image.fromarray(x)
        # do style transfer
        content = self.content_tf(im)
        # content = content.to(self.device).unsqueeze(0)
        # get the probable classes given the sample index
        style_classes = np.where(self._Y_h[index]>0)[0]
        
        # initialize a dict with mappings: {k, v} as {cls, list(stylizedImgs)}
        styleX = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6:[], 7: [], 8: [], 9: []}
        
        for stycls in style_classes: # for each test image, there can be multiple plausible classes
            # query the examplar texture image(s), note that there can be more than one examplar image.
            sty_paths = glob(f'./data/texture/{self._class_map[stycls]}.*')
            # print(sty_paths)
            # input()
            for sty_path in sty_paths:
                # TODO: style image generation; maybe categorize into human labeled and not human labeled
                style = self.style_tf(Image.open(sty_path).convert('RGB'))

                if self.preserve_color:
                    style = coral(style, content)
                style = style.to(self.device).unsqueeze(0)

                with torch.no_grad():
                    output = style_transfer(self._vgg, self._decoder, content.to(self.device).unsqueeze(0), style,
                                            self.stytf_alpha)
                output = output.cpu().squeeze(0)
                # normalize to 0-1 (per channel)
                for c in range(output.shape[0]):
                    output[c] = (output[c] - output[c].min()) / (output[c].max() - output[c].min()) 
                    # print(output[c].max(), output[c].min())
                if mode == 'ldim':   # return the larger-dimension synthesized image
                    return output, content.detach().cpu() # 512x512 style image
                # resize back to original input dim
                output = F.resize(output, im.size[0])
                
                if self.transform is not None:
                    output = self.transform(output)
                
                # for that specific style class, append that specific tenso    
                styleX[stycls].append(output)
                
                # print(f'appended classs {stycls} from {sty_path}')
                # input()

        if self.transform is not None:
            x = transforms.ToTensor()(im)
            x = self.transform(x)

        return x, y, styleX

    def __len__(self):
        return len(self._X)
    
    
def get_stylizedloader(cfg, data, targets, indices, transform=None):
    stydataset = StylizedSet(cfg, data, targets, indices, transform = transform)
    styloader = DataLoader(dataset=stydataset,
                          batch_size = cfg.test.batch_size,
                          num_workers=cfg.dataloader.num_workers,
                          shuffle=False)

    return styloader
