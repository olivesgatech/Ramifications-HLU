from .BBB.BBBLinear import BBBLinear as BBB_Linear
from .BBB.BBBConv import BBBConv2d as BBB_Conv2d

from .BBB_LRT.BBBLinear import BBBLinear as BBB_LRT_Linear
from .BBB_LRT.BBBConv import BBBConv2d as BBB_LRT_Conv2d

from .misc import FlattenLayer, ModuleWrapper

from .model import BBBResNet

def get_model(cfg):
    ## determine the type of Bayes
    if cfg.uncertain_quant.method=='BbBLRT':
        BBBLinear = BBB_LRT_Linear
        BBBConv2d = BBB_LRT_Conv2d
    elif cfg.uncertain_quant.method=='BbB':
        BBBLinear = BBB_Linear
        BBBConv2d = BBB_Conv2d
    else:
        raise ValueError("Undefined Bayesian layer type")
    BBBLayers = (BBBConv2d, BBBLinear)
    if cfg.uncertain_quant.backbone.arch=='resnet':          
        return BBBResNet(BBBLayers, cfg)
