from .model import mcdResNet

def get_model(cfg):
    if cfg.uncertain_quant.backbone.arch=='resnet':          
        return mcdResNet(cfg)
    