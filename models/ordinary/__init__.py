from .model import ordResNet

def get_model(cfg):
    if cfg.uncertain_quant.backbone.arch=='resnet':          
        return ordResNet(cfg.uncertain_quant.backbone.depth, cfg.trainset.num_classes)
    