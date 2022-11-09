# from models.Bayes_By_Backprop_Local_Reparametrization.bbp_lrt import BBP_LRT
from models import ordinary, mcd, Bayesian
# from models.Bayesian import BBB, BBB_LRT

_MODELS_ = {
    ## generative models
    'BbBLRT': Bayesian.get_model,  # Bayes-by-Backprop with Local Reparameterization 
    'BbB': Bayesian.get_model, # Bayes-by-Backprop
    # 'ensemble':  $bootstrap_ensemble 
    'MCD': mcd.get_model, # Monte-Carlo Dropout
    ## deterministic models
    'DUQ': None, # deterministic uncertainty quantification
    'Contrast': None,  # contrastive model
    'TENT': None, # 
    ## ordinary DNNs
    'ordinary': ordinary.get_model, # ordinary ResNet
}

def make_model(cfg):
    model = _MODELS_[cfg.uncertain_quant.method]
    print(f'using {cfg.uncertain_quant.method} method with {cfg.uncertain_quant.backbone.arch}-{cfg.uncertain_quant.backbone.depth}')
    return model(cfg)
