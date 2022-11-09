import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from scipy.spatial import distance
import pdb


# ELBO loss for variational inference methods
class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        assert not target.requires_grad
        return F.nll_loss(input, target, reduction='mean') * self.train_size + beta * kl

## Uncertainty Metrics ## 
# log-likelihood with deterministic labels
def get_ll(preds, targets):
    return np.log(1e-12 + preds[np.arange(len(targets)), targets]).mean()

## log-likelihood with human uncertain labels
def get_ll_h(preds, targets):
    return np.log(1e-12 + preds[np.arange(len(targets)), targets]).mean()

# Brier score
def get_brier(preds, targets):
    one_hot_targets = np.zeros(preds.shape)
    one_hot_targets[np.arange(len(targets)), targets] = 1.0
    return np.mean(np.sum((preds - one_hot_targets) ** 2, axis=1))

# Brier score with huma uncertain labels
def get_brier_h(preds, targets):
    # using the distribution labels, e.g., np.tile(targets, (15,1))
    return np.mean(np.sum((preds - targets) ** 2, axis=1))

def get_acc(preds, targets):
    yhat = np.argmax(preds, 1)
    accuracy = np.mean(yhat==targets)
    return accuracy

def get_entropy(probabilities, targets=None):
    # Shannon entropy
    logs = np.log2(1e-12+probabilities)
    mult = logs*probabilities
    entropy = (-1)*np.sum(mult, axis=1)
    return np.mean(entropy)

# def lr_linear(epoch_num, decay_start, total_epochs, start_value):
#     if epoch_num < decay_start:
#         return start_value
#     return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)
    
def get_align(x, y, alpha=2):
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if isinstance(y, np.ndarray): y = torch.from_numpy(y)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def get_uniform(x, t=2):
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def get_distance(preds, metric, topk=None): 
    # Compute averaged distance between each output and the centroids. the metric and topk can vary
    yhat = preds.argmax(1) # the predicted classes
    # print(np.unique(yhat))
    
    idx_list = dict() # indices of samples from each predicted class
    centroid_list = dict() # centroids of each predicted class 

    for c in range(10):
        idx_list[c]= np.where(yhat==c)[0]       
        if idx_list[c].size == 0:  # if no prediction on this class
            # assign a dummy feature vector
            centroid_list[c] = np.ones_like(preds[0])*100
        else:
            centroid_list[c] = preds[idx_list[c]].mean(0) # take the arithmic mean as the centroid

    centroid_array = np.vstack([centroid_list[c] for c in range(10)])

    dist = distance.cdist(preds, centroid_array, metric)
    # assert np.array_equal(dist.argsort(axis=1)[:,0], yhat)
    sorted_yhat = preds.argsort(axis=1)
    
    if topk == 1:
        ret = dist[range(len(dist)),sorted_yhat[:, -1]]
    else:
        ret = 0
        for k in range(2,topk+1):
            ret += dist[range(len(dist)),sorted_yhat[:, -k]]
        ret /= (topk-1)
    return ret.mean()

def get_margin(preds, targets=None):
    # yhat = preds.argmax(1)
    # get smallest margins
    sorted_probs = np.sort(preds, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    return margins.mean()
    
def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())

# KL divergence
def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

# Get weight beta for KL divergence
def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta
