import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import metrics
from utils import accuracy, logmeanexp


# Train the Model
def train(cfg, epoch, train_loader, model, optimizer):
    train_total=0
    train_correct=0
    model.train()
    for i, (images, labels, indexes) in enumerate(tqdm(train_loader)):

        batch_size = indexes.shape[0]
       
        images =images.to(cfg.device)
        labels =labels.to(cfg.device)
       
        # Forward + Backward + Optimize
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec
        loss = F.cross_entropy(logits, labels, reduce = True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % cfg.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  %(epoch+1, cfg.solver.num_epoch, i+1, len(train_loader.dataset)//batch_size, prec, loss.data))


    train_acc=float(train_correct)/float(train_total)
    return train_acc


# validate the Model
def validate(cfg, epoch, loader, model, save = False, best_acc = 0.0):
    model.eval()    # Change model to 'eval' mode.
    
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = Variable(images).to(cfg.device)
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)
    if save:
        if acc > best_acc:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            save_path= os.path.join(cfg.ckpt_dir, cfg.uncertain_quant.method,  cfg.trainset.name, cfg.trainset.noise_type, f'seed{cfg.seed}', 'best.pth.tar')
            torch.save(state,save_path)
            best_acc = acc
            print(f'model saved to {save_path}!')

    return acc


# Train Bayeisian Models
def train_bayesian(cfg, epoch, trainloader, net, optimizer, criterion):
    net.train()
    # get Bayesian configs
    num_ens = cfg.uncertain_quant.bayesian.train_ens
    beta_type=cfg.uncertain_quant.bayesian.beta_type
    num_epochs = cfg.solver.num_epoch
    
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels, _) in enumerate(tqdm(trainloader), 1):

        optimizer.zero_grad()

        inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
        outputs = torch.zeros(inputs.shape[0], cfg.trainset.num_classes, num_ens).to(cfg.device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)
        
        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
        
        if i % cfg.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f, KLD: %.4f'
                  %(epoch+1, cfg.solver.num_epoch, i, len(trainloader.dataset)//len(labels), accs[-1], loss.cpu().data, kl_list[-1]))

    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)


# validate Bayesian Models
def validate_bayesian(cfg, epoch, validloader, net, criterion, save = False, best_acc = 0.0):
    """Calculate ensemble accuracy and NLL Loss"""
    net.eval()
    # get Bayesian configs
    num_ens = cfg.uncertain_quant.bayesian.val_ens
    beta_type=cfg.uncertain_quant.bayesian.beta_type
    num_epochs = cfg.solver.num_epoch

    valid_loss = 0.0
    accs = []

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(validloader, 1):
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            outputs = torch.zeros(inputs.shape[0], cfg.trainset.num_classes, num_ens).to(cfg.device)
            kl = 0.0
            for j in range(num_ens):
                net_out, _kl = net(inputs)
                kl += _kl
                outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

            log_outputs = logmeanexp(outputs, dim=2)

            beta = metrics.get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
            valid_loss += criterion(log_outputs, labels, kl, beta).item()
            accs.append(metrics.acc(log_outputs, labels))

    if save:
        if np.mean(accs) > best_acc:
            state = {'state_dict': net.state_dict(),
                     'epoch':epoch,
                     'acc':np.mean(accs),
            }
            save_path= os.path.join(cfg.ckpt_dir, cfg.uncertain_quant.method,  cfg.trainset.name, cfg.trainset.noise_type, f'seed{cfg.seed}', 'best.pth.tar')
            torch.save(state,save_path)
            best_acc = np.mean(accs)
            print(f'model saved to {save_path}!')
            
    return valid_loss/len(validloader), np.mean(accs)