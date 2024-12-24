import argparse
import copy
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import open_world_cifar as datasets
import open_world_imagenet as datasets_imagenet
import open_world_tinyimagenet as datasetsTINYIMAGENET
import utils
from utils import *
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
import logging
import pdb
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def args_setting():
    parser = argparse.ArgumentParser(description='OwSSL')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str,default='name', help='name of the experiment in tensorboard')    
    parser.add_argument('--gpu', default='1', type=str,help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay of sgd')
    ## dataset setting
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--labeled-num', default=50, type=int, help='number of labeled classes')
    parser.add_argument('--labeled-ratio', default=0.5, type=float, help='ratio of labeled data in known classes')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',help='mini-batch size')
    parser.add_argument('--label-batch-ratio', default=-1, type=float, help= 'ratio of labeled data in a batch, default: -1, apportional to training set size')
    ## Sinkhorn-Knopp parameters setting
    parser.add_argument('--sk-epsilon', default=0.05, type=float, help='epsilon for Sinkhorn-Knopp algorithm')
    parser.add_argument('--sk-iters', default=10, type=int, help='number of iterations for Sinkhorn-Knopp algorithm')
    parser.add_argument('--adjust-prior', default=1, type=int, help='whether to adjust the prior of SK-code according to labeled ratio in a batch; False: use uniform prior')
    ## Structure setting
    ### Adaptive threshold
    parser.add_argument('--adathres', type=str, default='FreeMatch', 
                        help='OpenFree: adaptive threshold for seen and novel class groups \
                            FreeMatch: adaptive threshold for each class (1*10); \
                            trival: fix threshold')
    parser.add_argument('--adathres-init', type=float, default=-1, help='initial threshold for Confidence loss. -1: use 1/num_classes')
    parser.add_argument('--adathres-alpha', type=float, default=0.99, help='alpha for adaptive threshold')
    parser.add_argument('--fix-thres', type=float, default=0.95, help='threshold for Confidence loss, if adathres is trival')
    ### Multi-crop
    parser.add_argument('--n-crop', type=int, default=4, help='number of crop in multi-crop. default: 6')
    ### Logits Queue
    parser.add_argument('--queue-len', type=int, default=1024, help='length of the queue')
    ### Hard-logits (replace)
    parser.add_argument('--hard-logit', type=int, default=1, help='whether to use hard-logit')
    ### Confidence loss
    parser.add_argument('--has-confidence', type=int, default=1, help='whether to use confidence loss')
    ### Supervised loss
    parser.add_argument('--has-supervised', type=int, default=1, help='whether to use supervised loss')
    ### EMA for Infernece
    parser.add_argument('--ema-inf', type=int, default=0, help='whether to use EMA for inference. 1: true; 0: false')
    parser.add_argument('--ema-alpha', type=float, default=0.999, help='alpha for EMA')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:"+args.gpu if args.cuda else "cpu")
        
    return args

def setup_seed(seed):
    # Set the seed for generating random numbers to ensure reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, train_label_loader, train_unlabel_loader, optimizer, ema_optimizer, epoch, queue, queue2, adaThres):
    model.train()
    unlabel_loader_iter = cycle(train_unlabel_loader)

    for batch_idx, ((x, x2, xmc, x3), target) in enumerate(train_label_loader):

        optimizer.zero_grad()
        ((ux, ux2, uxmc, ux3), unlabel_tl) = next(unlabel_loader_iter)
        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        x3 = torch.cat([x3, ux3], 0)
        labeled_len = len(target)
        
        # `mc_list` is a list containing tensor of multicrop input
        if args.n_crop != 0:
            mc_list = []
            for i in range(len(xmc)):
                con_b = torch.cat([xmc[i], uxmc[i]], 0)
                mc_list.append(con_b)
            xmc_con = torch.cat(mc_list, 0)
        else:
            xmc_con = torch.tensor([]).to(args.device)  
        
        x, x2, x3, xmc, target, unlabel_tl = x.to(args.device), x2.to(args.device), x3.to(args.device), xmc_con.to(args.device), target.to(args.device), unlabel_tl.to(args.device)

        # output: logits, feat: embedding
        output, feat = model(x)
        output2, feat2 = model(x2)
        output3, feat3 = model(x3)
        if args.n_crop != 0:
            outputmc, featmc = model(xmc)

        # prediction of labeled data and unlabeled data
        prob = F.softmax(output, dim=1)
        prob2 = F.softmax(output2, dim=1)
            
        with torch.no_grad():
            # construct HARD logit for conditional self-labeling
            logit_weak_sk = output2.detach().clone()
            logit_weak_sk2 = output.detach().clone()
            if args.hard_logit:
                max_logit = 10
                # assign 10 (a max const.) to the corresponding class
                for i in range(labeled_len):
                    logit_weak_sk[i, target[i]] = max_logit
                    logit_weak_sk2[i, target[i]] = max_logit
                    
            # ---------- Generate self-label assignment by Sinkhorn-Knopp algorithm ----------
            ks_code = queue.get_SK_and_Update(logit_weak_sk)
            ks_code2 = queue2.get_SK_and_Update(logit_weak_sk2)
            # ---------- ################################################# ----------



        # ---------- Generate pseudo label by directly using softmax ----------
        
        # pseudo_label = torch.softmax(logit_weak.detach(), dim=-1)
        max_probs, targets_u = torch.max(prob2.detach(), dim=-1)

        # ---------- ----------Adaptive threshold---------- ----------
        threshold = adaThres.get()
        adaThres.update(prob2)
        conf_fm_b, hard_label_fm = torch.max(prob2, dim=-1)
        
        mask_fm = torch.zeros_like(conf_fm_b)
        mask_id, mask_ood = torch.zeros_like(conf_fm_b), torch.zeros_like(conf_fm_b)
        id_count, ood_count = 0, 0
        for i in range(args.num_classes):
            index_i = (hard_label_fm == i)
            mask_i =  index_i * (conf_fm_b > threshold[i]).float().detach()
            mask_fm += mask_i
            if i < args.labeled_num:
                mask_id += mask_i
                id_count += torch.sum(mask_i)
            else:
                mask_ood += mask_i
                ood_count += torch.sum(mask_i)

        # ---------- ----------##################---------- ----------
        


        ################################# Confidence loss #################################
        mask = mask_id + mask_ood
        if args.has_confidence:
            confidence_loss = (F.cross_entropy(output3,
                                                targets_u,
                                                reduction='none') * mask).mean()
        else:
            confidence_loss = torch.tensor(0.0)
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Confidence loss ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

        ################################# Clustering loss ###################################
        if args.n_crop != 0:
            cluster_loss = -torch.mean(torch.sum(ks_code * torch.log(prob), dim = 1))
            for mc_id in range(args.n_crop):
                probmc = F.softmax(outputmc[args.batch_size * mc_id: args.batch_size * (mc_id + 1)], dim = 1)
                cluster_loss -= torch.mean(torch.sum(ks_code * torch.log(probmc), dim = 1))
            cluster_loss /= args.n_crop + 1
        else:
            cluster_loss = (-torch.mean(torch.sum(ks_code * torch.log(prob), dim = 1)) - torch.mean(torch.sum(ks_code2 * torch.log(prob2), dim = 1)))/2
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Clustering loss ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#


        ################################# Supervised loss #####################################
        if args.has_supervised:
            supervisd_loss = (F.cross_entropy(output[:labeled_len], target, reduction='none')).mean()
        else:
            supervisd_loss = 0
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Supervised loss ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

        
        loss = confidence_loss + supervisd_loss + cluster_loss

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.ema_inf:
            ema_optimizer.step()

  

def test(args, model, test_loader, epoch):
    labeled_num = args.labeled_num
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    probs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(args.device), label.to(args.device)

            output, feature = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
            probs = np.append(probs, prob.cpu().numpy())

    targets = targets.astype(int)
    preds = preds.astype(int)

    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])

    if sum(unseen_mask) > 0:
        unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    else:
        unseen_acc = 0
    # unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))



def main():
    torch.cuda.empty_cache()
    args = args_setting()
    setup_seed(args.seed)

    if args.dataset == 'cifar10':
        train_label_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num,
                                                    labeled_ratio=args.labeled_ratio, download=True,
                                                    transform=TransformTwice('cifar10',args.n_crop))
        train_unlabel_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                                      labeled_ratio=args.labeled_ratio, download=True,
                                                      transform=TransformTwice('cifar10',args.n_crop),
                                                      unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                             labeled_ratio=args.labeled_ratio, download=True,
                                             transform=datasets.dict_transform['cifar_test'],
                                             unlabeled_idxs=train_label_set.unlabeled_idxs, train=False)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_label_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=True, labeled_num=args.labeled_num,
                                                     labeled_ratio=args.labeled_ratio, download=True,
                                                     transform=TransformTwice('cifar100',args.n_crop))
        train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                                       labeled_ratio=args.labeled_ratio, download=True,
                                                       transform=TransformTwice('cifar100',args.n_crop),
                                                       unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                              labeled_ratio=args.labeled_ratio, download=True,
                                              transform=datasets.dict_transform['cifar_test'],
                                              unlabeled_idxs=train_label_set.unlabeled_idxs, train=False)
        num_classes = 100
    elif args.dataset == 'tinyimagenet':
        train_label_set = datasetsTINYIMAGENET.TinyImageNet(root='./datasets/tiny-imagenet-200', labeled = True, labeled_num=args.labeled_num,
                                                        labeled_ratio=args.labeled_ratio,
                                                        transform=TransformTwice('tinyimagenet',args.n_crop))
        
        train_unlabel_set = datasetsTINYIMAGENET.TinyImageNet(root='./datasets/tiny-imagenet-200', labeled = False, labeled_num=args.labeled_num,
                                                            labeled_ratio=args.labeled_ratio,
                                                            transform=TransformTwice('tinyimagenet',args.n_crop),
                                                            unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasetsTINYIMAGENET.TinyImageNet(root='./datasets/tiny-imagenet-200', labeled = False, labeled_num=args.labeled_num,
                                                labeled_ratio=args.labeled_ratio,
                                                transform=datasetsTINYIMAGENET.dict_transform['tinyimagenet_test'],
                                                unlabeled_idxs=train_label_set.unlabeled_idxs, train=False)
        num_classes = 200
    else:
        warnings.warn('Dataset is not listed')
        return
    args.num_classes = num_classes


    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    print('labeled_len: ', labeled_len, 'unlabeled_len: ', unlabeled_len)
    if args.label_batch_ratio == -1:
        labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))
    else:
        labeled_batch_size = int(args.batch_size * args.label_batch_ratio)
    args.labeled_batch_size = labeled_batch_size

    for arg in vars(args):
        print(arg, getattr(args, arg))

    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True,
                                                     num_workers=64, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set,
                                                       batch_size=args.batch_size - labeled_batch_size, shuffle=True,
                                                       num_workers=64, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=64)

    # Set the model
    model = models.resnet_s.resnet18(num_classes=num_classes)
    if args.dataset == 'cifar10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    elif args.dataset == 'cifar100' or args.dataset == 'tinyimagenet':
        state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
    else:
        state_dict = None

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)

    if args.ema_inf:
        # copy the model parameters to ema_model
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        ema_model = ema_model.to(args.device)

    # Set the optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=args.weight_decay, momentum=0.9)
    else:
        warnings.warn('Optimizer is not listed')
        return
    if args.ema_inf:
        ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_alpha)
    else:
        ema_optimizer = None

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.queue_len > 0:
        queue = LogitQueue(args)
        queue2 = LogitQueue(args)
    else:
        queue = trivalQueue()

    if args.adathres == 'FreeMatch':
        adathres = AdaptiveThreshold(args)
    elif args.adathres == 'trivial':
        adathres = trivalThreshold(args)
    elif args.adathres == 'OpenFree':
        adathres = OpenFreeThreshold(args)
    else:
        raise ValueError('Adaptive threshold is not listed')

    for epoch in range(args.epochs):
        print('------------------- Epoch: %d -------------------' % epoch)
        train(args, model, train_label_loader, train_unlabel_loader, 
              optimizer, ema_optimizer, epoch, 
              queue, queue2, adathres)
        if args.ema_inf:
            test(args, ema_model, test_loader, epoch)
        else:
            test(args, model, test_loader, epoch)
        scheduler.step()



if __name__ == '__main__':
    main()
