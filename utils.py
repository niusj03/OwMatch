from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import os.path
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from randaugment import RandAugmentMC
import pdb
import warnings

from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps

'''
Updated: 2023-12-14
Supporting features:
- Multicrop augmentation
- Queue for SK-algorithm
- Adaptive threshold
- Labeled queue (not used yet)
- EMA model
    - Special Notes: EMA module not only add a ema model for reference, 
    - it also introduce a weight decay of 5e-4 on its student model
'''

class TransformTwice:
    def __init__(self, dataset, n_crop = 6):
        self.dataset = dataset
        self.n_crop = n_crop

        self.strong_cifar10 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),])

        self.strong_cifar100 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

        self.strong_tinyimagenet = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10, size=64),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))])

        self.cifar10_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),])

        self.cifar100_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

        self.tinyimagenet_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))])

        self.mc_cifar10 = transforms.Compose([
            transforms.RandomResizedCrop(size=18, scale=(0.3, 0.75)),
            transforms.RandomHorizontalFlip(),
            get_color_distortion_cifar(s=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),])
        
        self.mc_cifar100 = transforms.Compose([
            transforms.RandomResizedCrop(size=18, scale=(0.3, 0.75)),
            transforms.RandomHorizontalFlip(),
            get_color_distortion_cifar(s=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

        self.mc_tinyimagenet = transforms.Compose([
            transforms.RandomResizedCrop(size=36, scale=(0.3, 0.75)),
            transforms.RandomHorizontalFlip(),
            get_color_distortion_cifar(s=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))])
        
    def __call__(self, inp):
        outmc = () # init an empty tuple
        
        if self.dataset == 'cifar10':
            out1 = self.cifar10_train(inp)
            out2 = self.cifar10_train(inp)
            outs = self.strong_cifar10(inp)
            for _ in range(self.n_crop):
                outmc += (self.mc_cifar10(inp),)
            return out1, out2, outmc, outs
        elif self.dataset == 'cifar100':
            out1 = self.cifar100_train(inp)
            out2 = self.cifar100_train(inp)
            outs = self.strong_cifar100(inp)
            for _ in range(self.n_crop):
                outmc += (self.mc_cifar100(inp),)
            return out1, out2, outmc, outs
        elif self.dataset == 'imagenet100':
            out1 = self.imageNet100_train(inp)
            out2 = self.imageNet100_train(inp)
            outs = self.strong_imagenet(inp)
            outmc = tuple(self.mc_imageNet100(inp) for _ in range(self.n_crop))
            return out1, out2, outmc, outs
        elif self.dataset == 'tinyimagenet':
            out1 = self.tinyimagenet_train(inp)
            out2 = self.tinyimagenet_train(inp)
            outs = self.strong_tinyimagenet(inp)
            outmc = tuple(self.mc_tinyimagenet(inp) for _ in range(self.n_crop))
            return out1, out2, outmc, outs
        else:
            warnings.warn('Dataset is not listed')
            return

class WeightEMA(object):
    '''
    Copyright: @TRSSL
    '''
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 2e-5

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

        # print('WeightEMA: EMA model is created')
        # print('WeightEMA: EMA alpha is %f' % alpha)
        # print('WeightEMA: EMA weight decay is %f' % self.wd)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd) # there is already a weight decay in sgd, here it is a direct weight decay of the parameters.


class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class OpenFreeThreshold(object):
    '''
    OpenFreeThreshold creates two types of adaptive thresholds:
    'Close', for seen classes, and 'Open' for novel classes.
    '''
    def __init__(self, args):
        self.args = args
        args_close = deepcopy(args)
        args_open = deepcopy(args)

        # Set the number of classes for Close and Open
        args_close.num_classes = args.labeled_num
        args_open.num_classes = args.num_classes - args.labeled_num

        # Initialize AdaptiveThreshold instances for Close and Open
        self.close = AdaptiveThreshold(args_close)
        self.open = AdaptiveThreshold(args_open)

    def update(self, pred):
        pred = pred.detach()
        _, labels = torch.max(pred, dim=1)

        close_indices = labels < self.close.num_classes
        open_indices = labels >= self.close.num_classes

        # Slice the predictions based on these indices into close and open
        pred_close = pred[close_indices.nonzero(as_tuple=True)][:,:self.args.labeled_num]
        pred_open = pred[open_indices.nonzero(as_tuple=True)][:,self.args.labeled_num:]

        # Update thresholds for close and open sets
        if pred_close.nelement() > 0:  # Check if there are elements for 'close' to update
            self.close.update(pred_close)
        if pred_open.nelement() > 0:   # Check if there are elements for 'open' to update
            self.open.update(pred_open)

    def get(self):
        close_threshold = self.get_close_threshold()
        open_threshold = self.get_open_threshold()

        # warm up the threshold
        close_threshold = np.maximum(close_threshold, 0.3)
        open_threshold = np.maximum(open_threshold, 0.3)

        return np.concatenate([close_threshold,open_threshold],0)

    def get_close_threshold(self):
        # Get the threshold for the close set
        return self.close.get()

    def get_open_threshold(self):
        # Get the threshold for the open set
        return self.open.get()


class AdaptiveThreshold(object):
    '''
    Adaptive threshold is a combination of global and local thresholds.
    Global threshold := mean of max(confidence)
    Local threshold := mean of confidence of each class
    Final threshold := Global * MaxNorm(Local)
    '''
    def __init__(self, args):
        alpha = args.adathres_alpha
        num_classes = args.num_classes
        init_thres = args.adathres_init
        if init_thres is None:
            init_thres = 1/num_classes
        self.alpha = alpha
        self.num_classes = num_classes
        self.local = np.ones((num_classes)) * init_thres
        self.global_ = init_thres

    def update(self, pred):
        pred = pred.detach()
        conf, label = torch.max(pred, dim=1)
        # update global threshold
        self.global_ = self.alpha * self.global_ + (1 - self.alpha) * conf.mean().item()
        # update local threshold
        new_local = pred.mean(dim=0).cpu().numpy()
        self.local = self.alpha * self.local + (1 - self.alpha) * new_local

    def get(self):
        # tau_local = MaxNorm(local)
        tau_local = self.local/np.max(self.local)
        tau_global = self.global_
        return tau_local*tau_global
    
class trivalThreshold(object):
    def __init__(self, args):
        self.tau = args.fix_thres * np.ones((args.num_classes))
    def update(self, pred):
        pass
    def get(self):
        return self.tau

class LogitQueue(object):
    def __init__(self, args):
        # if args.swap:
        #     self.SIZE = args.queue_len * 2
        # else:
        self.SIZE = args.queue_len
        self.qlen = 0
        self.num_classes = args.num_classes
        self.queue = torch.tensor([]).to(args.device)
        self.args = args

    def update(self, new_logits):
        # assert new_logits.shape[0] == self.batch_size
        if self.qlen < self.SIZE:
            self.queue = torch.cat([self.queue, new_logits])
            self.qlen += new_logits.shape[0]
        else:
            new_len = new_logits.shape[0]
            self.queue = torch.cat([self.queue[new_len:], new_logits], dim=0)
            # [new] --> old[len:] -x-> old[:-len] |||| FIFO
    
    @torch.no_grad()
    def get_SK_and_Update(self, new_logits):
        self.update(new_logits)
        code_for_queue = sinkhorn(self.queue, self.args)
        return code_for_queue[-new_logits.shape[0]:].detach()

    
class trivalQueue(object):
    def __init__(self, args):
        self.args = args

    def update(self, new_logits):
        pass
    def get_SK_and_Update(self, new_logits):
        code_for_itself = sinkhorn(new_logits, self.args)
        return code_for_itself.detach()
    
                
print_once = True

@torch.no_grad()
def sinkhorn(logit_in, args):
    iters = args.sk_iters
    epsilon = args.sk_epsilon

    logit = logit_in.detach().clone()
    logit /= 10 # scale to [-1,1], the cosine with prototypes
    logit /= epsilon # temperature parameter in SwAV
    # if na
    if torch.isnan(logit).any():
        pdb.set_trace()
    Q = torch.exp(logit).t() # Q is K-by-B
    Q /= torch.sum(Q)

    c = torch.ones(Q.shape[1]).to(Q.device) / Q.shape[1] # Samples
    r = torch.ones(Q.shape[0]).to(Q.device) / Q.shape[0]

    for it in range(iters):
        u = torch.sum(Q, dim=1)
        Q *= (r / u).unsqueeze(1)
        Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
    Q /= torch.sum(Q, dim=0, keepdim=True)
    if torch.isnan(Q).any():
        pdb.set_trace()
    return Q.t()


def accuracy(output, target):
    
    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res

def cluster_acc(y_pred, y_true):

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size, "y_pred.size != y_true.size: "+str(y_pred.size)+" "+str(y_true.size)
    # pdb.set_trace()
    D = max(y_pred.max(), y_true.max()) + 1
    # if D != 10 or D != 100:
    #     pdb.set_trace()
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    # pdb.set_trace()
    return w[row_ind, col_ind].sum() / y_pred.size


def entropy(x):

    EPS = 1e-8
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class MarginLoss(nn.Module):
    
    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)

def count_assignment(pred, num_classes=10):
    import numpy as np
    prediction = np.zeros(num_classes)
    for i in range(num_classes):
        prediction[i] = torch.sum(pred == i)
    prediction = prediction / prediction.sum()
    # print to 2 decimals
    print(np.round(prediction.tolist(), 2))
    return prediction




def get_color_distortion_cifar(s=0.5):
    # print('_make_multicrop_cifar10_transforms distortion strength', s)
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    color_distort = transforms.Compose([rnd_color_jitter, Solarize(p=0.2), Equalize(p=0.2)])
    return color_distort

def get_color_distortion_imgnet(s=1.0):
    # print('_make_multicrop_imgnet_transforms distortion strength', s)
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class Solarize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        v = torch.rand(1) * 256
        return ImageOps.solarize(img, v)

class Equalize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        return ImageOps.equalize(img)
    
class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))