import os
import time
import random
import json
import numpy as np 
from sklearn import metrics
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

from collections import OrderedDict
import torch
import numpy as np 
# import scipy
from scipy.special import softmax as scipy_softmax
# import config 


def print_cz(str, f=None):
    if f is not None:
        print(str, file=f)
        if random.randint(0, 20) < 3:
            f.flush()
    print(str)

def time_mark():
    time_now = int(time.time())
    time_local = time.localtime(time_now)
    dt = time.strftime('%Y%m%d-%H%M%S', time_local)
    return(dt)

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def model_snapshot(model, new_file, old_file=None, save_dir='./', verbose=True, log_file=None):
    """
    :param model: network model to be saved
    :param new_file: new pth name
    :param old_file: old pth name
    :param verbose: more info or not
    :return: None
    """
    if os.path.exists(save_dir) is False:
        os.makedirs(expand_user(save_dir))
        print_cz(str='Make new dir:'+save_dir, f=log_file)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    for file in os.listdir(save_dir):
        if old_file in file:
            if verbose:
                print_cz(str="Removing old model  {}".format(expand_user(save_dir + file)), f=log_file)
            os.remove(save_dir + file) 
    if verbose:
        print_cz(str="Saving new model to {}".format(expand_user(save_dir + new_file)), f=log_file)
    torch.save(model, expand_user(save_dir + new_file))
    # torch.save(model.state_dict(),expand_user(save_dir + 'dict_'+new_file))

def adjust_learning_rate(optimizer, lr, epoch, lr_step=40, lr_gamma=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (lr_gamma ** (epoch // lr_step)) # 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0# value = current value
        self.avg = 0
        self.sum = 0# weighted sum
        self.count = 0# total sample num

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

######### curve #####
def curve_save(x, y, tag, yaxis, theme, save_dir):
    color = ['r', 'b', 'g', 'c', 'orange', 'lightsteelblue', 'cornflowerblue', 'indianred', 'lightgray', 'thistle']
    fig = plt.figure()
    # ax = plt.subplot()
    plt.grid(linestyle='-', linewidth=0.5, alpha=0.5)
    if isinstance(tag, list):
        for i, (y_term, tag_term) in enumerate(zip(y, tag)):
            plt.plot(x, y_term, color[i], label=tag_term, alpha=0.7)
    else:
        plt.plot(x, y, color[0], label=tag, alpha=0.7)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(yaxis, fontsize=12)
    plt.title('curve-{}'.format(theme), fontsize=14)
    plt.legend()
    fig.savefig(os.path.join(save_dir,'curve-{}.png'.format(theme)), dpi=300)
    plt.close('all') ####

def save_dict(info_dict, theme, save_dir):
    with open(os.path.join(save_dir, 'infodict-{}.json'.format(theme)), 'w') as f:
        f.write(json.dumps(info_dict))
        
def read_dict(filename):
    with open(filename, 'r') as f:
        info_dict = json.load(f)
    return info_dict

def init_dict(keys):
    d = {}
    for key in keys:
        d[key] = []
    return d



################
def convert_binary_1(matrix):
    TP = matrix[0, 0]
    FP = matrix[1, 0] + matrix[2, 0]
    FN = matrix[0, 1] + matrix[0, 2]
    TN = matrix[1, 1] + matrix[1, 2] + matrix[2, 1] + matrix[2, 2]
    return TP, FP, FN, TN

def convert_binary_2(matrix):
    TP = matrix[1, 1]
    FP = matrix[0, 1] + matrix[2, 1]
    FN = matrix[1, 0] + matrix[1, 2]
    TN = matrix[0, 0] + matrix[0, 2] + matrix[2, 2] + matrix[2, 0]
    return TP, FP, FN, TN

def convert_binary_3(matrix):
    TP = matrix[2, 2]
    FP = matrix[0, 2] + matrix[1, 2]
    FN = matrix[2, 0] + matrix[2, 1]
    TN = matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1]
    return TP, FP, FN, TN

def binary_precision(TP, FP, FN, TN):
    return 100.0*TP/(TP+FP)

def binary_sensitivity(TP, FP, FN, TN):
    """also recall"""
    return 100.0*TP/(TP+FN)

def binary_specificity(TP, FP, FN, TN):
    return 100.0*TN/(TN+FP)

def compute_precision(matrix):
    p1 = binary_precision(*convert_binary_1(matrix))
    p2 = binary_precision(*convert_binary_2(matrix))
    p3 = binary_precision(*convert_binary_3(matrix))
    p_avg = (p1 + p2 + p3)/3.0
    return p_avg, p1, p2, p3 

def compute_sensitivity(matrix):
    sen1 = binary_sensitivity(*convert_binary_1(matrix))
    sen2 = binary_sensitivity(*convert_binary_2(matrix))
    sen3 = binary_sensitivity(*convert_binary_3(matrix))
    sen_avg = (sen1 + sen2 + sen3)/3.0
    return sen_avg, sen1, sen2, sen3

def compute_specificity(matrix):
    spec1 = binary_specificity(*convert_binary_1(matrix))
    spec2 = binary_specificity(*convert_binary_2(matrix))
    spec3 = binary_specificity(*convert_binary_3(matrix))
    spec_avg = (spec1 + spec2 + spec3)/3.0
    return spec_avg, spec1, spec2, spec3

#########

# def returnCAM(feature, weight_softmax, class_idx):
#     """check feature_conv>0, following relu
#     """
#     B, C, Z = feature.shape #bz, nc, h, w = feature_conv.shape #原本的case，bz=1，所以后续随意的reshape
#     importance_classes = []
#     for idx in class_idx:
#         # 抹掉维度C
#         importance = np.sum(
#             (weight_softmax[idx]).reshape(1, -1, 1) * feature,
#             axis=1,
#             keepdims=False
#             ) #bz*nc*z -> bz*z
#         importance = scipy_softmax(importance, axis=-1) # 在z个Instance上进行归一化
#         importance_classes.append(importance)

#     importance_classes_npy = np.stack(importance_classes, axis=1) # bz*class*z
#     # print('importance_classes_npy.shape:\t', importance_classes_npy.shape)
#     # print('Z:\t', Z)
#     importance_std = np.zeros((B, Z))
#     for b in range(B):
#         for z in range(Z):
#             importance_std[b, z] = np.std(importance_classes_npy[b, :, z]) # std>=0
#         # importance_std[b] = (importance_std[b] - np.min(importance_std[b]))/(np.max(importance_std[b]) - np.min(importance_std[b]))
#         if np.sum(importance_std[b])>1e-10:
#             importance_std[b] = importance_std[b] /np.sum(importance_std[b]) 
#         else:
#             print('std sum problem!:\t', np.sum(importance_std[b]), importance_std[b])
#             importance_std[b] = np.ones((Z))/float(Z)
#     # importance_std = softmax(importance_std, axis=-1)
#     return importance_classes, importance_std

# #######################


def returnImportance(feature, weight_softmax, class_idx):
    """check feature_conv>0, following relu
    """
    B, C, K = feature.shape #
    importance_classes = []
    for idx in class_idx:
        importance = torch.sum(
            (weight_softmax[idx]).view(1,-1,1) * feature,
            dim=1,
            keepdim=False
            ) #bz*nc*K -> bz*K
        importance = torch.nn.functional.softmax(importance, dim=-1) # 
        importance_classes.append(importance)

    importance_classes_tensor = torch.stack(importance_classes, dim=1) # bz*class*K
    importance_std = torch.std(
        importance_classes_tensor,
        dim=1,
        keepdim=False
        )
    # 
    uniform_importance = torch.ones((K))/float(K) #
    if importance_std.is_cuda:
        uniform_importance.cuda()
    for b in range(B):
        if torch.sum(importance_std[b], dim=-1, keepdim=False)>1e-10:
            importance_std[b] = importance_std[b] /torch.sum(importance_std[b], dim=-1, keepdim=False)
        else:
            print('std sum problem!:\t', torch.sum(importance_std[b], dim=-1, keepdim=False), importance_std[b])
            importance_std[b] = uniform_importance

    return importance_classes, importance_std    # [B, 3, K], [B, K]



#######################
def seed_fix(args, logfile=None):
    torch.manual_seed(args.seed_idx)
    torch.cuda.manual_seed(args.seed_idx)
    np.random.seed(args.seed_idx)
    print_cz('seed fixed: {}'.format(args.seed_idx), f=logfile)

# ########################
# def returnImportance_norm1(feature, weight_softmax, class_idx):
#     """check feature_conv>0, following relu
#     """
#     B, C, K = feature.shape #bz, nc, h, w = feature_conv.shape #原本的case，bz=1，所以后续随意的reshape
#     importance_classes = []
#     for idx in class_idx:
#         tmp1 = (weight_softmax[idx]).view(1,-1,1)
#         # tmp1 = tmp1/torch.sum(torch.abs(tmp1), dim=1, keepdim=True)
#         tmp2 = feature 
#         # tmp2 = tmp2/torch.sum(torch.abs(tmp2), dim=1, keepdim=True)
#         importance = torch.sum(
#             tmp1*tmp2,
#             dim=1,
#             keepdim=False
#             ) #bz*nc*K -> bz*K
#         importance = torch.nn.functional.softmax(importance, dim=-1) # 在k个Instance上进行归一化
#         importance_classes.append(importance)

#     importance_classes_tensor = torch.stack(importance_classes, dim=1) # bz*class*K
#     # print('importance_classes_tensor.shape:\t', importance_classes_tensor.shape)
#     # print('K:\t', K)

#     #
#     importance_classes_tensor = importance_classes_tensor/torch.sum(importance_classes_tensor, dim=1, keepdim=True)

#     importance_std = torch.std(
#         importance_classes_tensor,
#         dim=1,
#         keepdim=False
#         )
#     # print(importance_std)
#     # 在K个instance的维度上归一化
#     uniform_importance = torch.ones((K))/float(K) #
#     if importance_std.is_cuda:
#         uniform_importance.cuda()
#     for b in range(B):
#         if torch.sum(importance_std[b], dim=-1, keepdim=False)>1e-10:
#             importance_std[b] = importance_std[b] /torch.sum(importance_std[b], dim=-1, keepdim=False)
#         else:
#             print('std sum problem!:\t', torch.sum(importance_std[b], dim=-1, keepdim=False), importance_std[b])
#             importance_std[b] = uniform_importance
#     # print(importance_std)
#     # _, index = torch.max(importance_std, dim=-1)
#     # print(index)
#     # print(torch.sum(importance_std, dim=-1, keepdim=False))
#     return importance_classes, importance_std
#     # [B, 3, K], [B, K]

