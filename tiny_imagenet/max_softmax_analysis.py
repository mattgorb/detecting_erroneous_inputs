# from robustness import model_utils, datasets, train, defaults
# from robustness.datasets import CIFAR
import torch as ch
import dill
from cox.utils import Parameters
import cox.store
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
import torchvision.models as models
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from sklearn.svm import SVC
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import sys
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.utils import shuffle
from torchvision.utils import save_image
import pickle
import numpy as np
import csv
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from tiny_imagenet.wideresnet import WideResNet





def load(file):
    dataset = np.load(file)
    print(dataset.shape)
    return dataset


name = 'wideresnet'

correct=load('tiny_imagenet/generated_data/correct_preds_'+str(name)+'.npy')
incorrect=load('tiny_imagenet/generated_data/incorrect_preds_'+str(name)+'.npy')
sun=load('tiny_imagenet/generated_data/sun'+str(name)+'.npy')
fgsm  = load('tiny_imagenet/generated_data/fgsm_'+str(name)+'.npy')

c_linf  = load('tiny_imagenet/generated_data/carlini_linf_0.3'+str(name)+'.npy')
c_l2  = load('tiny_imagenet/generated_data/carlini_l2'+str(name)+'.npy')
pgd  = load('tiny_imagenet/generated_data/pgd0.3'+str(name)+'.npy')
corrupted  = load('tiny_imagenet/generated_data/corrupted_softmax'+str(name)+'.npy')
places  = load('tiny_imagenet/generated_data/places365'+str(name)+'.npy')


correct=correct[np.arange(np.shape(correct)[0])[:,np.newaxis], np.argsort(correct)]
incorrect=incorrect[np.arange(np.shape(incorrect)[0])[:,np.newaxis], np.argsort(incorrect)]
sun=sun[np.arange(np.shape(sun)[0])[:,np.newaxis], np.argsort(sun)]
fgsm=fgsm[np.arange(np.shape(fgsm)[0])[:,np.newaxis], np.argsort(fgsm)]

c_linf=c_linf[np.arange(np.shape(c_linf)[0])[:,np.newaxis], np.argsort(c_linf)]
c_l2=c_l2[np.arange(np.shape(c_l2)[0])[:,np.newaxis], np.argsort(c_l2)]
pgd=pgd[np.arange(np.shape(pgd)[0])[:,np.newaxis], np.argsort(pgd)]
corrupted=corrupted[np.arange(np.shape(corrupted)[0])[:,np.newaxis], np.argsort(corrupted)]
places=places[np.arange(np.shape(places)[0])[:,np.newaxis], np.argsort(places)]

correct=correct[:,-1:]
incorrect=incorrect[:,-1:]
sun=sun[:,-1:]
fgsm=fgsm[:,-1:]
c_linf=c_linf[:,-1:]
c_l2=c_l2[:,-1:]
pgd=pgd[:,-1:]
corrupted=corrupted[:,-1:]
places=places[:,-1:]


print(np.mean(correct[:len(pgd)]))

print(np.mean(incorrect[:len(pgd)]))
print(np.mean(sun[:len(pgd)]))
print(np.mean(fgsm[:len(pgd)]))
print(np.mean(c_linf[:len(pgd)]))
print(np.mean(c_l2[:len(pgd)]))
print(np.mean(pgd[:len(pgd)]))
print(np.mean(corrupted[:len(pgd)]))
print(np.mean(places[:len(pgd)]))