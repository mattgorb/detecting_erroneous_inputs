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
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from utils.utils import *
import sys
import h5py
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

from sklearn.metrics import average_precision_score
from robustness.robustness.datasets import *
from robustness.robustness.cifar_models import *
from robustness.robustness.attacker import *
from robustness.robustness.attacker import *
from robustness.robustness.model_utils import *
from robustness.robustness.tools.vis_tools import show_image_row
from robustness.robustness.tools.label_maps import CLASS_DICT
from torchvision.utils import save_image
import pickle
import numpy as np
import csv
import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

model=torchvision.models.resnet50(pretrained=False)
from imagenet.resnet50 import resnet50
model=resnet50(pretrained=True)

ds = CIFAR('imagenet/data/')
device = 'cpu'

def load(file):
    dataset = np.load(file)
    print(dataset.shape)
    return dataset


def train(train_data,length, balance='balanced', fprN=False):
    #balance={0:5, 1:1}
    train_y = np.ones(len(train_data))
    train_y[:length] = 0
    train_data, train_y = shuffle(train_data, train_y, random_state=0)

    #clf = SVC(random_state=0, kernel='linear', class_weight=balance,probability=True)
    clf=LinearSVC(random_state=0, tol=1e-5,class_weight=balance)#
    clf.fit(train_data, train_y)

    y_pred = clf.decision_function(train_data)

    print("Roc Score")
    fpr, tpr, threshold = metrics.roc_curve(train_y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)

    print('Average Precision Score')
    print(average_precision_score(train_y, y_pred))


    if fprN:
        #for finding fpr at %...calculated manually
        y_pred =clf.decision_function(train_data)
        predict_mine = np.where(y_pred > 0.175, 1, 0)
        print(confusion_matrix(train_y, predict_mine))
        #sys.exit()




name = 'resnet50_2048_nat'
type = 'nat'

correct=load('imagenet/combined/correct_preds_'+str(name)+'.npy')
incorrect=load('imagenet/combined/incorrect_preds_'+str(name)+'.npy')
fgsm=load('imagenet/combined/fgsm_attacks_'+str(name)+'.npy')
sun=load('imagenet/combined/sun_'+str(name)+'.npy')
places=load('imagenet/combined/places_'+str(name)+'.npy')
corrupted=load('imagenet/combined/corrupted_'+str(name)+'.npy')
corrupted=corrupted[:len(correct)]

c_linf=load('imagenet/combined/carlini_linf_0.3'+str(name)+'.npy')
c_l2=load('imagenet/combined/carlini_l2'+str(name)+'.npy')
pgd=load('imagenet/combined/pgd0.3'+str(name)+'.npy')
a=load('imagenet/combined/a'+str(name)+'.npy')
o=load('imagenet/combined/o'+str(name)+'.npy')



msp=True
if msp:
    def score(train_data, length):
        train_y = np.ones(len(train_data))
        train_y[:length] = 0

        print("Roc Score")
        fpr, tpr, threshold = metrics.roc_curve(train_y, train_data)
        roc_auc = metrics.auc(fpr, tpr)
        print(roc_auc)

        print('Average Precision Score')
        print(average_precision_score(train_y, train_data))

    correct=correct[:,-1:]
    incorrect=incorrect[:,-1:]
    sun=sun[:,-1:]
    fgsm=fgsm[:,-1:]
    corrupted=corrupted[:,-1:]
    places=places[:,-1:]
    a=a[:,-1:]
    c_linf=c_linf[:,-1:]
    c_l2=c_l2[:,-1:]
    o = o[:, -1:]
    pgd=pgd[:,-1:]

    print('\ncorrect')
    print(np.mean(correct))
    print(np.var(correct))

    print('\nincorrect')
    print(np.mean(incorrect))
    print(np.var(incorrect))

    print('\nsun')
    print(np.mean(sun))
    print(np.var(sun))

    print('\nfgsm')
    print(np.mean(fgsm))
    print(np.var(fgsm))

    print('\ncorrupted')
    print(np.mean(corrupted))
    print(np.var(corrupted))

    print('\nplaces')
    print(np.mean(places))
    print(np.var(places))

    print('\na')
    print(np.mean(a))
    print(np.var(a))

    print('\nc_linf')
    print(np.mean(c_linf))
    print(np.var(c_linf))

    print('\nc_l2')
    print(np.mean(c_l2))
    print(np.var(c_l2))

    print('\npgd')
    print(np.mean(pgd))
    print(np.var(pgd))

    print('\no')
    print(np.mean(o))
    print(np.var(o))







