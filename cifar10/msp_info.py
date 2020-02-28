# from robustness import model_utils, datasets, train, defaults
# from robustness.datasets import CIFAR
import torch as ch
import dill
from cox.utils import Parameters
import cox.store
import numpy as np
from sklearn.svm import SVC
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from utils.utils import *
import sys
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

ds = CIFAR('cifar10/data/')
device = 'cpu'

def loadNPY(file):
    dataset = np.load(file)
    print(dataset.shape)
    return np.reshape(dataset, (dataset.shape[0], dataset.shape[1]))



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
        predict_mine = np.where(y_pred > -0.44, 1, 0)
        print(confusion_matrix(train_y, predict_mine))
        #sys.exit()




model, _ = make_and_restore_model(arch='resnet50', dataset=ds, device=device, resume_path='cifar10/cifar_nat.pt')

name = 'resnet50_2058'
type = 'nat'

correct = loadNPY('cifar10/generated_test/correct_preds_'+str(name)+'_'+str(type)+'.npy')
incorrect = loadNPY('cifar10/generated_test/incorrect_preds_'+str(name)+'_'+str(type)+'.npy')
cifar100 = loadNPY('cifar10/generated_test/unseen_'+str(name)+'_'+str(type)+'.npy')
corrupted = loadNPY('cifar10/generated_test/corrupted_'+str(name)+'_'+str(type)+'.npy')
sun = loadNPY('cifar10/generated_test/sun_'+str(name)+'_'+str(type)+'.npy')
sun=sun[:len(correct)]
fgsm  = loadNPY('cifar10/generated_test/fgsm_attacks_'+str(name)+'_'+str(type)+'.npy')

c_linf  = loadNPY('cifar10/generated_test/carlini_linf_0.3'+str(name)+'_'+str(type)+'.npy')
c_l2=loadNPY('cifar10/generated_test/carlini_l2'+str(name)+'_'+str(type)+'.npy')
pgd  = loadNPY('cifar10/generated_test/pgd0.3'+str(name)+'_'+str(type)+'.npy')





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
    cifar100=cifar100[:,-1:]
    c_linf=c_linf[:,-1:]
    c_l2=c_l2[:,-1:]

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

    print('\ncifar100')
    print(np.mean(cifar100))
    print(np.var(cifar100))

    print('\nc_linf')
    print(np.mean(c_linf))
    print(np.var(c_linf))

    print('\nc_l2')
    print(np.mean(c_l2))
    print(np.var(c_l2))

    print('\npgd')
    print(np.mean(pgd))
    print(np.var(pgd))




