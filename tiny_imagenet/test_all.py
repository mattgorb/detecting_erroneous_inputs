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
        x=-1
        cont=True
        while cont:
            predict_mine = np.where(y_pred > x, 1, 0)
            cm=confusion_matrix(train_y, predict_mine)
            if cm[1,0]/cm[1,1]>.0495 and cm[1,0]/cm[1,1]<.0595:
                cont=False
                print('fpr95')
                print(cm[0,1]/cm[0,0])
            #print(x)
            #print(cm[1,0]/cm[1,1])
            x+=.01



name = 'wideresnet'

model = WideResNet(40, 200, 2, dropRate=0.3)

network_state_dict = torch.load('tiny_imagenet/wrn_baseline_epoch_99.pt',map_location='cpu' )
model.load_state_dict(network_state_dict)
model.eval()




correct=load('tiny_imagenet/combined/correct_preds_'+str(name)+'.npy')
save=False
if save:
    names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    for i in names:
        incorrect = load('tiny_imagenet/combined/incorrect_preds_' + str(name) + '.npy')
        sun = load('tiny_imagenet/combined/sun' + str(name) + '.npy')
        fgsm = load('tiny_imagenet/combined/fgsm_' + str(name) + '.npy')
        places = load('tiny_imagenet/combined/places365' + str(name) + '.npy')
        corrupted = load('tiny_imagenet/combined/corrupted_softmax' + str(name) + '.npy')

        c_linf = load('tiny_imagenet/combined/carlini_linf_0.3' + str(name) + '.npy')
        c_l2 = load('tiny_imagenet/combined/carlini_l2' + str(name) + '.npy')
        pgd = load('tiny_imagenet/combined/pgd0.3' + str(name) + '.npy')
        incorrect = np.concatenate([incorrect, sun, corrupted, fgsm, places, c_linf, c_l2])
        np.random.shuffle(incorrect)
        np.save('tiny_imagenet/all_random/'+str(i)+'.npy', np.array(incorrect))
    sys.exit()

else:
    names=[1,2,3,4,5,6,7,8,9,0]
    for i in names:
        incorrect=load('tiny_imagenet/all_random/'+str(i)+'.npy')
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [incorrect[i] for i in range(int(len(correct)/5))]])
        train(train_data, len(correct),fprN=True)



