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

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)

    roc_total=[]
    prec_total=[]
    fpr95_total=[]
    for train_index, test_index in kf.split(train_data):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]

        #print(X_test.shape)
        #print(X_train.shape)


        clf=LinearSVC(random_state=0, tol=1e-5,class_weight=balance)#
        clf.fit(X_train, y_train)

        y_pred = clf.decision_function(X_test)

        #print("Roc Score")
        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        roc_total.append(roc_auc)
        #print(roc_auc)

        #print('Average Precision Score')
        #print(average_precision_score(y_test, y_pred))
        prec=average_precision_score(y_test, y_pred)
        prec_total.append(prec)

        if fprN:
            # for finding fpr at %...calculated manually
            y_pred = clf.decision_function(train_data)
            x = -1
            cont = True
            while cont:
                predict_mine = np.where(y_pred > x, 1, 0)
                cm = confusion_matrix(train_y, predict_mine)

                if cm[1, 0] / cm[1, 1] > .0495 and cm[1, 0] / cm[1, 1] < .0595:
                    cont = False
                    #print('fpr95')
                    #print(cm)
                    fpr95_total.append((cm[0, 1] / cm[0, 0]))
                x += .01

    #sys.exit()
    print('roc')
    print(np.mean(np.array(roc_total)))

    print('prec')
    print(np.mean(np.array(prec_total)))

    print('fpr')
    print(np.mean(np.array(fpr95_total)))

    print('roc var')
    print(np.var(np.array(roc_total)))

    print('prec var')
    print(np.var(np.array(prec_total)))

    print('fpr var')
    print(np.var(np.array(fpr95_total)))
    #sys.exit()



name = 'wideresnet'

model = WideResNet(40, 200, 2, dropRate=0.3)

network_state_dict = torch.load('tiny_imagenet/wrn_baseline_epoch_99.pt',map_location='cpu' )
model.load_state_dict(network_state_dict)
model.eval()



correct=load('tiny_imagenet/combined/correct_preds_'+str(name)+'.npy')
incorrect=load('tiny_imagenet/combined/incorrect_preds_'+str(name)+'.npy')
sun=load('tiny_imagenet/combined/sun'+str(name)+'.npy')
fgsm  = load('tiny_imagenet/combined/fgsm_'+str(name)+'.npy')
places  = load('tiny_imagenet/combined/places365'+str(name)+'.npy')
corrupted  = load('tiny_imagenet/combined/corrupted_softmax'+str(name)+'.npy')

c_linf  = load('tiny_imagenet/combined/carlini_linf_0.3'+str(name)+'.npy')
c_l2  = load('tiny_imagenet/combined/carlini_l2'+str(name)+'.npy')
pgd  = load('tiny_imagenet/combined/pgd0.3'+str(name)+'.npy')



latent=False
if latent:
    correct=correct[:,:128]
    incorrect=incorrect[:,:128]
    sun=sun[:,:128]
    fgsm=fgsm[:,:128]
    corrupted=corrupted[:,:128]
    places=places[:,:128]
    c_linf=c_linf[:,:128]
    c_l2=c_l2[:,:128]
    pgd=pgd[:,:128]

softmax_vector=False
if softmax_vector:
    correct=correct[:,-200:]
    incorrect=incorrect[:,-200:]
    sun=sun[:,-200:]
    fgsm=fgsm[:,-200:]
    corrupted=corrupted[:,-200:]
    places=places[:,-200:]
    c_linf=c_linf[:,-200:]
    c_l2=c_l2[:,-200:]
    pgd=pgd[:,-200:]


train_all=True
fpr_n=True
if train_all:
    '''print('\nIncorrect')
    train_data = np.concatenate([[correct[i] for i in range(len(incorrect))], [incorrect[i] for i in range(len(incorrect))]])
    train(train_data, len(incorrect),fprN=fpr_n)

    print('\nSun')
    train_data = np.concatenate([[correct[i] for i in range(len(sun))], [sun[i] for i in range(len(sun))]])
    train(train_data, len(sun),fprN=fpr_n)


    print('\nFGSM')
    train_data = np.concatenate([[correct[i] for i in range(len(fgsm))], [fgsm[i] for i in range(len(fgsm))]])
    train(train_data, len(fgsm),fprN=fpr_n)

    print('\nCorrupted')
    train_data = np.concatenate([[correct[i] for i in range(len(corrupted))], [corrupted[i] for i in range(len(corrupted))]])
    train(train_data, len(corrupted),fprN=fpr_n)

    print('\nPlaces')
    train_data = np.concatenate([[correct[i] for i in range(len(places))], [places[i] for i in range(len(places))]])
    train(train_data, len(places),fprN=fpr_n)'''

    #sys.exit()
    print('\npgd')
    train_data = np.concatenate([[correct[i] for i in range(len(pgd))], [pgd[i] for i in range(len(pgd))]])
    train(train_data, len(pgd),fprN=fpr_n)

    print('\nlinf')
    train_data = np.concatenate([[correct[i] for i in range(len(c_linf))], [c_linf[i] for i in range(len(c_linf))]])
    train(train_data, len(c_linf),fprN=fpr_n)

    print('\nl2')
    train_data = np.concatenate([[correct[i] for i in range(len(c_l2))], [c_l2[i] for i in range(len(c_l2))]])
    train(train_data, len(c_l2),fprN=fpr_n)




msp=False
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

    correct1=-correct[:,-1:]
    incorrect=-incorrect[:,-1:]
    sun=-sun[:,-1:]
    fgsm=-fgsm[:,-1:]
    corrupted=-corrupted[:,-1:]
    places=-places[:,-1:]
    c_linf=-c_linf[:,-1:]
    c_l2=-c_l2[:,-1:]

    correct2=correct[:,-1:]
    pgd=pgd[:,-1:]

    print('\nincorrect')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(incorrect))],[incorrect[i]  for i in range(len(incorrect))]])
    score(train_data, len(incorrect))

    print('\nsun')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(sun))],[sun[i]  for i in range(len(sun))]])
    score(train_data, len(sun))

    print('\nfgsm')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(fgsm))],[fgsm[i]  for i in range(len(fgsm))]])
    score(train_data, len(fgsm))

    print('\ncorrupted')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(corrupted))],[corrupted[i]  for i in range(len(corrupted))]])
    score(train_data, len(corrupted))

    print('\nplaces')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(places))],[places[i]  for i in range(len(places))]])
    score(train_data, len(places))

    print('\nc_linf')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(c_linf))],[c_linf[i]  for i in range(len(c_linf))]])
    score(train_data, len(c_linf))

    print('\nc_l2')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(c_l2))],[c_l2[i]  for i in range(len(c_l2))]])
    score(train_data, len(c_l2))

    print('\npgd')
    train_data = np.concatenate([ [correct2[i]  for i in range(len(pgd))],[pgd[i]  for i in range(len(pgd))]])
    score(train_data, len(pgd))

