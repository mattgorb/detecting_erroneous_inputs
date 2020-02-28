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

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)

    roc_total=[]
    prec_total=[]
    fpr95_total=[]
    fpr_results=True
    for train_index, test_index in kf.split(train_data):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]

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
            x = -10
            cont = True
            while cont:
                predict_mine = np.where(y_pred > x, 1, 0)
                cm = confusion_matrix(train_y, predict_mine)
                #print(cm)
                #print(cm[1, 0] / cm[1, 1])
                if cm[1, 0] / cm[1, 1] > .0495 and cm[1, 0] / cm[1, 1] < .0595:
                    cont = False
                    #print('fpr95')
                    #print(cm)
                    fpr95_total.append((cm[0, 1] / cm[0, 0]))
                x += .01
                #if x>10:
                    #fpr_results = False
                    #break


    #sys.exit()
    print('roc')
    print(np.mean(np.array(roc_total)))

    print('prec')
    print(np.mean(np.array(prec_total)))

    #if not fpr_results:
    print('fpr')
    print(np.mean(np.array(fpr95_total)))

    print('roc var')
    print(np.var(np.array(roc_total)))

    print('prec var')
    print(np.var(np.array(prec_total)))

    #if not fpr_results:
    print('fpr var')
    print(np.var(np.array(fpr95_total)))
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


latent=False
if latent:
    correct=correct[:,:2048]
    incorrect=incorrect[:,:2048]
    sun=sun[:,:2048]
    fgsm=fgsm[:,:2048]
    corrupted=corrupted[:,:2048]
    places=places[:,:2048]
    a=a[:,:2048]
    o=o[:,:2048]
    c_linf=c_linf[:,:2048]
    c_l2=c_l2[:,:2048]
    pgd=pgd[:,:2048]

softmax_vector=False
if softmax_vector:
    correct=correct[:,-1000:]
    incorrect=incorrect[:,-1000:]
    sun=sun[:,-1000:]
    fgsm=fgsm[:,-1000:]
    corrupted=corrupted[:,-1000:]
    places=places[:,-1000:]
    a=a[:,-1000:]
    o=o[:,-1000:]
    c_linf=c_linf[:,-1000:]
    c_l2=c_l2[:,-1000:]
    pgd=pgd[:,-1000:]

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
    train(train_data, len(places),fprN=fpr_n)

    print('\nA')
    train_data = np.concatenate([[correct[i] for i in range(len(a))], [a[i] for i in range(len(a))]])
    train(train_data, len(a),fprN=fpr_n)

    print('\no')
    train_data = np.concatenate([[correct[i] for i in range(len(o))], [o[i] for i in range(len(o))]])
    train(train_data, len(o),fprN=fpr_n)'''

    #print('\npgd')
    #train_data = np.concatenate([[correct[i] for i in range(len(pgd))], [pgd[i] for i in range(len(pgd))]])
    #train(train_data, len(pgd),fprN=fpr_n)

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
    a=-a[:,-1:]
    c_linf=-c_linf[:,-1:]
    c_l2=-c_l2[:,-1:]

    correct2=correct[:,-1:]
    o = o[:, -1:]
    pgd=pgd[:,-1:]

    '''print('\nincorrect')
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

    print('\na')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(a))],[a[i]  for i in range(len(a))]])
    score(train_data, len(a))'''

    print('\nc_linf')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(c_linf))],[c_linf[i]  for i in range(len(c_linf))]])
    score(train_data, len(c_linf))

    print('\nc_l2')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(c_l2))],[c_l2[i]  for i in range(len(c_l2))]])
    score(train_data, len(c_l2))

    print('\npgd')
    train_data = np.concatenate([ [correct2[i]  for i in range(len(pgd))],[pgd[i]  for i in range(len(pgd))]])
    score(train_data, len(pgd))

    print('\no')
    train_data = np.concatenate([ [correct2[i]  for i in range(len(o))],[o[i]  for i in range(len(o))]])
    score(train_data, len(o))







