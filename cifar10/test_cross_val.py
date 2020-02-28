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

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)

    roc_total=[]
    prec_total=[]
    fpr95_total=[]
    fpr_results=True
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
            x = -10
            cont = True
            while cont:
                predict_mine = np.where(y_pred > x, 1, 0)
                cm = confusion_matrix(train_y, predict_mine)
                #print(cm)
                #print(x)
                if cm[1, 0] / cm[1, 1] > .0495 and cm[1, 0] / cm[1, 1] < .0595:
                    cont = False
                    #print('fpr95')
                    #print(cm)
                    fpr95_total.append((cm[0, 1] / cm[0, 0]))
                x += .01
                if x>10:
                    fpr_results = False
                    break


    #sys.exit()
    print('roc')
    print(np.mean(np.array(roc_total)))

    print('prec')
    print(np.mean(np.array(prec_total)))

    if not fpr_results:
        print('fpr')
        print(np.mean(np.array(fpr95_total)))

    print('roc var')
    print(np.std(np.array(roc_total)))

    print('prec var')
    print(np.std(np.array(prec_total)))

    if not fpr_results:
        print('fpr var')
        print(np.std(np.array(fpr95_total)))
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



#correct=correct.append()




latent=False
if latent:
    correct=correct[:,:2048]
    incorrect=incorrect[:,:2048]
    sun=sun[:,:2048]
    fgsm=fgsm[:,:2048]
    corrupted=corrupted[:,:2048]
    cifar100=cifar100[:,:2048]
    c_linf=c_linf[:,:2048]
    c_l2=c_l2[:,:2048]
    pgd=pgd[:,:2048]

softmax_vector=False
if softmax_vector:
    correct=correct[:,-10:]
    incorrect=incorrect[:,-10:]
    sun=sun[:,-10:]
    fgsm=fgsm[:,-10:]
    corrupted=corrupted[:,-10:]
    cifar100=cifar100[:,-10:]
    c_linf=c_linf[:,-10:]
    c_l2=c_l2[:,-10:]
    pgd=pgd[:,-10:]


train_all=True
fpr=True
if train_all:
    print('\nIncorrect')
    train_data = np.concatenate([[correct[i] for i in range(len(incorrect))], [incorrect[i] for i in range(len(incorrect))]])
    train(train_data, len(incorrect),fprN=fpr)

    print('\nSun')
    train_data = np.concatenate([[correct[i] for i in range(len(sun))], [sun[i] for i in range(len(sun))]])
    train(train_data, len(sun),fprN=fpr)

    print('\nFGSM')
    train_data = np.concatenate([[correct[i] for i in range(len(fgsm))], [fgsm[i] for i in range(len(fgsm))]])
    train(train_data, len(fgsm),fprN=fpr)

    print('\nCorrupted')
    train_data = np.concatenate([[correct[i] for i in range(len(corrupted))], [corrupted[i] for i in range(len(corrupted))]])
    train(train_data, len(corrupted),fprN=fpr)

    print('\ncifar100')
    train_data = np.concatenate([[correct[i] for i in range(len(cifar100))], [cifar100[i] for i in range(len(cifar100))]])
    train(train_data, len(cifar100),fprN=fpr)

    print('\npgd')
    train_data = np.concatenate([[correct[i] for i in range(len(pgd))], [pgd[i] for i in range(len(pgd))]])
    train(train_data, len(pgd),fprN=fpr)

    print('\nlinf')
    train_data = np.concatenate([[correct[i] for i in range(len(c_linf))], [c_linf[i] for i in range(len(c_linf))]])
    train(train_data, len(c_linf),fprN=fpr)

    print('\nl2')
    train_data = np.concatenate([[correct[i] for i in range(len(c_l2))], [c_l2[i] for i in range(len(c_l2))]])
    train(train_data, len(c_l2),fprN=fpr)




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
    cifar100=-cifar100[:,-1:]
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

    print('\ncifar100')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(cifar100))],[cifar100[i]  for i in range(len(cifar100))]])
    score(train_data, len(cifar100))

    print('\nc_linf')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(c_linf))],[c_linf[i]  for i in range(len(c_linf))]])
    score(train_data, len(c_linf))

    print('\nc_l2')
    train_data = np.concatenate([ [correct1[i]  for i in range(len(c_l2))],[c_l2[i]  for i in range(len(c_l2))]])
    score(train_data, len(c_l2))

    print('\npgd')
    train_data = np.concatenate([ [correct2[i]  for i in range(len(pgd))],[pgd[i]  for i in range(len(pgd))]])
    score(train_data, len(pgd))




