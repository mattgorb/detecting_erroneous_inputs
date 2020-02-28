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


def train(train_data,length, labels=None,savefig=None,balance='balanced', fprN=False, svm=False, graph=False):
    #balance={0:5, 1:1}
    train_y = np.ones(len(train_data))
    train_y[:length] = 0
    train_data, train_y = shuffle(train_data, train_y, random_state=0)

    #clf = SVC(random_state=0, kernel='linear', class_weight=balance,probability=True)
    clf=LinearSVC(random_state=0, tol=1e-5,class_weight=balance)#
    clf.fit(train_data, train_y)

    if svm:
        y_pred = clf.predict(train_data)
    else:
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
        predict_mine = np.where(y_pred > -.47, 1, 0)
        print(confusion_matrix(train_y, predict_mine))

    if graph:

        coef = np.reshape(clf.coef_, (clf.coef_.shape[1], clf.coef_.shape[0]))
        test_values = []
        for h in range(len(y_pred)):
            test_values.append(train_data[h].dot(coef) + clf.intercept_)

        i = [test_values[i] for i in range(len(test_values)) if train_y[i] == 1 ]
        c = [test_values[i] for i in range(len(test_values)) if train_y[i] == 0 ]


        plt.clf()
        # plt.title('SVM Normal Vector of 4000 Data Points')
        plt.plot([j for j in range(350)], c[:350], 'x', label=labels[0])
        plt.plot([j for j in range(350)], i[:350], '+', label=labels[1])

        plt.axhline(0, color="gray")
        plt.ylabel('Mapped SVM')
        plt.xlabel('Sample Number')

        plt.legend(bbox_to_anchor=(0.78, .95), framealpha=1, frameon=True, prop={'size': 7})
        # plt.legend(loc='lower right', framealpha=1, frameon=True,prop={'size': 7})
        plt.savefig(savefig)
        #plt.show()



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




balance51=True
if balance51:
    correct=correct[:20000]
    incorrect=incorrect[:4000]
    sun=sun[:4000]
    fgsm=fgsm[:4000]
    corrupted=corrupted[:4000]
    places=places[:4000]
    correctc_linf=c_linf[:4000]
    c_l2=c_l2[:4000]
    pgd=pgd[:4000]

train_all=True
fpr_n=False
svm=False
graph=True
balanced=False
if train_all:
    print('\nIncorrect')

    if balanced:
        train_data = np.concatenate(
            [[correct[i] for i in range(len(incorrect))], [incorrect[i] for i in range(len(incorrect))]])
        train(train_data, len(incorrect),['Correct Prediction', 'Misclassified'],savefig='tiny_imagenet/normal_graphs/incorrect.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate(
            [[correct[i] for i in range(len(correct))], [incorrect[i] for i in range(len(incorrect))]])
        train(train_data, len(correct),['Correct Prediction', 'Misclassified'],savefig='tiny_imagenet/normal_graphs/incorrect.png',fprN=fpr_n, svm=svm, graph=graph)

    print('\nSun')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(sun))], [sun[i] for i in range(len(sun))]])
        train(train_data, len(sun),['Correct Prediction', 'Sun'],savefig='tiny_imagenet/normal_graphs/sun.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [sun[i] for i in range(len(sun))]])
        train(train_data, len(correct),['Correct Prediction', 'Sun'],savefig='tiny_imagenet/normal_graphs/sun.png',fprN=fpr_n, svm=svm, graph=graph)

    print('\nFGSM')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(fgsm))], [fgsm[i] for i in range(len(fgsm))]])
        train(train_data, len(fgsm),['Correct Prediction', 'FGSM'],savefig='tiny_imagenet/normal_graphs/fgsm.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [fgsm[i] for i in range(len(fgsm))]])
        train(train_data, len(correct),['Correct Prediction', 'FGSM'],savefig='tiny_imagenet/normal_graphs/fgsm.png',fprN=fpr_n, svm=svm, graph=graph)


    print('\nCorrupted')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(corrupted))], [corrupted[i] for i in range(len(corrupted))]])
        train(train_data, len(corrupted),['Correct Prediction', 'Tiny ImageNet-C'],savefig='tiny_imagenet/normal_graphs/corrupted.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [corrupted[i] for i in range(len(corrupted))]])
        train(train_data, len(correct),['Correct Prediction', 'Tiny ImageNet-C'],savefig='tiny_imagenet/normal_graphs/corrupted.png',fprN=fpr_n, svm=svm, graph=graph)

    print('\nPlaces')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(places))], [places[i] for i in range(len(places))]])
        train(train_data, len(places),['Correct Prediction', 'Places'],savefig='tiny_imagenet/normal_graphs/places.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [places[i] for i in range(len(places))]])
        train(train_data, len(correct),['Correct Prediction', 'Places'],savefig='tiny_imagenet/normal_graphs/places.png',fprN=fpr_n, svm=svm, graph=graph)

    #sys.exit()
    print('\npgd')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(pgd))], [pgd[i] for i in range(len(pgd))]])
        train(train_data, len(pgd),['Correct Prediction', 'PGD'],savefig='tiny_imagenet/normal_graphs/pgd.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [pgd[i] for i in range(len(pgd))]])
        train(train_data, len(correct),['Correct Prediction', 'PGD'],savefig='tiny_imagenet/normal_graphs/pgd.png',fprN=fpr_n, svm=svm, graph=graph)

    print('\nlinf')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(c_linf))], [c_linf[i] for i in range(len(c_linf))]])
        train(train_data, len(c_linf),['Correct Prediction', 'C & W L-Inf'],savefig='tiny_imagenet/normal_graphs/linf.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [c_linf[i] for i in range(len(c_linf))]])
        train(train_data, len(correct), ['Correct Prediction', 'C & W L-Inf'], savefig='tiny_imagenet/normal_graphs/linf.png',
          fprN=fpr_n, svm=svm, graph=graph)

    print('\nl2')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(c_l2))], [c_l2[i] for i in range(len(c_l2))]])
        train(train_data, len(c_l2),['Correct Prediction', 'C & W L2'],savefig='tiny_imagenet/normal_graphs/l2.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [c_l2[i] for i in range(len(c_l2))]])
        train(train_data, len(correct), ['Correct Prediction', 'C & W L2'], savefig='tiny_imagenet/normal_graphs/l2.png',
        fprN=fpr_n, svm=svm, graph=graph)


