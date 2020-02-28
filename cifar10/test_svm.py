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



def train(train_data,length, labels=None,savefig=None,balance='balanced', fprN=False, svm=False, graph=False):
    #balance={0:5, 1:1}
    train_y = np.ones(len(train_data))
    train_y[:length] = 0
    train_data, train_y = shuffle(train_data, train_y, random_state=0)

    #clf = SVC(random_state=0, kernel='rbf', class_weight=balance,probability=True)
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
        predict_mine = np.where(y_pred > 0, 1, 0)
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


balance51=False
if balance51:
    correct=correct[:8000]
    #incorrect=incorrect[:723]
    sun=sun[:1600]
    fgsm=fgsm[:1600]
    corrupted=corrupted[:1600]
    cifar100=cifar100[:1600]
    c_linf=c_linf[:1600]
    c_l2=c_l2[:1600]
    pgd=pgd[:1600]

train_all=True
fpr_n=False
svm=False
graph=False
balanced=True
if train_all:
    print('\nIncorrect')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(incorrect))], [incorrect[i] for i in range(len(incorrect))]])
        train(train_data, len(incorrect),['Correct Prediction', 'Misclassified'],savefig='cifar10/normal_graphs/incorrect.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [incorrect[i] for i in range(len(incorrect))]])
        train(train_data, len(correct),['Correct Prediction', 'Misclassified'],savefig='cifar10/normal_graphs/incorrect.png',fprN=fpr_n, svm=svm, graph=graph)

    print('\nSun')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(sun))], [sun[i] for i in range(len(sun))]])
        train(train_data, len(sun),['Correct Prediction', 'SUN'],savefig='cifar10/normal_graphs/sun.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [sun[i] for i in range(len(sun))]])
        train(train_data, len(correct),['Correct Prediction', 'SUN'],savefig='cifar10/normal_graphs/sun.png',fprN=fpr_n, svm=svm, graph=graph)

    print('\nFGSM')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(fgsm))], [fgsm[i] for i in range(len(fgsm))]])
        train(train_data, len(fgsm),['Correct Prediction', 'FGSM'],savefig='cifar10/normal_graphs/fgsm.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [fgsm[i] for i in range(len(fgsm))]])
        train(train_data, len(correct), ['Correct Prediction', 'FGSM'], savefig='cifar10/normal_graphs/fgsm.png', fprN=fpr_n,
          svm=svm, graph=graph)

    print('\nCorrupted')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(corrupted))], [corrupted[i] for i in range(len(corrupted))]])
        train(train_data, len(corrupted),['Correct Prediction', 'CIFAR10-C'],savefig='cifar10/normal_graphs/corrupted.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate(
        [[correct[i] for i in range(len(correct))], [corrupted[i] for i in range(len(corrupted))]])
        train(train_data, len(correct), ['Correct Prediction', 'CIFAR10-C'], savefig='cifar10/normal_graphs/corrupted.png',
          fprN=fpr_n, svm=svm, graph=graph)

    print('\ncifar100')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(cifar100))], [cifar100[i] for i in range(len(cifar100))]])
        train(train_data, len(cifar100),['Correct Prediction', 'CIFAR100'],savefig='cifar10/normal_graphs/cifar100.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [cifar100[i] for i in range(len(cifar100))]])
        train(train_data, len(correct), ['Correct Prediction', 'CIFAR100'], savefig='cifar10/normal_graphs/cifar100.png',
          fprN=fpr_n, svm=svm, graph=graph)

    print('\npgd')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(pgd))], [pgd[i] for i in range(len(pgd))]])
        train(train_data, len(pgd),['Correct Prediction', 'PGD'],savefig='cifar10/normal_graphs/pgd.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [pgd[i] for i in range(len(pgd))]])
        train(train_data, len(correct), ['Correct Prediction', 'PGD'], savefig='cifar10/normal_graphs/pgd.png', fprN=fpr_n, svm=svm,
          graph=graph)

    print('\nlinf')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(c_linf))], [c_linf[i] for i in range(len(c_linf))]])
        train(train_data, len(c_linf),['Correct Prediction', 'C & W L-Inf'],savefig='cifar10/normal_graphs/clinf.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [c_linf[i] for i in range(len(c_linf))]])
        train(train_data, len(correct), ['Correct Prediction', 'C & W L-Inf'], savefig='cifar10/normal_graphs/clinf.png',
          fprN=fpr_n, svm=svm, graph=graph)

    print('\nl2')
    if balanced:
        train_data = np.concatenate([[correct[i] for i in range(len(c_l2))], [c_l2[i] for i in range(len(c_l2))]])
        train(train_data, len(c_l2),['Correct Prediction', 'C & W L2'],savefig='cifar10/normal_graphs/cl2.png',fprN=fpr_n, svm=svm, graph=graph)
    else:
        train_data = np.concatenate([[correct[i] for i in range(len(correct))], [c_l2[i] for i in range(len(c_l2))]])
        train(train_data, len(correct), ['Correct Prediction', 'C & W L2'], savefig='cifar10/normal_graphs/cl2.png', fprN=fpr_n,
          svm=svm, graph=graph)





