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

generate_normal_graph = True
train = True


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




def train_predict(train_data, train_y ):
    for i in range(10):
        train_data, train_y = shuffle(train_data, train_y, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_y, test_size=0.3, random_state=109)

    clf = LinearSVC(random_state=0, tol=1e-5, class_weight='balanced')  #
    clf.fit([X_train[i] for i in range(len(X_train))], y_train)
    y_pred = clf.decision_function(X_test)
    return y_test,y_pred



train_data = np.concatenate([[correct[i] for i in range(len(incorrect))], [incorrect[i] for i in range(len(incorrect))]])
train_y = np.ones(len(train_data))
train_y[:len(incorrect)] = 0
incorrect_test,incorrect_pred =train_predict(train_data, train_y)

fpr1, tpr1, threshold1 = metrics.roc_curve(incorrect_test, incorrect_pred)
roc_auc1 = metrics.auc(fpr1, tpr1)


train_data = np.concatenate([[correct[i] for i in range(len(cifar100))], [cifar100[i] for i in range(len(cifar100))]])
train_y = np.ones(len(train_data))
train_y[:len(cifar100)] = 0
cifar100_test,cifar100_pred=train_predict(train_data, train_y)
fpr2, tpr2, threshold2 = metrics.roc_curve(cifar100_test,cifar100_pred)
roc_auc2 = metrics.auc(fpr2, tpr2)


train_data = np.concatenate([[correct[i] for i in range(len(fgsm))], [fgsm[i] for i in range(len(fgsm))]])
train_y = np.ones(len(train_data))
train_y[:len(fgsm)] = 0
test, pred=train_predict(train_data, train_y)
fpr3, tpr3, threshold = metrics.roc_curve(test, pred)
roc_auc3 = metrics.auc(fpr3, tpr3)


train_data = np.concatenate([[correct[i] for i in range(len(c_linf))], [c_linf[i] for i in range(len(c_linf))]])
train_y = np.ones(len(train_data))
train_y[:len(c_linf)] = 0
test, pred=train_predict(train_data, train_y)
fpr4, tpr4, threshold = metrics.roc_curve(test, pred)
roc_auc4 = metrics.auc(fpr4, tpr4)

train_data = np.concatenate([[correct[i] for i in range(len(corrupted))], [corrupted[i] for i in range(len(corrupted))]])
train_y = np.ones(len(train_data))
train_y[:len(corrupted)] = 0
test, pred=train_predict(train_data, train_y)
fpr4, tpr4, threshold = metrics.roc_curve(test, pred)
roc_auc4 = metrics.auc(fpr4, tpr4)

train_data = np.concatenate([[correct[i] for i in range(len(sun))], [sun[i] for i in range(len(sun))]])
train_y = np.ones(len(train_data))
train_y[:len(sun)] = 0
test, pred=train_predict(train_data, train_y)
fpr5, tpr5, threshold = metrics.roc_curve(test, pred)
roc_auc5 = metrics.auc(fpr5, tpr5)

plt.plot(fpr1, tpr1, 'b',marker='.', color='red',markevery=5, label='Misclassified = %0.2f' % roc_auc1)
plt.plot(fpr2, tpr2, 'b',marker='x', color='blue', markevery=10, label='CIFAR100 = %0.2f' % roc_auc2)
#plt.plot(fpr3, tpr3, 'b',marker='+', color='green',markevery=5,  label='FGSM = %0.2f' % roc_auc3)
plt.plot(fpr4, tpr4, 'b',marker='1', color='purple',markevery=5,  label='CIFAR10-C = %0.2f' % roc_auc4)
plt.plot(fpr5, tpr5, 'b',marker='+', color='orange',markevery=30,  label='SUN = %0.2f' % roc_auc5)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('cifar10/roc.png')
plt.show()