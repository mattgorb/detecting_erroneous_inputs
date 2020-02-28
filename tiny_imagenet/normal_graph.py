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


def load_tinyimagenet():

    train_transform = trn.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
    train_data = datasets.ImageFolder(
        root="tiny_imagenet/tiny-imagenet-200/train",
        transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=True)


    return train_loader


def load(file):
    dataset = np.load(file)
    print(dataset.shape)
    return dataset


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








def graph(non_learned, labels,fig_name):
    train_data = np.concatenate([[(correct[i], 3) for i in range(len(correct))], non_learned])
    train_y = np.ones(len(train_data))
    train_y[:len(correct)] = 0

    train_data, train_y = shuffle(train_data, train_y, random_state=0)


    #X_train, X_test, y_train, y_test = train_test_split(train_data, train_y, test_size=0.3, random_state=109)
    X_train=train_data
    X_test=train_data
    y_train=train_y
    y_test=train_y

    clf = LinearSVC(random_state=0, tol=1e-5, class_weight='balanced')  #
    clf.fit([X_train[i][0] for i in range(len(X_train))], y_train)




    y_pred = clf.predict([X_test[i][0] for i in range(len(X_test))])


    print("Roc Score")
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)

    print('Average Precision Score')
    print(average_precision_score(y_test, y_pred))


    print('Confusion Matrix')
    confusion_matrix(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))




    coef = np.reshape(clf.coef_, (clf.coef_.shape[1], clf.coef_.shape[0]))
    test_values = []
    for h in range(len(X_test)):
        test_values.append((X_test[h][0].dot(coef) + clf.intercept_, X_test[h][1]))




    i = [test_values[i][0] for i in range(len(test_values)) if y_test[i] == 1 and test_values[i][1] == 0]
    l2 = [test_values[i][0] for i in range(len(test_values)) if y_test[i] == 1 and test_values[i][1] == 1]
    #p = [test_values[i][0] for i in range(len(test_values)) if y_test[i] == 1 and test_values[i][1] == 2]
    c = [test_values[i][0] for i in range(len(test_values)) if y_test[i] == 0 and test_values[i][1] == 3]


    '''i=np.sort(i, axis=0)
    l2=np.sort(l2, axis=0)
    p=np.sort(p, axis=0)
    c=np.sort(c, axis=0)

    print(c.shape)'''

    #plt.title('SVM Normal Vector of 4000 Data Points')
    plt.plot([j for j in range(250)], i[:250], 'x',  label=labels[1])
    plt.plot([j for j in range(250)], l2[:250], '+', label=labels[2])
    #plt.plot([j for j in range(4000)], p[:4000], '1', label=labels[2])
    plt.plot( [j for j in range(250)],c[:250], '1',  label=labels[0])
    #plt.plot([j for j in range(500)], [0 for i in range(500)], '--', color='black')
    plt.axhline(0, color="gray")
    plt.xlabel('Sample Number')
    plt.ylabel('Mapped SVM')

    #plt.legend(bbox_to_anchor=(0.78, .95), framealpha=1, frameon=True,prop={'size': 12})
    plt.legend(loc='lower right', framealpha=1, frameon=True,prop={'size': 9})
    plt.savefig(fig_name)
    plt.xticks([])
    #plt.set_xticks([])
    #plt.show()
    #sys.exit()

#all models only trained with linear
non_learned=np.concatenate([
            [(fgsm[i],0) for i in range(len(fgsm))],
            [(c_l2[i],1) for i in range(len(c_l2))],
            #[(c_linf[i],1) for i in range(len(c_linf))]
])
labels=['Correct Prediction', 'FGSM', 'C & W L Inf','C & W L Inf']
graph(non_learned,labels,'tiny_imagenet/normal_adversarial_for_fig.png')

plt.clf()

non_learned=np.concatenate([
            [(corrupted[i],0) for i in range(len(corrupted))],
            [(sun[i],1) for i in range(len(sun))],
            #[(c_linf[i],2) for i in range(len(c_linf))]
])
labels=['Correct Prediction', 'Tiny ImageNet-C', 'SUN','']
graph(non_learned,labels,'tiny_imagenet/normal_ood_corrupted.png')