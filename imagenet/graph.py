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
import csv
import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


ds = CIFAR('imagenet/data/')
device = 'cpu'

def load(file):
    dataset = np.load(file)
    return dataset


model=torchvision.models.resnet50(pretrained=False)
from imagenet.resnet50 import resnet50
model=resnet50(pretrained=True)



name = 'gradient_data'
type = 'nat'



name = 'resnet50_2048'
type = 'nat'


correct=load('imagenet/generated_test/correct_preds_all'+str(name)+'_.npy')
incorrect=load('imagenet/generated_test/incorrect_preds_all'+str(name)+'_.npy')

correct=correct[-1,correct.argsort(axis=1)[:,:]]
incorrect=incorrect[-1,incorrect.argsort(axis=1)[:,:]]


correct=np.mean(correct, axis=0)
incorrect=np.mean(incorrect, axis=0)


plt.plot([i for i in range(len(correct))], [correct[i] for i in range(len(correct))] , 'x',color='royalblue',  label='Correct Prediction')
plt.plot([i for i in range(len(incorrect))], [incorrect[i] for i in range(len(incorrect))] , 'x', color='indianred', label='Incorrect Prediction',markevery=10)

#plt.plot([i for i in range(len(sun))], [incorrect[i] for i in range(len(sun))] , 'x', color='orange', label='Sun',markevery=11)
#plt.plot([i for i in range(len(cifar100))], [incorrect[i] for i in range(len(cifar100))] , 'x', color='purple', label='Cifar100',markevery=11)

plt.legend(loc='upper right', framealpha=1, frameon=True)
plt.xlabel('Index of Sorted Feature Vector (ResNet50)')
plt.ylabel('Values of Feature Vector')
#plt.savefig('cifar10/sum_var_incorrect.png')
plt.show()
plt.clf()
