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


ds = CIFAR('imagenet/data/')
device = 'cpu'

def load(file):
    dataset = np.load(file)
    print(dataset.shape)
    return dataset


model=torchvision.models.resnet50(pretrained=False)
from imagenet.resnet50 import resnet50
model=resnet50(pretrained=True)



name = 'resnet50_2048_nat'
type = 'nat'

correct=load('imagenet/generated_test/correct_preds_'+str(name)+'.npy')
incorrect=load('imagenet/generated_test/incorrect_preds_'+str(name)+'.npy')
fgsm=load('imagenet/generated_test/fgsm_attacks_'+str(name)+'.npy')
sun=load('imagenet/generated_test/sun_'+str(name)+'.npy')
places=load('imagenet/generated_test/places_'+str(name)+'.npy')
corrupted=load('imagenet/generated_test/corrupted_softmax'+str(name)+'.npy')

cw_linf=load('imagenet/generated_test/carlini_linf_0.3'+str(name)+'.npy')
cw_l2=load('imagenet/generated_test/carlini_l2'+str(name)+'.npy')
pgd=load('imagenet/generated_test/pgd0.3'+str(name)+'.npy')

correct=correct[np.arange(np.shape(correct)[0])[:,np.newaxis], np.argsort(correct)]
incorrect=incorrect[np.arange(np.shape(incorrect)[0])[:,np.newaxis], np.argsort(incorrect)]
fgsm=fgsm[np.arange(np.shape(fgsm)[0])[:,np.newaxis], np.argsort(fgsm)]
sun=sun[np.arange(np.shape(sun)[0])[:,np.newaxis], np.argsort(sun)]
corrupted=corrupted[np.arange(np.shape(corrupted)[0])[:,np.newaxis], np.argsort(corrupted)]
places=places[np.arange(np.shape(places)[0])[:,np.newaxis], np.argsort(places)]

cw_linf=cw_linf[np.arange(np.shape(cw_linf)[0])[:,np.newaxis], np.argsort(cw_linf)]
cw_l2=cw_l2[np.arange(np.shape(cw_l2)[0])[:,np.newaxis], np.argsort(cw_l2)]
pgd=pgd[np.arange(np.shape(pgd)[0])[:,np.newaxis], np.argsort(pgd)]




correct=correct[:,-1]
incorrect=incorrect[:,-1]
fgsm=fgsm[:,-1]
sun=sun[:,-1]
corrupted=corrupted[:,-1]
places=places[:,-1]
cw_linf=cw_linf[:,-1]
cw_l2=cw_l2[:,-1]
pgd=pgd[:,-1]




print(np.mean(correct[:1000]))
print(np.mean(incorrect[:len(incorrect)]))
print(np.mean(sun[:len(incorrect)]))
print(np.mean(fgsm[:len(incorrect)]))
print(np.mean(cw_linf[:len(incorrect)]))
print(np.mean(cw_l2[:len(incorrect)]))
print(np.mean(pgd[:len(incorrect)]))
print(np.mean(corrupted[:len(incorrect)]))
print(np.mean(places[:len(incorrect)]))



print('var')
print(np.var(correct[:1000]))
print(np.var(incorrect[:len(incorrect)]))
print(np.var(sun[:len(incorrect)]))
print(np.var(fgsm[:len(incorrect)]))
print(np.var(cw_linf[:len(incorrect)]))
print(np.var(cw_l2[:len(incorrect)]))
print(np.var(pgd[:len(incorrect)]))
print(np.var(corrupted[:len(incorrect)]))
print(np.var(places[:len(incorrect)]))




