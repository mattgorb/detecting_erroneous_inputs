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

correct=load('imagenet/combined/correct_preds_'+str(name)+'.npy')
incorrect=load('imagenet/combined/incorrect_preds_'+str(name)+'.npy')
fgsm=load('imagenet/combined/fgsm_attacks_'+str(name)+'.npy')
sun=load('imagenet/combined/sun_'+str(name)+'.npy')
places=load('imagenet/combined/places_'+str(name)+'.npy')
corrupted=load('imagenet/combined/corrupted_'+str(name)+'.npy')
corrupted=corrupted[:len(correct)]

#c_linf=load('imagenet/combined/carlini_linf_0.3'+str(name)+'.npy')
#c_l2=load('imagenet/combined/carlini_l2'+str(name)+'.npy')
#pgd=load('imagenet/combined/pgd0.3'+str(name)+'.npy')
a=load('imagenet/combined/a'+str(name)+'.npy')
o=load('imagenet/combined/o'+str(name)+'.npy')




correct=correct[:,-1]
incorrect=incorrect[:,-1]
fgsm=fgsm[:,-1]
sun=sun[:,-1]
corrupted=corrupted[:,-1]
places=places[:,-1]
#cw_linf=cw_linf[:,-1]
#cw_l2=cw_l2[:,-1]
#pgd=pgd[:,-1]
a=a[:,-1]
o=o[:,-1]


''''''
correct=np.sort(correct)
o=np.sort(o)
places=np.sort(places)
a=np.sort(a)
#pgd=np.sort(pgd)
#cw_linf=np.sort(cw_linf)
incorrect=np.sort(incorrect)
fgsm=np.sort(fgsm)
#cw_l2=np.sort(cw_l2)
corrupted=np.sort(corrupted)
sun=np.sort(sun)


correct=correct[0::4]
places=places[0::2]
a=a[0::3]
corrupted=corrupted[0::4]
sun=sun[0::4]

#pgd=np.sort(pgd)
#cw_linf=np.sort(cw_linf)
#cw_l2=np.sort(cw_l2)
#fgsm=np.sort(fgsm)


plt.plot([i for i in range(2000)], [correct[i] for i in range(2000)] , '-',  label='Correct Prediction')
plt.plot([i for i in range(2000)], [o[i] for i in range(2000)] , '.', label='ImageNet-O')
plt.plot([i for i in range(2000)], [places[i] for i in range(2000)] , '.',  label='Places')
plt.plot([i for i in range(2000)], [a[i] for i in range(2000)] , '.',  label='ImageNet-A')
plt.plot([i for i in range(2000)], [incorrect[i] for i in range(2000)] , '.',  label='Incorrect Prediction')
plt.plot([i for i in range(2000)], [corrupted[i] for i in range(2000)] , '.',  label='Corrupted')
plt.plot([i for i in range(2000)], [sun[i] for i in range(2000)] , '.', color='orange', label='Sun',markevery=11)

plt.legend(loc='lower right', framealpha=1, frameon=True)
plt.xlabel('Example Set Sorted by MSP')
plt.ylabel('MSP Value')
plt.savefig('imagenet/MSP_1.png')
plt.show()



plt.clf()

'''
plt.plot([i for i in range(len(correct))], [correct[i] for i in range(len(correct))] , '.',  label='Correct Prediction')
plt.plot([i for i in range(len(pgd))], [pgd[i] for i in range(len(pgd))] , '.',  label='PGDD')
plt.plot([i for i in range(len(cw_linf))], [cw_linf[i] for i in range(len(cw_linf))] , '.',  label='CW Linf')
plt.plot([i for i in range(len(fgsm))], [fgsm[i] for i in range(len(fgsm))] , '.',  label='FGSM')
plt.plot([i for i in range(len(cw_l2))], [cw_l2[i] for i in range(len(cw_l2))] , '.',  label='CW L2')
plt.legend(loc='lower right', framealpha=1, frameon=True)
plt.xlabel('Example Set Sorted by MSP')
plt.ylabel('MSP Value')
plt.savefig('imagenet/MSP_2.png')
plt.show()

'''



correct=load('imagenet/combined/correct_preds_'+str(name)+'.npy')
incorrect=load('imagenet/combined/incorrect_preds_'+str(name)+'.npy')
fgsm=load('imagenet/combined/fgsm_attacks_'+str(name)+'.npy')
sun=load('imagenet/combined/sun_'+str(name)+'.npy')
places=load('imagenet/combined/places_'+str(name)+'.npy')
corrupted=load('imagenet/combined/corrupted_'+str(name)+'.npy')
a=load('imagenet/combined/a'+str(name)+'.npy')
o=load('imagenet/combined/o'+str(name)+'.npy')
#cw_linf=load('imagenet/combined/carlini_linf_0.3'+str(name)+'.npy')
#cw_l2=load('imagenet/combined/carlini_l2'+str(name)+'.npy')
#pgd=load('imagenet/combined/pgd0.3'+str(name)+'.npy')



correct=correct[:,:2048]
incorrect=incorrect[:,:2048]
fgsm=fgsm[:,:2048]
sun=sun[:,:2048]
corrupted=corrupted[:,:2048]
places=places[:,:2048]
#cw_linf=cw_linf[:,:2048]
#cw_l2=cw_l2[:,:2048]
#pgd=pgd[:,:2048]
a=a[:,:2048]
o=o[:,:2048]

correct=np.sum(correct, axis=1)
incorrect=np.sum(incorrect, axis=1)
fgsm=np.sum(fgsm, axis=1)
sun=np.sum(sun, axis=1)
corrupted=np.sum(corrupted, axis=1)
places=np.sum(places, axis=1)
#cw_linf=np.sum(cw_linf, axis=1)
#cw_l2=np.sum(cw_l2, axis=1)
#pgd=np.sum(pgd, axis=1)
a=np.sum(a, axis=1)
o=np.sum(o, axis=1)


correct=correct[0::4]
places=places[0::2]
a=a[0::3]
corrupted=corrupted[0::4]
sun=sun[0::4]

#pgd=np.sort(pgd)
#cw_linf=np.sort(cw_linf)
#cw_l2=np.sort(cw_l2)
#fgsm=np.sort(fgsm)



correct=np.sort(correct)
incorrect=np.sort(incorrect)
sun=np.sort(sun)
corrupted=np.sort(corrupted)
places=np.sort(places)
a=np.sort(a)
o=np.sort(o)
fgsm=np.sort(fgsm)
#cw_linf=np.sort(cw_linf)
#cw_l2=np.sort(cw_l2)
#pgd=np.sort(pgd)

''''''

plt.clf()
plt.plot([i for i in range(2000)], [correct[i] for i in range(2000)] , 'x',  label='Correct Prediction')
plt.plot([i for i in range(2000)], [o[i] for i in range(2000)] , '.', label='ImageNet-O')
plt.plot([i for i in range(2000)], [places[i] for i in range(2000)] , '.',  label='Places')
plt.plot([i for i in range(2000)], [sun[i] for i in range(2000)] , '.',  label='Sun')
#plt.plot([i for i in range(2000)], [a[i] for i in range(2000)] , '.',  label='ImageNet-A')
#plt.plot([i for i in range(len(incorrect))], [incorrect[i] for i in range(len(incorrect))] , '.',  label='Incorrect Prediction')
#plt.plot([i for i in range(2000)], [corrupted[i] for i in range(2000)] , '.',  label='ImageNet-O')

'''plt.plot([i for i in range(2000)], [fgsm[i] for i in range(2000)] , '.',  label='FGSM')
plt.plot([i for i in range(2000)], [cw_l2[i] for i in range(2000)] , '.',  label='CW L2')

plt.plot([i for i in range(2000)], [pgd[i] for i in range(2000)] , '.',  label='PGDD')
plt.plot([i for i in range(2000)], [cw_linf[i] for i in range(2000)] , '.',  label='CW Linf')'''

#plt.plot([i for i in range(len(sun))], [incorrect[i] for i in range(len(sun))] , 'x', color='orange', label='Sun',markevery=11)
#plt.plot([i for i in range(len(cifar100))], [incorrect[i] for i in range(len(cifar100))] , 'x', color='purple', label='Cifar100',markevery=11)

plt.legend(loc='lower right', framealpha=1, frameon=True)
plt.xlabel('Example Set Sorted by Sum of Latent Vector')
plt.ylabel('Sum of Latent Vector')
plt.savefig('imagenet/latent_features1.png')
plt.show()


