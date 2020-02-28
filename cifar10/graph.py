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
    clf=LinearSVC(random_state=0, tol=1e-5,class_weight=balance)#
    clf.fit(train_data, train_y)

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
        predict_mine = np.where(y_pred > -0.44, 1, 0)
        print(confusion_matrix(train_y, predict_mine))
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
cw_linf  = loadNPY('cifar10/generated_test/carlini_linf_0.3'+str(name)+'_'+str(type)+'.npy')
cw_l2=loadNPY('cifar10/generated_test/carlini_l2'+str(name)+'_'+str(type)+'.npy')
pgd  = loadNPY('cifar10/generated_test/pgd0.3'+str(name)+'_'+str(type)+'.npy')



correct=correct[:,-1]
incorrect=incorrect[:,-1]
fgsm=fgsm[:,-1]
sun=sun[:,-1]
corrupted=corrupted[:,-1]
cifar100=cifar100[:,-1]
cw_linf=cw_linf[:,-1]
cw_l2=cw_l2[:,-1]
pgd=pgd[:,-1]



''''''
correct=np.sort(correct)
incorrect=np.sort(incorrect)
cifar100=np.sort(cifar100)
fgsm=np.sort(fgsm)
corrupted=np.sort(corrupted)
sun=np.sort(sun)
pgd=np.sort(pgd)
cw_linf=np.sort(cw_linf)
cw_l2=np.sort(cw_l2)

correct=correct[0::int(len(correct)/len(incorrect))]
cifar100=cifar100[0::int(len(cifar100)/len(incorrect))]
corrupted=corrupted[0::int(len(corrupted)/len(incorrect))]
sun=sun[0::int(len(sun)/len(incorrect))]
fgsm=fgsm[0::int(len(fgsm)/len(incorrect))]
pgd=pgd[0::int(len(pgd)/len(incorrect))]
cw_linf=cw_linf[0::int(len(cw_linf)/len(incorrect))]
cw_l2=cw_l2[0::int(len(cw_l2)/len(incorrect))]



'''
plt.plot([i for i in range(len(incorrect))], [cifar100[i] for i in range(len(incorrect))] , '.',  label='CIFAR100')
plt.plot([i for i in range(len(incorrect))], [sun[i] for i in range(len(incorrect))] , '.',  label='SUN')
plt.plot([i for i in range(len(incorrect))], [incorrect[i] for i in range(len(incorrect))] , '.',  label='Incorrect Prediction')
plt.plot([i for i in range(len(incorrect))], [corrupted[i] for i in range(len(incorrect))] , '.',  label='CIFAR10-C')
plt.plot([i for i in range(len(incorrect))], [correct[i] for i in range(len(incorrect))] , '-',  label='Correct Prediction')
plt.legend(loc='lower right', framealpha=1, frameon=True)
plt.xlabel('Example Set Sorted by MSP')
plt.ylabel('MSP Value')
plt.savefig('cifar10/MSP_1.png')
#plt.show()



plt.clf()



plt.plot([i for i in range(len(incorrect))], [pgd[i] for i in range(len(incorrect))] , '.',  label='PGD')
plt.plot([i for i in range(len(incorrect))], [cw_l2[i] for i in range(len(incorrect))] , '.',  label='C & W L2')
plt.plot([i for i in range(len(incorrect))], [cw_linf[i] for i in range(len(incorrect))] , '.',  label='C & W L-Inf')
plt.plot([i for i in range(len(incorrect))], [fgsm[i] for i in range(len(incorrect))] , '.',  label='FGSM')
plt.plot([i for i in range(len(incorrect))], [correct[i] for i in range(len(incorrect))] , '-',  label='Correct Prediction')
plt.legend(loc='lower right', framealpha=1, frameon=True)
plt.xlabel('Example Set Sorted by MSP')
plt.ylabel('MSP Value')
plt.savefig('cifar10/MSP_2.png')
'''



correct = loadNPY('cifar10/generated_test/correct_preds_'+str(name)+'_'+str(type)+'.npy')
incorrect = loadNPY('cifar10/generated_test/incorrect_preds_'+str(name)+'_'+str(type)+'.npy')
cifar100 = loadNPY('cifar10/generated_test/unseen_'+str(name)+'_'+str(type)+'.npy')
corrupted = loadNPY('cifar10/generated_test/corrupted_'+str(name)+'_'+str(type)+'.npy')
sun = loadNPY('cifar10/generated_test/sun_'+str(name)+'_'+str(type)+'.npy')
sun=sun[:len(correct)]
fgsm  = loadNPY('cifar10/generated_test/fgsm_attacks_'+str(name)+'_'+str(type)+'.npy')
cw_linf  = loadNPY('cifar10/generated_test/carlini_linf_0.3'+str(name)+'_'+str(type)+'.npy')
cw_l2=loadNPY('cifar10/generated_test/carlini_l2'+str(name)+'_'+str(type)+'.npy')
pgd  = loadNPY('cifar10/generated_test/pgd0.3'+str(name)+'_'+str(type)+'.npy')







correct=correct[:,:2048]
incorrect=incorrect[:,:2048]
fgsm=fgsm[:,:2048]
sun=sun[:,:2048]
corrupted=corrupted[:,:2048]
cifar100=cifar100[:,:2048]
cw_linf=cw_linf[:,:2048]
cw_l2=cw_l2[:,:2048]
pgd=pgd[:,:2048]


correct=np.sum(correct, axis=1)
incorrect=np.sum(incorrect, axis=1)
fgsm=np.sum(fgsm, axis=1)
sun=np.sum(sun, axis=1)
corrupted=np.sum(corrupted, axis=1)
cifar100=np.sum(cifar100, axis=1)
cw_linf=np.sum(cw_linf, axis=1)
cw_l2=np.sum(cw_l2, axis=1)
pgd=np.sum(pgd, axis=1)



correct=correct[0::int(len(correct)/len(incorrect))]
cifar100=cifar100[0::int(len(cifar100)/len(incorrect))]
corrupted=corrupted[0::int(len(corrupted)/len(incorrect))]
sun=sun[0::int(len(sun)/len(incorrect))]
fgsm=fgsm[0::int(len(fgsm)/len(incorrect))]

''''''
pgd=pgd[0::int(len(pgd)/len(incorrect))]
cw_linf=cw_linf[0::int(len(cw_linf)/len(incorrect))]
cw_l2=cw_l2[0::int(len(cw_l2)/len(incorrect))]




correct=np.sort(correct)
incorrect=np.sort(incorrect)
sun=np.sort(sun)
corrupted=np.sort(corrupted)
cifar100=np.sort(cifar100)
fgsm=np.sort(fgsm)
cw_linf=np.sort(cw_linf)
cw_l2=np.sort(cw_l2)
pgd=np.sort(pgd)

''''''

plt.clf()

plt.plot([i for i in range(len(incorrect))], [cifar100[i] for i in range(len(incorrect))] , '.',  label='CIFAR100')
plt.plot([i for i in range(len(incorrect))], [sun[i] for i in range(len(incorrect))] , '.',  label='SUN')

plt.plot([i for i in range(len(incorrect))], [incorrect[i] for i in range(len(incorrect))] , '.',  label='Incorrect Prediction')
plt.plot([i for i in range(len(incorrect))], [corrupted[i] for i in range(len(incorrect))] , '.',  label='CIFAR10-C')
plt.plot([i for i in range(len(incorrect))], [correct[i] for i in range(len(incorrect))] , '-',  label='Correct Prediction')

plt.legend(loc='upper left', framealpha=1, frameon=True)
plt.xlabel('Example Set Sorted by Sum of Latent Vector')
plt.ylabel('Sum of Latent Vector')
plt.savefig('cifar10/latent_features1.png')
#plt.show()

plt.clf()


plt.plot([i for i in range(len(incorrect))], [pgd[i] for i in range(len(incorrect))] , '.',  label='PGD')
plt.plot([i for i in range(len(incorrect))], [cw_l2[i] for i in range(len(incorrect))] , '.',  label='C & W L2')
plt.plot([i for i in range(len(incorrect))], [cw_linf[i] for i in range(len(incorrect))] , '.',  label='C & W L-Inf')
plt.plot([i for i in range(len(incorrect))], [fgsm[i] for i in range(len(incorrect))] , '.',  label='FGSM')
plt.plot([i for i in range(len(incorrect))], [correct[i] for i in range(len(incorrect))] , '-',  label='Correct Prediction')
plt.legend(loc='lower right', framealpha=1, frameon=True)
plt.legend(loc='upper left', framealpha=1, frameon=True)
plt.xlabel('Example Set Sorted by Sum of Latent Vector')
plt.ylabel('Sum of Latent Vector')
plt.savefig('cifar10/latent_features2.png')