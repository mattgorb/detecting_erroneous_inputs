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
from utils.utils import *
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

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def load_tinyimagenet_for_corrupted(file):

    train_transform = trn.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
    train_data = datasets.ImageFolder(
        root=file,
        transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=True)


    return train_loader


def corrupted_images(model,file):
    testset=load_tinyimagenet_for_corrupted(file)

    gaussian_data = []
    gaussian_data_latent=[]
    k=0
    for i,(data,target) in enumerate(testset):
        if k>150:
            break
        output=model(data)

        pred = output.max(1, keepdim=True)[1]



        if pred.eq(target.data.view_as(pred)).sum() > 0:
        #if pred.eq(target.data.view_as(pred)).sum() == 0:
            activations = model.get_activations(data).detach()
            x = torch.flatten(activations).numpy()

            out_vector = torch.flatten(F.softmax(output).detach()).numpy()
            out_vector = np.sort(out_vector)

            out_vector = np.append(x, out_vector)
            gaussian_data.append(out_vector)
            k+=1
            #print(k)

        else:

            continue
    return gaussian_data


model = WideResNet(40, 200, 2, dropRate=0.3)

network_state_dict = torch.load('tiny_imagenet/wrn_baseline_epoch_99.pt',map_location='cpu' )
model.load_state_dict(network_state_dict)
model.eval()
device='cpu'


tinyc_datasets=['brightness','contrast','defocus_blur','elastic_transform','fog','frost',
                   'gaussian_noise','glass_blur','impulse_noise',
                   'jpeg_compression','motion_blur', 'pixelate',
                   'shot_noise', 'zoom_blur','snow']
corrupted=[]
corrupted_latent=[]
num=0
for file in tinyc_datasets:
    for i in range(1,6):
        file_name='tiny_imagenet/Tiny-ImageNet-C/'+str(file)+'/'+str(i)
        out_vector=corrupted_images(model,file_name)
        corrupted.extend(out_vector)
        print(len(corrupted))

name = 'wideresnet'
np.save('tiny_imagenet/combined/corrupted_softmax'+str(name)+'.npy',np.array(corrupted))
#np.save('tiny_imagenet/generated_data/corrupted_latent'+str(name)+'.npy',np.array(corrupted_latent))