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

    train_transform = trn.Compose([torchvision.transforms.Resize(256),
                                                                          torchvision.transforms.ToTensor(),
                                                                          transforms.Normalize(
                                                                              mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])])
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
    combined=[]
    k=0
    for i,(data,target) in enumerate(testset):
        if k>10000:
            break
        output=model(data)

        pred = output.max(1, keepdim=True)[1]


        if pred.eq(target.data.view_as(pred)).sum() == 0:
            out_vector = torch.flatten(F.softmax(output).detach()).numpy()

            activations = model.get_activations(data).detach()
            activations = torch.flatten(activations).numpy()


            out_vector = np.sort(out_vector)

            out_vector = np.append(activations, out_vector)
            k += 1
            print(k)
            combined.append(out_vector)
        else:

            continue
    return  combined


model=torchvision.models.resnet50(pretrained=False)
from imagenet.resnet50 import resnet50
model=resnet50(pretrained=True)


model.eval()
device='cpu'
name = 'resnet50_2048'

combined_vector=corrupted_images(model,'imagenet/imagenet-a')
np.save('imagenet/combined/a'+str(name)+'.npy',np.array(combined_vector))

combined_vector=corrupted_images(model,'imagenet/imagenet-o')
np.save('imagenet/combined/o_meanMSP'+str(name)+'.npy',np.array(combined_vector))