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
from utils.utils import *
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def load_sun():
    data_path = 'tiny_imagenet/SUN2012'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([torchvision.transforms.Resize((64,64)),torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def sun(model):
    sun=[]
    model.eval()
    correct=0

    i=0

    for data, target in load_sun():
        i+=1
        print(i)
        if i>10000:
            break

        model.zero_grad()
        output = model(data)

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

        activations = model.get_activations(data).detach()
        x = torch.flatten(activations).numpy()

        out_vector=torch.flatten(F.softmax(output).detach()).numpy()
        out_vector = np.sort(out_vector)

        out_vector=np.append(x, out_vector)

        sun.append(out_vector)


    return np.array(sun)





def load_places365():
    data_path = 'tiny_imagenet/places365'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([torchvision.transforms.Resize((64,64)),torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def places365(model):
    places365_list=[]
    places365_latent=[]
    model.eval()
    correct=0

    i=0

    for data, target in load_places365():
        i+=1
        print(i)
        if i>5000:
            break

        model.zero_grad()
        output = model(data)

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()


        activations = model.get_activations(data).detach()
        x = torch.flatten(activations).numpy()

        out_vector=torch.flatten(F.softmax(output).detach()).numpy()
        out_vector = np.sort(out_vector)

        out_vector=np.append(x, out_vector)

        places365_list.append(out_vector)



    return np.array(places365_list)


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        path, target = self.imgs[index]


        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)



        return img, target, path

def load_tinyimagenet():

    train_transform = trn.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
    train_data = datasets.ImageFolder(
        root="tiny_imagenet/tiny-imagenet-200/train",
        transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=True)


    return train_loader





def generate_predictions(model):
    correct_list = []
    incorrect_list = []

    model.eval()

    i = 0
    # criterion = nn.CrossEntropyLoss()
    k=0
    for i,(data,target) in enumerate(load_tinyimagenet(),0):

        print('correct '+str(len(correct_list)))
        print('incorrect '+str(len(incorrect_list)))

        model.zero_grad()
        output = model(data)

        pred = output.data.max(1, keepdim=True)[1][0]
        output[:, pred].backward()

        activations = model.get_activations(data).detach()
        x = torch.flatten(activations).numpy()

        out_vector=torch.flatten(F.softmax(output).detach()).numpy()

        out_vector = np.sort(out_vector)

        out_vector=np.append(x, out_vector)

        if pred.eq(target.data.view_as(pred)).sum() > 0:
            correct_list.append(out_vector)
        else:
            incorrect_list.append(out_vector)

        if len(correct_list)>10000 and len(incorrect_list)>5000:
            break


    return np.array(correct_list), np.array(incorrect_list)





def generate_fgsm(model, epsilon):
    fgsm_data = []
    criterion = nn.CrossEntropyLoss()

    model.eval()
    i=0
    for i,(data,target) in enumerate(load_tinyimagenet(),0):
        data.requires_grad = True
        print(i)
        if i>10000:
            break
        # fgsm attack
        model.zero_grad()
        output = model(data)

        pred = output.max(1, keepdim=True)[1][0]

        if pred.eq(target.data.view_as(pred)).sum() == 0:
            print('incorrect pred, continuing')

            continue

        i+=1
        loss = criterion(output, target).to(device)

        loss.backward()
        model.zero_grad()
        data_grad = data.grad

        #epsilon = 0.05
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        model.zero_grad()
        output = model(perturbed_data,)
        perturbed_guess = output.max(1, keepdim=True)[1][0]
        if perturbed_guess.item() == pred.item():
            activations = model.get_activations(perturbed_data).detach()
            x = torch.flatten(activations).numpy()

            out_vector = torch.flatten(F.softmax(output).detach()).numpy()
            out_vector = np.sort(out_vector)

            out_vector = np.append(x, out_vector)
            fgsm_data.append(out_vector)


    return fgsm_data

model = WideResNet(40, 200, 2, dropRate=0.3)

network_state_dict = torch.load('tiny_imagenet/wrn_baseline_epoch_99.pt',map_location='cpu' )
model.load_state_dict(network_state_dict)
model.eval()
device='cpu'

def load(file):
    dataset = np.load(file)
    print(dataset.shape)
    return dataset
name = 'wideresnet'


fgsm_data=generate_fgsm(model, .01)
np.save('tiny_imagenet/combined/fgsm_correct'+str(name)+'.npy',np.array(fgsm_data))
sys.exit()

correct, incorrect=generate_predictions(model)
np.save('tiny_imagenet/combined/correct_preds_'+str(name)+'.npy',np.array(correct))
np.save('tiny_imagenet/combined/incorrect_preds_'+str(name)+'.npy',np.array(incorrect))

sun_list=sun(model)
np.save('tiny_imagenet/combined/sun'+str(name)+'.npy',np.array(sun_list))



places365_list=places365(model)
np.save('tiny_imagenet/combined/places365'+str(name)+'.npy',np.array(places365_list))
