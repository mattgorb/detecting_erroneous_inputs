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


'''from PIL import Image
img = Image.fromarray(x[10000], 'RGB')
img.show()

tensor_image = data.view(data.shape[1], data.shape[2], data.shape[3])
trans1 = transforms.ToPILImage()
plt.imshow(trans1(tensor_image))
plt.show()'''

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import numpy as np

ds = CIFAR('imagenet/data/')
device = 'cpu'
# train_loader, test_loader = ds.make_loaders(workers=0, batch_size=1)

'''testset = datasets.imagenet('imagenet/data', download=True, train=False,
                           transform=transforms.Compose([transforms.ToTensor()])
                           # transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
                           )
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

trainset = datasets.imagenet('imagenet/data', download=True, train=True,
                            transform=transforms.Compose([transforms.ToTensor()])
                            # transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
                            )
train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
'''


pass_classes=[3,13,15,19,29,31,34,36,38,42,43,44,50,58,64,65,66,74,75,80,81,85,88,89,90,97]

def load_sun():
    data_path = 'imagenet/SUN2012'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([torchvision.transforms.Resize(256),
                                                                          torchvision.transforms.ToTensor(),
                                                                          transforms.Normalize(
                                                                              mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )
    return train_loader




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

def load_imagenet():
    #data_dir = '/Volumes/My Passport/ILSVRC/Data/CLS-LOC/train'
    data_dir = 'imagenet/data'

    dataset = ImageFolderWithPaths(data_dir,transform=transforms.Compose([torchvision.transforms.Resize(256),
                                                                          torchvision.transforms.ToTensor(),
                                                                          transforms.Normalize(
                                                                              mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])]))  # our custom dataset

    train_loader = torch.utils.data.DataLoader(
        dataset,
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
        print(i)
        i+=1
        if i>8000:
            break

        model.zero_grad()
        output= model(data)

        pred = output.data.max(1, keepdim=True)[1]
        #correct += pred.eq(target.data.view_as(pred)).sum()

        #output[:, pred].backward()
        out_vector=torch.flatten(F.softmax(output).detach()).numpy()

        sun.append(out_vector)


    return sun




def generate_predictions(model):
    correct_list = []
    incorrect_list = []

    model.eval()

    i = 0
    # criterion = nn.CrossEntropyLoss()

    with open('imagenet/data/LOC_val_solution.csv', mode='r') as infile:
        reader = csv.reader(infile)
        #with open('coors_new.csv', mode='w') as outfile:
        #    writer = csv.writer(outfile)
        solutions = {rows[0]: rows[1].split(' ')[0] for rows in reader}



    with open('imagenet/data/LOC_synset_mapping.txt', mode='r') as infile:
        reader = csv.reader(infile)
        #with open('coors_new.csv', mode='w') as outfile:
            #writer = csv.writer(outfile)
        class_list = [rows[0].split(' ')[0] for rows in reader]


    def getClass(filename):
        try:
            searchItem=solutions[filename]
            return class_list.index(searchItem)
        except:
            return None

    k=0
    for i,(data,target,path) in enumerate(load_imagenet(),0):
        filename=path[0].split('/')[-1]
        filename=filename.split('.')[0]

        fill_class=getClass(filename)
        if fill_class is None:
            continue
        target=torch.Tensor([fill_class]).to(dtype=torch.int64)

        model.zero_grad()
        output = model(data)

        pred = output.data.max(1, keepdim=True)[1][0]


        out_vector = torch.flatten(F.softmax(output).detach()).numpy()

        if target==pred > 0:
            correct_list.append(out_vector)
        else:
            #print(torch.flatten(activations).numpy().shape)
            incorrect_list.append(out_vector)


        if len(correct_list)>10000 and len(incorrect_list)>7000:
            break

    return np.array(correct_list), np.array(incorrect_list)






def generate_fgsm(model, epsilon):
    fgsm_data = []
    criterion = nn.CrossEntropyLoss()

    model.eval()
    k=0


    with open('imagenet/data/LOC_val_solution.csv', mode='r') as infile:
        reader = csv.reader(infile)
        #with open('coors_new.csv', mode='w') as outfile:
        #    writer = csv.writer(outfile)
        solutions = {rows[0]: rows[1].split(' ')[0] for rows in reader}



    with open('imagenet/data/LOC_synset_mapping.txt', mode='r') as infile:
        reader = csv.reader(infile)
        #with open('coors_new.csv', mode='w') as outfile:
            #writer = csv.writer(outfile)
        class_list = [rows[0].split(' ')[0] for rows in reader]


    def getClass(filename):
        try:
            searchItem=solutions[filename]
            return class_list.index(searchItem)
        except:
            return None

    for i,(data,target,path) in enumerate(load_imagenet(),0):
        data.requires_grad = True

        filename=path[0].split('/')[-1]
        filename=filename.split('.')[0]

        fill_class=getClass(filename)
        if fill_class is None:
            continue


        target=torch.Tensor([fill_class]).to(dtype=torch.int64)
        if k>8000:
            break
        # fgsm attack
        model.zero_grad()
        output = model(data)
        pred = output.max(1, keepdim=True)[1][0]

        if pred.eq(target.data.view_as(pred)).sum() == 0:
            print('incorrect pred, continuing')

            continue


        loss = criterion(output, target).to(device)

        loss.backward()
        model.zero_grad()
        data_grad = data.grad

        #epsilon = 0.05
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        model.zero_grad()
        output = model(perturbed_data,)
        perturbed_guess = output.max(1, keepdim=True)[1][0]
        if perturbed_guess.item() != pred.item():

            out_vector = torch.flatten(F.softmax(output).detach()).numpy()

            fgsm_data.append(out_vector)
            print(k)
            k+=1


    return fgsm_data




def load_places365():
    data_path = 'tiny_imagenet/places365'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([torchvision.transforms.Resize(256),
                                                                          torchvision.transforms.ToTensor(),
                                                                          transforms.Normalize(
                                                                              mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])])
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

        output[:, pred].backward()
        out_vector=torch.flatten(F.softmax(output).detach()).numpy()

        activations = model.get_activations(data).detach()

        places365_list.append(out_vector)
        places365_latent.append(torch.flatten(activations).cpu().numpy())


    return np.array(places365_list), np.array(places365_latent)



def load(file):
    dataset = np.load(file)
    return dataset


#model, _ = make_and_restore_model(arch='resnet50', dataset=ds, device=device)#, resume_path='imagenet/cifar_nat.pt'


model=torchvision.models.resnet50(pretrained=False)
from imagenet.resnet50 import resnet50
model=resnet50(pretrained=True)





# final_layer, final_three_layers, final_two_layers, second_last_layer, second_last_third_last_layer,
name = 'resnet50_2048'

type = 'nat'



places, places_latent=places365(model)
np.save('imagenet/generated_test/places_'+str(name)+'_'+str(type)+'.npy',np.array(places))
np.save('imagenet/generated_test/placeslatent_'+str(name)+'_'+str(type)+'.npy',np.array(places_latent))
#sun=sun(model)
#np.save('imagenet/generated_test/sun_'+str(name)+'_'+str(type)+'.npy',np.array(sun))
#correct_list,incorrect_list = generate_predictions(model)

#print(relu_incorrect_list.shape)
#np.save('imagenet/generated_test/correct_preds_'+str(name)+'_'+str(type)+'.npy',correct_list)
#np.save('imagenet/generated_test/incorrect_preds_'+str(name)+'_'+str(type)+'.npy',incorrect_list)



#fgsm_data=generate_fgsm(model, .01)
#np.save('imagenet/generated_test/fgsm_attacks_'+str(name)+'_'+str(type)+'.npy',np.array(fgsm_data))
