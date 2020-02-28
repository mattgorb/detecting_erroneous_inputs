# from robustness import model_utils, datasets, train, defaults
# from robustness.datasets import CIFAR
import torch as ch
import dill
from cox.utils import Parameters
import cox.store
import numpy as np
from random import randrange
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



import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import numpy as np

ds = CIFAR('cifar10/data/')
device = 'cpu'
# train_loader, test_loader = ds.make_loaders(workers=0, batch_size=1)

testset = datasets.CIFAR10('cifar10/data', download=True, train=False,
                           transform=transforms.Compose([transforms.ToTensor()])
                           # transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
                           )
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

trainset = datasets.CIFAR10('cifar10/data', download=True, train=True,
                            transform=transforms.Compose([transforms.ToTensor()])
                            # transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
                            )
train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
pass_classes=[3,13,15,19,29,31,34,36,38,42,43,44,50,58,64,65,66,74,75,80,81,85,88,89,90,97]

def load_sun():
    data_path = 'cifar10/SUN2012'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
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
        if i>10000:
            break

        model.zero_grad()
        output, _ = model(data )

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()


        x = torch.flatten(model.module.model.get_activations(data).detach().cpu()).numpy()
        out_vector = torch.flatten(F.softmax(output).detach()).numpy()
        out_vector = np.sort(out_vector)
        out_vector = np.append(x, out_vector)

        sun.append(out_vector)

        print(i)
    return sun



def cifar100(model):
    testset = datasets.CIFAR100('cifar10/data_100', download=True, train=False,
                               transform=transforms.Compose([transforms.ToTensor()])
                               # transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
                               )
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    cifar100 = []

    model.eval()
    correct=0

    i=0
    #criterion = nn.CrossEntropyLoss()
    for data, target in test_loader:
        i+=1

        if target.item() in pass_classes:
            #print('here')
            continue

        model.zero_grad()
        output, _ = model(data)

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()


        x = torch.flatten(model.module.model.get_activations(data).detach().cpu()).numpy()
        out_vector = torch.flatten(F.softmax(output).detach()).numpy()
        out_vector = np.sort(out_vector)
        out_vector = np.append(x, out_vector)

        cifar100.append(out_vector)


    return cifar100



def cifar100_with_classes(model):
    testset = datasets.CIFAR100('cifar10/data_100', download=True, train=False,
                               transform=transforms.Compose([transforms.ToTensor()])
                               # transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
                               )
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    cifar100 = []

    model.eval()
    correct=0

    i=0
    #criterion = nn.CrossEntropyLoss()
    for data, target in test_loader:
        i+=1

        if target.item() in pass_classes:
            #print('here')
            continue

        print(i)
        if i>1000:
            break

        model.zero_grad()
        output, _ = model(data)

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()


        x = torch.flatten(model.module.model.get_activations(data).detach().cpu()).numpy()
        out_vector = torch.flatten(F.softmax(output).detach()).numpy()
        out_vector = np.sort(out_vector)
        out_vector = np.append(x, out_vector)

        cifar100.append((target.item(), out_vector))


    return cifar100



def corrupted_images(model,file,start):
    dataset = np.load(file)
    testset = datasets.CIFAR10('cifar10/data', download=True, train=False,
                               transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

    x, (data_original, labels_for_cifar10c) = next(enumerate(test_loader))

    extended_labels_for_cifar10c = []
    for i in range(5):
        extended_labels_for_cifar10c.extend(labels_for_cifar10c.tolist())

    gaussian_data = []
    correct = 0
    trans1 = transforms.Compose([
        transforms.ToTensor(),
    ])

    for i in range(start,start+1000):

        r=randrange(5)

        data = dataset[i+r*10000]

        data_original_tensor = data_original[i].view(1, data_original[i].shape[0], data_original[i].shape[1],
                                                     data_original[i].shape[2])
        original_preds, _ = model(data_original_tensor)
        pred_orig = original_preds.data.max(1, keepdim=True)[1]
        model.zero_grad()

        data = trans1(data)
        data = data.view(1, data.shape[0], data.shape[1], data.shape[2])
        output, _ = model(data)

        target = torch.from_numpy(np.array([[extended_labels_for_cifar10c[i]]]))
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        # we only want to use predictions that have changed between original image and corrupted
        if pred.eq(pred_orig.data.view_as(pred)).sum() == 0:

            x = torch.flatten(model.module.model.get_activations(data).detach().cpu()).numpy()
            out_vector = torch.flatten(F.softmax(output).detach()).numpy()
            out_vector = np.sort(out_vector)
            out_vector = np.append(x, out_vector)

            gaussian_data.append(out_vector)

        else:

            continue
    return gaussian_data



def generate_fgsm(model, epsilon):
    fgsm_data = []
    criterion = nn.CrossEntropyLoss()

    model.eval()
    i=0
    for data, target in test_loader:
        data.requires_grad = True
        print(i)
        if i>10000:
            break
        # fgsm attack
        model.zero_grad()
        output, _ = model(data)

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
        output, _ = model(perturbed_data,)
        perturbed_guess = output.max(1, keepdim=True)[1][0]
        if perturbed_guess.item() != pred.item():

            x = torch.flatten(model.module.model.get_activations(perturbed_data).detach().cpu()).numpy()
            out_vector = torch.flatten(F.softmax(output).detach()).numpy()
            out_vector = np.sort(out_vector)
            out_vector = np.append(x, out_vector)
            fgsm_data.append(out_vector)


    return fgsm_data

def generate_predictions(model):
    correct_list = []
    incorrect_list = []

    model.eval()

    i = 0
    # criterion = nn.CrossEntropyLoss()
    for data, target in test_loader:

        model.zero_grad()
        output,_ = model(data)

        pred = output.max(1, keepdim=True)[1][0]



        x = torch.flatten(model.module.model.get_activations(data).detach().cpu()).numpy()
        out_vector = torch.flatten(F.softmax(output).detach()).numpy()
        out_vector = np.sort(out_vector)
        out_vector = np.append(x, out_vector)

        if pred.eq(target.data.view_as(pred)).sum() > 0:
            correct_list.append(out_vector)
        else:
            incorrect_list.append(out_vector)
            i += 1
            print('incorrect '+str(len(incorrect_list)))
            print('correct '+str(len(correct_list)))
            continue
    return np.array(correct_list), np.array(incorrect_list)




model, _ = make_and_restore_model(arch='resnet50', dataset=ds, device=device, resume_path='cifar10/cifar_nat.pt',one_output=False)




name = 'resnet50_2058'
type = 'l2_0.25'


import pickle
unseen_preds=cifar100_with_classes(model)
#np.save('cifar10/generated_test/unseen_w_classes_'+str(name)+'_'+str(type)+'.npy',np.array(unseen_preds))
pickle.dump(unseen_preds, open('cifar10/generated_test/unseen_w_classes_.dump', 'wb'))

'''correct_list, incorrect_list=generate_predictions(model)
np.save('cifar10/generated_test/correct_preds_'+str(name)+'_'+str(type)+'.npy',np.array(correct_list))
np.save('cifar10/generated_test/incorrect_preds_'+str(name)+'_'+str(type)+'.npy',np.array(incorrect_list))

fgsm_data=generate_fgsm(model, .01)
np.save('cifar10/generated_test/fgsm_attacks_'+str(name)+'_'+str(type)+'.npy',np.array(fgsm_data))

sun=sun(model)
np.save('cifar10/generated_test/sun_'+str(name)+'_'+str(type)+'.npy',np.array(sun))

unseen_preds=cifar100(model)
np.save('cifar10/generated_test/unseen_'+str(name)+'_'+str(type)+'.npy',np.array(unseen_preds))


cifar10c_datasets=['brightness.npy','contrast.npy','defocus_blur.npy','elastic_transform.npy','fog.npy','frost.npy',
                   'gaussian_noise.npy','gaussian_blur.npy','glass_blur.npy','impulse_noise.npy','spatter.npy',
                   'jpeg_compression.npy','motion_blur.npy', 'pixelate.npy', 'saturate.npy', 'zoom_blur.npy',
                   'shot_noise.npy','speckle_noise.npy', 'zoom_blur.npy','snow.npy'
                   ]
corrupted_gradient_data=[]
num=0
for file in cifar10c_datasets:
    file_name='cifar10/CIFAR-10-C/'+str(file)
    print(len(corrupted_gradient_data))
    corrupted_gradient_data.extend(corrupted_images(model,file_name,(num*1000)))
    num+=1
    if num>9:
        num=0

np.save('cifar10/generated_test/corrupted_'+str(name)+'_'+str(type)+'.npy',np.array(corrupted_gradient_data))'''

