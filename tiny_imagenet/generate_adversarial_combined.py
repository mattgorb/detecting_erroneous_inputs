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
from tqdm import tqdm
from tiny_imagenet.wideresnet import WideResNet
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import sys
from robustness.robustness.datasets import *
from robustness.robustness.cifar_models import *
from robustness.robustness.attacker import *
from robustness.robustness.model_utils import *
from adversarial_robustness_toolbox.art.attacks import *

from adversarial_robustness_toolbox.art.classifiers import PyTorchClassifier

# from art.utils import load_mnist

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def load_tinyimagenet():

    train_transform = trn.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
    train_data = datasets.ImageFolder(
        root="tiny_imagenet/tiny-imagenet-200/train",
        transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=True)


    return train_loader

def generate_attacks(model, attack):
    attack_data = []
    attack_data_latent=[]
    criterion = nn.CrossEntropyLoss()

    # model.eval()
    k = 0
    for i,(data,target) in enumerate(load_tinyimagenet(),0):
        data=data.to(device)
        target=target.to(device)
        if len(attack_data) > 4000:
            break

        output = model(data)

        pred = output.max(1, keepdim=True)[1][0]

        if pred.eq(target.data.view_as(pred)).sum() == 0:
            print('incorrect pred, continuing')
            continue

        x_test_adv = attack.generate(x=data.cpu().numpy())

        output = model(torch.from_numpy(x_test_adv).to(device))
        perturbed_guess = output.max(1, keepdim=True)[1][0]

        if perturbed_guess.item() != pred.item():
            print(k)

            k += 1
            activations = model.get_activations(torch.from_numpy(x_test_adv).to(device)).detach()
            x = torch.flatten(activations).cpu().numpy()

            out_vector = torch.flatten(F.softmax(output).detach()).cpu().numpy()
            out_vector = np.sort(out_vector)

            out_vector = np.append(x, out_vector)

            attack_data.append(out_vector)
            #attack_data_latent.append(torch.flatten(activations).cpu().numpy())

    return attack_data#,attack_data_latent






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = WideResNet(40, 200, 2, dropRate=0.3)

network_state_dict = torch.load('tiny_imagenet/wrn_baseline_epoch_99.pt')
model.load_state_dict(network_state_dict)
model.eval()
model=model.to(device)

# sys.exit()
# model=resnet50()
# network_state_dict = torch.load('cifar10/cifar_nat.pt', map_location='cpu' )
# model.load_state_dict(network_state_dict)

name = 'resnet50_2058'
type = 'nat'

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()
classifier = PyTorchClassifier(model=model, loss=criterion, optimizer=optimizer, input_shape=(3, 64, 64), nb_classes=200)

carlini_linf = carlini.CarliniLInfMethod(classifier, confidence=0.0, targeted=False, learning_rate=0.01, max_iter=10,
                                         max_halving=5, max_doubling=5, eps=0.3, batch_size=128)
attack_data = generate_attacks(model, carlini_linf)
np.save('tiny_imagenet/combined/carlini_linf_0.3' + str(name) + '_' + str(type) + '.npy', np.array(attack_data))

pgd = ProjectedGradientDescent(classifier, norm=np.inf, eps=0.3, eps_step=0.1, max_iter=100, targeted=False,
                               num_random_init=0, batch_size=1)
attack_data = generate_attacks(model, pgd)
np.save('tiny_imagenet/combined/pgd0.3' + str(name) + '_' + str(type) + '.npy', np.array(attack_data))

carlini2 = CarliniL2Method(classifier, confidence=0.0, targeted=False, learning_rate=0.01, binary_search_steps=10,
                           max_iter=10, initial_const=0.01, max_halving=5, max_doubling=5, batch_size=1)
attack_data = generate_attacks(model, carlini2)
np.save('tiny_imagenet/combined/carlini_l2' + str(name) + '_' + str(type) + '.npy', np.array(attack_data))
