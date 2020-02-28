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

import sys
from robustness.robustness.datasets import *
from robustness.robustness.cifar_models import *
from robustness.robustness.attacker import *
from robustness.robustness.model_utils import *
from adversarial_robustness_toolbox.art.attacks import *

from adversarial_robustness_toolbox.art.classifiers import PyTorchClassifier
#from art.utils import load_mnist


ds = CIFAR('cifar10/data/')
device = 'cpu'

testset = datasets.CIFAR10('cifar10/data', download=True, train=False,
                           transform=transforms.Compose([transforms.ToTensor()])
                           # transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
                           )
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)





def generate_attacks(model, attack):
    attack_data = []
    attack_data_latent = []
    criterion = nn.CrossEntropyLoss()

    #model.eval()
    i=0
    k=0
    for data, target in test_loader:
        data=data.to(device)
        target=target.to(device)
        if len(attack_data)>3:
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

            out_vector = torch.flatten(F.softmax(output).detach()).cpu().numpy()
            out_vector = np.sort(out_vector)

            activations = torch.flatten(model.module.model.get_activations(torch.from_numpy(x_test_adv).to(device)).detach().cpu()).numpy()

            out_vector = np.append(activations, out_vector)
            attack_data.append(out_vector)

    return attack_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, _ = make_and_restore_model(arch='resnet50', dataset=ds, device='gpu', resume_path='cifar10/cifar_nat.pt',one_output=True)
model.eval()
model=model.to(device)

#sys.exit()
#model=resnet50()
#network_state_dict = torch.load('cifar10/cifar_nat.pt', map_location='cpu' )
#model.load_state_dict(network_state_dict)

name = 'resnet50_2048'
type = 'l2_0.25'

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()
classifier = PyTorchClassifier(model=model,loss=criterion,optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10)

''''''
carlini_linf=carlini.CarliniLInfMethod(classifier, confidence=0.0, targeted=False, learning_rate=0.01, max_iter=10, max_halving=5, max_doubling=5, eps=0.3, batch_size=128)
attack_data=generate_attacks(model,carlini_linf)
np.save('cifar10/combined/carlini_linf_0.3'+str(name)+'_'+str(type)+'.npy',np.array(attack_data))

pgd=ProjectedGradientDescent(classifier, norm=np.inf, eps=0.3, eps_step=0.1, max_iter=100, targeted=False, num_random_init=0, batch_size=1)
attack_data=generate_attacks(model,pgd)
np.save('cifar10/combined/pgd0.3'+str(name)+'_'+str(type)+'.npy',np.array(attack_data))

carlini2=CarliniL2Method(classifier, confidence=0.0, targeted=False, learning_rate=0.01, binary_search_steps=10, max_iter=10, initial_const=0.01, max_halving=5, max_doubling=5, batch_size=1)
attack_data=generate_attacks(model,carlini2)
np.save('cifar10/combined/carlini_l2'+str(name)+'_'+str(type)+'.npy',np.array(attack_data))

