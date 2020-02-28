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
from imagenet.resnet50 import resnet50
import sys
from robustness.robustness.datasets import *
from robustness.robustness.cifar_models import *
from robustness.robustness.attacker import *
from robustness.robustness.model_utils import *
from adversarial_robustness_toolbox.art.attacks import *
import csv
from adversarial_robustness_toolbox.art.classifiers import PyTorchClassifier

# from art.utils import load_mnist


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
                                                                              std=[0.229, 0.224, 0.225])
                                                                          ]))


    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )
    #for i, j in train_loader:
        #print(j)
        #sys.exit()

    return train_loader
def generate_attacks(model, attack):
    attack_data = []
    attack_data_latent=[]

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
    #with torch.no_grad()
    for i,(data,target,path) in enumerate(load_imagenet(),0):
        data=data.to(device)

        filename=path[0].split('/')[-1]
        filename=filename.split('.')[0]

        fill_class=getClass(filename)
        if fill_class is None:
            continue
        target=torch.Tensor([fill_class]).to(dtype=torch.int64).to(device)


        if len(attack_data) > 3500:
            break

        model.zero_grad()
        output = model(data)

        pred = output.max(1, keepdim=True)[1][0]

        if pred.eq(target.data.view_as(pred)).sum() == 0:
            print('incorrect pred, continuing')
            continue

        model.zero_grad()
        x_test_adv = attack.generate(x=data.cpu().numpy())#

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

    return attack_data


from imagenet.resnet50 import resnet50
model=resnet50(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=model.to(device)
model.eval()


name = 'resnet50_2058'
type = 'nat'

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()
classifier = PyTorchClassifier(model=model, loss=criterion, optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=1000)

carlini_linf = carlini.CarliniLInfMethod(classifier, confidence=0.0, targeted=False, learning_rate=0.01, max_iter=10,max_halving=5, max_doubling=5, eps=0.3, batch_size=1)

attack_data  = generate_attacks(model, carlini_linf)
np.save('imagenet/combined/carlini_linf_0.3' + str(name) + '_' + str(type) + '.npy', np.array(attack_data))


carlini2 = CarliniL2Method(classifier, confidence=0.0, targeted=False, learning_rate=0.01, binary_search_steps=10,max_iter=10, initial_const=0.01, max_halving=5, max_doubling=5, batch_size=1)
attack_data  = generate_attacks(model, carlini2)
np.save('imagenet/combined/carlini_l2' + str(name) + '_' + str(type) + '.npy', np.array(attack_data))

pgd = ProjectedGradientDescent(classifier, norm=np.inf, eps=0.3, eps_step=0.1, max_iter=100, targeted=False, num_random_init=0, batch_size=1)
attack_data = generate_attacks(model, pgd)
np.save('imagenet/combined/pgd0.3' + str(name) + '_' + str(type) + '.npy', np.array(attack_data))

