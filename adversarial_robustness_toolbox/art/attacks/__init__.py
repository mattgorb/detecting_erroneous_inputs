"""
Module providing adversarial attacks under a common interface.
"""
'''
from art.attacks.adversarial_patch import AdversarialPatch
from art.attacks.attack import Attack
from art.attacks.boundary import BoundaryAttack
from art.attacks.carlini import CarliniL2Method, CarliniLInfMethod
from art.attacks.deepfool import DeepFool
from art.attacks.elastic_net import ElasticNet
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.iterative_method import BasicIterativeMethod
from art.attacks.newtonfool import NewtonFool
from art.attacks.saliency_map import SaliencyMapMethod
from art.attacks.spatial_transformation import SpatialTransformation
from art.attacks.poisoning_attack_svm import PoisoningAttackSVM
from art.attacks.universal_perturbation import UniversalPerturbation
from art.attacks.virtual_adversarial import VirtualAdversarialMethod
from art.attacks.zoo import ZooAttack
from art.attacks.hop_skip_jump import HopSkipJump
from art.attacks.decision_tree_attack import DecisionTreeAttack
from art.attacks.hclu import HighConfidenceLowUncertainty
'''

from .adversarial_patch import AdversarialPatch
from .attack import Attack
from .boundary import BoundaryAttack
from .carlini import CarliniL2Method, CarliniLInfMethod
from .deepfool import DeepFool
from .elastic_net import ElasticNet
from .fast_gradient import FastGradientMethod
from .projected_gradient_descent import ProjectedGradientDescent
from .iterative_method import BasicIterativeMethod
from .newtonfool import NewtonFool
from .saliency_map import SaliencyMapMethod
from .spatial_transformation import SpatialTransformation
from .poisoning_attack_svm import PoisoningAttackSVM
from .universal_perturbation import UniversalPerturbation
from .virtual_adversarial import VirtualAdversarialMethod
from .zoo import ZooAttack
from .hop_skip_jump import HopSkipJump
from .decision_tree_attack import DecisionTreeAttack
from .hclu import HighConfidenceLowUncertainty