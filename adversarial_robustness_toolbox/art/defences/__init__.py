"""
Module implementing multiple types of defences against adversarial attacks.
"""
from .adversarial_trainer import AdversarialTrainer
from .feature_squeezing import FeatureSqueezing
from .gaussian_augmentation import GaussianAugmentation
from .jpeg_compression import JpegCompression
from .label_smoothing import LabelSmoothing
from .pixel_defend import PixelDefend
from .preprocessor import Preprocessor
from .spatial_smoothing import SpatialSmoothing
from .thermometer_encoding import ThermometerEncoding
from .variance_minimization import TotalVarMin
