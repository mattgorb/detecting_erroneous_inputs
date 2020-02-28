from robustness.robustness import model_utils, datasets, train, defaults
from robustness.robustness.datasets import CIFAR
import torch as ch

# We use cox (http://github.com/MadryLab/cox) to log, store and analyze
# results. Read more at https//cox.readthedocs.io.
from cox.utils import Parameters
import cox.store

# Hard-coded dataset, architecture, batch size, workers
ds = CIFAR('cifar10/data')
device='gpu'
m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,device='cpu')
train_loader, val_loader = ds.make_loaders(batch_size=64, workers=4)
print(m)


# Create a cox store for logging
out_store = cox.store.Store("train_out")

# Hard-coded base parameters
train_kwargs = {
    'out_dir': "train_out",
    'adv_train': 0,
    'constraint': '2',
    'eps': 0.0,
    'attack_lr': 1,
    'attack_steps': 0
}
train_args = Parameters(train_kwargs)

# Fill whatever parameters are missing from the defaults
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, CIFAR)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.PGD_ARGS, CIFAR)

# Train a model
train.train_model(train_args, m, (train_loader, val_loader), store=out_store)