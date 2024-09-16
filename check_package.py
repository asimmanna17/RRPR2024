# Standard libraries
import os
import math
import random
import pickle
import operator

# Third-party libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import pkg_resources

# PyTorch libraries
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data
from torchvision import transforms

# Setup CUDA environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# torch.cuda.set_device(0)

use_cuda = torch.cuda.is_available()

# Set random seeds
seed_value = 3407
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Print all versions at the end
print(f"pickle version: {pickle.format_version}")
print("numpy version:", np.__version__)
print("matplotlib version:", matplotlib.__version__)
print("tqdm version:", pkg_resources.get_distribution("tqdm").version)
print("Pillow version:", Image.__version__)
print("torch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)
