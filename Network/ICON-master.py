import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions.normal import Normal

from Modules.Interpolation import SpatialTransformer
# from .BaseNetwork import GenerativeRegistrationNetwork
from Network.Modules.BaseNetwork import GenerativeRegistrationNetwork
from Modules.Loss import LOSSDICT, gradient_loss

