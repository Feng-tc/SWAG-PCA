import torch.nn as nn
from .Interpolate2D import BilinearInterpolate
from Modules.Interpolation import SpatialTransformer
from .Meshgrid import meshgrid2D


class ExponentiationLayer(nn.Module):
    def __init__(self, size, factor=4):
        super(ExponentiationLayer, self).__init__()
        self.factor = factor
        self.interpolate = SpatialTransformer(size)

    def forward(self, v):

        for i in range(self.factor):
            v1 = self.interpolate(v, v)
            v = v + v1
        
        return v
