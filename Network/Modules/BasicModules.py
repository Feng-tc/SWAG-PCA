import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class Conv2dLayer(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super(Conv2dLayer, self).__init__()

        self.layer = OrderedDict()
        self.layer['Conv2d'] = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.layer['ReLU'] = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.layer = nn.Sequential(self.layer)

    def forward(self, x):
        return self.layer(x)


class Conv2dBlock(nn.Module):

    def __init__(self, n_blocks, in_dim, out_dim):
        super(Conv2dBlock, self).__init__()

        self.block = []
        for i in range(n_blocks):
            self.block.append(Conv2dLayer(out_dim if i else in_dim, out_dim))

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class Conv1dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, activation=True, batchNorm=True):
        super(Conv1dLayer, self).__init__()

        self.layer = OrderedDict()
        self.layer['Conv1d'] = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if batchNorm:
            self.layer['BatchNorm1d'] = nn.BatchNorm1d(out_channels)
        if activation:
            self.layer['ReLU'] = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.layer = nn.Sequential(self.layer)

    def forward(self, x):
        return self.layer(x)


class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLayer, self).__init__()
        self.layer = OrderedDict()
        self.layer['Linear'] = nn.Linear(in_channels, out_channels)
        self.layer['BatchNorm1d'] = nn.BatchNorm1d(out_channels)
        self.layer['ReLU'] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.layer = nn.Sequential(self.layer)

    def forward(self, x):
        return self.layer(x)
    

class InterpolationLayer(nn.Module):
    def __init__(self):
        super(InterpolationLayer, self).__init__()

    def forward(self, fm, cp_loc, scale):
        # cp_loc: (32, 64, 2)
        _, _, H, W = fm.shape  # (32, 64, 8, 8) 
        # 近似缩放至[0, 8]，这个8是特征域的大小，
        # 也就是在8 * 8的特征域中，根据控制点的坐标，
        # 即位置信息，插值出更好的特征。
        loc = (cp_loc + 1) / scale - 1
        # 近似映射至[-1, 1]
        loc[:, :, 0] = 2 * loc[:, :, 0] /  (W - 1) - 1
        loc[:, :, 1] = 2 * loc[:, :, 1] /  (H - 1) - 1
        loc = loc.unsqueeze(2)  # (32, 64, 1, 2)
        # 将8 * 8的特征插值成64 * 1的特征

        return F.grid_sample(fm, loc, align_corners=True).squeeze(3)



