import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()
    
    def forward(self, flow, penalty='l2'):
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        
        d = torch.mean(dx) + torch.mean(dy)

        return d / 2.0