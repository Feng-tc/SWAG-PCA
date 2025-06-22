import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class LocalCrossCorrelation2D(nn.Module):
    def __init__(self, win=[9, 9]):
        """Initialize the Local Cross Correlation (LCC) model for 2D images

        Args:
            win (list, optional): the size of the local windows. Defaults to [9, 9].
        """
        super(LocalCrossCorrelation2D, self).__init__()
        self.win = win
        self.win_size = self.win[0] * self.win[1]

    def set(self, win):
        self.win = win

    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """Push two images I and J through LCC2D block

        Args:
            I (torch.Tensor): A batch of 2D images with the shape of [BxCxHxW]
            J (torch.Tensor): Another batch of 2D images with the shape of [BxCxHxW]

        Returns:
            torch.Tensor: The results of LCC with the shape of [Bx1]
        """
        # 使用inplace操作计算平方和乘积
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I.mul(J)

        sum_filter = torch.ones([5, 1, self.win[0], self.win[1]], device=I.device)
        tmp = torch.cat((I, J, I2, J2, IJ), dim=1)
        tmp = F.conv2d(tmp, sum_filter, padding=self.win[0] // 2, groups=5)
        
        # 避免额外的unsqueeze操作，直接使用view重塈
        I_sum, J_sum = tmp[:, 0 : 1], tmp[:, 1 : 2]
        I2_sum, J2_sum, IJ_sum = tmp[:, 2 : 3], tmp[:, 3 : 4], tmp[:, 4 : 5]

        win_size_inv = 1.0 / self.win_size
        u_I = I_sum * win_size_inv
        u_J = J_sum * win_size_inv

        cross = IJ_sum - (u_J * I_sum + u_I * J_sum) + u_I * u_J * self.win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I.pow(2) * self.win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J.pow(2) * self.win_size

        # Here we filter the zero-value background to avoid NaN
        tmp_IJ_var = I_var * J_var
        bool_non_zero = tmp_IJ_var > np.power(np.e, -15)
        bool_zero = ~ bool_non_zero

        cross = bool_non_zero * cross + bool_zero
        I_var = bool_non_zero * I_var + bool_zero
        J_var = bool_non_zero * J_var + bool_zero

        cc = cross * cross / (I_var * J_var + np.power(np.e, -15))

        return -1.0 * torch.mean(cc, dim=[1, 2, 3]) + 1
    
    def get_cc(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """Push two images I and J through LCC2D block

        Args:
            I (torch.Tensor): A batch of 2D images with the shape of [BxCxHxW]
            J (torch.Tensor): Another batch of 2D images with the shape of [BxCxHxW]

        Returns:
            torch.Tensor: The results of LCC with the shape of [Bx1]
        """

        I2 = I * I
        J2 = J * J
        IJ = I * J

        sum_filter = torch.ones([5, 1, self.win[0], self.win[1]], device=I.device)
        tmp = torch.cat((I, J, I2, J2, IJ), dim=1)
        tmp = F.conv2d(tmp, sum_filter, padding=self.win[0] // 2, groups=5)
        I_sum, J_sum = tmp[:, 0, ...].unsqueeze(1), tmp[:, 1, ...].unsqueeze(1)
        I2_sum, J2_sum, IJ_sum = tmp[:, 2, ...].unsqueeze(1), tmp[:, 3, ...].unsqueeze(1), tmp[:, 4, ...].unsqueeze(1)

        u_I = I_sum / self.win_size
        u_J = J_sum / self.win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * self.win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * self.win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * self.win_size

        # Here we filter the zero-value background to avoid NaN
        tmp_IJ_var = I_var * J_var
        bool_non_zero = tmp_IJ_var > np.power(np.e, -15)
        bool_zero = ~ bool_non_zero

        cross = bool_non_zero * cross + bool_zero
        I_var = bool_non_zero * I_var + bool_zero
        J_var = bool_non_zero * J_var + bool_zero

        cc = cross * cross / (I_var * J_var + np.power(np.e, -15))

        return -1.0 * torch.mean(cc, dim=[1, 2, 3]) + 1, cc
    
    def get_cc_by_mask(self, I: torch.Tensor, J: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Push two images I and J through LCC2D block

        Args:
            I (torch.Tensor): A batch of 2D images with the shape of [BxCxHxW]
            J (torch.Tensor): Another batch of 2D images with the shape of [BxCxHxW]

        Returns:
            torch.Tensor: The results of LCC with the shape of [Bx1]
        """

        I2 = I * I
        J2 = J * J
        IJ = I * J

        sum_filter = torch.ones([5, 1, self.win[0], self.win[1]], device=I.device)
        tmp = torch.cat((I, J, I2, J2, IJ), dim=1)
        tmp = F.conv2d(tmp, sum_filter, padding=self.win[0] // 2, groups=5)
        I_sum, J_sum = tmp[:, 0, ...].unsqueeze(1), tmp[:, 1, ...].unsqueeze(1)
        I2_sum, J2_sum, IJ_sum = tmp[:, 2, ...].unsqueeze(1), tmp[:, 3, ...].unsqueeze(1), tmp[:, 4, ...].unsqueeze(1)


        u_I = I_sum / self.win_size
        u_J = J_sum / self.win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * self.win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * self.win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * self.win_size

        # Here we filter the zero-value background to avoid NaN
        tmp_IJ_var = I_var * J_var
        bool_non_zero = tmp_IJ_var > np.power(np.e, -15)
        bool_zero = ~ bool_non_zero

        cross = bool_non_zero * cross + bool_zero
        I_var = bool_non_zero * I_var + bool_zero
        J_var = bool_non_zero * J_var + bool_zero

        cc = cross * cross / (I_var * J_var + np.power(np.e, -15))

        total = torch.sum(mask, dim=[1, 2, 3])

        return torch.sum((1 - cc) * mask, dim=[1, 2, 3]) / total, torch.sum(cc * mask, dim=[1, 2, 3]) / total

        # return torch.mean((1 - cc) * mask, dim=[1, 2, 3]), -1.0 * torch.mean(cc, dim=[1, 2, 3]) + 1
    
    def get_cc_by_mask2(self, I: torch.Tensor, J: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Push two images I and J through LCC2D block

        Args:
            I (torch.Tensor): A batch of 2D images with the shape of [BxCxHxW]
            J (torch.Tensor): Another batch of 2D images with the shape of [BxCxHxW]

        Returns:
            torch.Tensor: The results of LCC with the shape of [Bx1]
        """

        I2 = I * I
        J2 = J * J
        IJ = I * J

        sum_filter = torch.ones([5, 1, self.win[0], self.win[1]], device=I.device)
        tmp = torch.cat((I, J, I2, J2, IJ), dim=1)
        tmp = F.conv2d(tmp, sum_filter, padding=self.win[0] // 2, groups=5)
        I_sum, J_sum = tmp[:, 0, ...].unsqueeze(1), tmp[:, 1, ...].unsqueeze(1)
        I2_sum, J2_sum, IJ_sum = tmp[:, 2, ...].unsqueeze(1), tmp[:, 3, ...].unsqueeze(1), tmp[:, 4, ...].unsqueeze(1)


        u_I = I_sum / self.win_size
        u_J = J_sum / self.win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * self.win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * self.win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * self.win_size

        # Here we filter the zero-value background to avoid NaN
        tmp_IJ_var = I_var * J_var
        bool_non_zero = tmp_IJ_var > np.power(np.e, -15)
        bool_zero = ~ bool_non_zero

        cross = bool_non_zero * cross + bool_zero
        I_var = bool_non_zero * I_var + bool_zero
        J_var = bool_non_zero * J_var + bool_zero

        cc = cross * cross / (I_var * J_var + np.power(np.e, -15))

        total = torch.sum(mask, dim=[1, 2, 3])

        return torch.sum((1 - cc) * mask, dim=[1, 2, 3]) / total, torch.sum(cc * mask, dim=[1, 2, 3]) / total, cc


class LocalCrossCorrelation3D(nn.Module):

    def __init__(self, win=[16, 9, 9]):

        super(LocalCrossCorrelation3D, self).__init__()

        self.win = win
        self.win_size = self.win[0] * self.win[1] * self.win[2]

    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:

        I2 = I * I
        J2 = J * J
        IJ = I * J

        sum_filter = torch.ones([5, 1, *self.win], device=I.device)
        tmp = torch.cat((I, J, I2, J2, IJ), dim=1)
        tmp = F.conv3d(tmp, sum_filter, padding=(0, self.win[1] // 2, self.win[2] // 2), groups=5)
        I_sum, J_sum = tmp[:, 0, ...], tmp[:, 1, ...]
        I2_sum, J2_sum, IJ_sum = tmp[:, 2, ...], tmp[:, 3, ...], tmp[:, 4, ...]

        u_I = I_sum / self.win_size
        u_J = J_sum / self.win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * self.win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * self.win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * self.win_size

        # Here we filter the zero-value background to avoid NaN
        tmp_IJ_var = I_var * J_var
        bool_non_zero = tmp_IJ_var > np.power(np.e, -15)
        bool_zero = ~ bool_non_zero

        cross = bool_non_zero * cross + bool_zero
        I_var = bool_non_zero * I_var + bool_zero
        J_var = bool_non_zero * J_var + bool_zero

        cc = cross * cross / (I_var * J_var + np.power(np.e, -15))

        return -1.0 * torch.mean(cc, dim=[1, 2, 3]) + 1, cc


class LocalCrossCorrelation2D_Uncertainty(nn.Module):
    def __init__(self, win=[9, 9]):
        """Initialize the Local Cross Correlation (LCC) model for 2D images

        Args:
            win (list, optional): the size of the local windows. Defaults to [9, 9].
        """
        super(LocalCrossCorrelation2D_Uncertainty, self).__init__()
        self.win = win

    def set(self, win):
        self.win = win

    def forward(self, I: torch.Tensor, J: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Push two images I and J through LCC2D block

        Args:
            I (torch.Tensor): A batch of 2D images with the shape of [BxCxHxW]
            J (torch.Tensor): Another batch of 2D images with the shape of [BxCxHxW]

        Returns:
            torch.Tensor: The results of LCC with the shape of [Bx1]
        """
        I2 = I * I
        J2 = J * J
        IJ = I * J

        sum_filter = torch.ones([1, 1, self.win[0], self.win[1]], device=I.device)

        I_sum = F.conv2d(I, sum_filter, padding=self.win[0] // 2)
        J_sum = F.conv2d(J, sum_filter, padding=self.win[0] // 2)
        I2_sum = F.conv2d(I2, sum_filter, padding=self.win[0] // 2)
        J2_sum = F.conv2d(J2, sum_filter, padding=self.win[0] // 2)
        IJ_sum = F.conv2d(IJ, sum_filter, padding=self.win[0] // 2)

        win_size = self.win[0] * self.win[1]
        # Average image intensity.
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Here we filter the zero-value background to avoid NaN
        non_zero = I_var * J_var > np.power(np.e, -15)
        zero = I_var * J_var <= np.power(np.e, -15)
        cross = non_zero * cross + zero
        I_var = non_zero * I_var + zero
        J_var = non_zero * J_var + zero

        cc = cross * cross / (I_var * J_var + np.power(np.e, -15))
        
        # ff1
        # return torch.mean(-1.0 * cc + 1, dim=[1, 2, 3]), torch.mean(-1.0 * cc * U + 1, dim=[1, 2, 3]), cc, cc * U

        # ff2 ff3 rc1
        return torch.mean(-1.0 * cc + 1, dim=[1, 2, 3]), torch.mean((-1.0 * cc + 1) * U, dim=[1, 2, 3]), cc, cc * U
        
        # 原
        # _cc = -1.0 * cc + 1

        # return torch.mean(_cc, dim=[1, 2, 3]), torch.mean(_cc / U, dim=[1, 2, 3]), _cc, _cc / U


class WeightedLocalCrossCorrelation2D(nn.Module):
    def __init__(self, alpha=0.02, win=[9, 9]):
        """Initialize the WeightedL Local Cross Correlation (WLCC) model for 2D images

        Args:
            alpha (float, optional): The factor of the WLCC. Defaults to 0.02.
            win (list, optional): the size of the local windows. Defaults to [9, 9].
        """
        super(WeightedLocalCrossCorrelation2D, self).__init__()
        self.win = win
        self.normal = Normal(0, alpha, validate_args=None)

    def set(self, alpha, win):
        self.win = win
        self.normal = Normal(0, alpha, validate_args=None)

    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """Push two images I and J through WLCC2D block

        Args:
            I (torch.Tensor): A batch of 2D images with the shape of [BxCxHxW]
            J (torch.Tensor): Another batch of 2D images with the shape of [BxCxHxW]

        Returns:
            torch.Tensor: The results of LCC with the shape of [Bx1]
        """
        I2 = I * I
        J2 = J * J
        IJ = I * J

        sum_filter = torch.ones([1, 1, self.win[0], self.win[1]],
                                device=I.device)

        I_sum = F.conv2d(I, sum_filter, padding=self.win[0] // 2)
        J_sum = F.conv2d(J, sum_filter, padding=self.win[0] // 2)
        I2_sum = F.conv2d(I2, sum_filter, padding=self.win[0] // 2)
        J2_sum = F.conv2d(J2, sum_filter, padding=self.win[0] // 2)
        IJ_sum = F.conv2d(IJ, sum_filter, padding=self.win[0] // 2)

        win_size = self.win[0] * self.win[1]

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Here we filter the zero-value background to avoid NaN
        non_zero = I_var * J_var > np.power(np.e, -15)
        zero = I_var * J_var <= np.power(np.e, -15)
        cross = non_zero * cross + zero
        I_var = non_zero * I_var + zero
        J_var = non_zero * J_var + zero

        cc = cross * cross / (I_var * J_var + np.power(np.e, -15))

        # calculating weight according the intensity difference
        P = self.normal.log_prob(torch.abs(I - J)).exp()
        weight = P / self.normal.log_prob(0).exp()

        dccp = weight + cc * (1 - weight)

        return -1.0 * torch.mean(dccp, dim=[1, 2, 3]) + 1
