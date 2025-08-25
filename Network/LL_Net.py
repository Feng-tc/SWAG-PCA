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

from torch.nn.utils import parameters_to_vector, vector_to_parameters

class LL_Net(GenerativeRegistrationNetwork):
    def __init__(self,
                 encoder_param,
                 i_size,
                 loss_weights,
                 similarity_loss='LCC',
                 similarity_loss_param={}):

        super(LL_Net, self).__init__(i_size)

        self.ll_net = Net(**encoder_param)
        self.loss_weights = loss_weights
        self.similarity_loss = LOSSDICT[similarity_loss](**similarity_loss_param)
        self.transformer = SpatialTransformer(i_size, need_grid=True)
        self.gradient_loss = gradient_loss()

    def test(self, src, tgt):

        flow = self.ll_net(src, tgt)
        warped_src = self.transformer(src, flow)

        return flow, warped_src

    def objective(self, src, tgt, segment=None):

        flow = self.ll_net(src, tgt)
        warped_src = self.transformer(src, flow)
        loss_similarity = self.similarity_loss(warped_src, tgt) * self.loss_weights[0]
        loss_gradient = self.gradient_loss(flow) * self.loss_weights[1]
        loss = loss_similarity + loss_gradient

        return {
            'loss': loss,
            'loss_similarity': loss_similarity,
            'loss_gradient': loss_gradient}

    def iter_params_with_lastConvBlockInEncoder(self):

        # lst_param_name = ['start_conv', 'start_act',
        #                   'encoder_stage_1', 'encoder_stage_2', 'encoder_stage_3', 'encoder_stage_4',
        #                   'mid_stage',
        #                   'decoder_stage_4', 'decoder_stage_3', 'decoder_stage_2', 'decoder_stage_1',
        #                   'FlowConv']
        lst_param_name = ['start_conv', 'start_act',
                          'encoder_stage_1', 'encoder_stage_2', 'encoder_stage_3', 'encoder_stage_4']
        # lst = []
        for name, param in self.named_parameters():
            # lst.append(name)
            if any(set(name.split('.')).intersection(set(lst_param_name))):
                yield param

        # print(lst)

    def collect_model(self, n, k):
        cap_theta = getattr(self, 'cap_theta')
        cap_theta_sq = getattr(self, 'cap_theta_sq')
        cap_D = getattr(self, 'cap_D')

        # theta = parameters_to_vector(self.encoder.parameters()).clone().detach()
        # BatchNorm层不计入BNN中
        iter_param = self.iter_params_with_lastConvBlockInEncoder()
        theta = parameters_to_vector(iter_param).clone().detach()

        theta_sq = theta.clone() ** 2
        cap_theta = (n * cap_theta + theta) / (n + 1)
        cap_theta_sq = (n * cap_theta_sq + theta_sq) / (n + 1)

        cap_D = torch.cat((cap_D, (theta - cap_theta).unsqueeze(0)), dim=0)

        if n >= k:
            cap_D = cap_D[1:, :]

        self.cap_theta = cap_theta
        self.cap_theta_sq = cap_theta_sq
        self.cap_D = cap_D

    def SWAG_test(self, src, tgt, nucp_loc, k, s):
        lst_phi = []  # 存储每次迭代的phi结果

        for _ in range(s):
            # 参数采样
            self.sampling_theta(k)

            # 直接运行整个网络获取形变场phi
            phi = self.ll_net(src, tgt)  # 调用完整的网络
            lst_phi.append(phi.unsqueeze(0))

        # 合并结果并计算均值
        all_phi = torch.cat(lst_phi, dim=0)
        phi_avg = all_phi.mean(dim=0)

        # 变形源图像
        w_src = self.transformer(src, phi_avg)

        return phi_avg, w_src, all_phi

    def sampling_theta(self, k):
        theta = getattr(self, 'cap_theta')
        theta_sq = getattr(self, 'cap_theta_sq')
        cov_diag = theta_sq - theta ** 2
        cov_diag = torch.clamp(cov_diag, min=0)

        z = torch.randn_like(theta)
        theta_apx = theta + cov_diag ** 0.5 * z

        iter_param = self.iter_params_with_lastConvBlockInEncoder()
        vector_to_parameters(theta_apx, iter_param)

class Conv_1x1(nn.Module):
    """
    一个简单的1x1卷积
    """

    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              stride=1, padding=0, groups=groups)

    def forward(self, x):
        out = self.conv(x)
        return out


class Conv_3x3(nn.Module):
    """
    一个简单的3x3卷积
    """

    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                              stride=1, padding=1, groups=groups)

    def forward(self, x):
        out = self.conv(x)
        return out


class Conv_Block(nn.Module):
    """
    一个简单的卷积
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)

    def forward(self, x):
        out = self.conv(x)
        return out


class H_Conv(nn.Module):
    """
    一个简单的单向卷积
    """

    def __init__(self, in_channels, out_channels, large_kernel, small_kernel, groups=1):
        super().__init__()
        kernel_size = (large_kernel, small_kernel)
        padding = ((large_kernel - 1) // 2, (small_kernel - 1) // 2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, groups=groups)
        self.groups = groups

    def forward(self, x):
        out = self.conv1(x)
        return out


class W_Conv(nn.Module):
    """
    一个简单的单向卷积
    """

    def __init__(self, in_channels, out_channels, large_kernel, small_kernel, groups=1):
        super().__init__()
        kernel_size = (small_kernel, large_kernel)
        padding = ((small_kernel - 1) // 2, (large_kernel - 1) // 2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, groups=groups)
        self.groups = groups

    def forward(self, x):
        out = self.conv1(x)
        return out

class Attention_Block(nn.Module):
    def __init__(self, in_channels, large_kernel, small_kernel, groups=1):
        super().__init__()
        self.local = Conv_Block(in_channels, in_channels, kernel_size=small_kernel, stride=1,
                                padding=(small_kernel - 1) // 2, groups=groups)

        self.AP = nn.AvgPool2d(kernel_size=2, stride=2)
        self.MP = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_3x3 = Conv_Block(in_channels, in_channels, kernel_size=3, stride=1,
                                padding=1, groups=groups)
        self.global_h = H_Conv(in_channels, in_channels, large_kernel, small_kernel, groups=groups)
        self.global_w = W_Conv(in_channels, in_channels, large_kernel, small_kernel, groups=groups)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')


        self.linear = Conv_1x1(in_channels, in_channels, groups=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        local = self.local(x)
        x_d = self.AP(x) + self.MP(x)
        global_f = self.global_h(x_d) + self.global_w(x_d) + self.global_3x3(x_d)
        attention = self.act(self.linear(local + self.up(global_f)))
        return attention * x


class MSA_Block(nn.Module):
    def __init__(self, in_channels, out_channels, large_kernel, small_kernel, groups=1):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels)
        self.linear_1 = Conv_1x1(in_channels, out_channels)
        self.act_1 = nn.PReLU()
        self.att = Attention_Block(out_channels, large_kernel, small_kernel, groups=groups)
        self.linear_2 = Conv_1x1(out_channels, out_channels)
        self.re_channels = Conv_1x1(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.norm(x)
        out = self.linear_1(out)
        out = self.act_1(out)
        out = self.att(out)
        out = self.linear_2(out)
        out += self.re_channels(x)
        return out


class SSA_Block(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.channel = Conv_1x1(in_channels, in_channels, groups=1)
        self.spatial = Conv_Block(in_channels, in_channels, kernel_size=kernel_size, stride=1,
                                  padding=(kernel_size - 1) // 2, groups=in_channels)
        self.fusion = Conv_1x1(in_channels, in_channels, groups=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        channel = self.channel(x)
        spatial = self.spatial(x)
        out = self.act(self.fusion(channel * spatial))
        out = out * x
        return out


class FFN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels)
        self.linear_1 = Conv_1x1(in_channels, out_channels)
        self.act = nn.PReLU()
        self.att = SSA_Block(out_channels, kernel_size)
        self.linear_2 = Conv_1x1(out_channels, out_channels)
        self.re_channels = Conv_1x1(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.norm(x)
        out = self.linear_1(out)
        out = self.act(out)
        out = self.att(out)
        out = self.linear_2(out)
        out += self.re_channels(x)
        return out


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, large_kernel, small_kernel, groups=1):
        super().__init__()
        self.msa = MSA_Block(in_channels, out_channels, large_kernel, small_kernel, groups=groups)

    def forward(self, x):
        out = self.msa(x)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.ffn = FFN_Block(in_channels, out_channels, kernel_size)

    def forward(self, x):
        out = self.ffn(x)
        return out


class Net(nn.Module):
    def __init__(self, start_channels, large_kernel, small_kernel, in_channels=2, out_channels=2):
        super().__init__()
        encoder_channels = [i * start_channels for i in [1, 1, 2, 4, 8, 16]]
        decoder_channels = [i * start_channels for i in [8, 4, 2, 1, 1, 1]]

        self.start_conv = nn.Conv2d(in_channels=in_channels, out_channels=encoder_channels[0], kernel_size=large_kernel, stride=1,
                                    padding=(large_kernel-1)//2)
        self.start_act = nn.PReLU()
        self.AP = nn.AvgPool2d(kernel_size=2, stride=2)
        self.MP = nn.MaxPool2d(kernel_size=2, stride=2)

        '''编码——下采样路径'''
        self.encoder_stage_1 = Stage(encoder_channels[0], encoder_channels[1], large_kernel, small_kernel, groups=encoder_channels[1])
        self.encoder_stage_2 = Stage(encoder_channels[1], encoder_channels[2], large_kernel, small_kernel, groups=encoder_channels[2])
        self.encoder_stage_3 = Stage(encoder_channels[2], encoder_channels[3], large_kernel, small_kernel, groups=encoder_channels[3])
        self.encoder_stage_4 = Stage(encoder_channels[3], encoder_channels[4], large_kernel, small_kernel, groups=encoder_channels[4])

        '''最底层的卷积'''
        self.mid_stage = Stage(encoder_channels[4], encoder_channels[5], large_kernel, small_kernel, groups=encoder_channels[5])

        '''解码——上采样路径'''
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.decoder_stage_4 = Decoder(encoder_channels[5] + encoder_channels[4], decoder_channels[0],
                                       kernel_size=small_kernel)
        self.decoder_stage_3 = Decoder(decoder_channels[0] + encoder_channels[3], decoder_channels[1],
                                       kernel_size=small_kernel)
        self.decoder_stage_2 = Decoder(decoder_channels[1] + encoder_channels[2], decoder_channels[2],
                                       kernel_size=small_kernel)
        self.decoder_stage_1 = Decoder(decoder_channels[2] + encoder_channels[1], decoder_channels[3],
                                       kernel_size=small_kernel)

        '''生成形变场的卷积'''
        self.FlowConv = nn.Conv2d(decoder_channels[3], out_channels, kernel_size=3, stride=1,
                                  padding=1)
        self.FlowConv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.FlowConv.weight.shape))
        self.FlowConv.bias = nn.Parameter(torch.zeros(self.FlowConv.bias.shape))
        self.final_act = nn.Softsign()

    def forward(self, m, f):
        x = torch.cat([m, f], dim=1)
        x = self.start_act(self.start_conv(x))

        encoder_feature = []
        '''编码路径'''
        # stage 1
        x = self.encoder_stage_1(x)
        encoder_feature.append(x)
        x = self.AP(x) + self.MP(x)

        # stage 2
        x = self.encoder_stage_2(x)
        encoder_feature.append(x)
        x = self.AP(x) + self.MP(x)

        # stage 3
        x = self.encoder_stage_3(x)
        encoder_feature.append(x)
        x = self.AP(x) + self.MP(x)

        # stage 4
        x = self.encoder_stage_4(x)
        encoder_feature.append(x)
        x = self.AP(x) + self.MP(x)

        '''中间路径'''
        x = self.mid_stage(x)
        
        '''解码路径'''
        x = self.upsample(x)
        x = torch.cat([x, encoder_feature.pop()], dim=1)
        x = self.decoder_stage_4(x)

        x = self.upsample(x)
        x = torch.cat([x, encoder_feature.pop()], dim=1)
        x = self.decoder_stage_3(x)

        x = self.upsample(x)
        x = torch.cat([x, encoder_feature.pop()], dim=1)
        x = self.decoder_stage_2(x)

        x = self.upsample(x)
        x = torch.cat([x, encoder_feature.pop()], dim=1)
        x = self.decoder_stage_1(x)

        '''最后的卷积层+形变场'''
        # flow = self.final_act(self.FlowConv(x))
        # flow = self.FlowConv(x)
        flow = self.FlowConv(self.final_act(x))
        return flow