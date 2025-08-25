from .Modules.BaseNetwork import GenerativeRegistrationNetwork
from .Modules.BasicModules import InterpolationLayer, Conv2dLayer, Conv2dBlock, Conv1dLayer
from .Modules.FeatureAndPointNet import FeatureEncoder, PointEncoder

from Modules.Interpolation import SpatialTransformer, RadialBasisLayer_MultiSamples, RadialBasisLayer_MultiSamples_G
from Modules.Loss import LOSSDICT, JacobianDeterminantLoss, MaxMinPointDist

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import parameters_to_vector, vector_to_parameters 


class NuNet(GenerativeRegistrationNetwork):
    def __init__(self,
                 encoder_param,
                 i_size,
                 c_factor,
                 cpoint_num,
                 nucpoint_num,
                 ucpoint_num,
                 similarity_loss='LCC',
                 similarity_loss_param={},
                 hyperparam={},
                 cropSize=64):

        super(NuNet, self).__init__(i_size)
        
        self.gap = 16   # 原128的图被裁成96，中心区域使用64大小

        self.similarity_factor = hyperparam['similarity_factor']
        self.jac = hyperparam['jac']
        self.i_size, self.c_factor, self.cpoint_num = i_size, c_factor, cpoint_num
        self.nucpoint_num, self.ucpoint_num = nucpoint_num, ucpoint_num
        
        self.encoder = RBFDCNUInterEncoder(self.gap, **encoder_param)
        self.similarity_loss = LOSSDICT[similarity_loss](**similarity_loss_param)
        self.transformer = SpatialTransformer(i_size, need_grid=True)
        self.jacobian_loss = JacobianDeterminantLoss()

        ucp_loc_vectors = [torch.linspace(0, i - 1, int(ucpoint_num ** 0.5)) for i in i_size]
        ucp_loc = torch.meshgrid(ucp_loc_vectors)
        ucp_loc = torch.stack(ucp_loc, 2)[:, :, [1, 0]]
        ucp_loc = torch.flatten(ucp_loc, start_dim=0, end_dim=1).float()
        self.ucp_loc = ucp_loc.unsqueeze(0).cuda()
        self.nudecoder = RadialBasisLayer_MultiSamples(i_size, c_factor, nucpoint_num)
        self.udecoder = RadialBasisLayer_MultiSamples_G(self.ucp_loc, i_size, c_factor, ucpoint_num)
        self.nu_cpoint_maxmin = MaxMinPointDist(nucpoint_num)
        self.u_cpoint_maxmin = MaxMinPointDist(ucpoint_num)
        self.B_u = self.calc_B(self.ucp_loc, 'u')

        # device = self.B_u.device
        # self.lst_nump_blocks, self.flag_modules = self.lst_blocks_iid(device)
        # self.log_prior_prec = nn.parameter.Parameter(torch.ones(self.lst_nump_blocks.shape), requires_grad=True)

        # generate a name
        self.name = 'lam' + str(int(self.similarity_factor / 1e3)) + 'k' + 'jac' + str(int(self.jac))

    def objective(self, src, tgt, nucp_loc):

        ucp_loc = self.ucp_loc.repeat(src.shape[0], 1, 1)

        nucp_loc = nucp_loc - self.gap
        u_mu, u_log_var, nu_mu, nu_log_var = self(src, tgt, nucp_loc, ucp_loc)
        nucp_loc = nucp_loc + self.gap

        self.nudecoder.get_weight(nucp_loc)

        nualpha = self.sample(nu_mu, nu_log_var)
        ualpha = self.sample(u_mu, u_log_var)

        loss, jacobian_loss, _lcc, kl, kl_nu, kl_u = self.calc_loss(src, tgt, nualpha, ualpha, nu_mu, nu_log_var, u_mu, u_log_var, nucp_loc)

        self.nudecoder.del_weight()

        return {'loss': loss,
                'loss_similarity': _lcc,
                'loss_kl': kl / self.similarity_factor,
                'loss_jacobian': jacobian_loss,
                'lcc': 1 - _lcc,
                'kl': kl,
                'nu_kl': kl_nu,
                'u_kl': kl_u}

    def test(self, src, tgt, nucp_loc):

        ucp_loc = self.ucp_loc.repeat(src.shape[0], 1, 1)

        ucp_loc = nucp_loc - self.gap
        uzMean, uzVariance, nuzMean, nuzVariance = self.encoder(src, tgt, nucp_loc, ucp_loc)
        nucp_loc = nucp_loc + self.gap
        self.nudecoder.get_weight(nucp_loc)

        nuflow = self.nudecoder(nuzMean.unsqueeze(1))  # nuzMean (B, 64, 2)
        uflow = self.udecoder(uzMean.unsqueeze(1))
        flow = nuflow + uflow
        flow = flow.squeeze(1)

        warped_src = self.transformer(src, flow)

        zMean = torch.cat([nuzMean, uzMean], 1)
        zVariance = torch.cat([nuzVariance, uzVariance], 1)

        self.nudecoder.del_weight()

        return flow, warped_src, (nucp_loc, ucp_loc), (zMean, zVariance)

    def calc_loss(self, src, tgt, nualpha, ualpha, nu_mu, nu_log_var, u_mu, u_log_var, nucp_loc):

        _lcc, jacobian_loss = self.calc_log_pxz(src, tgt, nualpha, ualpha)
        kl, kl_nu, kl_u = self.calc_closed_kl_loss(nu_mu, nu_log_var, u_mu, u_log_var, nucp_loc)
        loss = _lcc + kl / self.similarity_factor + jacobian_loss

        return loss, jacobian_loss, _lcc, kl, kl_nu, kl_u

    def calc_log_pxz(self, src, tgt, nualpha, ualpha):
        nuflow = self.nudecoder(nualpha.unsqueeze(1))
        uflow = self.udecoder(ualpha.unsqueeze(1))
        flow = nuflow + uflow
        flow = flow.squeeze(1)

        warped_src = self.transformer(src, flow, train=True)
        _lcc = self.similarity_loss(warped_src, tgt)

        if self.jac != 0:
            jacobian_loss = self.jacobian_loss(flow) / self.jac
        else:
            jacobian_loss = torch.zeros_like(_lcc)

        return _lcc, jacobian_loss
    
    def calc_closed_kl_loss(self, nu_mu, nu_log_var, u_mu, u_log_var, nucp_loc):
        
        B_nu = self.calc_B(nucp_loc, 'nu').unsqueeze(1)
        B_u = self.B_u.unsqueeze(1)

        nu_mu, nu_log_var = nu_mu.permute(0, 2, 1).unsqueeze(-2), nu_log_var.permute(0, 2, 1)
        u_mu, u_log_var = u_mu.permute(0, 2, 1).unsqueeze(-2), u_log_var.permute(0, 2, 1)

        kl_nu = 0.5 * (- torch.sum(nu_log_var, dim=-1) + torch.sum(torch.exp(nu_log_var), dim=-1) + torch.matmul(torch.matmul(nu_mu, B_nu), nu_mu.permute(0, 1, 3, 2)).flatten(1)).sum(-1)

        kl_u = 0.5 * (- torch.sum(u_log_var, dim=-1) + torch.sum(torch.exp(u_log_var), dim=-1) + torch.matmul(torch.matmul(u_mu, B_u), u_mu.permute(0, 1, 3, 2)).flatten(1)).sum(-1)

        kl = kl_nu + kl_u
        
        return kl, kl_nu, kl_u

    def calc_B(self, cp_loc, type):
        # 获取距离矩阵和缩放因子
        if type == 'nu':
            c, dist = self.nu_cpoint_maxmin(cp_loc)
        else:
            c, dist = self.u_cpoint_maxmin(cp_loc)
        
        # 计算缩放后的距离，使用inplace操作
        c = (c * self.c_factor).unsqueeze(1).unsqueeze(1)
        dist.div_(c)
        
        # 使用where操作替代显式的mask乘法，并合并计算步骤
        # 当dist >= 1时返回0，否则计算(1-dist)^4 * (4*dist + 1)
        dist_lt_1 = dist < 1
        dist_sub = torch.where(dist_lt_1, 1 - dist, torch.zeros_like(dist))
        B = dist_sub.pow_(4)
        B.mul_(torch.where(dist_lt_1, 4 * dist + 1, torch.zeros_like(dist)))
        
        return B

    def sample(self, mu, log_var):
        eps = torch.randn(mu.size(), device=mu.device)
        std = torch.exp(0.5 * log_var)
        return mu + std * eps

    def forward(self, src, tgt, nucp_loc, ucp_loc):
        uzMean, uzVariance, nuzMean, nuzVariance = self.encoder(src, tgt, nucp_loc, ucp_loc)
        return uzMean, uzVariance, nuzMean, nuzVariance
    
    def test(self, src, tgt, nucp_loc):
        
        ucp_loc = self.ucp_loc.repeat(src.shape[0], 1, 1)

        nucp_loc = nucp_loc - self.gap
        uzMean, uzVariance, nuzMean, nuzVariance = self.encoder(src, tgt, nucp_loc, ucp_loc)
        nucp_loc = nucp_loc + self.gap
        
        self.nudecoder.get_weight(nucp_loc)

        # nuflow = self.nudecoder(nuzMean.unsqueeze(1))  # nuzMean (B, 64, 2)
        # uflow = self.udecoder(uzMean.unsqueeze(1))

        nualpha = self.sample(nuzMean, nuzVariance)
        ualpha = self.sample(uzMean, uzVariance)
        nuflow = self.nudecoder(nualpha.unsqueeze(1))
        uflow = self.udecoder(ualpha.unsqueeze(1))

        flow = nuflow + uflow
        flow = flow.squeeze(1)

        warped_src = self.transformer(src, flow)

        zMean = torch.cat([nuzMean, uzMean], 1)
        zVariance = torch.cat([nuzVariance, uzVariance], 1)

        self.nudecoder.del_weight()

        return flow, warped_src, (nucp_loc, ucp_loc), (zMean, zVariance)


class RBFDCNUInterEncoder(nn.Module):
    
    def __init__(self,
                 gap=32,
                 dims=[16, 32, 32, 64, 64],
                 num_layers=[1, 1, 1, 1, 1],
                 local_dims=[16, 32, 32, 64],
                 local_num_layers=[2, 2, 2, 2]):
        super(RBFDCNUInterEncoder, self).__init__()

        self.gap = gap

        # 双线性插值层
        self.Interpolation = InterpolationLayer()

        # uniform branch
        self.cdb0 = []
        self.cdb0.append(Conv2dBlock(num_layers[0], 2, dims[0]))
        self.cdb0.append(Conv2dLayer(dims[0], dims[1], 3, 2, 1))
        self.cdb0 = nn.Sequential(*self.cdb0)

        self.cdb1 = []
        self.cdb1.append(Conv2dBlock(num_layers[1], dims[1], dims[1]))
        self.cdb1.append(Conv2dLayer(dims[1], dims[2], 3, 2, 1))
        self.cdb1 = nn.Sequential(*self.cdb1)

        self.cdb2 = []
        self.cdb2.append(Conv2dBlock(num_layers[2], dims[2], dims[2]))
        self.cdb2.append(Conv2dLayer(dims[2], dims[3], 3, 2, 1))
        self.cdb2 = nn.Sequential(*self.cdb2)

        self.cdb3 = []
        self.cdb3.append(Conv2dBlock(num_layers[3], dims[3], dims[3]))
        self.cdb3.append(Conv2dLayer(dims[3], dims[4], 5, 1, 0))
        self.cdb3 = nn.Sequential(*self.cdb3)

        self.cb4 = Conv2dBlock(num_layers[4], dims[4], dims[4])

        # uniform alpha
        self.uzMean_Layer = []
        self.uzMean_Layer.append(Conv1dLayer(64, 8, 1, stride=1, activation=False, batchNorm=False))
        self.uzMean_Layer.append(Conv1dLayer(8, 2, 1, stride=1, activation=False, batchNorm=False))
        self.uzMean_Layer = nn.Sequential(*self.uzMean_Layer)

        self.uzVariance_Layer = []
        self.uzVariance_Layer.append(Conv1dLayer(64, 8, 1, stride=1, activation=False, batchNorm=False))
        self.uzVariance_Layer.append(Conv1dLayer(8, 2, 1, stride=1, activation=False, batchNorm=False))
        self.uzVariance_Layer = nn.Sequential(*self.uzVariance_Layer)
        
        # nonuniform branch
        self.dcb0 = Conv2dLayer(2, local_dims[0], 3)  # 64

        self.dcdb1 = []
        self.dcdb1.append(Conv2dBlock(local_num_layers[0], local_dims[0], local_dims[0]))
        self.dcdb1.append(Conv2dLayer(local_dims[0], local_dims[1], 3, 2, 1))
        self.dcdb1 = nn.Sequential(*self.dcdb1)

        self.dcdb2 = []
        self.dcdb2.append(Conv2dBlock(local_num_layers[1], local_dims[1], local_dims[1]))
        self.dcdb2.append(Conv2dLayer(local_dims[1], local_dims[2], 3, 2, 1))
        self.dcdb2 = nn.Sequential(*self.dcdb2)

        self.dcdb3 = []
        self.dcdb3.append(Conv2dBlock(local_num_layers[2], local_dims[2], local_dims[2]))
        self.dcdb3.append(Conv2dLayer(local_dims[2], local_dims[3], 3, 2, 1))
        self.dcdb3 = nn.Sequential(*self.dcdb3)

        self.dcb4 = Conv2dBlock(local_num_layers[3], local_dims[3], local_dims[3])

        # nonuniform alpha
        self.featureEncoder = FeatureEncoder(local_dims[3])
        self.pointEncoder = PointEncoder(2)

        self.dc_layer = []
        self.dc_layer.append(Conv1dLayer(1088 * 2, 512, 1))
        self.dc_layer.append(Conv1dLayer(512, 256, 1))
        self.dc_layer.append(Conv1dLayer(256, 128, 1))
        self.dc_layer.append(Conv1dLayer(128, 64, 1))
        self.dc_layer.append(Conv1dLayer(64, 8, 1))
        self.dc_layer = nn.Sequential(*self.dc_layer)

        # 在下分支编码器的末尾加1个conv1d卷积块
        self.dcb5 = []
        self.dcb5.append(Conv1dLayer(8, 8, 1, activation=True, batchNorm=False))
        self.dcb5.append(Conv1dLayer(8, 8, 1, activation=True, batchNorm=False))
        self.dcb5 = nn.Sequential(*self.dcb5)

        self.nuzMean_Layer = Conv1dLayer(8, 2, 1, stride=1, activation=False, batchNorm=False)
        self.nuzVariance_Layer = Conv1dLayer(8, 2, 1, stride=1, activation=False, batchNorm=False)
    
    def forward(self, src, tgt, nucp_loc, ucp_loc):
        
        # uniform branch
        x_in = torch.cat((src, tgt), 1)

        x0 = self.cdb0(x_in)
        x1 = self.cdb1(x0)
        x2 = self.cdb2(x1)
        x3 = self.cdb3(x2)
        x4 = self.cb4(x3)

        # uniform alpha
        xi = self.Interpolation(x4, ucp_loc, int(src.shape[-1] / ucp_loc.shape[1] ** 0.5))  # (32, 64, 8, 8) -> (32, 64, 64)

        uzMean = self.uzMean_Layer(xi) # (32, 2, 64)
        uzVariance = self.uzVariance_Layer(xi)

        # nonuniform branch
        dx0 = self.dcb0(x_in[:, :, self.gap: src.shape[-1] - self.gap, self.gap: src.shape[-1] - self.gap])
        
        dx1 = self.dcdb1(dx0)
        dx2 = self.dcdb2(dx1)
        dx3 = self.dcdb3(dx2)

        dx4 = self.dcb4(dx3)

        # nonuniform alpha
        dxi = self.Interpolation(dx4, nucp_loc, int((src.shape[-1] - 2 * self.gap) / nucp_loc.shape[1] ** 0.5))  # (32, 64, 8, 8) -> (32, 64, 64)

        dxfe = self.featureEncoder(dxi)  # (32, 64, 64) -> (32, 1088, 64)
        dxpe = self.pointEncoder(nucp_loc / (src.shape[-1] - 2 * self.gap))  # (32, 2, 64) -> (32, 1088, 64)
        dxe = torch.cat([dxfe, dxpe], dim=1)

        dxe = self.dc_layer(dxe)

        # 新加的conv1d卷积块
        dxe = self.dcb5(dxe)

        nuzMean = self.nuzMean_Layer(dxe)
        nuzVariance = self.nuzVariance_Layer(dxe)

        uzMean = uzMean.transpose(1, 2)
        uzVariance = uzVariance.transpose(1, 2)
        nuzMean = nuzMean.transpose(1, 2)
        nuzVariance = nuzVariance.transpose(1, 2)

        return uzMean, uzVariance, nuzMean, nuzVariance







