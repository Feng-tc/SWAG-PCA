import torch
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np

import matplotlib.pyplot as plt
from Utils import plt_everything


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, size, need_grid=True):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
            :param need_grid: to determine whether the transformer create the sampling grid
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        if need_grid:
            vectors = [torch.arange(0, s) for s in size]
            grids = torch.meshgrid(vectors)
            grid = torch.stack(grids)[[1, 0] if len(size) == 2 else
                                      [1, 0, 2]]  # y, x, z ==> x, y, z
            grid = torch.unsqueeze(grid, 0)  # add batch
            grid = grid.type(torch.FloatTensor)
            self.register_buffer('grid', grid)

        self.need_grid = need_grid

    def forward(self, src, flow, mode='bilinear', align_corners=True, train=False):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        if self.need_grid:
            new_locs = self.grid + flow
        else:
            new_locs = flow * 1.0

        shape = flow.shape[2:]
        if len(shape) == 2:
            shape = [shape[1], shape[0]]
        elif len(shape) == 3:
            shape = [shape[1], shape[0], shape[2]]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)

        # return nnf.grid_sample(src, new_locs, mode=mode, align_corners=align_corners)
        # return self.bilinear_grid_sample(src, new_locs, align_corners=True)

        # 通过torch.autograd.grad计算grid_sample函数的Hessian矩阵时，其值是不准确的。
        
        if train:
            return self.bilinear_grid_sample(src, new_locs, align_corners=True)
        else:
            return nnf.grid_sample(src, new_locs, mode=mode, align_corners=align_corners)
    
    def bilinear_grid_sample(self, im, grid, align_corners=False):
        """Given an input and a flow-field grid, computes the output using input
        values and pixel locations from grid. Supported only bilinear interpolation
        method to sample the input pixels.

        Args:
            im (torch.Tensor): Input feature map, shape (N, C, H, W)
            grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
            align_corners {bool}: If set to True, the extrema (-1 and 1) are
                considered as referring to the center points of the input’s
                corner pixels. If set to False, they are instead considered as
                referring to the corner points of the input’s corner pixels,
                making the sampling more resolution agnostic.

        Returns:
            torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
        """
        n, c, h, w = im.shape
        gn, gh, gw, _ = grid.shape
        assert n == gn

        x = grid[:, :, :, 0]
        y = grid[:, :, :, 1]

        if align_corners:
            x = ((x + 1) / 2) * (w - 1)
            y = ((y + 1) / 2) * (h - 1)
        else:
            x = ((x + 1) * w - 1) / 2
            y = ((y + 1) * h - 1) / 2

        x = x.view(n, -1)
        y = y.view(n, -1)

        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = x0 + 1
        y1 = y0 + 1

        wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
        wb = ((x1 - x) * (y - y0)).unsqueeze(1)
        wc = ((x - x0) * (y1 - y)).unsqueeze(1)
        wd = ((x - x0) * (y - y0)).unsqueeze(1)

        # Apply default for grid_sample function zero padding
        im_padded = nnf.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
        padded_h = h + 2
        padded_w = w + 2
        # save points positions after padding
        x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

        # Clip coordinates to padded image size
        x0 = torch.where(x0 < 0, torch.tensor(0).cuda(), x0)
        x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1).cuda(), x0)
        x1 = torch.where(x1 < 0, torch.tensor(0).cuda(), x1)
        x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1).cuda(), x1)
        y0 = torch.where(y0 < 0, torch.tensor(0).cuda(), y0)
        y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1).cuda(), y0)
        y1 = torch.where(y1 < 0, torch.tensor(0).cuda(), y1)
        y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1).cuda(), y1)

        im_padded = im_padded.view(n, c, -1)

        x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

        Ia = torch.gather(im_padded, 2, x0_y0)
        Ib = torch.gather(im_padded, 2, x0_y1)
        Ic = torch.gather(im_padded, 2, x1_y0)
        Id = torch.gather(im_padded, 2, x1_y1)

        return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)

    '''
    def forward(self, src, tgt, segement, flow, bps_loc, mode='bilinear', align_corners=True):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        if self.need_grid:
            new_locs = self.grid + flow
        else:
            new_locs = flow * 1.0

        shape = flow.shape[2:]
        if len(shape) == 2:
            shape = [shape[1], shape[0]]
        elif len(shape) == 3:
            shape = [shape[1], shape[0], shape[2]]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i,
                     ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)

        warped_src = nnf.grid_sample(src, new_locs, mode=mode, align_corners=align_corners)


        i = 0
        bps_mask = torch.zeros_like(src[i, 0])
        bps_flow = []
        # bps_loc: (h, w)
        for idx in bps_loc:
            bps_mask[idx[0], idx[1]] = 1
            bps_flow.append(flow[i, :, idx[0], idx[1]])
        bps_flow = torch.vstack(bps_flow).detach().cpu().numpy()

        # plt.scatter(bps_loc[:, 1], bps_loc[:, 0], s=3)
        # plt.quiver(bps_loc[:, 1], bps_loc[:, 0], bps_flow[:, 0], bps_flow[:, 1], headwidth=2)
        # plt.xlim([0, 128])
        # plt.ylim([128, 0])
        # plt.savefig('img/1.png', dpi=300)

        # colors = plt.cm.viridis(src[0, 0].detach().cpu().numpy().flatten())
        # x, y = np.meshgrid(np.arange(128), np.arange(128))
        # x = x.flatten()
        # y = y.flatten()
        # plt.scatter(x, y, s=2, c=colors)
        # plt.savefig('img/2.png', dpi=300)
        # plt.close()

        warped_bsp_loc = np.transpose(np.vstack((bps_loc[:, 0] + bps_flow[:, 1], bps_loc[:, 1] + bps_flow[:, 0])))
        
        plt.scatter(bps_loc[:, 1], bps_loc[:, 0], s=3, c='red')
        plt.scatter(warped_bsp_loc[:, 1], warped_bsp_loc[:, 0], s=3, c='green')
        plt.quiver(bps_loc[:, 1], bps_loc[:, 0], bps_flow[:, 0], bps_flow[:, 1], headwidth=2, angles='xy', color='blue')
        plt.scatter(bps_loc[0, 1], bps_loc[0, 0], s=3, c='black')
        plt.scatter(warped_bsp_loc[0, 1], warped_bsp_loc[0, 0], s=3, c='m')
        plt.xlim([0, 128])
        plt.ylim([128, 0])
        plt.savefig('img/1.png', dpi=300)

        h_loc, w_loc = bps_loc[0]
        warped_h, warped_w = h_loc + bps_flow[0][1], w_loc + bps_flow[0][0]
        left, right, top, bottom = int(np.floor(warped_w)), int(np.ceil(warped_w)), int(np.floor(warped_h)), int(np.ceil(warped_h))
        q_lt, q_rt, q_lb, q_rb = src[i, 0, top, left].item(), src[i, 0, top, right].item(), \
                                 src[i, 0, bottom, left].item(), src[i, 0, bottom, right].item()
        t = warped_src[i, 0, h_loc, w_loc]
        t_cap = self.f(q_lt, q_rt, q_lb, q_rb, warped_h, warped_w, left, right, top, bottom)
        print('1')

    def f(self, q_lt, q_rt, q_lb, q_rb, h_loc, w_loc, left, right, top, bottom):
        p_top = (right - w_loc) / (right - left) * q_lt + (w_loc - left) / (right - left) * q_rt
        p_bottom = (right - w_loc) / (right - left) * q_lb + (w_loc - left) / (right - left) * q_rb
        p = (bottom - h_loc) / (bottom - top) * p_top + (h_loc - top) / (bottom - top) * p_bottom
        return p
    '''


    