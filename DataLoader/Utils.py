import numpy as np
import math
import torch


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def crop(img, center, vol_size):
    center_x, center_y = center
    return img[center_y - int(vol_size[0] / 2):center_y + int(vol_size[0] / 2),
               center_x - int(vol_size[1] / 2):center_x + int(vol_size[1] / 2)]


def randomTransform(degree_interval, translate_interval, scale_interval,
                    image_size):
    degree = np.random.uniform(degree_interval[0], degree_interval[1])
    degree = degree * math.pi / 180
    scale = np.random.uniform(scale_interval[0], scale_interval[1])
    tx = np.random.uniform(translate_interval[0], translate_interval[1]) / image_size[1] * 2
    ty = np.random.uniform(translate_interval[0], translate_interval[1]) / image_size[0] * 2

    rm = np.array([[math.cos(degree), -math.sin(degree), 0], [math.sin(degree), math.cos(degree), 0], [0, 0, 1]])
    tm = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    sm = np.array([[scale, 0, tx], [0, scale, ty], [0, 0, 1]])
    # combine the transforms
    m = np.matmul(sm, np.matmul(tm, rm))
    # remove the last row; it's not used by affine transform
    return m[0:2, 0:3]


class ContourPointExtractor():
    def __init__(self, img_size, gap, num):
        super(ContourPointExtractor, self).__init__()

        self.img_size = img_size
        self.gap = gap
        self.num = num
        self.nu_sqrt_cps = int(num ** 0.5)
        
    def farthest_point_sample(self, x, y):
        distance = np.ones(len(x)) * 1e10
        select_index = [0]
        currentX, currentY = x[0], y[0]
        for i in range(self.num - 1):
            dist = (x - currentX) ** 2 + (y - currentY) ** 2
            mask = dist < distance
            distance[mask] = dist[mask]
            ind = np.argmax(distance)
            select_index.append(ind)
            currentX, currentY = x[ind], y[ind]
        return x[select_index], y[select_index]

    def paddingCP(self, num, x, y, diff):
        
        if num == 0:
            pointX = np.linspace(0, 63, int(self.num ** 0.5), dtype = int)[:, np.newaxis].repeat(int(self.num ** 0.5), 1).T.flatten()
            pointY = np.linspace(0, 63, int(self.num ** 0.5), dtype = int)[:, np.newaxis].repeat(int(self.num ** 0.5), 1).flatten()
            return pointX, pointY

        left = np.min(x)
        right = np.max(x)
        top = np.min(y)
        bottom = np.max(y)
        
        gapX = right - left + 1
        gapY = bottom - top + 1
        if gapX < self.nu_sqrt_cps:
            if left + gapX - self.nu_sqrt_cps >= 0:
                left = left + gapX - self.nu_sqrt_cps
            else:
                left = 0
                right = right + self.nu_sqrt_cps - gapX
        if gapY < self.nu_sqrt_cps:
            if top + gapY - self.nu_sqrt_cps >= 0:
                top = top + gapY - self.nu_sqrt_cps
            else:
                top = 0
                bottom = bottom + self.nu_sqrt_cps - gapY
                
        # 取出最小包围盒中的非轮廓点
        tempX, tempY = np.where(diff[left: right + 1, top: bottom + 1] == 0) 
        tempX = tempX + left
        tempY = tempY + top

        # 填充非轮廓点
        index = np.linspace(0, len(tempX) - 1, self.num - num, dtype = int)
        pointX = np.append(x, tempX[index])
        pointY = np.append(y, tempY[index])

        return pointX, pointY

    def contourPoints(self, seg):
        h, w = seg.shape
        seg_h = np.concatenate((seg[:, 1: w], np.expand_dims(seg[:, w - 1], 1)), axis=1)
        diffH = np.abs(seg_h - seg)
        seg_v = np.concatenate((seg[1: h, :], np.expand_dims(seg[h - 1, :], 0)), axis=0)
        diffV = np.abs(seg_v - seg)
        diff = diffH + diffV

        return diff

    def getControlPoint(self, segement_input):
        
        segement = segement_input[self.gap: self.img_size[0] - self.gap, self.gap: self.img_size[0] - self.gap]
        diff = self.contourPoints(segement)

        x, y = np.where(diff != 0)
        num = len(x)

        if num < self.num:
            pointX, pointY = self.paddingCP(num, x, y, diff)
        else:
            pointX, pointY = self.farthest_point_sample(x, y)
        
        nucp_loc = torch.from_numpy(np.expand_dims(np.transpose(np.vstack((pointY, pointX))), 0)).float()
        
        return nucp_loc + self.gap
