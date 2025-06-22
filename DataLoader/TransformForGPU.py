import math
import torch
import numpy as np
import torchvision.transforms as transforms


class CentralCropTensor(object):
    def __init__(self, img_size):
        self.trans = transforms.CenterCrop(img_size)

    def __call__(self, pair):
        original_img_size = pair['src']['img'].shape[-1]

        imgs = torch.cat((pair['src']['img'], pair['tgt']['img'], pair['src']['seg'], pair['tgt']['seg']), dim=1)
        if pair['src']['genSeg'] != []:
            imgs = torch.cat((imgs, pair['src']['genSeg'], pair['tgt']['genSeg']), dim=1)

        imgs = self.trans(imgs)

        pair['src']['img'] = imgs[:, 0 : 1, ...]
        pair['tgt']['img'] = imgs[:, 1 : 2, ...]
        pair['src']['seg'] = imgs[:, 2 : 3, ...]
        pair['tgt']['seg'] = imgs[:, 3 : 4, ...]

        if pair['src']['genSeg'] != []:
            pair['src']['genSeg'] = imgs[:, 4 : 5, ...]
            pair['tgt']['genSeg'] = imgs[:, 5 : 6, ...]

        if pair['src'].get('NuPs') != None:
            new_img_size = self.trans.size[0]
            crop_size = (original_img_size - new_img_size) / 2.0

            nu_cps_src = pair['src']['NuPs'] - crop_size
            nu_cps_tgt = pair['tgt']['NuPs'] - crop_size

            pair['src']['NuPs'] = nu_cps_src
            pair['tgt']['NuPs'] = nu_cps_tgt

            pass
        
        return pair


class RandomAffineTransform(transforms.RandomAffine):
    def __init__(self,
                 degrees=[-45, 45],
                 translate=[0.1, 0.1],
                 scale=[0.8, 1.2],
                 img_size=[200, 200]):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.img_size = img_size

    def __call__(self, pair):

        imgs = torch.cat((pair['src']['img'], pair['tgt']['img']), dim=1)
        masks = torch.cat((pair['src']['seg'], pair['tgt']['seg']), dim=1)


        if pair['src']['genSeg'] != []:
            masks = torch.cat((masks, pair['src']['genSeg'], pair['tgt']['genSeg']), dim=1)
        
        # angle, translations, scale, shear=None
        params = self.get_params(degrees=self.degrees, 
                                 translate=self.translate, 
                                 scale_ranges=self.scale,
                                 shears=None,
                                 img_size=self.img_size)
        
        imgs = transforms.functional.affine(imgs, *params, interpolation=transforms.InterpolationMode.BILINEAR)
        masks = transforms.functional.affine(masks, *params, interpolation=transforms.InterpolationMode.NEAREST)

        pair['src']['img'] = imgs[:, 0 : 1, ...]
        pair['tgt']['img'] = imgs[:, 1 : 2, ...]

        pair['src']['seg'] = masks[:, 0 : 1, ...]
        pair['tgt']['seg'] = masks[:, 1 : 2, ...]

        if pair['src']['genSeg'] != []:
            pair['src']['genSeg'] = masks[:, 2 : 3, ...]
            pair['tgt']['genSeg'] = masks[:, 3 : 4, ...]
        
        if pair['src'].get('NuPs') != None:
            nu_cps_src = pair['src']['NuPs']
            nu_cps_tgt = pair['tgt']['NuPs']

            affine_matrix = torch.from_numpy(self.calc_affine_matrix(params)).float().unsqueeze(0).unsqueeze(0).to(nu_cps_src.device)

            ones_matrix = torch.ones_like(nu_cps_src)[:, :, :1]
            nu_cps_src = torch.cat((nu_cps_src, ones_matrix), dim=-1).unsqueeze(-1)
            nu_cps_src = torch.matmul(affine_matrix, nu_cps_src).squeeze(-1)
            
            nu_cps_tgt = torch.cat((nu_cps_tgt, ones_matrix), dim=-1).unsqueeze(-1)
            nu_cps_tgt = torch.matmul(affine_matrix, nu_cps_tgt).squeeze(-1)

            pair['src']['NuPs'] = nu_cps_src
            pair['tgt']['NuPs'] = nu_cps_tgt

        return pair
    
    def calc_affine_matrix(self, params):

        rot = math.radians(params[0])
        sx = math.radians(params[3][0])
        sy = math.radians(params[3][1])

        cx, cy = 100, 100
        tx, ty = params[1][0], params[1][1]

        a = math.cos(rot - sy) / math.cos(sy)
        b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
        c = math.sin(rot - sy) / math.cos(sy)
        d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * params[2] for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

        matrix = np.array(matrix).reshape(2, 3)

        return matrix


class RandomMirrorTensor2D(object):

    def __init__(self):
        self.trans_HF = transforms.RandomHorizontalFlip(p=1)
        self.trans_VF = transforms.RandomVerticalFlip(p=1)

    def __call__(self, pair):

        width = pair['src']['img'].shape[-1]
        
        imgs = torch.cat((pair['src']['img'], pair['tgt']['img'], pair['src']['seg'], pair['tgt']['seg']), dim=1)
        if pair['src']['genSeg'] != []:
            imgs = torch.cat((imgs, pair['src']['genSeg'], pair['tgt']['genSeg']), dim=1)

        coin_x = np.random.randint(0, 2)
        coin_y = np.random.randint(0, 2)

        if coin_x:
            imgs = self.trans_HF(imgs)
        if coin_y:
            imgs = self.trans_VF(imgs)

        pair['src']['img'] = imgs[:, 0 : 1, ...]
        pair['tgt']['img'] = imgs[:, 1 : 2, ...]
        pair['src']['seg'] = imgs[:, 2 : 3, ...]
        pair['tgt']['seg'] = imgs[:, 3 : 4, ...] 

        if pair['src']['genSeg'] != []:
            pair['src']['genSeg'] = imgs[:, 4 : 5, ...]
            pair['tgt']['genSeg'] = imgs[:, 5 : 6, ...]

        if pair['src'].get('NuPs') != None:
            nu_cps_src_x = pair['src']['NuPs'][:, :, :1]
            nu_cps_src_y = pair['src']['NuPs'][:, :, 1:]
            nu_cps_tgt_x = pair['tgt']['NuPs'][:, :, :1]
            nu_cps_tgt_y = pair['tgt']['NuPs'][:, :, 1:]

            if coin_x:
                nu_cps_src_x = width - 1 - nu_cps_src_x
                nu_cps_tgt_x = width - 1 - nu_cps_tgt_x
            if coin_y:
                nu_cps_src_y = width - 1 - nu_cps_src_y
                nu_cps_tgt_y = width - 1 - nu_cps_tgt_y
            
            nu_cps_src = torch.cat((nu_cps_src_x, nu_cps_src_y), dim=-1)
            nu_cps_tgt = torch.cat((nu_cps_tgt_x, nu_cps_tgt_y), dim=-1)

            nu_cps_src = torch.clamp(nu_cps_src, min=0, max=width - 1)
            nu_cps_tgt = torch.clamp(nu_cps_tgt, min=0, max=width - 1)

            pair['src']['NuPs'] = nu_cps_src
            pair['tgt']['NuPs'] = nu_cps_tgt

        return pair


class NormalizeTensor(object):
    def normalize(self, tensor):
        return (tensor - torch.min(tensor, dim=0)[0]) / (
            torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0])

    def __call__(self, pair):
        pair['src']['img'] = self.normalize(pair['src']['img'])
        pair['tgt']['img'] = self.normalize(pair['tgt']['img'])
        return pair
