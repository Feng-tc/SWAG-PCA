import os

import numpy as np
import torch

from .Utils import crop, normalize


class Caseset(torch.utils.data.Dataset):
    def __init__(self,
                 pair_list_path,
                 pair_dir,
                 resolution_path,
                 NuPs=None, 
                 CPE=None,
                 outimg_size=(128, 128)):
        pair_list = np.loadtxt(pair_list_path).astype(np.int32)

        # TODO 训练集只包含york
        # pair_list = pair_list[pair_list[:, 0] <= 33]

        self.pair_dir = pair_dir
        self.outimg_size = outimg_size
        resolutions = np.loadtxt(resolution_path)

        self.case_list = []
        self.packed_case_pair = {}
        count = 0
        for c_no, s, ed, es in pair_list:
            ed_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, ed, s)))
            es_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, es, s)))
            src_seg = ed_unit['seg']
            tgt_seg = es_unit['seg']

            if np.sum(src_seg) == 0 or np.sum(tgt_seg) == 0:
                continue
            # for old version, need to #
            if len(np.unique(src_seg)) is not self.getDatasetName(c_no) or len(
                    np.unique(tgt_seg)) is not self.getDatasetName(c_no):
                continue

            if len(ed_unit.files) == 3 and np.sum(es_unit['genSeg']) == 0:
                count += 1
                continue

            
            src_img = normalize(ed_unit['img'])
            tgt_img = normalize(es_unit['img'])
            inimg_size = src_img.shape
            center = (inimg_size[1] // 2, inimg_size[0] // 2)

            src_img = crop(src_img, center, self.outimg_size)
            tgt_img = crop(tgt_img, center, self.outimg_size)
            src_seg = crop(src_seg, center, self.outimg_size)
            tgt_seg = crop(tgt_seg, center, self.outimg_size)

            # nu_cps_src = None
            # nu_cps_tgt = None

            if NuPs != None:

                nu_cps_src = CPE.getControlPoint(src_seg)
                nu_cps_tgt = CPE.getControlPoint(tgt_seg)

            src_img = torch.tensor(src_img).unsqueeze(0).unsqueeze(0)
            tgt_img = torch.tensor(tgt_img).unsqueeze(0).unsqueeze(0)
            src_seg = torch.tensor(src_seg).unsqueeze(0).unsqueeze(0)
            tgt_seg = torch.tensor(tgt_seg).unsqueeze(0).unsqueeze(0)

            if c_no not in self.case_list:
                self.packed_case_pair[c_no] = {
                    's_list': [],
                    'src_img': [],
                    'tgt_img': [],
                    'src_seg': [],
                    'tgt_seg': [],
                    'src_NuPs': [],
                    'tgt_NuPs': [],
                    'src_genSeg': [],
                    'tgt_genSeg': [],
                    'resolution': resolutions[c_no - 1][0]
                }
                self.case_list.append(c_no)
            self.packed_case_pair[c_no]['s_list'].append(s)
            self.packed_case_pair[c_no]['src_img'].append(src_img)
            self.packed_case_pair[c_no]['tgt_img'].append(tgt_img)
            self.packed_case_pair[c_no]['src_NuPs'].append(nu_cps_src)
            self.packed_case_pair[c_no]['tgt_NuPs'].append(nu_cps_tgt)
            self.packed_case_pair[c_no]['src_seg'].append(src_seg)
            self.packed_case_pair[c_no]['tgt_seg'].append(tgt_seg)

            # 添加生成Mask
            if len(ed_unit.files) == 3:
                src_genSeg = ed_unit['genSeg']
                tgt_genSeg = es_unit['genSeg']

                src_genSeg = crop(src_genSeg, center, self.outimg_size)
                tgt_genSeg = crop(tgt_genSeg, center, self.outimg_size)

                src_genSeg = torch.tensor(src_genSeg).unsqueeze(0).unsqueeze(0)
                tgt_genSeg = torch.tensor(tgt_genSeg).unsqueeze(0).unsqueeze(0)

                self.packed_case_pair[c_no]['src_genSeg'].append(src_genSeg)
                self.packed_case_pair[c_no]['tgt_genSeg'].append(tgt_genSeg)
   
        self.case_list.sort()

    def getDatasetName(self, case_no):
        if case_no <= 33:
            return 3
        elif case_no > 33 and case_no <= 78:
            return 2
        else:
            return 4

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        c_no = self.case_list[idx]
        return self.getByCaseNo(c_no)

    def getByCaseNo(self, c_no):
        packed_src_img = torch.cat(self.packed_case_pair[c_no]['src_img'], 0)
        packed_tgt_img = torch.cat(self.packed_case_pair[c_no]['tgt_img'], 0)
        packed_src_seg = torch.cat(self.packed_case_pair[c_no]['src_seg'], 0)
        packed_tgt_seg = torch.cat(self.packed_case_pair[c_no]['tgt_seg'], 0)
        packed_src_NuPs = torch.cat(self.packed_case_pair[c_no]['src_NuPs'], 0)
        packed_tgt_NuPs = torch.cat(self.packed_case_pair[c_no]['tgt_NuPs'], 0)
        resolution = self.packed_case_pair[c_no]['resolution']

        packed_src_genSeg = []
        packed_tgt_genSeg = []

        if len(self.packed_case_pair[c_no]['src_genSeg']) != 0:
            packed_src_genSeg = torch.cat(self.packed_case_pair[c_no]['src_genSeg'], 0)
            packed_tgt_genSeg = torch.cat(self.packed_case_pair[c_no]['tgt_genSeg'], 0)
        
        return {
            'src': packed_src_img,
            'tgt': packed_tgt_img,
            'src_seg': packed_src_seg,
            'tgt_seg': packed_tgt_seg,
            'src_NuPs': packed_src_NuPs,
            'tgt_NuPs': packed_tgt_NuPs,
            'src_genSeg': packed_src_genSeg,
            'tgt_genSeg': packed_tgt_genSeg,
            'case_no': c_no,
            'slice': self.packed_case_pair[c_no]['s_list'],
            'resolution': resolution
        }
