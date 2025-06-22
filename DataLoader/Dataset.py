import os

import numpy as np
import torch
from .Utils import normalize

class Dataset(torch.utils.data.Dataset):
    def __init__(self, pair_list_path, pair_dir, NuPs=None, CPE=None, transform=None):
        self.pair_list = np.loadtxt(pair_list_path).astype(np.int32)

        # TODO 训练集只包含york
        # self.pair_list = self.pair_list[self.pair_list[:, 0] <= 33]

        self.transform = transform
        self.pair_dir = pair_dir
        self.preload(NuPs, CPE)

    def preload(self, NuPs, CPE):
        self.dataset = {}
        idx_del_list = []
        flag = -1
        for c_no, s, ed, es in self.pair_list:
            
            ed_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, ed, s)))
            es_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, es, s)))
            if c_no not in self.dataset:
                self.dataset[c_no] = {}

            flag += 1
            if len(ed_unit.files) == 3 and torch.sum(torch.tensor(es_unit['genSeg'].astype(np.float32)).unsqueeze(0).unsqueeze(0).float()) ==0:
                idx_del_list.append(flag)
                continue
            
            # Fusion数据集 黑图
            #       训练集  验证集  测试集
            # 10%:  9       1      6
            # 15%:  3       1      3

            # M&Ms数据集 黑图
            #       训练集  验证集  测试集
            # 10%:  9       0      2 
            # 15%:  3       0      2

            if s not in self.dataset[c_no]:
                self.dataset[c_no][s] = {
                    ed: {
                        'img': torch.tensor(normalize(ed_unit['img'].astype(np.float32))).unsqueeze(0).unsqueeze(0).float(),
                        'seg': torch.tensor(ed_unit['seg'].astype(np.float32)).unsqueeze(0).unsqueeze(0).float(),
                        'NuPs': CPE.getControlPoint(ed_unit['seg'].astype(np.float32)) if NuPs != None else None, # 这里写死为真实mask，没有冗余生成mask
                        'genSeg': torch.tensor(ed_unit['genSeg'].astype(np.float32)).unsqueeze(0).unsqueeze(0).float() if len(ed_unit.files) == 3 else [],
                    },
                    es: {
                        'img': torch.tensor(normalize(es_unit['img'].astype(np.float32))).unsqueeze(0).unsqueeze(0).float(),
                        'seg': torch.tensor(es_unit['seg'].astype(np.float32)).unsqueeze(0).unsqueeze(0).float(),
                        'NuPs': CPE.getControlPoint(es_unit['seg'].astype(np.float32)) if NuPs != None else None,
                        'genSeg': torch.tensor(es_unit['genSeg'].astype(np.float32)).unsqueeze(0).unsqueeze(0).float() if len(es_unit.files) == 3 else [],
                    },
                }
            else:
                self.dataset[c_no][s][ed] = {
                        'img': torch.tensor(normalize(ed_unit['img'].astype(np.float32))).unsqueeze(0).unsqueeze(0).float(),
                        'seg': torch.tensor(ed_unit['seg'].astype(np.float32)).unsqueeze(0).unsqueeze(0).float(),
                        'NuPs': CPE.getControlPoint(ed_unit['seg'].astype(np.float32)) if NuPs != None else None,
                        'genSeg': torch.tensor(ed_unit['genSeg'].astype(np.float32)).unsqueeze(0).unsqueeze(0).float() if len(ed_unit.files) == 3 else [],
                    }
                self.dataset[c_no][s][es] = {
                        'img': torch.tensor(normalize(es_unit['img'].astype(np.float32))).unsqueeze(0).unsqueeze(0).float(),
                        'seg': torch.tensor(es_unit['seg'].astype(np.float32)).unsqueeze(0).unsqueeze(0).float(),
                        'NuPs': CPE.getControlPoint(es_unit['seg'].astype(np.float32)) if NuPs != None else None,
                        'genSeg': torch.tensor(es_unit['genSeg'].astype(np.float32)).unsqueeze(0).unsqueeze(0).float() if len(es_unit.files) == 3 else [],
                    }
        # 在生成Mask数据中，删除tgt生成Mask为空的slice.
        self.pair_list = np.delete(self.pair_list, idx_del_list, axis=0)
        print('Done')

    def __len__(self):
        return self.pair_list.shape[0]

    def __getitem__(self, idx):
        c_no, s, ed, es = self.pair_list[idx]

        output = {
            'src': self.dataset[c_no][s][ed],
            'tgt': self.dataset[c_no][s][es]
        }

        if self.transform:
            output = self.transform(output)

        return output


class Collate(object):
    def __call__(self, batch):
        output = {
            'src': {
                'img': torch.cat([d['src']['img'] for d in batch], 0),
                'NuPs': torch.cat([d['src']['NuPs'] for d in batch], 0),
                'seg': torch.cat([d['src']['seg'] for d in batch], 0),
            },
            'tgt': {
                'img': torch.cat([d['tgt']['img'] for d in batch], 0),
                'NuPs': torch.cat([d['tgt']['NuPs'] for d in batch], 0),
                'seg': torch.cat([d['tgt']['seg'] for d in batch], 0),
            }
        }

        return output


class CollateGPU(object):
    def __init__(self, transforms=None):
        self.transforms = transforms

    def collate(self, batch):
        output = {
            'src': {
                'img': torch.cat([d['src']['img'] for d in batch], 0).cuda(),
                'seg': torch.cat([d['src']['seg'] for d in batch], 0).cuda(),
                'NuPs': torch.cat([d['src']['NuPs'] for d in batch], 0).cuda(),
                'genSeg': torch.cat([d['src']['genSeg'] for d in batch], 0).cuda() if batch[0]['src']['genSeg'] !=[] else [],
            },
            'tgt': {
                'img': torch.cat([d['tgt']['img'] for d in batch], 0).cuda(),
                'seg': torch.cat([d['tgt']['seg'] for d in batch], 0).cuda(),
                'NuPs': torch.cat([d['tgt']['NuPs'] for d in batch], 0).cuda(),
                'genSeg': torch.cat([d['tgt']['genSeg'] for d in batch], 0).cuda() if batch[0]['tgt']['genSeg'] !=[] else [],
            }
        }
        return output

    def __call__(self, batch):
        batch = self.collate(batch)
        if self.transforms:
            batch = self.transforms(batch)
        return batch
