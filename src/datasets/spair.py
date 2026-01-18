r"""SPair-71k dataset"""
import os
import json
import copy
import random
import itertools
import numpy as np
import os.path as osp
from PIL import Image

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from mmengine import DATASETS
from mmengine.dataset import Compose

def deal_kps(kps):
    out_kps = []
    out_ids = []
    for kp_id, xy in kps.items():
        if xy is not None:
            out_kps.append(xy)
            out_ids.append(int(kp_id))
    return np.array(out_kps), np.array(out_ids)

def read_all_kps(ann_file):
    data = json.load(open(ann_file))
    out_kps = []
    for kp_id, xy in data['kps'].items():
        if xy is not None:
            out_kps.append(xy)

    return np.array(out_kps).T

def read_from_json(ann_file):
    data = json.load(open(ann_file))
    src_kps = np.array(data['src_kps']).T
    trg_kps = np.array(data['trg_kps']).T
    n_pts = src_kps.shape[-1]
    src_bbox = np.array(data['src_bndbox']).astype(np.float32)
    trg_bbox = np.array(data['trg_bndbox']).astype(np.float32)
    kps_ids = np.array(data['kps_ids']).astype(int)
    return src_kps, trg_kps, n_pts, src_bbox, trg_bbox, kps_ids


def read_file():
    path = '/home/jinhl/PythonWorkspace/ours/SimpleMatch/b.txt'
    arr = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            arr.append(line)
    return arr

@DATASETS.register_module()
class SPairDataset(Dataset):

    def __init__(self, data_root, split, pipeline, demo_sample=-1):
        super(SPairDataset, self).__init__()
        # basic path
        self.split = split
        self.demo_sample = demo_sample
        dataset_dir = osp.join(data_root, 'SPair-71k')
        self.data_path = osp.join(dataset_dir, 'Layout/large', split + '.txt')
        self.img_dir = osp.join(dataset_dir, 'JPEGImages')
        self.ann_dir = osp.join(dataset_dir, 'PairAnnotation', split)
        self.img_ann_dir = osp.join(dataset_dir, 'ImageAnnotation')
        self.permutation_dir = osp.join(dataset_dir, 'Permutation', split)

        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                        'car', 'cat', 'chair', 'cow', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']

        self.pipeline = Compose(pipeline)
        self.data_list, self.data_dict = self.load_data_list()

    def load_data_list(self):
        # read data
        train_data = open(self.data_path).read().split('\n')
        train_data = train_data[:len(train_data) - 1]

        data_list = []
        data_dict = dict()

        for filename in train_data:  # 000009-2008_001546-2009_000327:aeroplane
            pair_name, category = filename.split(':')
            _, src_name, trg_name = pair_name.split('-')
            ann_file = osp.join(self.ann_dir, filename + '.json')  # # 000009-2008_001546-2009_000327:aeroplane.json
            src_kps, trg_kps, n_pts, src_bbox, trg_bbox, kps_ids = read_from_json(ann_file)
            sample = {
                'category': category,
                'category_id': self.classes.index(category),
                'src_img_path': osp.join(self.img_dir, category, src_name + '.jpg'),
                'trg_img_path': osp.join(self.img_dir, category, trg_name + '.jpg'),
                'pair_name': f'{src_name}-{trg_name}:{category}',
                'src_name': src_name,
                'trg_name': trg_name,
                'src_kps': src_kps,
                'trg_kps': trg_kps,
                'n_pts': n_pts,
                'src_bbox': src_bbox,
                'trg_bbox': trg_bbox,
                'kps_ids': kps_ids,
            }
            data_list.append(sample)
            # if category == 'person':
            #     data_list.append(sample)
            # if self.split == 'test' and f'{src_name}-{trg_name}' in names:
            #     data_list.append(sample)
            
            # if self.split != 'test':
            #     data_list.append(sample)
                
            # all_src_kps = read_all_kps(osp.join(self.img_ann_dir, category, src_name + '.json'))
            # sample['all_src_kps'] = all_src_kps
            # sample['all_n_pts'] = all_src_kps.shape[-1]

        # random.shuffle(data_list)
        return data_list, data_dict

    def __len__(self):
        if self.demo_sample > 0:
            return self.demo_sample
        else:
            return len(self.data_list)


    def __getitem__(self, idx):
        batch = copy.deepcopy(self.data_list[idx])

        batch = self.pipeline(batch)

        return batch

