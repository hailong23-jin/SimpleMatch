r"""PF-PASCAL dataset"""
import copy
import json
import scipy.io as sio
import pandas as pd
import numpy as np
import os.path as osp
import random
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset

from mmengine import DATASETS
from mmengine.dataset import Compose


@DATASETS.register_module()
class PFPascalDataset(Dataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, data_root, split, pipeline, demo_sample=-1):
        r"""PF-PASCAL dataset constructor"""
        super(PFPascalDataset, self).__init__()
        self.split = split
        self.demo_sample = demo_sample
        dataset_dir = osp.join(data_root, 'PF-PASCAL')
        self.data_path = osp.join(dataset_dir, split + '_pairs.csv')
        self.img_dir = osp.join(dataset_dir, 'JPEGImages')
        self.ann_dir = osp.join(dataset_dir, 'Annotations')

        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.pipeline = Compose(pipeline)

        if split == 'trn' or split == 'val':
            self.data_list = self.load_data_list()
        else:
            self.data_list = self.load_data_list()

    def load_train_data_list(self):
        with open('data/PF-PASCAL/traindata.json') as f:
            data = json.load(f)

        max_samples = 500
        data_list = []
        for category, filenames in data.items():
            if len(filenames) > max_samples:
                filenames = random.sample(filenames, max_samples)
            for src_name, trg_name in filenames:
                src_kps, kps_ids, src_bbox = read_from_mat(osp.join(self.ann_dir, category, src_name + '.mat'))
                trg_kps, _, trg_bbox = read_from_mat(osp.join(self.ann_dir, category, trg_name + '.mat'))
                sample = {
                    'category': category,
                    'category_id': self.cls.index(category),
                    'src_img_path': osp.join(self.img_dir, src_name + '.jpg'),
                    'trg_img_path': osp.join(self.img_dir, trg_name + '.jpg'),
                    'pair_name': f'{src_name}-{trg_name}:{category}',
                    'src_kps': src_kps,
                    'trg_kps': trg_kps,
                    'kps_ids': kps_ids,
                    'n_pts': src_kps.shape[-1],
                    'src_bbox': src_bbox,
                    'trg_bbox': trg_bbox,
                    'is_flip': 0
                }
                data_list.append(sample)

        return data_list

    def load_data_list(self):
        self.train_data = pd.read_csv(self.data_path)
        src_imnames = np.array(self.train_data.iloc[:, 0])
        trg_imnames = np.array(self.train_data.iloc[:, 1])
        cls_ids = self.train_data.iloc[:, 2].values.astype('int') - 1
        if self.split == 'trn':
            flip = self.train_data.iloc[:, 3].values.astype('int')
        else:
            flip = [0] * len(src_imnames)

        data_list = []
        for src_path, trg_path, category_id, is_flip in zip(src_imnames, trg_imnames, cls_ids, flip):
            category = self.cls[category_id]
            src_imname = osp.basename(src_path)[:-4]
            trg_imname = osp.basename(trg_path)[:-4]
            src_anns = osp.join(self.ann_dir, category, src_imname + '.mat')
            trg_anns = osp.join(self.ann_dir, category, trg_imname + '.mat')

            src_kps, kps_ids, src_bbox = read_from_mat(src_anns)
            trg_kps, _, trg_bbox = read_from_mat(trg_anns)

            sample = {
                'category': category,
                'category_id': category_id,
                'src_img_path': osp.join(self.img_dir, src_imname + '.jpg'),
                'trg_img_path': osp.join(self.img_dir, trg_imname + '.jpg'),
                'pair_name': f'{src_imname}-{trg_imname}:{category}',
                'src_kps': src_kps,
                'trg_kps': trg_kps,
                'kps_ids': kps_ids,
                'n_pts': src_kps.shape[-1],
                'src_bbox': src_bbox,
                'trg_bbox': trg_bbox,
                'is_flip': int(is_flip)
            }
            # if 'cat' in sample['category']:
            data_list.append(sample)
            # if category not in data_dict:
            #     data_dict[category] = [src_imname, trg_imname]
            # else:
            #     data_dict[category] += [src_imname, trg_imname]
        
        # print(f'-------{self.split}--------')
        # for k, v in data_dict.items():
        #     print(k, len(set(v)), len(v)/2)
        # print('=======================')
        return data_list


    def __len__(self):
        if self.demo_sample > 0:
            return self.demo_sample
        else:
            return len(self.data_list)

    def __getitem__(self, idx):
        batch = copy.deepcopy(self.data_list[idx])

        sample = copy.deepcopy(self.data_list[random.randint(0, len(self.data_list) - 1)])
        batch['sample'] = {
            'src_img_path': sample['src_img_path'],
            'trg_img_path': sample['trg_img_path'],
        }

        batch = self.pipeline(batch)
        
        return batch


@DATASETS.register_module()
class UnsupervisePFPascalDataset(Dataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, data_root, split, pipeline, demo_sample=-1):
        r"""PF-PASCAL dataset constructor"""
        super(UnsupervisePFPascalDataset, self).__init__()
        self.split = split
        self.demo_sample = demo_sample
        dataset_dir = osp.join(data_root, 'PF-PASCAL')
        self.data_path = osp.join(dataset_dir, split + '_pairs.csv')
        self.img_dir = osp.join(dataset_dir, 'JPEGImages')
        self.ann_dir = osp.join(dataset_dir, 'Annotations')
        self.ckpt_dir = osp.join(dataset_dir, 'tensors_image/test_dino_840x840')

        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.pipeline = Compose(pipeline)

        if split == 'trn' or split == 'val':
            self.data_list = self.load_data_list()
        else:
            self.data_list = self.load_data_list()
            images = self.load_unsupervised_train_data_list()
            self.name_to_tensor = self.pre_load_tensors(images)
            self.name_to_image = self.pred_load_images(images)

    def load_train_data_list(self):
        with open('data/PF-PASCAL/traindata.json') as f:
            data = json.load(f)

        max_samples = 500
        data_list = []
        for category, filenames in data.items():
            if len(filenames) > max_samples:
                filenames = random.sample(filenames, max_samples)
            for src_name, trg_name in filenames:
                src_kps, kps_ids, src_bbox = read_from_mat(osp.join(self.ann_dir, category, src_name + '.mat'))
                trg_kps, _, trg_bbox = read_from_mat(osp.join(self.ann_dir, category, trg_name + '.mat'))
                sample = {
                    'category': category,
                    'category_id': self.cls.index(category),
                    'src_img_path': osp.join(self.img_dir, src_name + '.jpg'),
                    'trg_img_path': osp.join(self.img_dir, trg_name + '.jpg'),
                    'pair_name': f'{src_name}-{trg_name}:{category}',
                    'src_kps': src_kps,
                    'trg_kps': trg_kps,
                    'kps_ids': kps_ids,
                    'n_pts': src_kps.shape[-1],
                    'src_bbox': src_bbox,
                    'trg_bbox': trg_bbox,
                    'is_flip': 0
                }
                data_list.append(sample)

        return data_list

    def load_data_list(self):
        self.train_data = pd.read_csv(self.data_path)
        src_imnames = np.array(self.train_data.iloc[:, 0])
        trg_imnames = np.array(self.train_data.iloc[:, 1])
        cls_ids = self.train_data.iloc[:, 2].values.astype('int') - 1
        if self.split == 'trn':
            flip = self.train_data.iloc[:, 3].values.astype('int')
        else:
            flip = [0] * len(src_imnames)

        data_list = []
        for src_path, trg_path, category_id, is_flip in zip(src_imnames, trg_imnames, cls_ids, flip):
            category = self.cls[category_id]
            src_imname = osp.basename(src_path)[:-4]
            trg_imname = osp.basename(trg_path)[:-4]
            src_anns = osp.join(self.ann_dir, category, src_imname + '.mat')
            trg_anns = osp.join(self.ann_dir, category, trg_imname + '.mat')

            src_kps, kps_ids, src_bbox = read_from_mat(src_anns)
            trg_kps, _, trg_bbox = read_from_mat(trg_anns)

            sample = {
                'category': category,
                'category_id': category_id,
                'src_img_path': osp.join(self.img_dir, src_imname + '.jpg'),
                'trg_img_path': osp.join(self.img_dir, trg_imname + '.jpg'),
                'pair_name': f'{src_imname}-{trg_imname}:{category}',
                'src_kps': src_kps,
                'trg_kps': trg_kps,
                'kps_ids': kps_ids,
                'n_pts': src_kps.shape[-1],
                'src_bbox': src_bbox,
                'trg_bbox': trg_bbox,
                'is_flip': int(is_flip)
            }
            # if 'cat' in sample['category']:
            data_list.append(sample)
            # if category not in data_dict:
            #     data_dict[category] = [src_imname, trg_imname]
            # else:
            #     data_dict[category] += [src_imname, trg_imname]
        
        # print(f'-------{self.split}--------')
        # for k, v in data_dict.items():
        #     print(k, len(set(v)), len(v)/2)
        # print('=======================')
        return data_list

    def load_unsupervised_train_data_list(self):
        train_data = pd.read_csv(self.data_path)
        src_imnames = np.array(train_data.iloc[:, 0])
        trg_imnames = np.array(train_data.iloc[:, 1])
        cls_ids = train_data.iloc[:, 2].values.astype('int') - 1

        data_dict = dict()
        for src_path, trg_path, cls_id in zip(src_imnames, trg_imnames, cls_ids):
            src_name = osp.basename(src_path)[:-4]
            trg_name = osp.basename(trg_path)[:-4]
            if cls_id not in data_dict:
                data_dict[cls_id] = [src_name, trg_name]
            else:
                data_dict[cls_id] += [src_name, trg_name]

        data_list = []
        for cls_id, v in data_dict.items():
            arr = sorted(list(set(v)))
            for src_name in arr:
                data_list.append(osp.join(self.img_dir, src_name + '.jpg'))
        return data_list

    def pre_load_tensors(self, images):
        name_to_tensor = dict()
        for img_path in tqdm(images, ncols=80):
            name = osp.basename(img_path)[:-4]
            name_to_tensor[name] = torch.load(osp.join(self.ckpt_dir, name + '.pth'), map_location='cpu').squeeze(0)
        return name_to_tensor
    
    def pred_load_images(self, images):
        name_to_image = dict()
        for img_path in tqdm(images, ncols=80):
            name = osp.basename(img_path)[:-4]
            name_to_image[name] = np.array(Image.open(img_path).convert('RGB'))
        return name_to_image

    def __len__(self):
        if self.demo_sample > 0:
            return self.demo_sample
        else:
            return len(self.data_list)

    def __getitem__(self, idx):
        batch = copy.deepcopy(self.data_list[idx])

        if self.split == 'test':
            src_name = osp.basename(batch['src_img_path'])[:-4]
            trg_name = osp.basename(batch['trg_img_path'])[:-4]
            batch['src_img'] = self.name_to_image[src_name]
            batch['trg_img'] = self.name_to_image[trg_name]
            batch['src_feat'] = self.name_to_tensor[src_name]
            batch['trg_feat'] = self.name_to_tensor[trg_name]

        batch = self.pipeline(batch)
        
        return batch

def read_from_mat(path):  # kps bbox
    data = sio.loadmat(path)

    kps = []
    kps_idx = []
    for idx, kp in enumerate(data['kps']):
        if True in np.isnan(kp):
            continue
        else:
            kps.append(kp)
            kps_idx.append(idx)

    kps = np.stack(kps, axis=1)
    kps_idx = np.array(kps_idx)
    bbox = np.array(data['bbox'][0]).astype(np.float64)

    return kps, kps_idx, bbox