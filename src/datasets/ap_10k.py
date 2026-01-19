r"""AP10k dataset"""
import json
import os
import os.path as osp
import torch
import copy
from glob import glob
import numpy as np
from tqdm import tqdm

from mmengine import DATASETS
from mmengine.dataset import Compose


@DATASETS.register_module()
class AP10kDataset(torch.nn.Module):
    def __init__(self, data_root, eval_type, split, pipeline, demo_sample=-1):
        super().__init__()
        self.demo_sample = demo_sample
        dataset_dir = osp.join(data_root, 'ap-10k')
        self.img_ann_dir = osp.join(dataset_dir, 'ImageAnnotation')
        self.pair_ann_dir = osp.join(dataset_dir, 'PairAnnotation')
        self.img_dir = osp.join(dataset_dir, 'JPEGImages')

        categories, split = self.get_categories(eval_type, split)
        self.classes = categories
        self.data_list = self.load_data_list(categories, split)
        self.pipeline = Compose(pipeline)


    def get_categories(self, eval_type, split):
        categories = []
        subfolders = os.listdir(self.img_ann_dir)
        if eval_type == 'intra-species':
            categories = [folder for subfolder in subfolders for folder in os.listdir(os.path.join(self.img_ann_dir, subfolder))]
        elif eval_type == 'cross-species':
            categories = [subfolder for subfolder in subfolders if len(os.listdir(os.path.join(self.img_ann_dir, subfolder))) > 1]
            split += '_cross_species'
        elif eval_type == 'cross-family':
            categories = ['all']
            split += '_cross_family'
        categories = sorted(categories)

        return categories, split
    
    def load_data_list(self, categories, split):
        pairs = []
        for category in categories:
            pairs += sorted(glob(f'{self.pair_ann_dir}/{split}/*:{category}.json'))
        data_list = []
        for pair in tqdm(pairs, ncols=80):
            with open(pair) as f:
                data = json.load(f)
            category = pair.split(':')[1][:-5]
            source_json_path = data["src_json_path"]
            target_json_path = data["trg_json_path"]
            src_img_path = source_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')
            trg_img_path = target_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')

            with open(source_json_path) as f:
                src_file = json.load(f)
            with open(target_json_path) as f:
                trg_file = json.load(f)

            source_bbox = np.asarray(src_file["bbox"]).astype(float)  # l t w h
            target_bbox = np.asarray(trg_file["bbox"]).astype(float)

            source_kps = torch.tensor(src_file["keypoints"]).view(-1, 3).float()
            target_kps = torch.tensor(trg_file["keypoints"]).view(-1, 3).float()
            used_kps, = torch.where(source_kps[:, 2] * target_kps[:, 2]>0)
            source_kps = source_kps[used_kps, :2]
            target_kps = target_kps[used_kps, :2]

            sample = {
                    'category': category,
                    'category_id': self.classes.index(category),
                    'src_img_path': src_img_path,
                    'trg_img_path': trg_img_path,
                    'pair_name': osp.basename(pair)[:-5],
                    'src_kps': source_kps.T.numpy(),
                    'trg_kps': target_kps.T.numpy(),
                    'n_pts': target_kps.shape[0],
                    'src_bbox': source_bbox,
                    'trg_bbox': target_bbox,
                }
            
            data_list.append(sample)

        return data_list

    def __len__(self):
        if self.demo_sample > 0:
            return self.demo_sample
        else:
            return len(self.data_list)


    def __getitem__(self, idx):  #
        batch = copy.deepcopy(self.data_list[idx])
        
        batch = self.pipeline(batch)

        batch['pckthres'] = max(batch['trg_bbox'][3], batch['trg_bbox'][2])


        return batch
    
