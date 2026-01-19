r"""PF-PASCAL dataset"""
import copy
import scipy.io as sio
import pandas as pd
import numpy as np
import os.path as osp
from torch.utils.data import Dataset

from mmengine import DATASETS
from mmengine.dataset import Compose


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


@DATASETS.register_module()
class PFPascalDataset(Dataset):
    def __init__(self, data_root, split, pipeline, demo_sample=-1):
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
        self.data_list = self.load_data_list()

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
            data_list.append(sample)
           
        return data_list

    def __len__(self):
        if self.demo_sample > 0:
            return self.demo_sample
        else:
            return len(self.data_list)

    def __getitem__(self, idx):
        batch = copy.deepcopy(self.data_list[idx])

        batch = self.pipeline(batch)
        
        return batch


