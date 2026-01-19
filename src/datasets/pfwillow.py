r"""PF-WILLOW dataset"""
import copy
import numpy as np
import pandas as pd
import os.path as osp

from torch.utils.data import Dataset

from mmengine.registry import DATASETS
from mmengine.dataset import Compose


@DATASETS.register_module()
class PFWillowDataset(Dataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, data_root, split, pipeline):
        r"""PF-WILLOW dataset constructor"""
        super(PFWillowDataset, self).__init__()

        dataset_dir = osp.join(data_root, 'PF-WILLOW')
        self.data_path = osp.join(dataset_dir, 'test_pairs.csv')
        self.img_dir = dataset_dir

        self.cls = ['car(G)', 'car(M)', 'car(S)', 'duck(S)',
                    'motorbike(G)', 'motorbike(M)', 'motorbike(S)',
                    'winebottle(M)', 'winebottle(wC)', 'winebottle(woC)']
        
        self.pipeline = Compose(pipeline)
        self.data_list = self.load_data_list()

    def load_data_list(self):
        train_data = pd.read_csv(self.data_path)

        data_list = []
        for _, row in train_data.iterrows():
            src_imname = osp.basename(row['imageA'])[:-4]
            trg_imname = osp.basename(row['imageB'])[:-4]
            category = row['imageA'].split('/')[1]
            category_id = self.cls.index(category)
            src_kps = row[2:22].values.reshape(2, 10).astype(np.float32)
            trg_kps = row[22:].values.reshape(2, 10).astype(np.float32)

            sample = {
                'category': category,
                'category_id': category_id,
                'src_img_path': osp.join(self.img_dir, category, src_imname + '.png'),
                'trg_img_path': osp.join(self.img_dir, category, trg_imname + '.png'),
                'pair_name': f'{src_imname}-{trg_imname}:{category}',
                'src_kps': src_kps,
                'trg_kps': trg_kps,
                'n_pts': 10,
            }
            data_list.append(sample)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        batch = copy.deepcopy(self.data_list[idx])
        batch = self.pipeline(batch)
        return batch
