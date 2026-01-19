import cv2
import random
import numpy as np 
import albumentations as A

import torch
from torchvision import transforms

from mmengine.registry import TRANSFORMS


def check_kps(img, kps):
    h, w = img.shape[:2]
    assert kps[0].max() < w
    assert kps[1].max() < h

@TRANSFORMS.register_module()
class Resize:
    def __init__(self, size) -> None:
        self.size = size

    def resize_kps(self, kps, img_size):
        '''
        img_size: h x w
        '''
        kps[0, :] = kps[0, :] * (self.size[1] / img_size[1])
        kps[1, :] = kps[1, :] * (self.size[0] / img_size[0])
        return kps

    def __call__(self, batch):
        batch['src_img'] = cv2.resize(batch['src_img'], dsize=self.size)
        batch['trg_img'] = cv2.resize(batch['trg_img'], dsize=self.size)

        batch['src_kps'] = self.resize_kps(batch['src_kps'], batch['src_imsize'])
        batch['trg_kps'] = self.resize_kps(batch['trg_kps'], batch['trg_imsize'])

        return batch

@TRANSFORMS.register_module()
class RandomCrop:
    def __init__(self, crop_size, p=0.5) -> None:
        self.crop_size = crop_size
        self.p = p

    def random_crop(self, img, kps, bbox):
        h, w, _ = img.shape
        kps = kps.T
        left = random.randint(0, bbox[0])
        top = random.randint(0, bbox[1])
        height = random.randint(bbox[3], h) - top
        width = random.randint(bbox[2], w) - left
        crop_img = img[top:(top+height), left:(left+width), :]
        crop_img = cv2.resize(crop_img, dsize=self.crop_size)
        
        resized_kps = np.zeros_like(kps, dtype=np.float32)
        resized_kps[:, 0] = (kps[:, 0] - left) * (self.crop_size[1] / width)
        resized_kps[:, 1] = (kps[:, 1] - top) * (self.crop_size[0] / height)
        resized_kps = np.clip(resized_kps, 0, self.crop_size[0] - 1)
        return crop_img, resized_kps.T
    
    def random_crop_mask(self, img, mask, kps, bbox):
        h, w, _ = img.shape
        kps = kps.T
        left = random.randint(0, bbox[0])
        top = random.randint(0, bbox[1])
        height = random.randint(bbox[3], h) - top
        width = random.randint(bbox[2], w) - left
        crop_img = img[top:(top+height), left:(left+width), :]
        crop_msk = mask[top:(top+height), left:(left+width), :]
        crop_img = cv2.resize(crop_img, dsize=self.crop_size)
        crop_msk = cv2.resize(crop_msk, dsize=self.crop_size)
        
        resized_kps = np.zeros_like(kps, dtype=np.float32)
        resized_kps[:, 0] = (kps[:, 0] - left) * (self.crop_size[1] / width)
        resized_kps[:, 1] = (kps[:, 1] - top) * (self.crop_size[0] / height)
        resized_kps = np.clip(resized_kps, 0, self.crop_size[0] - 1)
        return crop_img, crop_msk, resized_kps.T

    def resize(self, img, kps):
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=self.crop_size)
        kps[0, :] = kps[0, :] * (self.crop_size[1] / w)
        kps[1, :] = kps[1, :] * (self.crop_size[0] / h)
        return img, kps
    

    def __call__(self, batch):
        if random.uniform(0, 1) > self.p:
            batch['src_img'], batch['src_kps'] = self.resize(batch['src_img'], batch['src_kps'])
            batch['trg_img'], batch['trg_kps'] = self.resize(batch['trg_img'], batch['trg_kps'])
        else:
            batch['src_img'], batch['src_kps'] = self.random_crop(batch['src_img'], batch['src_kps'], batch['src_bbox'].copy().astype(int))
            batch['trg_img'], batch['trg_kps'] = self.random_crop(batch['trg_img'], batch['trg_kps'], batch['trg_bbox'].copy().astype(int))
        return batch

@TRANSFORMS.register_module()
class RandomRotation:
    def __init__(self, p=0.5, target_size=448):
        self.p = p
        self.target_size = target_size

    def rotate_img_points(self, image, key_points, angle):
        center = ((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
 
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])

        scale = 1.0
        new_width = int(scale * (image.shape[1] * cos + image.shape[0] * sin))
        new_height = int(scale * (image.shape[1] * sin + image.shape[0] * cos))

        rotation_matrix[0, 2] += ((new_width - 1) / 2) - center[0]
        rotation_matrix[1, 2] += ((new_height - 1) / 2) - center[1]

        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

        rotated_key_points = key_points.copy()
        rotated_key_points[0, :] = key_points[0, :] * rotation_matrix[0, 0] + key_points[1, :] * rotation_matrix[0, 1] + rotation_matrix[0, 2]
        rotated_key_points[1, :] = key_points[0, :] * rotation_matrix[1, 0] + key_points[1, :] * rotation_matrix[1, 1] + rotation_matrix[1, 2]

        return rotated_image, rotated_key_points

    def resize(self, img, kps):
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(self.target_size, self.target_size))
        kps[0, :] = kps[0, :] * (self.target_size / w)
        kps[1, :] = kps[1, :] * (self.target_size / h)
        return img, kps

    def __call__(self, batch):
        if random.uniform(0, 1) < self.p:
            if batch['category'] == 'bottle':
                angle = random.random() * 360 - 180
            else:
                angle = random.random() * 60 - 30

            rand_num = random.randint(0, 2)
            if rand_num == 0 or rand_num == 2:
                batch['src_img'], batch['src_kps'] = self.rotate_img_points(batch['src_img'], batch['src_kps'], angle)
            if rand_num == 1 or rand_num == 2:
                batch['trg_img'],  batch['trg_kps'] = self.rotate_img_points(batch['trg_img'], batch['trg_kps'], angle)

        batch['src_img'], batch['src_kps'] = self.resize(batch['src_img'], batch['src_kps'])
        batch['trg_img'], batch['trg_kps'] = self.resize(batch['trg_img'], batch['trg_kps'])
        return batch
        
@TRANSFORMS.register_module()
class NormalAug:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.ToGray(p=0.1),
            A.Posterize(p=0.2),
            A.Equalize(p=0.2),
            A.augmentations.transforms.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Solarize(p=0.2),
            A.ColorJitter(p=0.2),
        ])

    def __call__(self, batch):
        batch['src_img'] = self.transform(image=batch['src_img'])['image']
        batch['trg_img'] = self.transform(image=batch['trg_img'])['image']
        return batch


@TRANSFORMS.register_module()
class PadKeyPoints:
    def __init__(self, max_num):
        self.max_num = max_num

    def pad_kps(self, kps):
        pad_num = self.max_num - kps.shape[-1]
        pad_pts = np.ones((2, pad_num)) * -1
        kps = np.concatenate([kps, pad_pts], axis=1)
        return kps
        
    def pad_kps_ids(self, kps_ids):
        pad_num = self.max_num - kps_ids.shape[-1]
        pad_vals = np.ones(pad_num, dtype=int) * -1
        kps_ids = np.concatenate([kps_ids, pad_vals], axis=0)
        return kps_ids

    def __call__(self, batch):
        batch['src_kps'] = self.pad_kps(batch['src_kps'])
        batch['trg_kps'] = self.pad_kps(batch['trg_kps'])
        if 'kps_ids' in batch:
            batch['kps_ids'] = self.pad_kps_ids(batch['kps_ids'])
        return batch


@TRANSFORMS.register_module()
class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, batch):
        batch['src_img'] = torch.from_numpy(batch['src_img'].copy()).float()
        batch['trg_img'] = torch.from_numpy(batch['trg_img'].copy()).float()
        batch['src_kps'] = torch.from_numpy(batch['src_kps'].copy()).float()
        batch['trg_kps'] = torch.from_numpy(batch['trg_kps'].copy()).float()
        
        if 'kps_ids' in batch:
            batch['kps_ids'] = torch.from_numpy(batch['kps_ids']).long()
        if 'src_imsize' in batch:
            batch['src_imsize'] = torch.tensor(batch['src_imsize']).long()
        if 'trg_imsize' in batch:
            batch['trg_imsize'] = torch.tensor(batch['trg_imsize']).long()
        if 'category_id' in batch:
            batch['category_id'] = torch.tensor(batch['category_id']).long()
        if 'pckthres' in batch:
            batch['pckthres'] = torch.tensor(batch['pckthres']).float()
        
        return batch


@TRANSFORMS.register_module()
class Normalize:
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, batch):
        batch['src_img'] = self.normalize(batch['src_img'].permute(2, 0, 1) / 255.0)
        batch['trg_img'] = self.normalize(batch['trg_img'].permute(2, 0, 1) / 255.0)
        return batch


def resize_kps(kps, imside, img_size):
    '''
    imside: 448
    img_size: h x w
    '''
    kps[0, :] = kps[0, :] * (imside / img_size[1])
    kps[1, :] = kps[1, :] * (imside / img_size[0])
    return kps

@TRANSFORMS.register_module()
class ResizeTransform:
    def __init__(self, target_size):
        self.target_size = target_size

    def resize(self, img):
        img = cv2.resize(img, dsize=(self.target_size, self.target_size))
        return img
    
    def __call__(self, batch):
        batch['src_img'] = self.resize(batch['src_img'])
        batch['trg_img'] = self.resize(batch['trg_img'])
        batch['src_kps'] = resize_kps(batch['src_kps'], self.target_size, batch['src_imsize'])
        batch['trg_kps'] = resize_kps(batch['trg_kps'], self.target_size, batch['trg_imsize'])

        if 'trg_bbox' in batch:  # PF-WILLOW dataset has no trg_bbox
            h, w = batch['trg_imsize']
            bbox = batch['trg_bbox'].copy()
            h_scale = self.target_size / h
            w_scale = self.target_size / w
            bbox[0::2] *= w_scale
            bbox[1::2] *= h_scale
            batch['trg_bbox'] = bbox

        return batch


@TRANSFORMS.register_module()
class BoundingBoxThreshold:
    def __call__(self, batch):
        bbox = batch['trg_bbox'].copy()
        bbox_w = (bbox[2] - bbox[0])
        bbox_h = (bbox[3] - bbox[1])
        pckthres = max(bbox_w, bbox_h)
        batch['pckthres'] = pckthres
        return batch

@TRANSFORMS.register_module()
class ImageThreshold:
    def __call__(self, batch):
        imsize_t = batch['trg_img'].shape[:2]
        pckthres = max(imsize_t[0], imsize_t[1])
        batch['pckthres'] = pckthres
        return batch

@TRANSFORMS.register_module()
class WILLOWThreshold:        
    def __call__(self, batch):
        batch['pckthres'] = max(batch['trg_kps'].max(1) - batch['trg_kps'].min(1))  # kps
        return batch


@TRANSFORMS.register_module()
class Exchange:
    def __init__(self, p=0.5):
        self.p = p

    def exchange(self, a, b):
        tmp = a 
        a = b
        b = tmp
        return a, b
    
    def __call__(self, batch):
        if random.uniform(0, 1) < self.p:
            batch['src_img'], batch['trg_img'] = self.exchange(batch['src_img'], batch['trg_img'])
            batch['src_kps'], batch['trg_kps'] = self.exchange(batch['src_kps'], batch['trg_kps'])
            if 'src_bbox' in batch:
                batch['src_bbox'], batch['trg_bbox'] = self.exchange(batch['src_bbox'], batch['trg_bbox'])
        
        return batch


