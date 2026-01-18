import os
import cv2
import math
import random
import numpy as np 
import albumentations as A
from functools import partial

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
        imside: 448
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
class ResizeForVisualization:
    def __init__(self, target_size=840):
        self.target_size = target_size
        self.max_num = 40

    def resize(self, img, kps):
        h, w, c = img.shape
        img = cv2.resize(img, dsize=(self.target_size, self.target_size))
        kps = resize_kps(kps, self.target_size, [h, w])
        kps = self.pad_kps(kps)
        return img, kps

    def pad_kps(self, kps):
        pad_num = self.max_num - kps.shape[-1]
        pad_pts = np.ones((2, pad_num)) * -1
        kps = np.concatenate([kps, pad_pts], axis=1)
        return kps

    def __call__(self, batch):
        batch['orgin_src_img'], batch['orgin_src_kps'] = self.resize(batch['src_img'], batch['src_kps'].copy())
        batch['orgin_trg_img'], batch['orgin_trg_kps'] = self.resize(batch['trg_img'], batch['trg_kps'].copy())
        return batch

@TRANSFORMS.register_module()
class KeypointCenteredRandomCrop:
    def __init__(self, crop_size, target_size, p=0.5, resize_scope=(0.8, 1.2)) -> None:
        self.crop_size = crop_size
        self.target_size = target_size
        self.resize_scope = resize_scope
        self.p = p

    def rearrange(self, kps, kps_ids):
        '''
        kps: k x 2
        kps_ids: k
        '''
        out_kps = []
        out_ids = []
        for (x, y), idx in zip(kps, kps_ids):
            if x >= 0 and x < self.target_size and y >= 0 and y < self.target_size:
                out_kps.append([x, y])
                out_ids.append(idx)

        return np.stack(out_kps).T, out_ids

    def random_crop(self, img, kps, kps_ids):
        select_idx = random.randint(0, len(kps_ids) - 1)
        x, y = kps[:, select_idx]
        # 随机选择框的大小
        # scale_h = random.uniform(*self.resize_scope)
        # scale_w = random.uniform(*self.resize_scope)
        # trg_h = int(self.crop_size * scale_h)
        # trg_w = int(self.crop_size * scale_w)
        h, w = img.shape[:2]
        trg_h = random.randint(min(int(h * 0.8), 256), h)
        trg_w = random.randint(min(int(w * 0.8), 256), w)
        # 随机选择框的位置
        # left = random.randint(int(max(0, x - trg_w + 14)), int(max(0, x - 14)))
        # top = random.randint(int(max(0, y - trg_h + 14)), int(max(0, y - 14)))
        left = int(max(0, x - trg_w // 2))
        top = int(max(0, y - trg_h // 2))
        
        # crop + resize
        crop_img = img[top:(top + trg_h), left:(left + trg_w)]
        height, width = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, dsize=(self.target_size, self.target_size))

        # resize kps
        resized_kps = np.zeros_like(kps, dtype=np.float32)
        resized_kps[0, :] = (kps[0, :] - left) * (self.target_size / width)
        resized_kps[1, :] = (kps[1, :] - top) * (self.target_size / height)

        resized_kps, kps_ids = self.rearrange(resized_kps.T, kps_ids)
        return crop_img, resized_kps, kps_ids, select_idx

    def random_crop2(self, img, kps, kps_ids, select_idx):
        x, y = kps[:, select_idx]
        # 随机选择框的大小
        # scale_h = random.uniform(*self.resize_scope)
        # scale_w = random.uniform(*self.resize_scope)
        # trg_h = int(self.crop_size * scale_h)
        # trg_w = int(self.crop_size * scale_w)
        h, w = img.shape[:2]
        trg_h = random.randint(min(int(h * 0.8), 256), h)
        trg_w = random.randint(min(int(w * 0.8), 256), w)
        # 随机选择框的位置
        # left = random.randint(int(max(0, x - trg_w + 14)), int(max(0, x - 14)))
        # top = random.randint(int(max(0, y - trg_h + 14)), int(max(0, y - 14)))
        left = int(max(0, x - trg_w // 2))
        top = int(max(0, y - trg_h // 2))

        # crop + resize
        crop_img = img[top:(top + trg_h), left:(left + trg_w)]
        height, width = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, dsize=(self.target_size, self.target_size))

        # resize kps
        resized_kps = np.zeros_like(kps, dtype=np.float32)
        resized_kps[0, :] = (kps[0, :] - left) * (self.target_size / width)
        resized_kps[1, :] = (kps[1, :] - top) * (self.target_size / height)

        resized_kps, kps_ids = self.rearrange(resized_kps.T, kps_ids)
        return crop_img, resized_kps, kps_ids
    
    def random_crop3(self, img, kps, bbox):
        h, w, _ = img.shape
        kps = kps.T
        left = random.randint(0, bbox[0])
        top = random.randint(0, bbox[1])
        height = random.randint(bbox[3], h) - top
        width = random.randint(bbox[2], w) - left
        crop_img = img[top:(top+height), left:(left+width), :]
        target_size = self.target_size
        crop_img = cv2.resize(crop_img, dsize=(target_size, target_size))
        
        resized_kps = np.zeros_like(kps, dtype=np.float32)
        resized_kps[:, 0] = (kps[:, 0] - left) * (target_size / width)
        resized_kps[:, 1] = (kps[:, 1] - top) * (target_size / height)
        resized_kps = np.clip(resized_kps, 0, target_size - 1)
        return crop_img, resized_kps.T

    def resize(self, img, kps):
        h, w, _ = img.shape
        target_size = self.target_size
        img = cv2.resize(img, dsize=(target_size, target_size))
        kps[0, :] = kps[0, :] * (target_size / w)
        kps[1, :] = kps[1, :] * (target_size / h)
        return img, kps
    
    def __call__(self, batch):
        p = random.uniform(0, 1)
        if p < 0.5:
            select_idx = None
            batch['src_img'], src_kps, kps_ids_src, select_idx = self.random_crop(batch['src_img'], batch['src_kps'], batch['kps_ids'])
            batch['trg_img'], trg_kps, kps_ids_trg = self.random_crop2(batch['trg_img'], batch['trg_kps'], batch['kps_ids'], select_idx)
            batch['n_pts'] = batch['src_kps'].shape[-1]

            kps_ids = list(set(kps_ids_src) & set(kps_ids_trg))
            batch['src_kps'] = np.stack([src_kps[:, kps_ids_src.index(val)] for val in kps_ids]).T
            batch['trg_kps'] = np.stack([trg_kps[:, kps_ids_trg.index(val)] for val in kps_ids]).T
            batch['kps_ids'] = np.array(kps_ids)
        # elif p >= 0.2 and p < 0.7:
        #     batch['src_img'], batch['src_kps'] = self.random_crop3(batch['src_img'], batch['src_kps'], batch['src_bbox'].copy().astype(int))
        #     batch['trg_img'], batch['trg_kps'] = self.random_crop3(batch['trg_img'], batch['trg_kps'], batch['trg_bbox'].copy().astype(int))

        else:
            batch['src_img'], batch['src_kps'] = self.resize(batch['src_img'], batch['src_kps'])
            batch['trg_img'], batch['trg_kps'] = self.resize(batch['trg_img'], batch['trg_kps'])

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
class KeypointCutMix:
    def __init__(self, p=0.3):
        self.p = p

    def keypoint_cut_mix(self, img, kps, kps_ids, aux_img, aux_kps, aux_kps_ids):
        kps_ids = kps_ids.tolist()
        aux_kps_ids = aux_kps_ids.tolist()
        union = list(set(kps_ids) & set(aux_kps_ids))
        for kp_id in union:
            if random.uniform(0, 1) < 0.5:
                idx = kps_ids.index(kp_id)
                aux_idx = aux_kps_ids.index(kp_id)
                x, y = kps[:, idx].astype(int)
                aux_x, aux_y = aux_kps[:, aux_idx].astype(int)

                l = random.randint(7, 21)
                h, w = img.shape[:2]
                aux_h, aux_w = aux_img.shape[:2]
                if (y-l) < 0 or (x-l) < 0 or (aux_x-l) < 0 or (aux_y-l) < 0 or \
                    (y+l) > h or (x+l) > w or (aux_x+l) > aux_w or (aux_y+l) > aux_h:
                    continue

                img[(y-l):(y+l), (x-l):(x+l)] = aux_img[(aux_y-l):(aux_y+l), (aux_x-l):(aux_x+l)]

        # cv2.imwrite(f'{random.randint(10, 100)}.png', img)

        return img
    
    def keypoint_blur(self, img, kps):
        for x, y in kps.T:
            if random.uniform(0, 1) < 0.5:
                l = random.randint(7, 21)
                h, w = img.shape[:2]
                if (x - l) < 0 or (y - l) < 0:
                    continue
                
                kernel = random.sample([7, 11, 15], k=1)[0]
                img[(y-l):(y+l), (x-l):(x+l)] = cv2.blur(img[(y-l):(y+l), (x-l):(x+l)], (kernel, kernel))

        # cv2.imwrite(f'{random.randint(10, 100)}.png', img)
        return img

    def __call__(self, batch):
        if random.uniform(0, 1) < self.p:
            batch['src_img'] = self.keypoint_cut_mix(
                batch['src_img'], batch['src_kps'], 
                batch['kps_ids'], batch['aux_src_img'], 
                batch['sample']['src_kps'],
                batch['sample']['kps_ids'])

        if random.uniform(0, 1) < self.p:
            batch['trg_img'] = self.keypoint_cut_mix(
                batch['trg_img'], batch['trg_kps'], 
                batch['kps_ids'], batch['aux_trg_img'], 
                batch['sample']['trg_kps'],
                batch['sample']['kps_ids'])

        if random.uniform(0, 1) < 0.2:
            batch['trg_img'] = self.keypoint_blur(batch['trg_img'], batch['trg_kps'])

        return batch

@TRANSFORMS.register_module()
class RandomRotation:
    def __init__(self, p=0.5, target_size=448):
        self.p = p
        self.target_size = target_size

    def rotate_img_points(self, image, key_points, angle):
        center = ((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # 计算旋转后的图像宽度和高度
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])

        scale = 1.0
        new_width = int(scale * (image.shape[1] * cos + image.shape[0] * sin))
        new_height = int(scale * (image.shape[1] * sin + image.shape[0] * cos))

        # 调整旋转矩阵以确保中心点位于输出图像中心
        rotation_matrix[0, 2] += ((new_width - 1) / 2) - center[0]
        rotation_matrix[1, 2] += ((new_height - 1) / 2) - center[1]

        # mask = np.ones_like(image).astype(np.uint8)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        # rotated_mask = cv2.warpAffine(mask, rotation_matrix, (new_width, new_height))

        rotated_key_points = key_points.copy()
        rotated_key_points[0, :] = key_points[0, :] * rotation_matrix[0, 0] + key_points[1, :] * rotation_matrix[0, 1] + rotation_matrix[0, 2]
        rotated_key_points[1, :] = key_points[0, :] * rotation_matrix[1, 0] + key_points[1, :] * rotation_matrix[1, 1] + rotation_matrix[1, 2]

        return rotated_image, rotated_key_points

    def resize(self, img, kps):
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(self.target_size, self.target_size))
        # mask = cv2.resize(mask, dsize=(self.target_size, self.target_size))
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

        # visualize_img_with_kp(batch['src_img'], batch['src_kps'].T.astype(int), f"visual_results/tmp/{os.path.basename(batch['src_img_path'])[:-4]}_src.png")
        # visualize_img_with_kp(batch['trg_img'], batch['trg_kps'].T.astype(int), f"visual_results/tmp/{os.path.basename(batch['src_img_path'])[:-4]}_trg.png")
        return batch
        
@TRANSFORMS.register_module()
class CutRotationMix:  # 0.3
    def __init__(self, p=0.3, resize_scope=(0.8, 1.3), target_size=448):
        self.p = p
        self.resize_scope = resize_scope
        self.target_size = target_size

    def merge_bounding_box(self, box1, box2):
        box = box1.copy()
        box[0] = box1[0] if box1[0] < box2[0] else box2[0] 
        box[1] = box1[1] if box1[1] < box2[1] else box2[1]
        box[2] = box1[2] if box1[2] > box2[2] else box2[2]
        box[3] = box1[3] if box1[3] > box2[3] else box2[3]
        return box  
    
    def expand_bounding_box(self, bbox, size=10):
        bbox[0] = bbox[0] - size if (bbox[0] - size) >= 0 else 0
        bbox[1] = bbox[1] - size if (bbox[1] - size) >= 0 else 0
        bbox[2] = bbox[2] + size 
        bbox[3] = bbox[3] + size 
        return bbox

    def rotate_img_points(self, image, key_points, angle):
        center = ((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # 计算旋转后的图像宽度和高度
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])

        scale = 1.0
        new_width = int(scale * (image.shape[1] * cos + image.shape[0] * sin))
        new_height = int(scale * (image.shape[1] * sin + image.shape[0] * cos))

        # 调整旋转矩阵以确保中心点位于输出图像中心
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        mask = np.ones_like(image).astype(np.uint8)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (new_width, new_height))

        rotated_key_points = key_points.copy()
        rotated_key_points[0, :] = key_points[0, :] * rotation_matrix[0, 0] + key_points[1, :] * rotation_matrix[0, 1] + rotation_matrix[0, 2]
        rotated_key_points[1, :] = key_points[0, :] * rotation_matrix[1, 0] + key_points[1, :] * rotation_matrix[1, 1] + rotation_matrix[1, 2]

        return rotated_image, rotated_key_points, rotated_mask

    def crop_and_random_resize(self, img, kps, bbox):
        kp_bbox = get_bounding_box(kps.copy().astype(int))
        bbox = self.merge_bounding_box(bbox, kp_bbox)
        bbox = self.expand_bounding_box(bbox)

        obj_area = img[bbox[1]:(bbox[3]+1), bbox[0]:(bbox[2]+1)]
        h, w = obj_area.shape[:2]
        scale_h = random.uniform(*self.resize_scope)
        scale_w = random.uniform(*self.resize_scope)
        trg_h = int(min(h * scale_h, self.target_size))
        trg_w = int(min(w * scale_w, self.target_size))

        obj_area = cv2.resize(obj_area, dsize=(trg_w, trg_h))
        kps[0, :] = (kps[0, :] - bbox[0]) * (trg_w / w)
        kps[1, :] = (kps[1, :] - bbox[1]) * (trg_h / h)

        return obj_area, kps

    def cutmix(self, category, img, bg_img, kps, bbox, img_path):
        # crop and random resize
        obj_area, kps = self.crop_and_random_resize(img, kps, bbox)
        # rotate image
        if category == 'bottle':
            angle = random.random() * 360 - 180
        else:
            # p = np.array([0.7, 0.3])
            # angle = np.random.choice([random.random() * 60 - 30, 180], p=p.ravel())
            angle = random.random() * 60 - 30

        obj_area, kps, mask = self.rotate_img_points(obj_area, kps, angle=angle)

        # adjust image and keypoints
        h, w = obj_area.shape[:2]
        scale = min(self.target_size / h, self.target_size / w)
        scale = min(scale, 1.0)
        trg_h = int(h * scale)
        trg_w = int(w * scale)
        mask = cv2.resize(mask, dsize=(trg_w, trg_h))
        obj_area = cv2.resize(obj_area, dsize=(trg_w, trg_h))
        kps = kps * scale

        # paste cropped image to another image
        h, w = obj_area.shape[:2]
        left = random.randint(0, self.target_size - w)
        top = random.randint(0, self.target_size - h)
        new_img = bg_img.copy()

        new_img[top:(top+h), left:(left+w)] = obj_area
        new_img[top:(top+h), left:(left+w)][mask==0] = bg_img[top:(top+h), left:(left+w)][mask==0]
        kps[0, :] = kps[0, :] + left
        kps[1, :] = kps[1, :] + top

        # visualize_img_with_kp(new_img, kps.T.astype(int), f'visual_results/tmp/{random.randint(10, 30)}.png')
        return new_img, kps

    def resize(self, img, kps):
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(self.target_size, self.target_size))
        kps[0, :] = kps[0, :] * (self.target_size / w)
        kps[1, :] = kps[1, :] * (self.target_size / h)
        return img, kps
    
    def random_crop(self, img, kps, bbox):
        h, w, _ = img.shape
        kps = kps.T
        left = random.randint(0, bbox[0])
        top = random.randint(0, bbox[1])
        height = random.randint(bbox[3], h) - top
        width = random.randint(bbox[2], w) - left
        crop_img = img[top:(top+height), left:(left+width), :]
        crop_img = cv2.resize(crop_img, dsize=(self.target_size, self.target_size))
        
        resized_kps = np.zeros_like(kps, dtype=np.float32)
        resized_kps[:, 0] = (kps[:, 0] - left) * (self.target_size / width)
        resized_kps[:, 1] = (kps[:, 1] - top) * (self.target_size / height)
        resized_kps = np.clip(resized_kps, 0, self.target_size - 1)
        return crop_img, resized_kps.T

    def __call__(self, batch):
        p = random.uniform(0, 1)
        if p < 0.3:  # resize
            batch['src_img'], batch['src_kps'] = self.resize(batch['src_img'], batch['src_kps'])
            batch['trg_img'], batch['trg_kps'] = self.resize(batch['trg_img'], batch['trg_kps'])
        elif p >= 0.3 and p < 0.6:  # random crop
            batch['src_img'], batch['src_kps'] = self.random_crop(batch['src_img'], batch['src_kps'], batch['src_bbox'].copy().astype(int))
            batch['trg_img'], batch['trg_kps'] = self.random_crop(batch['trg_img'], batch['trg_kps'], batch['trg_bbox'].copy().astype(int))
        else:  # cut rotation mix
            rand_num = random.randint(0, 2)
            if rand_num == 0 or rand_num == 2:
                batch['src_img'], batch['src_kps'] = self.cutmix(
                    batch['category'],
                    batch['src_img'], 
                    batch['bg_src_img'], 
                    batch['src_kps'], 
                    batch['src_bbox'].astype(int).copy(),
                    os.path.basename(batch['src_img_path'])
                )
            if rand_num == 1 or rand_num == 2:
                batch['trg_img'], batch['trg_kps'] = self.cutmix(
                    batch['category'],
                    batch['trg_img'], 
                    batch['bg_trg_img'], 
                    batch['trg_kps'], 
                    batch['trg_bbox'].astype(int).copy(),
                    os.path.basename(batch['trg_img_path'])
                )
            batch['src_img'], batch['src_kps'] = self.resize(batch['src_img'], batch['src_kps'])
            batch['trg_img'], batch['trg_kps'] = self.resize(batch['trg_img'], batch['trg_kps'])
        
        return batch

@TRANSFORMS.register_module()
class MixUp:
    def __init__(self, p=0.2):
        self.p = p

    def mixup(self, img, bg_img):
        alpha = random.uniform(0.8, 1)
        out_img = alpha * img + (1 - alpha) * bg_img
        return out_img.astype(np.uint8)

    def __call__(self, batch):
        if random.uniform(0, 1) < self.p:
            batch['src_img'] = self.mixup(batch['src_img'], batch['bg_src_img'])
        if random.uniform(0, 1) < self.p:
            batch['trg_img'] = self.mixup(batch['trg_img'], batch['bg_trg_img'])
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
class KeypointMask:
    def __init__(self, size, downfactor):
        self.size = size
        self.downfactor = downfactor

    def gen_mask(self, kps):
        feat_size = self.size // self.downfactor
        mask = torch.zeros(size=[feat_size, feat_size])
        kps = (kps/self.downfactor).astype(int).T
        for x, y in kps:
            mask[y, x] = 1
        return mask
    
    def gen_multi_scale_mask(self, kps):
        feat_size = self.size // self.downfactor
        masks = []
        for feat_size in [16, 32, 64]:
            mask = torch.zeros(size=[feat_size, feat_size])
            down_factor = self.size / feat_size
            scaled_kps = (kps.copy()/down_factor).astype(int).T
            for x, y in scaled_kps:
                mask[y, x] = 1
            masks.append(mask)
        return masks

    def gen_all_masks(self, kps):
        feat_size = self.size // self.downfactor
        kps = (kps/self.downfactor).astype(int).T
        masks = []
        for x, y in kps:
            mask = torch.zeros(size=[feat_size, feat_size])
            mask[y, x] = 1
            masks.append(mask)
        masks = torch.stack(masks)
        masks = torch.cat([masks, torch.zeros(40-kps.shape[0], feat_size, feat_size)], dim=0)
        return masks

    def __call__(self, batch):
        batch['src_mask_16'], batch['src_mask_32'], batch['src_mask_64'] = self.gen_multi_scale_mask(batch['src_kps'].copy())
        batch['trg_mask_16'], batch['trg_mask_32'], batch['trg_mask_64'] = self.gen_multi_scale_mask(batch['trg_kps'].copy())
        return batch


@TRANSFORMS.register_module()
class GeoKepoints:

    def sample_points_on_line(self, p1, p2, t):
        """
        在线段L1上按照距离阈值采样点
        """
        (x1, y1), (x2, y2) = p1, p2
        points = []
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        num_samples = max(int(length / t), 1)  # 根据线段长度和阈值确定大致采样数量

        for _ in range(num_samples):
            # 在[0,1]区间均匀采样一个参数lambda来确定点在线段上的位置
            lam = random.uniform(0, 1)
            x = x1 + lam * (x2 - x1)
            y = y1 + lam * (y2 - y1)

            # 检查新采样点与已有采样点及端点的距离是否满足阈值要求
            valid = True
            for px, py in points:
                dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                if dist < t:
                    valid = False
                    break
            if valid:
                points.append([x, y])
        return np.array(points)


    def get_corresponding_points(self, l1_points, p1, p2, p3, p4):
        """
        根据线段L1上的采样点及两条线段的端点坐标，获取L2上对应的点坐标
        """
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
        l2_points = []
        l1_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        l2_length = math.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2)
        for px, py in l1_points:
            # 计算L1上点到起点的距离占L1长度的比例
            ratio = math.sqrt((px - x1) ** 2 + (py - y1) ** 2) / l1_length
            # 根据比例确定L2上对应的点坐标
            x = x3 + ratio * (x4 - x3)
            y = y3 + ratio * (y4 - y3)
            l2_points.append([x, y])
        return np.array(l2_points)

    def __call__(self, batch):
        mapping = batch['mapping']
        src_kps = batch['src_kps']
        trg_kps = batch['trg_kps']
        kps_ids = batch['kps_ids'].tolist()
        src_outs = []
        trg_outs = []
        for idx1, idx2 in mapping:
            if idx1 in kps_ids and idx2 in kps_ids:
                src_p1 = src_kps[:, kps_ids.index(idx1)]
                src_p2 = src_kps[:, kps_ids.index(idx2)]
                trg_p1 = trg_kps[:, kps_ids.index(idx1)]
                trg_p2 = trg_kps[:, kps_ids.index(idx2)]
                src_dist = np.sqrt(np.sum((src_p1-src_p2) ** 2))
                trg_dist = np.sqrt(np.sum((trg_p1-trg_p2) ** 2))
                if src_dist < 28 or trg_dist < 28:
                    continue

                threshold = 14
                src_geo_pts = self.sample_points_on_line(src_p1, src_p2, threshold)
                trg_geo_pts = self.get_corresponding_points(src_geo_pts, src_p1, src_p2, trg_p1, trg_p2)
                src_outs.append(src_geo_pts)
                trg_outs.append(trg_geo_pts)
        
        if len(src_outs) > 0:
            src_geo_pts = np.concatenate(src_outs, axis=0).T
            trg_geo_pts = np.concatenate(trg_outs, axis=0).T
            
            n1 = src_kps.shape[-1]
            n2 = src_geo_pts.shape[-1]
            kps_weights = np.concatenate([np.ones(n1), np.ones(n2) * (n1/n2) ])
            src_kps = np.concatenate([src_kps, src_geo_pts], axis=1)
            trg_kps = np.concatenate([trg_kps, trg_geo_pts], axis=1)
            batch['n_pts'] = batch['n_pts'] + src_geo_pts.shape[-1]
            batch['kps_weights'] = kps_weights
        else:
            batch['kps_weights'] = np.ones(src_kps.shape[-1])

        batch['src_kps'] = src_kps
        batch['trg_kps'] = trg_kps

        return batch


@TRANSFORMS.register_module()
class MultiScaleImage:
    def __init__(self, img_size, down_factor) -> None:
        self.resolution1_4 = (img_size // 4) * down_factor
        self.resolution1_8 = (img_size // down_factor) * 2 * down_factor

    def resize(self, img):
        img_upscale1 = cv2.resize(img, dsize=(self.resolution1_4, self.resolution1_4))
        img_upscale2 = cv2.resize(img, dsize=(self.resolution1_8, self.resolution1_8))
        return img_upscale1, img_upscale2

    def __call__(self, batch):
        batch['src_img_1_4'], batch['src_img_1_8'] = self.resize(batch['src_img'])
        batch['trg_img_1_4'], batch['trg_img_1_8'] = self.resize(batch['trg_img'])
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

    def pad_weights(self, weights):
        pad_num = self.max_num - weights.shape[0]
        pad_vals = np.ones(pad_num, dtype=int) * -1
        weights = np.concatenate([weights, pad_vals], axis=0)
        return weights

    def __call__(self, batch):
        batch['src_kps'] = self.pad_kps(batch['src_kps'])
        if 'trg_kps' in batch:
            batch['trg_kps'] = self.pad_kps(batch['trg_kps'])
        if 'all_src_kps' in batch:
            batch['all_src_kps'] = self.pad_kps(batch['all_src_kps'])
        
        if 'kps_weights' in batch:
            batch['kps_weights'] = self.pad_weights(batch['kps_weights'])
        if 'kps_ids' in batch:
            batch['kps_ids'] = self.pad_kps_ids(batch['kps_ids'])
        return batch


@TRANSFORMS.register_module()
class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, batch):
        batch['src_img'] = torch.from_numpy(batch['src_img'].copy()).float()
        batch['src_kps'] = torch.from_numpy(batch['src_kps'].copy()).float()
        batch['trg_img'] = torch.from_numpy(batch['trg_img'].copy()).float()
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
        if 'all_src_kps' in batch:
            batch['all_src_kps'] = torch.from_numpy(batch['all_src_kps'].copy()).float()

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

def get_bounding_box(kps):
    x_min = kps[0, :].min()
    x_max = kps[0, :].max()
    y_min = kps[1, :].min()
    y_max = kps[1, :].max()
    return [x_min, y_min, x_max, y_max]


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
        if 'all_src_kps' in batch:
            batch['all_src_kps'] = resize_kps(batch['all_src_kps'], self.target_size, batch['src_imsize'])

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
class PFPASCALFlip:
    
    def flip(self, batch):
        tmp = batch['src_bbox'][0].copy()
        width = batch['src_img'].shape[1]
        batch['src_bbox'][0] = width - 1 - batch['src_bbox'][2]
        batch['src_bbox'][2] = width - 1 - tmp

        tmp = batch['trg_bbox'][0].copy()
        width = batch['trg_img'].shape[1]
        batch['trg_bbox'][0] = width - 1 - batch['trg_bbox'][2]
        batch['trg_bbox'][2] = width - 1 - tmp

        batch['src_kps'][0][:batch['n_pts']] = width - 1 - batch['src_kps'][0][:batch['n_pts']]
        batch['trg_kps'][0][:batch['n_pts']] = width - 1 - batch['trg_kps'][0][:batch['n_pts']]

        batch['src_img'] = np.flip(batch['src_img'], axis=1)
        batch['trg_img'] = np.flip(batch['trg_img'], axis=1)
        return batch
    
    def exchange(self, a, b):
        tmp = a 
        a = b
        b = tmp
        return a, b

    def __call__(self, batch):
        # if batch['is_flip']:
        #     self.flip(batch)

        if random.uniform(0, 1) < 0.5:
            batch['src_img'], batch['trg_img'] = self.exchange(batch['src_img'], batch['trg_img'])
            batch['src_kps'], batch['trg_kps'] = self.exchange(batch['src_kps'], batch['trg_kps'])
            batch['src_bbox'], batch['trg_bbox'] = self.exchange(batch['src_bbox'], batch['trg_bbox'])

        # visualize_img_with_kp(batch['src_img'].copy(), batch['src_kps'].T.astype(int), f"visual_results/tmp/{os.path.basename(batch['src_img_path'])[:-4]}_src.png")
        # visualize_img_with_kp(batch['trg_img'].copy(), batch['trg_kps'].T.astype(int), f"visual_results/tmp/{os.path.basename(batch['trg_img_path'])[:-4]}_trg.png")
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


