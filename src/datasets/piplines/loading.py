
import cv2
import numpy as np 
from PIL import Image
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImage:
    def __call__(self, batch):
        src_img = self.read_img(batch['src_img_path'])
        trg_img = self.read_img(batch['trg_img_path'])

        batch['src_img'] = src_img
        batch['trg_img'] = trg_img
        batch['src_imsize'] = np.array(src_img.shape[:2])  # [h, w]
        batch['trg_imsize'] = np.array(trg_img.shape[:2])

        return batch

    def read_img(self, path):
        return np.array(Image.open(path).convert('RGB'))
    
    def read_mask(self, path):
        img = cv2.imread(path) / 255
        return img
