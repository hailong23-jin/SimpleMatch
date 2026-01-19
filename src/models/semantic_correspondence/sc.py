import torch
import torch.nn as nn 
import torch.nn.functional as F

import math
from collections import OrderedDict
from mmengine.registry import MODELS

from src.utils import cosine_similarity_BNC

class ResBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x + self.block1(x)
        x = x + self.block2(x)
        return x


@MODELS.register_module()
class CorrespondenceModel(nn.Module):
    def __init__(self, pair_image_encoder_cfg, task_cfg, window_size):
        super().__init__()
        num_features = 768
        self.window_size = window_size
        self.pair_image_encoder = MODELS.build(pair_image_encoder_cfg)
        self.dropout = nn.Dropout(p=0.3)

        self.trans_conv1 = nn.ConvTranspose2d(num_features, num_features, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(num_features, num_features, kernel_size=2, stride=2)

        self.conv_refine1 = ResBlock(num_features)
        self.conv_refine2 = ResBlock(num_features)

        self.proj = nn.Linear(num_features, num_features)

        self.task = MODELS.build(task_cfg)

    def state_dict(self):
        ckpt = dict()

        ckpt1 = self.pair_image_encoder.state_dict()
        for k, v in ckpt1.items():
            ckpt[f'pair_image_encoder.image_encoder.{k}'] = v

        for name, param in self.named_parameters():
            if 'pair_image_encoder' not in name:
                ckpt[name] = param

        for name, param in self.named_buffers():
            ckpt[name] = param

        return ckpt

    def get_kps_feat(self, src_feat, src_kps, n_pts, down_factor):
        B, N, C = src_feat.shape
        W = int(math.sqrt(N))
        src_kps = (src_kps.clone() / down_factor).long()
        out_feats = []
        for feat, kps, num in zip(src_feat, src_kps, n_pts):
            kps = kps[:, :num]  # k x 2
            kp_index = kps[0, :] + kps[1, :] * W
            kp_feats = feat[kp_index]
            kp_feats_pad = torch.cat([kp_feats, torch.zeros(30-num, C).to(kp_feats.device)], dim=0)
            out_feats.append(kp_feats_pad)

        return torch.stack(out_feats)

    def forward(self, src_img, trg_img, src_kps, n_pts):
        _, _, H, W = src_img.shape
        # extract image features
        src_feat, trg_feat = self.pair_image_encoder(src_img, trg_img)
        B, N, C = trg_feat.shape
        w = h = int(math.sqrt(N))

        src_feat = self.dropout(src_feat)
        trg_feat = self.dropout(trg_feat)

        # up scale x2
        up_h, up_w = h*2, w*2
        src_feat = src_feat.transpose(-2, -1).reshape(B, C, h, w)
        trg_feat = trg_feat.transpose(-2, -1).reshape(B, C, h, w)
        src_feat = F.interpolate(src_feat, (up_h, up_w), None, 'bilinear', False) + self.trans_conv1(src_feat)
        trg_feat = F.interpolate(trg_feat, (up_h, up_w), None, 'bilinear', False) + self.trans_conv1(trg_feat)
        src_feat = self.conv_refine1(src_feat)
        trg_feat = self.conv_refine1(trg_feat)

        # up scale to 1/4 resolution
        up_h, up_w = H//4, W//4
        src_feat = F.interpolate(src_feat, (up_h, up_w), None, 'bilinear', False) 
        trg_feat = F.interpolate(trg_feat, (up_h, up_w), None, 'bilinear', False)
        src_feat = self.conv_refine2(src_feat).flatten(2).transpose(-2, -1)  # to B x N x C
        trg_feat = self.conv_refine2(trg_feat).flatten(2).transpose(-2, -1)

        src_kps_feat = self.get_kps_feat(src_feat, src_kps, n_pts, down_factor=4)
        src_kps_feat = self.proj(src_kps_feat)
        trg_feat = self.proj(trg_feat)
        corr = cosine_similarity_BNC(src_kps_feat, trg_feat)

        return {
            'corr': corr,
        }

    def forward_step(self, batch):
        outs = self.forward(batch['src_img'], batch['trg_img'], batch['src_kps'], batch['n_pts'])

        pred_trg_kps = self.task.compute_trg_kps(outs['corr'], window_size=self.window_size)
        loss = self.task.compute_loss(pred_trg_kps, batch['trg_kps'], batch['n_pts'])

        return OrderedDict(
            pred_trg_kps=pred_trg_kps, 
            total_loss=loss,
        )