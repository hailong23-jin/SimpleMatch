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
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU()
        )

    def forward(self, x):
        x = x + self.block1(x)
        x = x + self.block2(x)
        return x


@MODELS.register_module()
class CorrespondenceModelResNet(nn.Module):
    def __init__(self, pair_image_encoder_cfg, task_cfg):
        super().__init__()
        num_features = 512
        self.backbone_type = pair_image_encoder_cfg['type'] 
        self.pair_image_encoder = MODELS.build(pair_image_encoder_cfg)

        self.trans_conv1 = nn.ConvTranspose2d(num_features, num_features, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(num_features, num_features, kernel_size=2, stride=2)

        self.conv_refine1 = ResBlock(num_features)
        self.conv_refine2 = ResBlock(num_features)
        self.conv_refine3 = ResBlock(num_features)

        self.proj1 = nn.Linear(num_features, num_features)
        self.proj2 = nn.Linear(num_features, num_features)
        self.proj3 = nn.Linear(num_features, num_features)

        self.task = MODELS.build(task_cfg)
    
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
        src_feats, trg_feats = self.pair_image_encoder(src_img, trg_img)

        # 16x16
        src_feat = src_feats[-2] + self.trans_conv1(src_feats[-1])
        trg_feat = trg_feats[-2] + self.trans_conv1(trg_feats[-1])
        src_feat = self.conv_refine1(src_feat)
        trg_feat = self.conv_refine1(trg_feat)

        src_kps_feat = self.get_kps_feat(src_feat.flatten(2).transpose(-2, -1), src_kps, n_pts, down_factor=16)
        x1 = self.proj1(src_kps_feat)
        x2 = self.proj1(trg_feat.flatten(2).transpose(-2, -1))
        corr1 = cosine_similarity_BNC(x1, x2)

        # up scale 32x32
        src_feat = src_feats[-3] + self.trans_conv2(src_feat)
        trg_feat = trg_feats[-3] + self.trans_conv2(trg_feat)
        src_feat = self.conv_refine2(src_feat)
        trg_feat = self.conv_refine2(trg_feat)
        src_kps_feat = self.get_kps_feat(src_feat.flatten(2).transpose(-2, -1), src_kps, n_pts, down_factor=8)
        x1 = self.proj2(src_kps_feat)
        x2 = self.proj2(trg_feat.flatten(2).transpose(-2, -1))
        corr2 = cosine_similarity_BNC(x1, x2)

        # up scale 64x64
        up_h, up_w = H//4, W//4
        src_feat = src_feats[-4] + F.interpolate(src_feat, (up_h, up_w), None, 'bilinear', True) 
        trg_feat = trg_feats[-4] + F.interpolate(trg_feat, (up_h, up_w), None, 'bilinear', True) 
        src_feat = self.conv_refine3(src_feat)
        trg_feat = self.conv_refine3(trg_feat)
        src_kps_feat = self.get_kps_feat(src_feat.flatten(2).transpose(-2, -1), src_kps, n_pts, down_factor=4)
        x1 = self.proj3(src_kps_feat)
        x2 = self.proj3(trg_feat.flatten(2).transpose(-2, -1))
        corr3 = cosine_similarity_BNC(x1, x2)

        return {
            'corr1': corr1,
            'corr2': corr2,
            'corr3': corr3,
        }

    def forward_step(self, batch):
        outs = self.forward(batch['src_img'], batch['trg_img'], batch['src_kps'], batch['n_pts'])

        pred_trg_kps1 = self.task.compute_trg_kps(outs['corr1'], window_size=None)
        pred_trg_kps2 = self.task.compute_trg_kps(outs['corr2'], window_size=15)
        pred_trg_kps3 = self.task.compute_trg_kps(outs['corr3'], window_size=30)

        loss1 = self.task.compute_loss(pred_trg_kps1, batch['trg_kps'], batch['n_pts'])
        loss2 = self.task.compute_loss(pred_trg_kps2, batch['trg_kps'], batch['n_pts'])
        loss3 = self.task.compute_loss(pred_trg_kps3, batch['trg_kps'], batch['n_pts'])
        total_loss =  loss1 + loss2 + loss3

        return OrderedDict(
            pred_trg_kps=pred_trg_kps3, 
            total_loss=total_loss,
        )
    