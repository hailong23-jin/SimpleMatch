import torch
import torch.nn as nn 
from mmengine.registry import MODELS


@MODELS.register_module()
class PairImageEncoderResNet(nn.Module):
    def __init__(self, backbone_cfg) -> None:
        super().__init__()
        self.backbone = MODELS.build(backbone_cfg)
        self.dim = 512
        
        align12 = nn.Conv2d(256, 512, kernel_size=1)
        align32 = nn.Conv2d(1024, 512, kernel_size=1)
        align42 = nn.Conv2d(2048, 512, kernel_size=1)
        self.aligns = torch.nn.ModuleList([align12, nn.Identity(), align32, align42])

    def patch_embedding(self, img):
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        return feat

    def forward(self, src_img, trg_img):
        imgs = torch.cat([src_img, trg_img], dim=0)

        src_feats = []
        trg_feats = []
        B = imgs.shape[0] // 2
        feat = self.patch_embedding(imgs)  

        for i in range(1, 5):
            feat = self.backbone.__getattr__(f'layer{i}').forward(feat)
            feat_align = self.aligns[i-1](feat)

            src_feat, trg_feat = torch.split(feat_align, B, dim=0)
            src_feats.append(src_feat)
            trg_feats.append(trg_feat)

        return src_feats, trg_feats
        


        
   