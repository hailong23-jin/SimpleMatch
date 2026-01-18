import torch
import torch.nn as nn 

import loralib as lora
from mmengine.registry import MODELS

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


@MODELS.register_module()
class PairImageEncoder(nn.Module):
    def __init__(self, backbone_cfg, checkpoint_path, fine_tune_type) -> None:
        super().__init__()
        self.fine_tune_type = fine_tune_type
        self.image_encoder = MODELS.build(backbone_cfg)
        
        if fine_tune_type == 'lora':
            self.set_lora()
            lora.mark_only_lora_as_trainable(self.image_encoder)
        
        self.image_encoder.load_ckpt(checkpoint_path)

    def state_dict(self):
        if self.fine_tune_type == 'lora':
            ckpt = lora.lora_state_dict(self.image_encoder)
        else:
            ckpt = self.image_encoder.state_dict()
        return ckpt

    def set_lora(self):
        block_nums = list(range(6, 12)) #+ [2, 5]
        for name, module in self.image_encoder.named_modules():
            for num in block_nums:
                if f'blocks.{num}.mlp.fc2' == name:
                    _set_module(self.image_encoder, name, lora.Linear(module.in_features, module.out_features, r=16, merge_weights=False))

    def forward(self, src_img, trg_img):
        src_feat = self.image_encoder.patch_embedding(src_img)
        trg_feat = self.image_encoder.patch_embedding(trg_img)

        for i, block in enumerate(self.image_encoder.blocks):
            src_feat = block(src_feat)
            trg_feat = block(trg_feat)

        return src_feat[:, 1:, :], trg_feat[:, 1:, :]
