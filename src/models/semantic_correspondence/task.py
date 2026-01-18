import torch
import torch.nn.functional as F

import math
from mmengine.registry import MODELS
from get_kxk_window.kxk_window_optimized import get_kxk_window_optimized

@MODELS.register_module()
class SparseCorrespondenceTask:
    def __init__(self, img_size) -> None:
        self.img_size = img_size

    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        '''
        corr: B x 40 x h x w  src x trg
        '''
        b, _, h, w = corr.shape
        device = corr.device

        corr = corr.view(b, -1, h*w)
        corr = F.softmax(corr/beta, dim=-1)
        corr = corr.view(b, -1, h, w) 
        # b x 1 x w
        x_normal = torch.linspace(-1, 1, w).expand(b, w).view(b, 1, w).to(device)
        grid_x = corr.sum(dim=2, keepdim=False) # marginalize to x-coord.  B x 40 x w
        grid_x = (grid_x * x_normal).sum(dim=-1, keepdim=False) # b x 40 x 1
        # b x 1 x h
        y_normal = torch.linspace(-1, 1, h).expand(b, h).view(b, 1, h).to(device)
        grid_y = corr.sum(dim=3, keepdim=False) # marginalize to y-coord.
        grid_y = (grid_y * y_normal).sum(dim=-1, keepdim=False) # b x 40 x 1

        return grid_x, grid_y

    def compute_trg_kps(self, corr, src_kps, n_pts, window_size):
        # corr: B x 40 x N  src x trg
        b, _, N = corr.size()
        h = w = int(math.sqrt(N))
        H = W = self.img_size
        down_factor = H/h

        # arr = []
        # for M, src_kp, num in zip(corr, src_kps, n_pts):
        #     src_kp = (src_kp[:, :num] / down_factor).long()
        #     src_kp_index = src_kp[0, :] + src_kp[1, :] * w
        #     M = M[src_kp_index]
        #     M = torch.cat([M, torch.zeros(size=[30-num, N]).cuda()], dim=0)
        #     arr.append(M)
        # corr = torch.stack(arr)
        # window_size = h

        if window_size is None or h < window_size:
            window_size = h
        corr, top_left_coords = get_kxk_window_optimized(corr.view(b, -1, h, w), k=window_size) 

        grid_x, grid_y = self.soft_argmax(corr.view(b, -1, window_size, window_size))
        # grid_x, grid_y = self.soft_argmax(corr.view(b, -1, h, w))
        pred_xy = torch.stack([grid_x, grid_y], dim=1)  #  B x 2 x 40
        pred_xy[:, 0] = (pred_xy[:, 0].float().clone() + 1) * (window_size - 1) / 2.0 
        pred_xy[:, 1] = (pred_xy[:, 1].float().clone() + 1) * (window_size - 1) / 2.0
        pred_xy = pred_xy + top_left_coords.transpose(-2, -1)

        # arr = []
        # for xy, src_kp, num in zip(pred_xy, src_kps, n_pts):
        #     src_kp = (src_kp[:, :num] / down_factor).long()
        #     src_kp_index = src_kp[0, :] + src_kp[1, :] * w
            
        #     xy = xy[:, src_kp_index]
        #     xy = torch.cat([xy, torch.zeros(size=[2, 30-num]).cuda()], dim=1)
        #     arr.append(xy)
        # pred_xy = torch.stack(arr)

        return pred_xy * down_factor

    def compute_loss(self, pred_kps, trg_kps, n_pts):
        total_loss = torch.zeros(0).to(pred_kps.device)
        for pred_kp, trg_kp, num in zip(pred_kps, trg_kps, n_pts):
            pred_kp = pred_kp[:, :num]
            trg_kp = trg_kp[:, :num]
            sample_loss = torch.norm(pred_kp - trg_kp, p=2, dim=0)
            total_loss = torch.cat([total_loss, sample_loss], dim=0)
        return total_loss.mean()
    

    def compute_loss_flow(self, pred_kps, trg_kps, src_kps, n_pts):
        total_loss = torch.zeros(0).to(pred_kps.device)
        pred_flow = pred_kps - src_kps
        gt_flow = trg_kps - src_kps
        for flow1, flow2, num in zip(pred_flow, gt_flow, n_pts):
            flow1 = flow1[:, :num]
            flow2 = flow2[:, :num]
            sample_loss = torch.norm(flow1 - flow2, p=2, dim=0)
            total_loss = torch.cat([total_loss, sample_loss], dim=0)
        return total_loss.mean()

