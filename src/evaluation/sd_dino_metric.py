import torch
import torch.nn.functional as F
from collections import OrderedDict
from mmengine.registry import METRICS
import math
import numpy as np 

def where(predicate):
    r"""Predicate must be a condition on nd-tensor"""
    matching_indices = predicate.nonzero()
    if len(matching_indices) != 0:
        matching_indices = matching_indices.t().squeeze(0)
    return matching_indices


@METRICS.register_module()
class SDDINOCorrespondenceMetric:
    def __init__(self, alpha, img_size) -> None:
        self.img_size = (img_size, img_size)
        self.alpha = alpha
        self.pcks = []
        self.pck_dict = dict()
        self.pck_alpha = []
        self.all_values = dict()

        self.category_correspondences = dict()
        self.loss_buff = OrderedDict()

        feat_size = 16
        arr = torch.linspace(-1, 1, feat_size).cuda()
        self.grid_x = arr.view(1, -1).repeat(feat_size, 1).view(-1)
        self.grid_y = arr.view(-1, 1).repeat(1, feat_size).view(-1)

    def mutual_nn_filter(self, correlation_matrix, eps=1e-30):
        r""" Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18 )"""
        corr_src_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=1, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += eps
        corr_trg_max[corr_trg_max == 0] += eps

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)
    

    def classify_prd(self, prd_kps, trg_kps, pckthres):
        r"""Compute the number of correctly transferred key-points"""
        l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5)
        thres = pckthres.expand_as(l2dist).float() * self.alpha
        correct_pts = torch.le(l2dist, thres)

        correct_ids = where(correct_pts == 1)
        incorrect_ids = where(correct_pts == 0)
        correct_dist = l2dist[correct_pts]

        return correct_dist, correct_ids, incorrect_ids

    def update_correspondence(self, prd_kps, batch):
        pck = []
        for idx, (pk, tk, category) in enumerate(zip(prd_kps, batch['trg_kps'], batch['category'])):
            thres = batch['pckthres'][idx]
            npt = batch['n_pts'][idx]
            _, correct_ids, _ = self.classify_prd(pk[:, :npt], tk[:, :npt], thres)

            pck.append(len(correct_ids)/npt)
        return pck
    
    def evaluate_all_alpha(self, prd_kps, batch, alpha=[0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30]):
        r""" Compute percentage of correct key-points (PCK) with multiple alpha {0.05, 0.1, 0.15 }"""
        alpha = torch.tensor(alpha).unsqueeze(1).cuda()
        # pcks = []
        for idx, (pk, tk, category) in enumerate(zip(prd_kps, batch['trg_kps'], batch['category'])):
            pckthres = batch['pckthres'][idx].cuda()
            npt = batch['n_pts'][idx]
            prd_kps = pk[:, :npt].cuda()
            trg_kps = tk[:, :npt].cuda()

            l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5).unsqueeze(0).repeat(len(alpha), 1)
            thres = pckthres.expand_as(l2dist).float() * alpha
            cmp_res = torch.le(l2dist, thres)
            if category not in self.category_correspondences:
                self.category_correspondences[category] = cmp_res
            else:
                self.category_correspondences[category] = torch.cat([self.category_correspondences[category], cmp_res], dim=-1)
            
            # pck = torch.le(l2dist, thres).sum(dim=1) / float(npt)
            # if len(pck) == 1: pck = pck[0]
            # pcks.append(pck)

        # print('abc')
        # return pcks

    def get_trg_kps(self, grid, src_kps, n_pts):
        image_size = self.img_size
        # grid = grid[:, 1:-1, 1:-1, :]
        # grid[:,:,:,0] = grid[:,:,:,0] * (image_size[0]//2)/(image_size[0]//2 - 1)
        # grid[:,:,:,1] = grid[:,:,:,1] * (image_size[1]//2)/(image_size[1]//2 - 1)

        grid = grid.permute(0, 3, 1, 2)  # trg x src  B x 2 x h x w
        grid = F.interpolate(grid, size=image_size, mode='bilinear', align_corners=True)  # bicubic
        grid = grid.permute(0, 2, 3, 1)  # B x h x w x 2
        
        pred_kps = []
        for pos_map, kps, n_pt in zip(grid, src_kps, n_pts):
            est_kps = torch.zeros_like(kps)
            kps = kps[:, :n_pt]
            for idx, (point_x, point_y) in enumerate(kps.t()):
                point_x = torch.round(point_x).long()
                point_y = torch.round(point_y).long()
                if point_x == image_size[0]:
                    point_x = point_x - 1
                if point_y == image_size[1]:
                    point_y = point_y - 1

                est_x = (pos_map[point_y, point_x, 0] + 1) * (image_size[0] - 1) / 2
                est_y = (pos_map[point_y, point_x, 1] + 1) * (image_size[1] - 1) / 2
                est_kps[:, idx] = torch.tensor([est_x, est_y]).to(est_kps.device)
            pred_kps.append(est_kps)
        return torch.stack(pred_kps, dim=0)
    
    def get_trg_kps_nn(self, corrs, src_kps, n_pts):
        image_size = self.img_size

        pred_kps = []
        B, N = corrs.shape[:2]
        h = w = int(math.sqrt(N))
        down_factor = (image_size[0] / w)
        corrs = torch.argmax(corrs, dim=-1)
        for corr, kps, n_pt in zip(corrs, src_kps, n_pts):
            est_kps = torch.zeros_like(kps)
            kps = (kps[:, :n_pt] / down_factor).long()
            src_idx = (w * kps[1, :] + kps[0, :])
            est_trg_idx = corr[src_idx]
            est_y = torch.div(est_trg_idx, w, rounding_mode='floor')
            est_x = est_trg_idx % w

            for idx, (x, y) in enumerate(zip(est_x, est_y)):
                est_kps[:, idx] = torch.tensor([x, y]).to(est_kps.device) * down_factor  + down_factor // 2 - 0.5

            pred_kps.append(est_kps)

        pred_kps = torch.stack(pred_kps, dim=0) 

        return pred_kps

    def get_trg_kps2(self, corrs, src_kps, n_pts):
        pred_kps = []
        corrs = torch.argmax(corrs, dim=-1)
        w = 512
        for corr, kps, n_pt in zip(corrs, src_kps, n_pts):
            est_kps = torch.zeros_like(kps)
            est_y = torch.div(corr, w, rounding_mode='floor')
            est_x = corr % w
            for idx, (x, y) in enumerate(zip(est_x, est_y)):
                est_kps[:, idx] = torch.tensor([x, y]).to(est_kps.device)
            pred_kps.append(est_kps)

        pred_kps = torch.stack(pred_kps, dim=0)
        return pred_kps 

    def update_metrics(self, outputs, batch):
        estimated_kps = outputs['pred_trg_kps']
        self.evaluate_all_alpha(estimated_kps, batch)
        # pck = self.update_correspondence(estimated_kps, batch)
        # self.pcks += pck

        for key in outputs.keys():
            if 'loss' in key:
                if key not in self.loss_buff:
                    self.loss_buff[key] = []
                self.loss_buff[key].append(outputs[key].item())

        results = self.get_metrics()
        return results

    def get_metrics(self):
        results = OrderedDict()
        for key, val in self.loss_buff.items():
            results[key] = sum(val) / (len(val) + 1e-10)

        category_pck = dict()
        arr = []
        for category, correspondences in self.category_correspondences.items():
            pck = correspondences.sum(dim=-1) / correspondences.shape[-1] * 100
            category_pck[category] = pck[2]  # 取 alpha = 0.1 对应的pck
            arr.append(pck)
        
        results['category_pck'] = category_pck
        alpha_pcks = sum(arr) / len(arr)
        results['pck'] = alpha_pcks[2]
        results['pck_alpha'] = alpha_pcks.cpu().numpy()
                
        return results