import torch
from collections import OrderedDict
from .metric_base import MetricBase
from mmengine.registry import METRICS


def where(predicate):
    r"""Predicate must be a condition on nd-tensor"""
    matching_indices = predicate.nonzero()
    if len(matching_indices) != 0:
        matching_indices = matching_indices.t().squeeze(0)
    return matching_indices


@METRICS.register_module()
class CorrespondenceMetric(MetricBase):
    def __init__(self, alpha=0.1, img_size=256) -> None:
        self.img_size = (img_size, img_size)
        self.alpha = alpha
        self.pcks = []
        self.pck_dict = dict()
        self.pck_alpha = []
        self.loss_buff = OrderedDict()

        self.acc_dict = dict()
        self.accs = []

    def classify_prd(self, prd_kps, trg_kps, pckthres):
        r"""Compute the number of correctly transferred key-points"""
        l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5)
        thres = pckthres.expand_as(l2dist).float() * self.alpha
        correct_pts = torch.le(l2dist, thres)

        correct_ids = where(correct_pts == 1)
        incorrect_ids = where(correct_pts == 0)
        correct_dist = l2dist[correct_pts]

        return correct_dist, correct_ids, incorrect_ids

    def eval_kps_transfer(self, prd_kps, batch):
        pck = []
        for idx, (pk, tk) in enumerate(zip(prd_kps, batch['trg_kps'])):
            thres = batch['pckthres'][idx]
            npt = batch['n_pts'][idx]
            _, correct_ids, _ = self.classify_prd(pk[:, :npt], tk[:, :npt], thres)

            pck.append((len(correct_ids) / (npt.item())) * 100)

        return pck

    def evaluate_all_alpha(self, prd_kps, batch, alpha=[0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30]):
        r""" Compute percentage of correct key-points (PCK) with multiple alpha {0.05, 0.1, 0.15 }"""
        alpha = torch.tensor(alpha).unsqueeze(1).cuda()
        pcks = []
        for idx, (pk, tk) in enumerate(zip(prd_kps, batch['trg_kps'])):
            pckthres = batch['pckthres'][idx].cuda()
            npt = batch['n_pts'][idx]
            prd_kps = pk[:, :npt].cuda()
            trg_kps = tk[:, :npt].cuda()

            l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5).unsqueeze(0).repeat(len(alpha), 1)
            thres = pckthres.expand_as(l2dist).float() * alpha
            pck = torch.le(l2dist, thres).sum(dim=1) / float(npt)
            if len(pck) == 1: pck = pck[0]
            pcks.append(pck)

        return pcks

    def update_metrics(self, outputs, batch):
        estimated_kps = outputs['pred_trg_kps']
        pck = self.eval_kps_transfer(estimated_kps, batch)
        pck_alpha = self.evaluate_all_alpha(estimated_kps, batch)
        self.pcks += pck
        self.pck_alpha += pck_alpha

        for pck_val, category in zip(pck, batch['category']):
            if category not in self.pck_dict:
                self.pck_dict[category] = [pck_val]
            else:
                self.pck_dict[category] += [pck_val]

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

        results['pck'] = sum(self.pcks) / len(self.pcks)

        category_pck = dict()
        for key, val in self.pck_dict.items():
            category_pck[key] = sum(val) / len(val)
        results['category_pck'] = category_pck

        results['pck_alpha'] = torch.stack(self.pck_alpha).mean(dim=0).cpu().numpy() * 100

        return results