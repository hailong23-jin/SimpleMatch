
import copy
import os.path as osp
from prettytable import PrettyTable
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmengine.config import Config, ConfigDict
from mmengine.registry import RUNNERS, MODELS, METRICS, DATASETS
from mmengine.logging import MMLogger

from src.utils import set_random_seed, to_cuda, Timer


@RUNNERS.register_module()
class Runner:
    def __init__(self,
            model: Union[nn.Module, Dict],
            work_dir: str,
            train_dataloader: Optional[Dict] = None,
            val_dataloader: Optional[Dict] = None,
            test_dataloader: Optional[Dict] = None,
            optimizer: Optional[Dict] = None, 
            lr_scheduler: Optional[Dict] = None,
            metric: Optional[Dict] = None, 
            log_level: str = 'INFO',
            resume: bool = False,
            cfg: Optional[Union[Dict, Config, ConfigDict]] = None
        ):
        self.work_dir = work_dir
        self.max_epochs = cfg.max_epochs
        self.interval = cfg.interval

        self.logger = self.build_logger(log_level, log_name=cfg.log_name)
        self.metric = copy.deepcopy(metric)
        self.cfg = copy.deepcopy(cfg)

        # build modules
        set_random_seed(123)
        self.model = self.build_model(model, cfg.get('model_path', None))
        self.optimizer = self.build_optimizer(self.model, optimizer)
        self.lr_scheduler = self.build_lr_scheduler(self.optimizer, lr_scheduler)
        self.train_dataloader = self.build_dataloader(train_dataloader)
        self.val_dataloader = self.build_dataloader(val_dataloader)
        self.test_dataloader = self.build_dataloader(test_dataloader)

        self.logger.info(f'work_dir: {work_dir}')
        if self.train_dataloader is not None:
            self.logger.info(f'batch size: {self.train_dataloader.batch_size}')
            self.logger.info(f'train data size: {len(self.train_dataloader)}')
    
        if self.val_dataloader is not None:
            self.logger.info(f'val data size: {len(self.val_dataloader)}')
        self.logger.info(f'test data size: {len(self.test_dataloader)}')
        
        self.best_pck = 0.0

    @classmethod
    def from_cfg(cls, cfg: Union[Dict, Config, ConfigDict]):
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model = cfg['model'],
            work_dir = cfg['work_dir'],
            train_dataloader = cfg.get('train_dataloader'),
            val_dataloader = cfg.get('val_dataloader'),
            test_dataloader = cfg.get('test_dataloader'),
            optimizer = cfg.get('optimizer'),
            lr_scheduler = cfg.get('lr_scheduler'),
            metric = cfg.get('metric'),
            log_level = cfg.get('log_level'),
            resume = cfg.get('resume'),
            cfg = cfg,
        )
        return runner
    
    def build_model(self, model_cfg: Dict, model_path: str=None):
        model_cfg = copy.deepcopy(model_cfg)
        model = MODELS.build(model_cfg).cuda()
        if model_path is not None:
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt, strict=False)
        return model
        
    def build_optimizer(self, model, optimizer_cfg):
        optimizer_cfg = copy.deepcopy(optimizer_cfg)
        optim_type = optimizer_cfg.pop('type')
        optimizer_cfg['params'] = model.parameters()

        if optim_type == 'SGD':
            optimizer = torch.optim.SGD(**optimizer_cfg)
        elif optim_type == 'Adam':
            optimizer = torch.optim.Adam(**optimizer_cfg)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(**optimizer_cfg)
        else: 
            raise ValueError('Do not support such type of the optimizer.')
        return optimizer
    
    def build_lr_scheduler(self, optimizer, lr_scheduler_cfg):
        lr_scheduler_cfg = copy.deepcopy(lr_scheduler_cfg)
        lr_scheduler_type = lr_scheduler_cfg.pop('type')
        if lr_scheduler_type == 'StepLR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_scheduler_cfg)
        elif lr_scheduler_type == 'LinearLR':
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **lr_scheduler_cfg)
        elif lr_scheduler_type == 'ExponentialLR':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **lr_scheduler_cfg)
        else:
            raise ValueError('Do not support such type of lr scheduler.')
        return lr_scheduler
   
    def build_dataloader(self, dataloader: Optional[Dict] = None):
        if dataloader is None:
            return None
        
        dataloader_cfg = copy.deepcopy(dataloader)
        dataset_cfg = dataloader_cfg.pop('dataset')
        dataset = DATASETS.build(dataset_cfg)
        data_loader = DataLoader(dataset=dataset, **dataloader_cfg)
        return data_loader

    def build_logger(self,
                     log_level: Union[int, str] = 'INFO',
                     log_name: str = None,
                     **kwargs) -> MMLogger:
        
        log_file = osp.join(self.work_dir, f'{log_name}.txt')
        log_cfg = dict(log_level=log_level, log_file=log_file, **kwargs)
        log_cfg.setdefault('name', 'abc')
        log_cfg.setdefault('file_mode', 'a+')

        return MMLogger.get_instance(**log_cfg)  # type: ignore

    def resume(self):
        latest_ckpt_path = osp.join(self.work_dir, 'latest.pth')
        ckpt = torch.load(latest_ckpt_path)

        self.best_pck = ckpt['pck']
        self.start_epoch = ckpt['epoch'] + 1
        self.best_epoch = ckpt['best_epoch']

        self.model.load_state_dict(ckpt['model'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer'])

        self.logger.info(f'resume from: {latest_ckpt_path}')
        self.logger.info(f'best PCK: {self.best_pck}')
        self.logger.info(f'continue training...')

    def run(self):
        for epoch in range(0, self.max_epochs):
            # train one epoch and evaluate the model
            results_train = self.train(epoch)
            results_val = self.val()
            self.record_epoch(results_train, results_val)
            self.save_best_checkpoint(results_val, epoch)

        self.logger.info(f'Best PCK: {self.best_pck}')

    def train(self, epoch):
        # build metric
        metric = METRICS.build(self.metric)

        timer = Timer()
        timer.record()
        self.model.train()
        total_iter = len(self.train_dataloader)
        for idx, batch in enumerate(self.train_dataloader): 
            to_cuda(batch)
            outputs = self.model.forward_step(batch)

            loss = outputs['total_loss']
            # update parameters
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()
            
            results = metric.update_metrics(outputs, batch)

            timer.record()
            time_info = timer.infer_eta_time(idx, total_iter)
            self.record_train_iter(results, time_info, epoch, idx, total_iter)

        results = metric.get_metrics()
        return results

    def val(self):
        # build metric
        metric = METRICS.build(self.metric)

        model = self.model
        model.eval()
        for idx, batch in enumerate(self.val_dataloader):
            # forward model
            to_cuda(batch)
            with torch.no_grad():
                outputs = model.forward_step(batch)
            
            # compute metrics
            results = metric.update_metrics(outputs, batch)
            self.record_val_iter(results, idx, len(self.val_dataloader))

        results = metric.get_metrics()
        return results

    def inference(self):
        # set evaluation metric
        metric = METRICS.build(self.metric)
        self.model.eval()
        times = []
        for idx, batch in enumerate(self.test_dataloader):  # test_dataloader
            # forward model
            to_cuda(batch)
            with torch.no_grad():
                outputs = self.model.forward_step(batch)

            # compute metrics
            results = metric.update_metrics(outputs, batch)
            self.record_val_iter(results, idx, len(self.test_dataloader))

        results = metric.get_metrics()
        self.record_infer(results)
        return results

    def record_train_iter(self, results, time_info, epoch, iter, max_iter):
        # if use gradient accumulation, skip current step
        if (iter + 1) % self.interval != 0:
            return 
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        process = 'Epoch:[{:03d}/{}] Iter:[{:04d}/{}] '.format(epoch + 1, self.max_epochs, iter + 1, max_iter)
        eta = 'eta:{:02d}:{:02d}:{:02d} '.format(time_info['hours'], time_info['minutes'], time_info['seconds'])
        res = ' '.join([f'{key}: {val:.4f}' for key, val in results.items() if 'loss' in key])  # for loss, mIoU, acc, etc.
        res += f" pck: {results['pck']:.3f}"
        msg = process + eta + res + f' lr: {lr:.4e}'
        self.logger.info(msg)

    def record_val_iter(self, results, iter, max_iter):
        # if use gradient accumulation, skip current step
        if (iter + 1) % self.interval != 0:
            return 
        process = 'Iter:[{:04d}/{}] '.format(iter + 1, max_iter)
        res = ' '.join([f'{key}: {val:.4f}' for key, val in results.items() if 'loss' in key])
        res += f" pck: {results['pck']:.3f}"
        msg = process + res
        self.logger.info(msg)

    def record_epoch(self, results_train, results_val):
        table = PrettyTable(['mode'] + ['total_loss', 'pck'])
        table.add_row(['Training'] + [f"{results_train['total_loss']:.4f}", f"{results_train['pck']:.4f}"])
        table.add_row(['Validation'] + [f"{results_val['total_loss']:.4f}", f"{results_val['pck']:.4f}"])
        self.logger.info('\n' + table.get_string())

    def record_infer(self, results):
        table = PrettyTable(['total_loss', 'pck'])
        table.add_row([f"{results['total_loss']:.3f}", f"{results['pck']:.3f}"])
        self.logger.info('\n' + table.get_string())
        
        for key, val in results['category_pck'].items():
            self.logger.info(f'{key}\t{val:.1f}')

        alphas=[0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30]
        for alpha, pck in zip(alphas, results['pck_alpha']):
            self.logger.info(f'{alpha}: {pck:.1f}')


    def save_best_checkpoint(self, results_val, epoch):
        cur_indicator = results_val['pck']
        if cur_indicator > self.best_pck:
            self.best_pck = cur_indicator
            save_path = osp.join(self.work_dir, 'best_model.pth')
            torch.save(self.model.state_dict(), save_path)
            self.best_epoch = epoch
            self.logger.info('Save checkpoint')
            self.logger.info(f'Epoch:{epoch + 1} Best PCK:{cur_indicator:.2f}')

    def save_last_checkpoint(self, epoch):
        latest_ckpt = {
            'epoch': epoch,  # current epoch
            'best_epoch': self.best_epoch,
            'PCK': self.best_pck,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        latest_ckpt_path = osp.join(self.work_dir, f'latest.pth')
        torch.save(latest_ckpt, latest_ckpt_path)
