import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from src.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--model_path', help='Path to the model checkpoint.')
    parser.add_argument('--work-dir', help='the directory to save logs and models')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--log_name', type=str, default='test_tmp')  
    args = parser.parse_args()

    return args


def main():
    # init config
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    cfg.log_name = args.log_name
    if args.model_path is not None:
        cfg.work_dir = osp.dirname(args.model_path)
        cfg.model_path = args.model_path
    elif args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = 'work_dirs/tmp'

    runner = Runner.from_cfg(cfg)
    runner.inference()


if __name__ == '__main__':
    main()
#