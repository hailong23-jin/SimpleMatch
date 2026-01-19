import os
import argparse
import os.path as osp

from mmengine.config import Config, DictAction

from src.runner import Runner
from src.utils import save_code

def parse_args():
    parser = argparse.ArgumentParser(description='Train a semantic correspondence model')
    parser.add_argument('--config',  help='train config file path')
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
    parser.add_argument('--log_name', type=str, default='log')
    args = parser.parse_args()

    return args


def main():
    # init config
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # merge config
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./work_dirs/tmp')

    # create work directory
    if not osp.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir) 
    
    # save current version code
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    save_code('./src', osp.join(cfg.work_dir, 'src'))
    
    cfg.log_name = args.log_name

    # start running
    runner = Runner.from_cfg(cfg)
    runner.run()

if __name__ == '__main__':
    main()
