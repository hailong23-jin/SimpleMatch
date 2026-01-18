import torch
import shutil


def save_code(src_dir, dst_dir):
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

