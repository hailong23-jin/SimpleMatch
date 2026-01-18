# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='kxk_window_cuda_optimized',
    ext_modules=[
        CUDAExtension('kxk_window_cuda_optimized', [
            'kxk_window_optimized.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)