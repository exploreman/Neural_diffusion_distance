from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='SpixelAggr_avr',
    ext_modules=[
        CUDAExtension('SpixelAggr_avr_cuda', [
            'SpixelAggr_avr_cuda.cpp',
            'SpixelAggr_avr_cuda_ker.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })