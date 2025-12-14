from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# 检测系统并设置 OpenMP 参数
import sys
import os
import torch


torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
torch_include = os.path.join(torch.__path__[0], "include")

if sys.platform == 'darwin':  # macOS
    # macOS 需要安装 libomp（可以通过 Homebrew 安装：brew install libomp）
    extra_compile_args = ['-Xpreprocessor', '-fopenmp']
    extra_link_args = ['-lomp']
else:  # Linux 或其他平台
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp', f'-Wl,-rpath,{torch_lib_path}']


setup(
    name='my_cache_load',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            'my_cache_load._C',
            sources=[
                'src/cpu_cache.cpp',
                # 'src/cache_bench.cpp',
                'src/cache_api.cpp',
                'pybind_wrapper.cpp'
            ],
            include_dirs=['src', torch_include],
            extra_compile_args=extra_compile_args + ['-O2', '-Wall'],
            extra_link_args=extra_link_args
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)