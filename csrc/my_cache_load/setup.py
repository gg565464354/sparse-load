from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths

import sys
import os
import torch

# 获取 PyTorch 包含路径
torch_include = os.path.join(torch.__path__[0], "include")
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')

# 根据平台设置 OpenMP 参数（仅用于 C++ 文件）
if sys.platform == 'darwin':  # macOS
    # macOS 上需要安装 libomp：brew install libomp
    cxx_extra_compile_args = ['-Xpreprocessor', '-fopenmp']
    extra_link_args = ['-lomp']
else:  # Linux
    cxx_extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp', f'-Wl,-rpath,{torch_lib_path}']

# 设置编译参数（只为 C++ 添加 OpenMP，CUDA 不加）
extra_compile_args = {
    'cxx': ['-O2', '-Wall'] + cxx_extra_compile_args,
    'nvcc': ['-O2']
}

setup(
    name='my_cache_load',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            name='my_cache_load._C',
            sources=[
                'src/cpu_cache.cpp',
                # 'src/cuda_ops.cu',
                'src/cache_api.cpp',
                'pybind_wrapper.cpp'
            ],
            include_dirs=['src'] + include_paths(),
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)

