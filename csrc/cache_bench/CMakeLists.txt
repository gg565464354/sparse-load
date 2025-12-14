cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cache_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED Torch_DIR)
    message(FATAL_ERROR "Torch_DIR is not set.")
endif()

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 设置 TORCH_INSTALL_PREFIX 到 libtorch 根目录
get_filename_component(TORCH_INSTALL_PREFIX "${Torch_DIR}/../.." ABSOLUTE)

# 可选：打印调试信息
message(STATUS "TORCH_INSTALL_PREFIX = ${TORCH_INSTALL_PREFIX}")
message(STATUS "TORCH_INCLUDE_DIRS = ${TORCH_INCLUDE_DIRS}")

# ✅ 不再需要手动添加 ATen 等路径，因为代码中只使用 torch/extension.h

# 设置 CUDA 编译参数
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

# 添加源文件
set(SOURCES
    # cpu_cache.cpp
    # cuda_ops.cu   # 替换为你实际使用的 .cu 文件名
    cache_bench.cpp
)

add_executable(cache_bench ${SOURCES})

# 自动链接 PyTorch 包含的 CUDA 库
target_link_libraries(cache_bench 
    PRIVATE 
      ${TORCH_LIBRARIES}
      OpenMP::OpenMP_CXX
      CUDA::cudart
)

# 包含头文件路径（适用于 C++ 源文件）
target_include_directories(cache_bench PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 输出目录和编译选项
set_property(TARGET cache_bench PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_compile_options(cache_bench PRIVATE -Wall -O2 -fopenmp)