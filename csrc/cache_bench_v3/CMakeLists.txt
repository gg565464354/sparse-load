cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cache_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(NOT DEFINED Torch_DIR)
    message(FATAL_ERROR "Torch_DIR is not set.")
endif()

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

# —— 新增：查找 CUDA Runtime （包含 cuda_runtime.h） —— 
# 推荐使用 CMake 3.17+ 自带的 CUDAToolkit 包
find_package(CUDAToolkit REQUIRED)

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(SOURCES cpu_cache.cpp cache_bench.cpp)
add_executable(cache_bench ${SOURCES})

#################### set Torch
set(TORCH_INSTALL_PREFIX "${Torch_DIR}/../..")

set(TORCH_CUDA_LIBRARIES
    "${TORCH_INSTALL_PREFIX}/lib/libc10_cuda.so"
    "${TORCH_INSTALL_PREFIX}/lib/libtorch_cuda.so"
)


# 链接 libtorch、OpenMP，以及 CUDA Runtime 库 (cudart)
target_link_libraries(cache_bench 
    PRIVATE 
      ${TORCH_LIBRARIES}
      ${TORCH_CUDA_LIBRARIES}
      OpenMP::OpenMP_CXX
      CUDA::cudart
)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 包含 CUDA 头文件路径（find_package(CUDAToolkit) 会设置这个变量）
get_filename_component(LIBTORCH_BASE ${Torch_DIR} DIRECTORY)

target_include_directories(cache_bench PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 输出目录和编译选项
set_property(TARGET cache_bench PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_compile_options(cache_bench PRIVATE -Wall -O2 -fopenmp)
