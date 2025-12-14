cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cache_bench LANGUAGES CXX)

# 设置 C++ 版本
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 确保找到 LibTorch
if(NOT DEFINED Torch_DIR)
    message(FATAL_ERROR "Torch_DIR is not set. Please set it to your LibTorch installation path.")
endif()

find_package(Torch REQUIRED)

# 添加 TBB 支持
find_package(TBB REQUIRED)
if(TBB_FOUND)
    message(STATUS "TBB found: ${TBB_VERSION}")
endif()

# 添加 OpenMP 支持
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
endif()

# 源文件
set(SOURCES cpu_cache.cpp cache_bench.cpp)

# 生成可执行文件
add_executable(cache_bench ${SOURCES})

# 统一使用关键字格式链接所有库
target_link_libraries(cache_bench
    PRIVATE
    "${TORCH_LIBRARIES}"
    TBB::tbb
    OpenMP::OpenMP_CXX
)

# 确保 RPATH 设置正确
set_property(TARGET cache_bench PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 设置编译选项
target_compile_options(cache_bench 
    PRIVATE 
    -Wall 
    -O2 
    -fopenmp
)

# 设置包含路径
target_include_directories(cache_bench 
    PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)