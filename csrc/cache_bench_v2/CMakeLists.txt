cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cache_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED Torch_DIR)
    message(FATAL_ERROR "Torch_DIR is not set.")
endif()

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(SOURCES cpu_cache.cpp cache_bench.cpp)
add_executable(cache_bench ${SOURCES})

# 链接库
target_link_libraries(cache_bench "${TORCH_LIBRARIES}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(cache_bench OpenMP::OpenMP_CXX)
endif()

# 设置输出目录和编译选项
set_property(TARGET cache_bench PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_compile_options(cache_bench PRIVATE -Wall -O2 -fopenmp)
target_include_directories(cache_bench PRIVATE ${TORCH_INCLUDE_DIRS})