#include "cuda_ops.h"
#include <torch/extension.h>  // 包含必要的 PyTorch 扩展头文件
#include <ATen/cuda/CUDAContext.h>  // 新增：包含 aten cuda 相关的头文件

// 获取当前 CUDA 流
at::cuda::CUDAStream getCurrentCUDAStream() {
    return at::cuda::getCurrentCUDAStream();  // 使用新的 API
}

CUDATransferResult transfer_cache_to_gpu(
    int group_num,
    const std::vector<torch::Tensor>& cache_keys,
    const std::vector<torch::Tensor>& cache_values) {

    // 使用正确的 API 获取 CUDA 流
    std::vector<at::cuda::CUDAStream> group_streams;  // 修改类型为 at::cuda::CUDAStream
    std::vector<torch::Tensor> cached_k_gpu(group_num);
    std::vector<torch::Tensor> cached_v_gpu(group_num);

    for (int i = 0; i < group_num; ++i) {
        // 使用 at::cuda::getCurrentCUDAStream 获取当前 CUDA 流
        group_streams.push_back(getCurrentCUDAStream());
    }

    for (int i = 0; i < group_num; ++i) {
        auto& stream = group_streams[i];
        // 使用 at::cuda::CUDAGuard 进行设备和流保护
        at::cuda::CUDAGuard device_guard(stream.device());

        // 设置当前流
        at::cuda::CUDAStreamGuard stream_guard(stream);

        cached_k_gpu[i] = cache_keys[i].to(torch::kCUDA, true /* non-blocking */).contiguous();
        cached_v_gpu[i] = cache_values[i].to(torch::kCUDA, true /* non-blocking */).contiguous();
    }

    return {cached_k_gpu, cached_v_gpu};
}

CUDAUnhitTransferResult transfer_unhits_to_gpu(
    int group_num,
    const std::vector<torch::Tensor>& unhit_k_list,
    const std::vector<torch::Tensor>& unhit_v_list) {

    // 使用正确的 API 获取 CUDA 流
    std::vector<at::cuda::CUDAStream> group_streams;  // 修改类型为 at::cuda::CUDAStream
    std::vector<torch::Tensor> unhit_k_gpu(group_num);
    std::vector<torch::Tensor> unhit_v_gpu(group_num);

    for (int i = 0; i < group_num; ++i) {
        // 使用 at::cuda::getCurrentCUDAStream 获取当前 CUDA 流
        group_streams.push_back(getCurrentCUDAStream());
    }

    for (int i = 0; i < group_num; ++i) {
        auto& stream = group_streams[i];
        // 使用 at::cuda::CUDAGuard 进行设备和流保护
        at::cuda::CUDAGuard device_guard(stream.device());

        // 设置当前流
        at::cuda::CUDAStreamGuard stream_guard(stream);

        unhit_k_gpu[i] = unhit_k_list[i].to(torch::kCUDA, true /* non-blocking */).contiguous();
        unhit_v_gpu[i] = unhit_v_list[i].to(torch::kCUDA, true /* non-blocking */).contiguous();
    }

    return {unhit_k_gpu, unhit_v_gpu};
}



