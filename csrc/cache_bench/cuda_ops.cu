// cuda_ops.cu
#include "cuda_ops.h"
#include <torch/extension.h>  // 替代 ATen/cuda/CUDAStream.h
#include <vector>

CUDATransferResult transfer_cache_to_gpu(
    int group_num,
    const std::vector<torch::Tensor>& cache_keys,
    const std::vector<torch::Tensor>& cache_values) {

    std::vector<c10::cuda::CUDAStream> group_streams;
    std::vector<torch::Tensor> cached_k_gpu(group_num);
    std::vector<torch::Tensor> cached_v_gpu(group_num);

    for (int i = 0; i < group_num; ++i) {
        group_streams.push_back(c10::cuda::getDefaultCUDAStream());
    }

    for (int i = 0; i < group_num; ++i) {
        auto& stream = group_streams[i];
        at::cuda::CUDAStreamGuard guard(stream);

        cached_k_gpu[i] = cache_keys[i].to(torch::kCUDA, true /*non-blocking*/);
        cached_v_gpu[i] = cache_values[i].to(torch::kCUDA, true /*non-blocking*/);
    }

    return {cached_k_gpu, cached_v_gpu};
}

CUDAUnhitTransferResult transfer_unhits_to_gpu(
    int group_num,
    const std::vector<torch::Tensor>& unhit_k_list,
    const std::vector<torch::Tensor>& unhit_v_list) {

    std::vector<c10::cuda::CUDAStream> group_streams;
    std::vector<torch::Tensor> unhit_k_gpu(group_num);
    std::vector<torch::Tensor> unhit_v_gpu(group_num);

    for (int i = 0; i < group_num; ++i) {
        group_streams.push_back(c10::cuda::getDefaultCUDAStream());
    }

    for (int i = 0; i < group_num; ++i) {
        auto& stream = group_streams[i];
        at::cuda::CUDAStreamGuard guard(stream);

        unhit_k_gpu[i] = unhit_k_list[i].to(torch::kCUDA, true /*non-blocking*/);
        unhit_v_gpu[i] = unhit_v_list[i].to(torch::kCUDA, true /*non-blocking*/);
    }

    return {unhit_k_gpu, unhit_v_gpu};
}