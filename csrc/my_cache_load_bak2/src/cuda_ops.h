// cuda_ops.h
#pragma once

#include <vector>
#include <torch/torch.h>

struct CUDATransferResult {
    std::vector<torch::Tensor> cached_k_gpu;
    std::vector<torch::Tensor> cached_v_gpu;
};

struct CUDAUnhitTransferResult {
    std::vector<torch::Tensor> unhit_k_gpu;
    std::vector<torch::Tensor> unhit_v_gpu;
};

CUDATransferResult transfer_cache_to_gpu(
    int group_num,
    const std::vector<torch::Tensor>& cache_keys,
    const std::vector<torch::Tensor>& cache_values);

CUDAUnhitTransferResult transfer_unhits_to_gpu(
    int group_num,
    const std::vector<torch::Tensor>& unhit_k_list,
    const std::vector<torch::Tensor>& unhit_v_list);