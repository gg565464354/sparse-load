#ifndef CACHE_API_H
#define CACHE_API_H

#include <vector>
#include <unordered_set>
#include <torch/torch.h>
#include "cpu_cache.h"

// CPUCache ALLCache; 

// int init_cache(int bh, const torch::Tensor& cache_idx, const std::vector<int64_t>& cache_shape); // return 0-success, 1-fail update
int init_cache(int bh, const torch::Tensor& cache_idx, const std::vector<int64_t>& cache_shape,  const std::vector<std::vector<int>>& class_group_ids); // return 0-success, 1-fail
int update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache); // return 0-success, 1-uninitial cache
int update_cache_idx(const torch::Tensor& prefetch_idx); // return 0-success, 1-uninitial cache
std::vector<int64_t> show_cache_shape();
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> cache_load(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_load(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);

// select k/v and upload to gpu, the selected k/v can be used as next cache
int update_group_cache(const torch::Tensor& prefetch_idx, const std::vector<torch::Tensor>& group_k_cache, const std::vector<torch::Tensor>& group_v_cache);
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    generate_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> 
    generate_update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);

#endif  // CPU_CACHE_H