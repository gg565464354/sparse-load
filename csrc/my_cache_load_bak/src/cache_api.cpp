#ifndef CACHE_API_CPP
#define CACHE_API_CPP

#include <vector>
#include <unordered_set>
#include <torch/torch.h>
#include "cache_api.h"
#include "cpu_cache.h"


CPUCache ALLCache = CPUCache(); 

// return 0-success, 1-fail
int init_cache(int bh, const torch::Tensor& cache_idx, const std::vector<int64_t>& cache_shape,  const std::vector<std::vector<int>>& class_group_ids) {
    ALLCache = CPUCache(bh, cache_idx, cache_shape, class_group_ids);
    return 0;
} 

// return 0-success, 1-uninitial cache instance
int update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache) {
    if (ALLCache.cur_cache_shape[0] == 0) {
        return 1;
    }

    int stat = ALLCache.update_cache(prefetch_idx, k_cache, v_cache);
    return stat;
}

// return 0-success, 1-uninitial cache instance
int update_group_cache(const torch::Tensor& prefetch_idx, const std::vector<torch::Tensor>& group_k_cache, const std::vector<torch::Tensor>& group_v_cache) {
    if (ALLCache.cur_cache_shape[0] == 0) {
        return 1;
    }

    int stat = ALLCache.update_cache(prefetch_idx, group_k_cache, group_v_cache);
    return stat;
}

// return 0-success, 1-uninitial cache instance
int update_cache_idx(const torch::Tensor& prefetch_idx) {
    // only update cached_idx
    if (ALLCache.cur_cache_shape[0] == 0) {
        return 1;
    }

    int stat = ALLCache.update_cache_map(prefetch_idx);
    return stat;
}

std::vector<int64_t> show_cache_shape() {
    return ALLCache.show_cache_shape();
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> cache_load(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache) {
    return ALLCache.load_with_cached(prefetch_idx, k_cache, v_cache);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_load(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache) {
    return ALLCache.direct_load(prefetch_idx, k_cache, v_cache);
}


std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    generate_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache) {
    return ALLCache.generate_cache(prefetch_idx, k_cache, v_cache);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    generate_update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache) {
    return ALLCache.generate_update_cache(prefetch_idx, k_cache, v_cache);
}

#endif