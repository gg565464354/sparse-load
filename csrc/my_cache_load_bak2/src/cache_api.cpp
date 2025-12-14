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

    int stat = ALLCache.update_group_cache(prefetch_idx, group_k_cache, group_v_cache);
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

// std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> cache_load_v2(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache) {
//     return ALLCache.load_with_cached_v2(prefetch_idx, k_cache, v_cache);
// }

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_cached_kv() {
    return ALLCache.get_cached_kv();
}
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> get_unhit_kv(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache) {
    return ALLCache.get_unhit_kv(prefetch_idx, k_cache, v_cache);
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



std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> 
    static_get_unhit_kv(CPUCache& cache, const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values)
{
    std::cout << "[c++] get_unhit_kv: start step 1" << std::endl;

    // 对每个group 分开处理
    std::vector<torch::Tensor> group_unhit_k;
    std::vector<torch::Tensor> group_unhit_v;
    group_unhit_k.reserve(cache.group_num);
    group_unhit_v.reserve(cache.group_num);


    // 原逻辑
    // Step 2: 获取未命中的索引列表
    auto pure_unhit_list = cache.get_unhit(prefetch_idx);
    auto group_unhits = cache.split_unhit(pure_unhit_list);

    
    std::cout << "[c++] get_unhit_kv: unhit get success" << std::endl;

    for (int i = 0; i < cache.group_num; i ++) {
        auto tmp_unhit = group_unhits[i];

        // Step 3: 使用 pad_and_convert_unhits 填充未命中索引
        auto unhit_tensor = cache.pad_and_convert_unhits(tmp_unhit);
        auto unhit_tensor_int = unhit_tensor.to(torch::kInt);  // 确保 unhit_tensor 是 Int 类型

        // Step 5: 选择未命中的 KV 缓存
        auto [un_cached_k, un_cached_v] = cache.select_kv_v2(unhit_tensor_int, keys, values, cache.class_groups_[i]);

        // step 7: 添加到输出结果内
        group_unhit_k.push_back(un_cached_k);
        group_unhit_v.push_back(un_cached_v);
    }
    
    std::cout << "[c++] get_unhit_kv: finish" << std::endl;

    
    return {group_unhit_k, group_unhit_v, group_unhits};

    // return {group_unhit_k, group_unhit_v, null};
}




#endif