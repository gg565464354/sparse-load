#include "cpu_cache.h"
#include <iostream>
#include <unordered_set>
#include <vector>
#include <torch/torch.h>
#include <omp.h>
#include <chrono>

CPUCache::CPUCache(int bh, const torch::Tensor& prefetch_idx, const std::vector<int64_t>& cache_shape)
    : bh(bh), cache_token_size(cache_shape[0]), head_dim(cache_shape[2]) {
    
    // 初始化 cache_keys 和 cache_values 为 pinned memory
    cache_keys = torch::empty(cache_shape, torch::dtype(torch::kFloat32).pinned_memory(true));
    cache_values = torch::empty(cache_shape, torch::dtype(torch::kFloat32).pinned_memory(true));

    // 初始化 cache_maps，使用 unordered_set 提高查询效率
    cache_maps.resize(bh);
    _update_cache_map(prefetch_idx);
}

void CPUCache::_update_cache_map(const torch::Tensor& prefetch_idx) {
    auto bh_index = prefetch_idx.permute({2, 1, 0}).contiguous().view({bh, -1});

    for (int i = 0; i < bh; i++) {
        cache_maps[i].clear();

        auto bh_tensor = bh_index[i].contiguous();  // 创建左值
        auto bh_idx_list = bh_tensor.accessor<int, 1>();  // 现在可以安全使用 accessor

        for (int j = 0; j < bh_idx_list.size(0); j++) {
            cache_maps[i].insert(bh_idx_list[j]);
        }
    }
}


std::vector<std::vector<int>> CPUCache::get_unhit(const torch::Tensor& prefetch_idx) {
    int token_num = prefetch_idx.size(0);

    // 使用 vector + OpenMP 进行并行化，每个线程有自己的 unhit 列表
    std::vector<std::vector<std::vector<int>>> thread_local_unhits(omp_get_max_threads(), std::vector<std::vector<int>>(bh));

    #pragma omp parallel for
    for (int tid = 0; tid < token_num; tid++) {
        int thread_id = omp_get_thread_num();
        auto accessor = prefetch_idx.accessor<int, 3>();

        for (int i = 0; i < bh; i++) {
            const auto& bh_cache_set = cache_maps[i];
            int cur_token = accessor[tid][0][i];

            if (bh_cache_set.find(cur_token) == bh_cache_set.end()) {
                thread_local_unhits[thread_id][i].push_back(cur_token);
            }
        }
    }

    // 合并所有线程的结果
    std::vector<std::vector<int>> pure_unhit_list(bh);
    for (const auto& thread_unhits : thread_local_unhits) {
        for (int i = 0; i < bh; i++) {
            pure_unhit_list[i].insert(pure_unhit_list[i].end(), thread_unhits[i].begin(), thread_unhits[i].end());
        }
    }

    return pure_unhit_list;
}

at::Tensor CPUCache::pad_and_convert_unhits(const std::vector<std::vector<int>>& pure_unhit_list) {
    // Step 1: 检查 pure_unhit_list 是否为空
    if (pure_unhit_list.empty()) {
        // 返回一个空张量，形状为 (0, 1, 1)
        return torch::empty({1, 1, 1}, at::kInt);
    }

    // Step 2: 找到最大 unhit 长度，最短也应该是2
    size_t max_unhit_len = 2;
    bool all_empty = true;  // 标志变量，用于判断是否所有 unhit 长度为 0

    for (const auto& unhit : pure_unhit_list) {
        if (!unhit.empty()) {
            all_empty = false;
            max_unhit_len = std::max(max_unhit_len, unhit.size());
        }
    }

    // std::cout << "pad_and_convert_unhits max_unhit_len = " << max_unhit_len << std::endl;

    // 如果所有 unhit 长度为 0，强制设置 max_unhit_len = 1
    if (all_empty) {
        max_unhit_len = 1;
    }

    // Step 3: 创建一个 flat vector 存储所有填充后的值
    std::vector<int> flat_data;
    flat_data.reserve(pure_unhit_list.size() * max_unhit_len);

    // Step 4: 并行化填充和转换过程（仅在非全空情况下执行）
    if (!all_empty) {
        #pragma omp parallel
        {
            std::vector<int> local_flat_data;  // 每个线程的本地存储
            local_flat_data.reserve(pure_unhit_list.size() / omp_get_num_threads() * max_unhit_len);

            #pragma omp for nowait
            for (size_t i = 0; i < pure_unhit_list.size(); ++i) {
                const auto& unhit = pure_unhit_list[i];
                for (size_t j = 0; j < unhit.size(); ++j) {
                    local_flat_data.push_back(unhit[j]);
                }
                for (size_t j = unhit.size(); j < max_unhit_len; ++j) {
                    local_flat_data.push_back(0);  // 填充 0
                }
            }

            // 安全地将本地数据合并到全局 flat_data
            #pragma omp critical
            {
                flat_data.insert(flat_data.end(), local_flat_data.begin(), local_flat_data.end());
            }
        }
    } else {
        // 如果所有 unhit 长度为 0，直接填充 0
        for (size_t i = 0; i < pure_unhit_list.size(); ++i) {
            for (size_t j = 0; j < max_unhit_len; ++j) {
                flat_data.push_back(0);  // 填充 0
            }
        }
    }

    // Step 5: 从 flat_data 创建张量，形状为 (bh, 1, n)
    at::Tensor unhit_tensor = torch::from_blob(
        flat_data.data(),
        {static_cast<long>(pure_unhit_list.size()), 1, static_cast<long>(max_unhit_len)},
        at::kInt
    ).clone();  // Clone 确保张量拥有自己的数据

    // Step 6: 将张量从 (bh, 1, n) 转换为 (n, 1, bh)
    unhit_tensor = unhit_tensor.permute({2, 1, 0});  // 转换维度顺序

    return unhit_tensor;
}



std::tuple<torch::Tensor, torch::Tensor> CPUCache::select_kv(const torch::Tensor& prefetch_idx,
                                                   const torch::Tensor& k_cache,
                                                   const torch::Tensor& v_cache) {
    // Step 1: 确保 prefetch_idx 是 2 维，并将其移动到与 k_cache 相同的设备
    auto squeezed_idx = prefetch_idx.squeeze().to(k_cache.device());

    // 如果 prefetch_idx 的形状为 (bh,)，添加一个维度使其变为 (1, bh)
    if (squeezed_idx.dim() == 1) {
        squeezed_idx = squeezed_idx.unsqueeze(0);  // Shape: (1, bh)
    }
    
    // std::cout << "squeezed_idx shape = " << squeezed_idx.sizes() << std::endl;

    // Step 2: 计算 ind = prefetch_idx * k_cache.size(1) + arange(k_cache.size(1))
    int bh = k_cache.size(1);  // 获取 bh（batch size * head number）
    auto arange_tensor = torch::arange(bh, torch::TensorOptions().dtype(torch::kInt).device(k_cache.device()));

    
    // std::cout << "arange_tensor shape = " << arange_tensor.sizes() << std::endl;
    
    // 使用广播机制直接计算 ind，避免多余的 unsqueeze
    auto ind = (squeezed_idx * bh + arange_tensor).to(torch::kInt);  // Shape: (n', bh)

    // Step 3: 将 k_cache 和 v_cache 展平为 (n * bh, d)
    auto flat_k_cache = k_cache.view({-1, k_cache.size(2)});  // Shape: (n * bh, d)
    auto flat_v_cache = v_cache.view({-1, v_cache.size(2)});  // Shape: (n * bh, d)
    
    // std::cout << "flat_k_cache shape = " << flat_k_cache.sizes() << std::endl;

    // Step 4: 使用 embedding 操作选择对应的缓存
    auto selected_k = torch::embedding(flat_k_cache, ind);  // Shape: (n', bh, d)
    auto selected_v = torch::embedding(flat_v_cache, ind);  // Shape: (n', bh, d)

    return {selected_k, selected_v};
}

std::tuple<std::vector<std::vector<int>>, double> CPUCache::load_with_cached(const torch::Tensor& prefetch_idx,
                                                                             const torch::Tensor& keys,
                                                                             const torch::Tensor& values) {



    
    // Step 1: 开始计时
    torch::cuda::synchronize();
    auto start_time = std::chrono::high_resolution_clock::now();

    // Step 2: 获取未命中的索引列表
    auto pure_unhit_list = get_unhit(prefetch_idx);
    
    // Step 4: 将缓存的 key 和 value 从 pinned memory 传输到 GPU
    // 重叠通信和计算
    auto gpu_cached_k = cache_keys.to(torch::kCUDA);
    auto gpu_cached_v = cache_values.to(torch::kCUDA);
    

    // Step 3: 使用 pad_and_convert_unhits 填充未命中索引
    auto unhit_tensor = pad_and_convert_unhits(pure_unhit_list);
    auto unhit_tensor_int = unhit_tensor.to(torch::kInt);  // 确保 unhit_tensor 是 Int 类型
    // std::cout << "unhit_tensor shape = " << unhit_tensor.sizes() << std::endl;    
    

    // Step 5: 选择未命中的 KV 缓存
    auto [un_cached_k, un_cached_v] = select_kv(unhit_tensor_int, keys, values);

    // torch::cuda::synchronize();
    // std::cout << "here 2";

    // 将未命中的 key 和 value 传输到 GPU
    auto gpu_uncached_k = un_cached_k.to(torch::kCUDA);
    auto gpu_uncached_v = un_cached_v.to(torch::kCUDA);

    // Step 6: 在 GPU 上拼接缓存和未命中的 KV 缓存
    auto final_k = torch::cat({gpu_cached_k, gpu_uncached_k}, /*dim=*/0);
    auto final_v = torch::cat({gpu_cached_v, gpu_uncached_v}, /*dim=*/0);

    // torch::cuda::synchronize();
    // std::cout << "here 3";

    // Step 7: 结束计时并计算通信时间
    torch::cuda::synchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> communication_time = end_time - start_time;

    // 打印调试信息
    // std::cout << "Unhit index shape: " << unhit_tensor_int.sizes() << std::endl;
    // std::cout << "Uncached K shape: " << gpu_uncached_k.sizes() << std::endl;
    // std::cout << "Final K shape: " << final_k.sizes() << std::endl;
    // std::cout << "load_with_cached time: " << communication_time.count() << " ms" << std::endl;

    return {pure_unhit_list, communication_time.count()};
    // return {{{0}, {1}}, 1};
}


double CPUCache::direct_load(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values) {

    // Step 1: 开始计时
    torch::cuda::synchronize();
    auto start_time = std::chrono::high_resolution_clock::now();

    // Step 2: 选择预取的 KV 缓存
    auto [prefetch_k, prefetch_v] = select_kv(prefetch_idx, keys, values);

    // 将未命中的 key 和 value 传输到 GPU
    auto gpu_prefetch_k = prefetch_k.to(torch::kCUDA);
    auto gpu_prefetch_v = prefetch_v.to(torch::kCUDA);

    // Step 7: 结束计时并计算通信时间
    torch::cuda::synchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> communication_time = end_time - start_time;

    // 打印调试信息
    // std::cout << "prefetch_idx shape: " << prefetch_idx.sizes() << std::endl;
    // std::cout << "gpu_prefetch_k shape: " << gpu_prefetch_k.sizes() << std::endl;
    // std::cout << "direct_load time: " << communication_time.count() << " ms" << std::endl;

    return communication_time.count();
}