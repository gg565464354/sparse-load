#include "cpu_cache.h"
#include <iostream>
#include <unordered_set>
#include <vector>
#include <torch/torch.h>
#include <omp.h>
#include <chrono>
#include <stdexcept>

CPUCache::CPUCache(int bh, const torch::Tensor& prefetch_idx, const std::vector<int64_t>& cache_shape)
    : bh(bh), cache_token_size(cache_shape[0]), head_dim(cache_shape[2]) {
    
    torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU)
        .pinned_memory(true);

    // 初始化 cache_keys 和 cache_values 为 pinned memory
    cache_keys = torch::empty(cache_shape, options);
    cache_values = torch::empty(cache_shape, options);
    cur_cache_shape = cache_shape;

    // 初始化 cache_maps，使用 unordered_set 提高查询效率
    cache_maps.resize(bh);
    update_cache_map(prefetch_idx);
}


CPUCache::CPUCache() {
    std::vector<int64_t> tmp = {0,0,0};
    cur_cache_shape = tmp;
}

std::vector<int64_t> CPUCache::show_cache_shape() {
    return cur_cache_shape;
}

int CPUCache::update_cache_map(const torch::Tensor& prefetch_idx) {
    auto bh_index = prefetch_idx.permute({2, 1, 0}).contiguous().view({bh, -1});

    for (int i = 0; i < bh; i++) {
        cache_maps[i].clear();

        auto bh_tensor = bh_index[i].contiguous();  // 创建左值
        auto bh_idx_list = bh_tensor.accessor<int, 1>();  // 现在可以安全使用 accessor

        for (int j = 0; j < bh_idx_list.size(0); j++) {
            cache_maps[i].insert(bh_idx_list[j]);
        }
    }

    return 0;
}

int CPUCache::update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& new_k_cache, const torch::Tensor& new_v_cache) {
    update_cache_map(prefetch_idx);
    cache_keys = new_k_cache;
    cache_values = new_v_cache;

    return 0;
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
    if (pure_unhit_list.empty()) {
        return torch::empty({1, 1, 1}, at::kInt);
    }

    size_t max_unhit_len = 2;
    bool all_empty = true;

    for (const auto& unhit : pure_unhit_list) {
        if (!unhit.empty()) {
            all_empty = false;
            max_unhit_len = std::max(max_unhit_len, unhit.size());
        }
    }

    if (all_empty) {
        max_unhit_len = 1;
    }

    std::vector<int> flat_data(pure_unhit_list.size() * max_unhit_len, 0);

    #pragma omp parallel for
    for (size_t i = 0; i < pure_unhit_list.size(); ++i) {
        const auto& unhit = pure_unhit_list[i];
        size_t offset = i * max_unhit_len;
        for (size_t j = 0; j < unhit.size(); ++j) {
            flat_data[offset + j] = unhit[j];
        }
    }

    at::Tensor unhit_tensor = torch::from_blob(
        flat_data.data(),
        {static_cast<long>(pure_unhit_list.size()), 1, static_cast<long>(max_unhit_len)},
        at::kInt
    ).clone();

    return unhit_tensor.permute({2, 1, 0});
}



std::tuple<torch::Tensor, torch::Tensor, std::vector<std::vector<int>>> CPUCache::load_with_cached(const torch::Tensor& prefetch_idx,
                                                                            const torch::Tensor& keys,
                                                                            const torch::Tensor& values) {

    // Step 1: 开始计时
    // torch::cuda::synchronize();
    // auto start_time = std::chrono::high_resolution_clock::now();

    // Step 4: 将缓存的 key 和 value 从 pinned memory 传输到 GPU
    // 重叠通信和计算
    auto gpu_cached_k = cache_keys.to(torch::kCUDA);
    auto gpu_cached_v = cache_values.to(torch::kCUDA);

    // Step 2: 获取未命中的索引列表
    auto pure_unhit_list = get_unhit(prefetch_idx);

    // Step 3: 使用 pad_and_convert_unhits 填充未命中索引
    auto unhit_tensor = pad_and_convert_unhits(pure_unhit_list);
    auto unhit_tensor_int = unhit_tensor.to(torch::kInt);  // 确保 unhit_tensor 是 Int 类型
    // std::cout << "unhit_tensor shape = " << unhit_tensor.sizes() << std::endl;    


    // Step 5: 选择未命中的 KV 缓存
    auto [un_cached_k, un_cached_v] = select_kv(unhit_tensor_int, keys, values);

    // 将未命中的 key 和 value 传输到 GPU
    auto gpu_uncached_k = un_cached_k.to(torch::kCUDA);
    auto gpu_uncached_v = un_cached_v.to(torch::kCUDA);

    // Step 6: 在 GPU 上拼接缓存和未命中的 KV 缓存
    auto final_k = torch::cat({gpu_cached_k, gpu_uncached_k}, 0);
    auto final_v = torch::cat({gpu_cached_v, gpu_uncached_v}, 0);

    // Step 7: 结束计时并计算通信时间
    // torch::cuda::synchronize();
    // auto end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> communication_time = end_time - start_time;

    return {final_k, final_v, pure_unhit_list};
}


std::tuple<std::vector<std::vector<int>>, double> CPUCache::load_with_cached_test(const torch::Tensor& prefetch_idx,
                                                                             const torch::Tensor& keys,
                                                                             const torch::Tensor& values) {

    // Step 1: 开始计时
    torch::cuda::synchronize();
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Step 4: 将缓存的 key 和 value 从 pinned memory 传输到 GPU
    // 重叠通信和计算
    auto gpu_cached_k = cache_keys.to(torch::kCUDA);
    auto gpu_cached_v = cache_values.to(torch::kCUDA);

    // Step 2: 获取未命中的索引列表
    auto pure_unhit_list = get_unhit(prefetch_idx);

    // Step 3: 使用 pad_and_convert_unhits 填充未命中索引
    auto unhit_tensor = pad_and_convert_unhits(pure_unhit_list);
    auto unhit_tensor_int = unhit_tensor.to(torch::kInt);  // 确保 unhit_tensor 是 Int 类型
    // std::cout << "unhit_tensor shape = " << unhit_tensor.sizes() << std::endl;    
    

    // Step 5: 选择未命中的 KV 缓存
    auto [un_cached_k, un_cached_v] = select_kv(unhit_tensor_int, keys, values);

    // 将未命中的 key 和 value 传输到 GPU
    auto gpu_uncached_k = un_cached_k.to(torch::kCUDA);
    auto gpu_uncached_v = un_cached_v.to(torch::kCUDA);

    // Step 6: 在 GPU 上拼接缓存和未命中的 KV 缓存
    auto final_k = torch::cat({gpu_cached_k, gpu_uncached_k}, 0);
    auto final_v = torch::cat({gpu_cached_v, gpu_uncached_v}, 0);

    // Step 7: 结束计时并计算通信时间
    torch::cuda::synchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> communication_time = end_time - start_time;

    return {pure_unhit_list, communication_time.count()};
}


double CPUCache::direct_load_test(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values) {

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

    return communication_time.count();
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CPUCache::direct_load(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values) {

    // // Step 1: 开始计时
    // torch::cuda::synchronize();
    // auto start_time = std::chrono::high_resolution_clock::now();

    // Step 2: 选择预取的 KV 缓存
    auto [prefetch_k, prefetch_v] = select_kv(prefetch_idx, keys, values);

    // 将未命中的 key 和 value 传输到 GPU
    auto gpu_prefetch_k = prefetch_k.to(torch::kCUDA);
    auto gpu_prefetch_v = prefetch_v.to(torch::kCUDA);

    // // Step 7: 结束计时并计算通信时间
    // torch::cuda::synchronize();
    // auto end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> communication_time = end_time - start_time;

    return {gpu_prefetch_k, gpu_prefetch_v, prefetch_k, prefetch_v};
}


///////////////////////////////////////// 

// 辅助函数：分类头
std::vector<int> CPUCache::classify_heads(const std::vector<std::vector<int>>& pure_unhit_list, const int idx_len) {
    std::vector<int> classes(bh);
    for (int i = 0; i < bh; ++i) {
        int count = pure_unhit_list[i].size();
        // 示例分类策略，可根据实际情况调整
        if (count <= 5) {
            classes[i] = 0;
        } else if (count <= 10) {
            classes[i] = 1;
        } else {
            classes[i] = 2;
        }
    }
    return classes;
}

// 辅助函数：按类别分组头
std::unordered_map<int, std::vector<int>> CPUCache::group_heads_by_class(const std::vector<int>& head_classes) {
    std::unordered_map<int, std::vector<int>> groups;
    for (int i = 0; i < head_classes.size(); ++i) {
        groups[head_classes[i]].push_back(i);
    }
    return groups;
}

// 辅助函数：填充类别的未命中索引
torch::Tensor CPUCache::pad_class_unhits(const std::vector<std::vector<int>>& class_unhit_list, int max_unhit, int num_heads) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor unhit_tensor = torch::full({max_unhit, num_heads}, -1, options);

    for (int h = 0; h < num_heads; ++h) {
        const auto& unhits = class_unhit_list[h];
        int num_unhits = unhits.size();
        if (num_unhits > 0) {
            auto indices = torch::from_blob(const_cast<int*>(unhits.data()), num_unhits, options);
            unhit_tensor.slice(0, 0, num_unhits).select(1, h) = indices;
        }
    }

    return unhit_tensor;
}

// std::tuple<torch::Tensor, torch::Tensor, std::vector<std::vector<int>>> CPUCache::load_with_cached_v2(
//     const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values) {

//     // Step 1: 获取未命中列表
//     auto pure_unhit_list = get_unhit(prefetch_idx);

//     // Step 2: 分类头
//     int idx_len = prefetch_idx.size(0)
//     std::vector<int> head_classes = classify_heads(pure_unhit_list, idx_len);

//     // Step 3: 分组头
//     auto class_to_heads = group_heads_by_class(head_classes);

//     // Step 4: 处理每个类别的未命中数据
//     std::vector<torch::Tensor> all_uncached_k, all_uncached_v;
//     int global_max_unhit = 0;

//     for (const auto& [class_id, heads] : class_to_heads) {
//         std::vector<std::vector<int>> class_unhit_list;
//         for (int head : heads) {
//             class_unhit_list.push_back(pure_unhit_list[head]);
//         }

//         // 计算该组的最大未命中数
//         int max_unhit = 0;
//         for (const auto& list : class_unhit_list) {
//             max_unhit = std::max(max_unhit, static_cast<int>(list.size()));
//         }
//         global_max_unhit = std::max(global_max_unhit, max_unhit);

//         // 填充并转换为张量
//         auto class_unhit_tensor = pad_class_unhits(class_unhit_list, max_unhit, heads.size());
//         auto [un_cached_k, un_cached_v] = select_kv(class_unhit_tensor.to(torch::kInt), keys, values);

//         // 传输到GPU并保存
//         all_uncached_k.push_back(un_cached_k.to(torch::kCUDA));
//         all_uncached_v.push_back(un_cached_v.to(torch::kCUDA));
//     }

//     // Step 5: 合并所有类别的未命中数据
//     auto options = torch::TensorOptions().device(torch::kCUDA).dtype(keys.dtype());
//     torch::Tensor final_uncached_k = torch::zeros({global_max_unhit, bh, keys.size(2)}, options);
//     torch::Tensor final_uncached_v = torch::zeros({global_max_unhit, bh, values.size(2)}, options);

//     for (size_t i = 0; i < all_uncached_k.size(); ++i) {
//         const auto& [class_id, heads] = *std::next(class_to_heads.begin(), i);
//         const auto& class_k = all_uncached_k[i];
//         const auto& class_v = all_uncached_v[i];
        
//         int class_max_unhit = class_k.size(0);
//         auto slice = torch::indexing::Slice(0, class_max_unhit);

//         // 将当前类别的数据复制到对应头的位置
//         final_uncached_k.index_put_({slice, torch::tensor(heads)}, class_k);
//         final_uncached_v.index_put_({slice, torch::tensor(heads)}, class_v);
//     }

//     // Step 6: 拼接缓存和未命中数据
//     auto gpu_cached_k = cache_keys.to(torch::kCUDA);
//     auto gpu_cached_v = cache_values.to(torch::kCUDA);
//     auto final_k = torch::cat({gpu_cached_k, final_uncached_k}, 0);
//     auto final_v = torch::cat({gpu_cached_v, final_uncached_v}, 0);

//     return {final_k, final_v, pure_unhit_list};
// }

// std::tuple<torch::Tensor, torch::Tensor> CPUCache::select_kv_v2(const torch::Tensor& prefetch_idx,
//                                                    const torch::Tensor& k_cache,
//                                                    const torch::Tensor& v_cache) {
//     // ...（原有代码）

//     // 使用padding_idx处理无效索引
//     auto selected_k = torch::embedding(flat_k_cache, ind, -1, /*padding_idx=*/-1); 
//     auto selected_v = torch::embedding(flat_v_cache, ind, -1, /*padding_idx=*/-1);

//     return {selected_k, selected_v};
// }


// 新增初始化接口
void CPUCache::initialize_head_classes(const std::vector<int>& head_classes) {
    head_classes_ = head_classes;
    class_groups_.clear();
    for (int i = 0; i < head_classes_.size(); ++i) {
        class_groups_[head_classes_[i]].push_back(i);
    }
}

// std::tuple<torch::Tensor, torch::Tensor, std::vector<std::vector<int>>> CPUCache::load_with_cached_v2(
//                 const torch::Tensor& prefetch_idx,
//                 const torch::Tensor& keys,
//                 const torch::Tensor& values) {
//     // 步骤1：获取未命中列表
//     auto pure_unhit_list = get_unhit(prefetch_idx);

//     // 步骤2：分类处理每组head
//     std::vector<torch::Tensor> class_uncached_k, class_uncached_v;
//     int global_max_unhit = 0;

//     // 遍历所有预定义类别
//     for (const auto& [class_id, heads] : class_groups_) {
//         // 步骤3：收集当前类别的未命中数据
//         std::vector<std::vector<int>> class_unhit_list;
//         for (int head : heads) {
//             class_unhit_list.push_back(pure_unhit_list[head]);
//         }

//         // 步骤4：计算当前类别的最大未命中数
//         int max_unhit = 0;
//         for (const auto& list : class_unhit_list) {
//             max_unhit = std::max(max_unhit, static_cast<int>(list.size()));
//         }
//         global_max_unhit = std::max(global_max_unhit, max_unhit);

//         // 步骤5：填充并处理当前类别
//         auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
//         torch::Tensor class_unhit_tensor = torch::full({max_unhit, heads.size()}, -1, options);
        
//         // 填充实际未命中索引
//         for (size_t h = 0; h < heads.size(); ++h) {
//             const auto& unhits = class_unhit_list[h];
//             if (!unhits.empty()) {
//                 class_unhit_tensor.slice(0, 0, unhits.size()).select(1, h) = 
//                     torch::from_blob(const_cast<int*>(unhits.data()), unhits.size(), options);
//             }
//         }

//         // 步骤6：选择并传输KV
//         auto [k, v] = select_kv_v2(class_unhit_tensor.to(torch::kInt), keys, values, heads);
//         class_uncached_k.push_back(k.to(torch::kCUDA));
//         class_uncached_v.push_back(v.to(torch::kCUDA));
//     }

//     // 步骤7：在GPU上合并所有类别数据
//     auto options = torch::TensorOptions().device(torch::kCUDA).dtype(keys.dtype());
//     torch::Tensor final_uncached_k = torch::zeros({global_max_unhit, bh, keys.size(2)}, options);
//     torch::Tensor final_uncached_v = torch::zeros({global_max_unhit, bh, values.size(2)}, options);

//     for (size_t i = 0; i < class_uncached_k.size(); ++i) {
//         const auto& heads = class_groups_[i];  // 假设class_id从0开始连续
//         const auto& class_k = class_uncached_k[i];
//         const auto& class_v = class_uncached_v[i];

//         int class_max_unhit = class_k.size(0);
//         final_uncached_k.slice(0, 0, class_max_unhit).index_put_(
//             {torch::indexing::Slice(), torch::tensor(heads)},
//             class_k
//         );
//         final_uncached_v.slice(0, 0, class_max_unhit).index_put_(
//             {torch::indexing::Slice(), torch::tensor(heads)},
//             class_v
//         );
//     }

//     // 步骤8：合并缓存和未命中数据
//     auto gpu_cached_k = cache_keys.to(torch::kCUDA);
//     auto gpu_cached_v = cache_values.to(torch::kCUDA);
//     auto final_k = torch::cat({gpu_cached_k, final_uncached_k}, 0);
//     auto final_v = torch::cat({gpu_cached_v, final_uncached_v}, 0);

//     return {final_k, final_v, pure_unhit_list};
// }
// 源文件 (cpu_cache.cpp)
std::vector<torch::Tensor> CPUCache::SplitGroupIdx(
    std::vector<std::vector<int>>& unhit_list, 
    std::unordered_map<int, std::vector<int>>& group_head_ids) {
    
    std::vector<torch::Tensor> grouped_indices;
    int global_max = 0;
    int type_num = group_head_ids.size();

    for (int i = 0; i < type_num; ++i) {
        // 确定该组的最大未命中数
        int max_unhit = 0;
        for (int head : group_head_ids.at(i)) { // 使用 at() 避免越界
            max_unhit = std::max(max_unhit, (int)unhit_list[head].size());
        }
        
        // 创建填充后的张量
        auto options = torch::TensorOptions().dtype(torch::kInt32);
        torch::Tensor group_idx = torch::full({max_unhit, (int)group_head_ids.at(i).size()}, 0, options);
        
        // 填充每个头的数据
        for (size_t h = 0; h < group_head_ids.at(i).size(); ++h) {
            int head = group_head_ids.at(i)[h];
            auto& unhits = unhit_list[head]; // 使用 const 引用避免拷贝
            
            // 拷贝有效数据并填充
            if (!unhits.empty()) {
                auto head_tensor = torch::from_blob(unhits.data(), {(int)unhits.size()}, options);
                group_idx.slice(0, 0, unhits.size()).select(1, h) = head_tensor;
            }
        }
        grouped_indices.push_back(group_idx.unsqueeze(1)); // 添加中间维度

        // 获取全局最大batch size
        global_max = std::max(global_max, max_unhit);
    }

    return grouped_indices; // 修复缺少分号的问题
}


std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<int>>> CPUCache::load_with_cached_v2(
                const torch::Tensor& prefetch_idx,
                const torch::Tensor& keys,
                const torch::Tensor& values) {
    // 步骤1: 获取未命中列表
    auto pure_unhit_list = get_unhit(prefetch_idx);

    // 步骤2: 构造不同head的idx
    auto group_head_ids = SplitGroupIdx(pure_unhit_list, class_groups_);

    // 步骤3: 对每个group进行select和传输
    std::vector<torch::Tensor> v2_k_results;
    std::vector<torch::Tensor> v2_v_results;
    for (size_t g=0; g<group_head_ids.size(); ++g) {
        auto [k, v] = select_kv_v2(
            group_head_ids[g], 
            keys,
            values,
            class_groups_[g]
        );
        auto gpu_k = k.cuda();
        auto gpu_v = v.cuda();
        v2_k_results.push_back(gpu_k);
        v2_v_results.push_back(gpu_v);
    }
    // 步骤3: 传输结果
    return {v2_k_results, v2_v_results, pure_unhit_list};
}



/////////////////////////////////////////


// std::tuple<torch::Tensor, torch::Tensor> CPUCache::select_kv_v2(
//     const torch::Tensor& prefetch_idx,  // 形状: (max_unhit, num_heads_in_class)
//     const torch::Tensor& k_cache,       // 形状: (n, bh_total, d)
//     const torch::Tensor& v_cache,
//     const std::vector<int>& head_group // 新增参数：当前组的头索引列表
// ) {
//     // Step 1: 确保 prefetch_idx 是 2 维，并将其移动到与 k_cache 相同的设备
//     auto squeezed_idx = prefetch_idx.to(k_cache.device());
//     if (squeezed_idx.dim() == 1) {
//         squeezed_idx = squeezed_idx.unsqueeze(0); // (1, num_heads_in_class)
//     }

    
//     // std::cout<< "squeezed_idx = " << squeezed_idx << std::endl;

//     // Step 2: 检查 prefetch_idx 的有效性
//     int n = k_cache.size(0);
//     auto invalid_indices = torch::logical_or(squeezed_idx < 0, squeezed_idx >= n);
//     if (invalid_indices.any().item<bool>()) {
//         throw std::out_of_range("Indices out of range in prefetch_idx");
//     }

//     // Step 3: 计算每个头在展平缓存中的偏移量
//     const int bh_total = k_cache.size(1);
//     const int num_heads_in_class = head_group.size();
//     torch::Tensor head_offsets = torch::tensor(head_group, torch::kInt32)
//                                     .to(k_cache.device())
//                                     .unsqueeze(0); // (1, num_heads_in_class)

//     // Step 4: 计算索引 ind = prefetch_idx * bh_total + head_offset
//     auto ind = (squeezed_idx * bh_total + head_offsets).to(torch::kInt); // (max_unhit, num_heads_in_class)

//     // Step 5: 展平缓存
//     auto flat_k_cache = k_cache.view({-1, k_cache.size(2)}); // (n*bh_total, d)
//     auto flat_v_cache = v_cache.view({-1, v_cache.size(2)});

//     // Step 6: 使用 embedding 选择数据（支持填充）
//     auto selected_k = torch::embedding(flat_k_cache, ind, -1, /*padding_idx=*/-1);
//     auto selected_v = torch::embedding(flat_v_cache, ind, -1, /*padding_idx=*/-1);

//     return {selected_k, selected_v};
// }

std::tuple<torch::Tensor, torch::Tensor> CPUCache::select_kv_v2(
    const torch::Tensor& prefetch_idx,  // 输入形状应为 (max_unhit, 1, num_heads_in_class)
    const torch::Tensor& k_cache,       // 形状: (n, bh_total, d)
    const torch::Tensor& v_cache,
    const std::vector<int>& head_group // 当前组的头索引列表
) {
    // Step 1: 调整输入维度为 (max_unhit, num_heads_in_class)
    auto processed_idx = prefetch_idx.squeeze(1).to(k_cache.device());  // 去除中间的单一维度
    
    // Step 2: 检查有效性（保持原有逻辑）
    int n = k_cache.size(0);
    auto invalid_indices = torch::logical_or(processed_idx < 0, processed_idx >= n);
    if (invalid_indices.any().item<bool>()) {
        throw std::out_of_range("Indices out of range in prefetch_idx");
    }

    // Step 3: 计算索引
    const int bh_total = k_cache.size(1);
    const int num_heads_in_class = head_group.size();
    torch::Tensor head_offsets = torch::tensor(head_group, torch::kInt32)
                                    .to(k_cache.device())
                                    .view({1, -1});  // 形状 (1, num_heads_in_class)

    auto ind = (processed_idx * bh_total + head_offsets).to(torch::kInt);  // 形状 (max_unhit, num_heads_in_class)

    // Step 4: 展平缓存并选择
    auto flat_k_cache = k_cache.view({-1, k_cache.size(2)});
    auto flat_v_cache = v_cache.view({-1, v_cache.size(2)});

    std::cout << "ind = " << ind << std::endl;
    
    // 选择后增加维度以保持维度一致
    auto selected_k = torch::embedding(flat_k_cache, ind, -1, -1); 
    auto selected_v = torch::embedding(flat_v_cache, ind, -1, -1);

    return {selected_k, selected_v};
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

    // Step 2: 检查 prefetch_idx 的有效性
    int n = k_cache.size(0);
    auto invalid_indices = torch::logical_or(squeezed_idx < 0, squeezed_idx >= n);
    if (invalid_indices.any().item<bool>()) {
        throw std::out_of_range("Indices out of range in prefetch_idx");
    }

    // Step 3: 计算 ind = prefetch_idx * k_cache.size(1) + arange(k_cache.size(1))
    int bh = k_cache.size(1);  // 获取 bh（batch size * head number）
    auto arange_tensor = torch::arange(bh, torch::TensorOptions().dtype(torch::kInt).device(k_cache.device()));

    // 使用广播机制直接计算 ind，避免多余的 unsqueeze
    auto ind = (squeezed_idx * bh + arange_tensor).to(torch::kInt);  // Shape: (n', bh)

    // Step 4: 将 k_cache 和 v_cache 展平为 (n * bh, d)
    auto flat_k_cache = k_cache.view({-1, k_cache.size(2)});  // Shape: (n * bh, d)
    auto flat_v_cache = v_cache.view({-1, v_cache.size(2)});  // Shape: (n * bh, d)

    // Step 5: 使用 embedding 操作选择对应的缓存
    auto selected_k = torch::embedding(flat_k_cache, ind, -1, /*padding_idx=*/-1);
    auto selected_v = torch::embedding(flat_v_cache, ind, -1, /*padding_idx=*/-1);

    return {selected_k, selected_v};
}



