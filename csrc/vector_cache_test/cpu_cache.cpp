#include "cpu_cache.h"
// #include "cuda_ops.h"

#include <iostream>
#include <unordered_set>
#include <vector>
#include <torch/torch.h>
#include <omp.h>
#include <chrono>
#include <stdexcept>

#include <iomanip> // for std::setw

using namespace std;
using namespace std::chrono;

// #include <ATen/cuda/CUDAContext.h>   // for getStreamFromPool
// #include <ATen/cuda/CUDAStream.h>    // for CUDAStream + CUDAStreamGuard

CPUCache::CPUCache(int _bh, const torch::Tensor& prefetch_idx, const std::vector<int64_t>& cache_shape, const std::vector<int>& head_classes) {
    
    bh = _bh; 
    cache_token_size = cache_shape[0]; 
    head_dim = cache_shape[2];
    head_num = cache_shape[1];
    cur_cache_shape = cache_shape;

    torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(torch::kCPU)
        .pinned_memory(true);
    
    // 初始化 group
    initialize_head_classes(head_classes);
    
    // 初始化cache keys
    cache_keys = std::vector<torch::Tensor>{};
    cache_values = std::vector<torch::Tensor>{};
    auto tmp_shape = cache_shape; 
    for (int i = 0; i < group_num; i ++) {
        int group_len = class_groups_[i].size();
        tmp_shape[1] = group_len;

        cache_keys.push_back(torch::empty(tmp_shape, options));
        cache_values.push_back(torch::empty(tmp_shape, options));
    }

    // 初始化 cache_maps，使用 unordered_set 提高查询效率
    cache_maps.resize(bh);
    update_cache_map(prefetch_idx);
}


CPUCache::CPUCache(
    int _bh, 
    const torch::Tensor& prefetch_idx, 
    const std::vector<int64_t>& cache_shape, 
    const std::vector<std::vector<int>>& class_group_ids
) {
    bh = _bh; 
    cache_token_size = cache_shape[0]; 
    head_dim = cache_shape[2];
    head_num = cache_shape[1];
    cur_cache_shape = cache_shape;

    torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU)
        .pinned_memory(true);
    
    // 初始化 group
    initialize_head_classes(class_group_ids);
    
    // 初始化cache keys
    cache_keys = std::vector<torch::Tensor>{};
    cache_values = std::vector<torch::Tensor>{};
    auto tmp_shape = cache_shape; 
    for (int i = 0; i < group_num; i ++) {
        int group_len = class_groups_[i].size();
        tmp_shape[1] = group_len;

        cache_keys.push_back(torch::empty(tmp_shape, options));
        cache_values.push_back(torch::empty(tmp_shape, options));
    }

    // 初始化 cache_maps，使用 unordered_set 提高查询效率
    cache_maps.resize(bh);
    update_cache_map(prefetch_idx);
}


CPUCache::CPUCache() {
    std::vector<int64_t> tmp = {0,0,0};
    cur_cache_shape = tmp;
}

// 新增初始化接口
void CPUCache::initialize_head_classes(const std::vector<int>& head_classes) {
    head_classes_ = head_classes;
    class_groups_.clear();
    for (size_t i = 0; i < head_classes_.size(); ++i) {
        class_groups_[head_classes_[i]].push_back(i);
    }
    group_num = class_groups_.size();

    // 转化成tensor，方便后续使用
    // 遍历并转换
    for (const auto& [key, vec] : class_groups_) {
        // 将 std::vector<int> 转换为 torch::Tensor
        torch::Tensor tensor = torch::tensor(vec, torch::kInt32);
        // 插入到新的 unordered_map 中
        class_groups_tensor_[key] = tensor;
    }
}

void CPUCache::initialize_head_classes(const std::vector<std::vector<int>>& class_group_ids) {
    class_groups_.clear();
    for (size_t i = 0; i < class_group_ids.size(); ++i) {
        class_groups_[i] = class_group_ids[i];
    }
    group_num = class_groups_.size();

    // 转化成tensor，方便后续使用
    // 遍历并转换
    for (const auto& [key, vec] : class_groups_) {
        // 将 std::vector<int> 转换为 torch::Tensor
        torch::Tensor tensor = torch::tensor(vec, torch::kInt32);
        // 插入到新的 unordered_map 中
        class_groups_tensor_[key] = tensor;
    }

    // for update_v2
    std::vector<int> all_indices;
    for (const auto& [key, vec] : class_groups_) {
        all_indices.insert(all_indices.end(), vec.begin(), vec.end());
    }
    combined_group_ids_tensor = torch::tensor(all_indices, torch::kInt32);
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

    for (int i = 0; i < group_num; i ++) {
        cache_keys[i] =  torch::index_select(new_k_cache, 1, class_groups_tensor_[i]);
        cache_values[i] = torch::index_select(new_v_cache, 1, class_groups_tensor_[i]);
    }

    // 考虑复制到pin memory中，暂时先不做，后面再说

    return 0;
}


int CPUCache::update_cache_v2(const torch::Tensor& prefetch_idx, const torch::Tensor& new_k_cache, const torch::Tensor& new_v_cache) {
    update_cache_map(prefetch_idx);

    // update v2
    torch::Tensor selected_keys = torch::index_select(new_k_cache, 1, combined_group_ids_tensor);
    torch::Tensor selected_values = torch::index_select(new_v_cache, 1, combined_group_ids_tensor);

    int offset = 0;
    for (const auto& [key, vec] : class_groups_) {
        int group_size = vec.size();
        cache_keys[key] = selected_keys.index({torch::indexing::Slice(), torch::indexing::Slice(offset, offset + group_size)});
        cache_values[key] = selected_values.index({torch::indexing::Slice(), torch::indexing::Slice(offset, offset + group_size)});
        offset += group_size;
    }

    return 0;
}


std::vector<std::vector<std::vector<int>>> CPUCache::split_unhit(std::vector<std::vector<int>> pure_unhit_list) {
    std::vector<std::vector<std::vector<int>>> result;

    for (int i = 0; i < group_num; i ++) {
        std::vector<std::vector<int>> tmp;

        for (size_t j = 0; j < class_groups_[i].size(); j ++) {
            tmp.push_back(pure_unhit_list[class_groups_[i][j]]);
        }
        
        result.push_back(tmp);
        
        // std::cout << "######## split unhit \n" << std::endl;
        // std::cout << "tmp size = " << tmp.size() << std::endl;
    }

    return result;
}


torch::Tensor CPUCache::pad_and_convert_unhits(const std::vector<std::vector<int>>& pure_unhit_list) {
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

    torch::Tensor unhit_tensor = torch::from_blob(
        flat_data.data(),
        {static_cast<long>(pure_unhit_list.size()), 1, static_cast<long>(max_unhit_len)},
        at::kInt
    ).clone();

    return unhit_tensor.permute({2, 1, 0});
}



std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> CPUCache::load_with_cached(
    const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values) {

    // Step 1: 将缓存的 key 和 value 从 pinned memory 传输到 GPU
    // 重叠通信和计算
    std::vector<torch::Tensor> group_cached_k;
    std::vector<torch::Tensor> group_cached_v;
    group_cached_v.reserve(group_num);
    group_cached_v.reserve(group_num);
    
    for (int i = 0; i < group_num; i ++) {
        group_cached_k.push_back(cache_keys[i].to(torch::kCUDA, /*non_blocking=*/true));
        group_cached_v.push_back(cache_values[i].to(torch::kCUDA, /*non_blocking=*/true));
    }
    
    // Step 2: 获取未命中的索引列表
    auto pure_unhit_list = get_unhit(prefetch_idx);
    auto group_unhits = split_unhit(pure_unhit_list);

    // torch::cuda::synchronize();
    // auto unhit_time = std::chrono::high_resolution_clock::now();

    // 对每个group 分开处理
    std::vector<torch::Tensor> group_final_k;
    std::vector<torch::Tensor> group_final_v;
    group_final_k.reserve(group_num);
    group_final_v.reserve(group_num);

    std::vector<torch::Tensor> group_unhit_k;
    std::vector<torch::Tensor> group_unhit_v;
    group_unhit_k.reserve(group_num);
    group_unhit_v.reserve(group_num);

    for (int i = 0; i < group_num; i ++) {
        auto tmp_unhit = group_unhits[i];

        // Step 3: 使用 pad_and_convert_unhits 填充未命中索引
        auto unhit_tensor = pad_and_convert_unhits(tmp_unhit);
        auto unhit_tensor_int = unhit_tensor.to(torch::kInt);  // 确保 unhit_tensor 是 Int 类型

        // Step 5: 选择未命中的 KV 缓存
        auto [un_cached_k, un_cached_v] = select_kv_v2(unhit_tensor_int, keys, values, class_groups_[i]);

        // 将未命中的 key 和 value 传输到 GPU
        auto gpu_unhit_k = un_cached_k.to(torch::kCUDA, /*non_blocking=*/true);
        auto gpu_unhit_v = un_cached_v.to(torch::kCUDA, /*non_blocking=*/true);

        auto final_k = torch::cat({group_cached_k[i], gpu_unhit_k}, 0);
        auto final_v = torch::cat({group_cached_v[i], gpu_unhit_v}, 0);

        // step 7: 添加到输出结果内
        group_final_k.push_back(final_k);
        group_final_v.push_back(final_v);
    }

    return {group_final_k, group_final_v, group_unhits};
}


std::tuple<std::vector<std::vector<int>>, double> CPUCache::load_with_cached_test(const torch::Tensor& prefetch_idx,
                                                                             const torch::Tensor& keys,
                                                                             const torch::Tensor& values) {

    // 开始计时
    torch::cuda::synchronize();
    auto start_time = std::chrono::high_resolution_clock::now();

    
    
    // Step 1: 将缓存的 key 和 value 从 pinned memory 传输到 GPU
    // 重叠通信和计算
    std::vector<torch::Tensor> group_cached_k;
    std::vector<torch::Tensor> group_cached_v;
    group_cached_v.reserve(group_num);
    group_cached_v.reserve(group_num);
    
    for (int i = 0; i < group_num; i ++) {
        group_cached_k.push_back(cache_keys[i].to(torch::kCUDA, /*non_blocking=*/true));
        group_cached_v.push_back(cache_values[i].to(torch::kCUDA, /*non_blocking=*/true));
    }
    
    // torch::cuda::synchronize();
    // auto cache_time = std::chrono::high_resolution_clock::now();

    // Step 2: 获取未命中的索引列表
    auto pure_unhit_list = get_unhit(prefetch_idx);
    auto group_unhits = split_unhit(pure_unhit_list);

    // torch::cuda::synchronize();
    // auto unhit_time = std::chrono::high_resolution_clock::now();

    // 对每个group 分开处理
    std::vector<torch::Tensor> group_final_k;
    std::vector<torch::Tensor> group_final_v;
    group_final_k.reserve(group_num);
    group_final_v.reserve(group_num);

    std::vector<torch::Tensor> group_unhit_k;
    std::vector<torch::Tensor> group_unhit_v;
    group_unhit_k.reserve(group_num);
    group_unhit_v.reserve(group_num);

    for (int i = 0; i < group_num; i ++) {
        auto tmp_unhit = group_unhits[i];

        // Step 3: 使用 pad_and_convert_unhits 填充未命中索引
        auto unhit_tensor = pad_and_convert_unhits(tmp_unhit);
        auto unhit_tensor_int = unhit_tensor.to(torch::kInt);  // 确保 unhit_tensor 是 Int 类型

        // Step 5: 选择未命中的 KV 缓存
        auto [un_cached_k, un_cached_v] = select_kv_v2(unhit_tensor_int, keys, values, class_groups_[i]);

        // std::cout << "select success!" << std::endl;

        // 将未命中的 key 和 value 传输到 GPU
        auto gpu_unhit_k = un_cached_k.to(torch::kCUDA, /*non_blocking=*/true);
        auto gpu_unhit_v = un_cached_v.to(torch::kCUDA, /*non_blocking=*/true);

        // group_unhit_k.push_back(gpu_unhit_k);
        // group_unhit_v.push_back(gpu_unhit_v);

        auto final_k = torch::cat({group_cached_k[i], gpu_unhit_k}, 0);
        auto final_v = torch::cat({group_cached_v[i], gpu_unhit_v}, 0);

        // step 7: 添加到输出结果内
        group_final_k.push_back(final_k);
        group_final_v.push_back(final_v);
    }

    // torch::cuda::synchronize();
    // auto cuda_time = std::chrono::high_resolution_clock::now();

    // Step 7: 结束计时并计算通信时间
    torch::cuda::synchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> communication_time = end_time - start_time;


    return {pure_unhit_list, communication_time.count()};
}


// std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> CPUCache::load_with_cached_v2(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values) {

//     // 创建一个新的 CUDA 流用于数据传输
//     c10::cuda::CUDAStream transfer_stream = c10::cuda::getStreamFromPool(true);

//     // Step 1: 将缓存的 key 和 value 从 pinned memory 传输到 GPU
//     // 使用 non_blocking=true 实现异步传输，并指定 stream
//     std::vector<torch::Tensor> group_cached_k;
//     std::vector<torch::Tensor> group_cached_v;
//     group_cached_k.reserve(group_num);
//     group_cached_v.reserve(group_num);

//     for (int i = 0; i < group_num; ++i) {
//         // 先将张量移动到指定的CUDA流中
//         auto gpu_tensor_k = cache_keys[i].to(transfer_stream);
//         auto gpu_tensor_v = cache_values[i].to(transfer_stream);

//         // 再将张量转换到GPU设备，并设置non_blocking=true
//         at::TensorOptions options = gpu_tensor_k.options().device(torch::kCUDA);
//         group_cached_k.push_back(gpu_tensor_k.to(options, /*non_blocking=*/true));
//         group_cached_v.push_back(gpu_tensor_v.to(options, /*non_blocking=*/true));
//     }

//     // Step 2: 获取未命中的索引列表
//     auto pure_unhit_list = get_unhit(prefetch_idx);
//     auto group_unhits = split_unhit(pure_unhit_list);

//     // 对每个group 分开处理
//     std::vector<torch::Tensor> group_final_k;
//     std::vector<torch::Tensor> group_final_v;
//     group_final_k.reserve(group_num);
//     group_final_v.reserve(group_num);

//     std::vector<torch::Tensor> group_unhit_k;
//     std::vector<torch::Tensor> group_unhit_v;
//     group_unhit_k.reserve(group_num);
//     group_unhit_v.reserve(group_num);

//     for (int i = 0; i < group_num; ++i) {
//         auto tmp_unhit = group_unhits[i];

//         // Step 3: 使用 pad_and_convert_unhits 填充未命中索引
//         auto unhit_tensor = pad_and_convert_unhits(tmp_unhit);
//         auto unhit_tensor_int = unhit_tensor.to(torch::kInt);  // 确保 unhit_tensor 是 Int 类型

//         // Step 5: 选择未命中的 KV 缓存
//         auto [un_cached_k, un_cached_v] = select_kv_v2(unhit_tensor_int, keys, values, class_groups_[i]);

//         // 将未命中的 key 和 value 传输到 GPU
//         // 先将张量移动到指定的CUDA流中
//         auto gpu_unhit_k = un_cached_k.to(transfer_stream);
//         auto gpu_unhit_v = un_cached_v.to(transfer_stream);

//         // 再将张量转换到GPU设备，并设置non_blocking=true
//         at::TensorOptions options = gpu_unhit_k.options().device(torch::kCUDA);
//         gpu_unhit_k = gpu_unhit_k.to(options, /*non_blocking=*/true);
//         gpu_unhit_v = gpu_unhit_v.to(options, /*non_blocking=*/true);

//         // 等待传输完成
//         c10::cuda::currentCUDAStream()->waitStream(transfer_stream);

//         // Step 6: 合并缓存和未命中的 key 和 value
//         auto final_k = torch::cat({group_cached_k[i], gpu_unhit_k}, 0);
//         auto final_v = torch::cat({group_cached_v[i], gpu_unhit_v}, 0);

//         // Step 7: 添加到输出结果内
//         group_final_k.push_back(final_k);
//         group_final_v.push_back(final_v);
//     }

//     return {group_final_k, group_final_v, group_unhits};
// }

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> CPUCache::get_cached_kv() {
    return {cache_keys, cache_values};
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


//////////////////////////// select kv

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
    // const int num_heads_in_class = head_group.size();

    // TODO: use class_groups_tensor_ to replace
    torch::Tensor head_offsets = torch::tensor(head_group, torch::kInt32)
                                    .to(k_cache.device())
                                    .view({1, -1});  // 形状 (1, num_heads_in_class)

    
    // std::cout << "ind generate begin " << std::endl;

    // std::cout << "processed_idx = " << processed_idx.sizes() << std::endl;
    // std::cout << "prefetch_idx = " << prefetch_idx.sizes() << std::endl;

    auto ind = (processed_idx * bh_total + head_offsets).to(torch::kInt);  // 形状 (max_unhit, num_heads_in_class)

    // std::cout << "ind generate end " << std::endl;

    // Step 4: 展平缓存并选择
    auto flat_k_cache = k_cache.view({-1, k_cache.size(2)});
    auto flat_v_cache = v_cache.view({-1, v_cache.size(2)});

    // std::cout << "ind = " << ind << std::endl;
    
    // 选择后增加维度以保持维度一致
    auto selected_k = torch::embedding(flat_k_cache, ind, -1, -1); 
    auto selected_v = torch::embedding(flat_v_cache, ind, -1, -1);

    return {selected_k, selected_v};
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



std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> 
    CPUCache::get_unhit_kv(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values)
{
    // std::cout << "[c++] get_unhit_kv: start step 1" << std::endl;

    // 对每个group 分开处理
    std::vector<torch::Tensor> group_unhit_k;
    std::vector<torch::Tensor> group_unhit_v;
    group_unhit_k.reserve(group_num);
    group_unhit_v.reserve(group_num);

    try {
        // 原逻辑
        // Step 2: 获取未命中的索引列表
        auto pure_unhit_list = get_unhit(prefetch_idx);
        auto group_unhits = split_unhit(pure_unhit_list);

        
        std::cout << "[c++] get_unhit_kv: unhit get success" << std::endl;

        for (int i = 0; i < group_num; i ++) {
            auto tmp_unhit = group_unhits[i];

            // Step 3: 使用 pad_and_convert_unhits 填充未命中索引
            auto unhit_tensor = pad_and_convert_unhits(tmp_unhit);
            auto unhit_tensor_int = unhit_tensor.to(torch::kInt);  // 确保 unhit_tensor 是 Int 类型

            // Step 5: 选择未命中的 KV 缓存
            auto [un_cached_k, un_cached_v] = select_kv_v2(unhit_tensor_int, keys, values, class_groups_[i]);

            // step 7: 添加到输出结果内
            group_unhit_k.push_back(un_cached_k);
            group_unhit_v.push_back(un_cached_v);
        }
        
        // std::cout << "[c++] get_unhit_kv: finish" << std::endl;

        
        return {group_unhit_k, group_unhit_v, group_unhits};

    } catch (const std::exception& e) {
        std::cerr << "C++ Exception: " << e.what() << std::endl;
        throw;
    }


    // return {group_unhit_k, group_unhit_v, null};
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


///////////////////////////////////////// 不一定用得上的功能

// 辅助函数：分类头，一半也用不上
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
    for (size_t i = 0; i < head_classes.size(); ++i) {
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


// 划分不同group的id
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


////////////////////////////////////////////// update cache
std::vector<torch::Tensor> CPUCache::SplitIdx(const torch::Tensor& prefetch_idx) {
    std::vector<torch::Tensor> group_idx;

    for (int i = 0; i < group_num; i ++) {
        auto tmp_idx =  torch::index_select(prefetch_idx, -1, class_groups_tensor_[i]);
        group_idx.push_back(tmp_idx);
    }

    return group_idx;
}


// generate the group cache for cache update
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> 
    CPUCache::generate_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values) 
{
    // Step 1: 将缓存的 key 和 value 从 pinned memory 传输到 GPU
    // 重叠通信和计算
    std::vector<torch::Tensor> group_cpu_k;
    std::vector<torch::Tensor> group_cpu_v;

    std::vector<torch::Tensor> group_gpu_k;
    std::vector<torch::Tensor> group_gpu_v;

    std::vector<torch::Tensor> group_idx = SplitIdx(prefetch_idx);

    for (int i = 0; i < group_num; i ++) {
        auto idx_tensor_int = group_idx[i].to(torch::kInt);  // 确保 idx_tensor_int 是 Int 类型

        // 选择需要的 KV 缓存
        auto [tmp_k, tmp_v] = select_kv_v2(idx_tensor_int, keys, values, class_groups_[i]);

        // 将未命中的 key 和 value 传输到 GPU
        auto gpu_tmp_k = tmp_k.to(torch::kCUDA);
        auto gpu_tmp_v = tmp_v.to(torch::kCUDA);

        group_cpu_k.push_back(tmp_k);
        group_cpu_v.push_back(tmp_v);

        group_gpu_k.push_back(gpu_tmp_k);
        group_gpu_v.push_back(gpu_tmp_v);
    }

    // step 6: 返回结果
    return {group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v};
}


// generate cache without upload
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> 
    CPUCache::generate_update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values) 
{
    // Step 1: 将缓存的 key 和 value 从 pinned memory 传输到 GPU
    // 重叠通信和计算
    std::vector<torch::Tensor> group_cpu_k;
    std::vector<torch::Tensor> group_cpu_v;

    std::vector<torch::Tensor> group_idx = SplitIdx(prefetch_idx);

    for (int i = 0; i < group_num; i ++) {
        auto idx_tensor_int = group_idx[i].to(torch::kInt);  // 确保 idx_tensor_int 是 Int 类型

        // 选择需要的 KV 缓存
        auto [tmp_k, tmp_v] = select_kv_v2(idx_tensor_int, keys, values, class_groups_[i]);

        group_cpu_k.push_back(tmp_k);
        group_cpu_v.push_back(tmp_v);

    }

    // step 6: 返回结果
    return {group_cpu_k, group_cpu_v};
}


// use group keys and values to update cpu cache
int CPUCache::update_group_cache(const torch::Tensor& prefetch_idx, 
    const std::vector<torch::Tensor>& group_cpu_k, 
    const std::vector<torch::Tensor>& group_cpu_v) 
{
    update_cache_map(prefetch_idx);
    cache_keys = group_cpu_k;
    cache_values = group_cpu_v;

    return 0;
}

// only update cache withou upload the cache
int CPUCache::asyn_update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values) 
{
    // Step 1: 将缓存的 key 和 value 从 pinned memory 传输到 GPU
    // 重叠通信和计算
    std::vector<torch::Tensor> group_cpu_k;
    std::vector<torch::Tensor> group_cpu_v;

    std::vector<torch::Tensor> group_idx = SplitIdx(prefetch_idx);

    for (int i = 0; i < group_num; i ++) {
        auto idx_tensor_int = group_idx[i].to(torch::kInt);  // 确保 idx_tensor_int 是 Int 类型

        // 选择需要的 KV 缓存
        auto [tmp_k, tmp_v] = select_kv_v2(idx_tensor_int, keys, values, class_groups_[i]);

        group_cpu_k.push_back(tmp_k);
        group_cpu_v.push_back(tmp_v);

    }

    return update_group_cache(prefetch_idx, group_cpu_k, group_cpu_v);
}


/////////////////////////// 基于vector的get_unhit 实现


std::vector<std::vector<int>> CPUCache::get_unhit_vector(const std::vector<std::vector<int>>& prefetch_idx_vec) {
    int token_num = prefetch_idx_vec.size();  // shape: [n][bh]
    std::vector<std::vector<std::vector<int>>> thread_local_unhits(omp_get_max_threads(), std::vector<std::vector<int>>(bh));

    #pragma omp parallel for
    for (int tid = 0; tid < token_num; tid++) {
        int thread_id = omp_get_thread_num();
        const auto& bh_tokens = prefetch_idx_vec[tid];

        for (int i = 0; i < bh; i++) {
            int cur_token = bh_tokens[i];
            const auto& bh_cache_set = cache_maps[i];

            if (bh_cache_set.find(cur_token) == bh_cache_set.end()) {
                thread_local_unhits[thread_id][i].push_back(cur_token);
            }
        }
    }

    // 合并线程结果
    std::vector<std::vector<int>> pure_unhit_list(bh);
    for (const auto& thread_unhits : thread_local_unhits) {
        for (int i = 0; i < bh; i++) {
            pure_unhit_list[i].insert(pure_unhit_list[i].end(), thread_unhits[i].begin(), thread_unhits[i].end());
        }
    }

    return pure_unhit_list;
}


std::tuple<
    std::vector<std::vector<std::vector<std::vector<float>>>>,
    std::vector<std::vector<std::vector<std::vector<float>>>>,
    std::vector<std::vector<std::vector<int>>>>
CPUCache::get_unhit_kv_vector(
    const std::vector<std::vector<int>>& prefetch_idx, // [seq][bh_total]
    const std::vector<std::vector<std::vector<float>>>& all_keys,       // [n][bh_total][d]
    const std::vector<std::vector<std::vector<float>>>& all_values      // [n][bh_total][d]
) {
    // std::cout << "[c++] get_unhit_kv: start step 1" << std::endl;

    std::vector<std::vector<std::vector<std::vector<float>>>> group_unhit_k;
    std::vector<std::vector<std::vector<std::vector<float>>>> group_unhit_v;

    group_unhit_k.reserve(group_num);
    group_unhit_v.reserve(group_num);

    try {
        // Step 1: 获取未命中的索引
        auto pure_unhit_list = get_unhit_vector(prefetch_idx);       // [batch][idx]
        auto group_unhit_list = split_unhit(pure_unhit_list); // [group][head][idx]

        for (int i = 0; i < group_num; ++i) {
            const auto& group_unhit = group_unhit_list[i]; // [batch][head][idx]

            // Step 2: padding & 变换成矩阵 [max_unhit][num_heads_in_group]
            std::vector<std::vector<int>> padded_idx = pad_and_convert_unhits_vector(group_unhit); // 每列是一个 head

            // Step 3: KV选择（使用vector版本）
            auto [selected_k, selected_v] = select_kv_vector_v2(padded_idx, all_keys, all_values, class_groups_[i]);

            group_unhit_k.push_back(std::move(selected_k));
            group_unhit_v.push_back(std::move(selected_v));
        }

        // std::cout << "[c++] get_unhit_kv: finish" << std::endl;
        return {group_unhit_k, group_unhit_v, group_unhit_list};

    } catch (const std::exception& e) {
        std::cerr << "C++ Exception: " << e.what() << std::endl;
        throw;
    }
}


std::tuple<
    std::vector<std::vector<std::vector<std::vector<float>>>>,
    std::vector<std::vector<std::vector<std::vector<float>>>>,
    std::vector<std::vector<std::vector<int>>>>
CPUCache::get_unhit_kv_vector_test(
    const std::vector<std::vector<int>>& prefetch_idx, // [seq][bh_total]
    const std::vector<std::vector<std::vector<float>>>& all_keys,       // [n][bh_total][d]
    const std::vector<std::vector<std::vector<float>>>& all_values      // [n][bh_total][d]
) {
    std::vector<std::vector<std::vector<std::vector<float>>>> group_unhit_k;
    std::vector<std::vector<std::vector<std::vector<float>>>> group_unhit_v;

    group_unhit_k.reserve(group_num);
    group_unhit_v.reserve(group_num);

    try {
        auto t_start = high_resolution_clock::now();

        // Step 1: 获取未命中的索引
        auto pure_unhit_list = get_unhit_vector(prefetch_idx);       // [batch][idx]
        auto t_get_unhit = high_resolution_clock::now();

        // Step 2: 分组处理
        auto group_unhit_list = split_unhit(pure_unhit_list);
        auto t_split = high_resolution_clock::now();

        // Step 3~4: 循环处理每个 group
        double step3_total = 0.0;

        for (int i = 0; i < group_num; ++i) {
            const auto& group_unhit = group_unhit_list[i];

            std::cout << "group_unhit shape = (" << group_unhit.size() << ", " << group_unhit[0].size() << ")\n";

            // Step 3: padding & 变换

            auto t_pad_start_2 = high_resolution_clock::now();
            std::vector<std::vector<int>> padded_idx_2 = pad_and_convert_unhits_vector_test(group_unhit);
            auto t_pad_end_2 = high_resolution_clock::now();

            auto t_pad_start = high_resolution_clock::now();
            std::vector<std::vector<int>> padded_idx = pad_and_convert_unhits_vector(group_unhit);
            auto t_pad_end = high_resolution_clock::now();


            // Step 4: KV选择
            
            // 性能测试
            auto t_select_start_2 = high_resolution_clock::now();
            std::vector<float> selected_k_2;
            std::vector<float> selected_v_2;
            select_kv_list_test(padded_idx, all_keys, all_values, class_groups_[i], selected_k_2, selected_v_2);
            auto t_select_end_2 = high_resolution_clock::now();

            auto t_select_start = high_resolution_clock::now();
            std::vector<std::vector<std::vector<float>>> selected_k;
            std::vector<std::vector<std::vector<float>>> selected_v;
            select_kv_vector_v3(padded_idx, all_keys, all_values, class_groups_[i], selected_k, selected_v);
            auto t_select_end = high_resolution_clock::now();


            step3_total += duration_cast<duration<double, milli>>(t_select_end - t_pad_start).count();

            double pad_cost = duration_cast<duration<double, milli>>(t_pad_end - t_pad_start).count();
            double pad_cost_2 = duration_cast<duration<double, milli>>(t_pad_end_2 - t_pad_start_2).count();
            double select_cost = duration_cast<duration<double, milli>>(t_select_end - t_select_start).count();
            double select_cost_2 = duration_cast<duration<double, milli>>(t_select_end_2 - t_select_start_2).count();

            // std::cout << "pad_cost = " << pad_cost << "ms, select_cost = " << select_cost << "ms\n";
            std::cout << "###################### select_cost 1 = " << select_cost << "ms, select_cost 2 = " << select_cost_2 << "ms\n";
            std::cout << "###################### pad_cost 1 = " << pad_cost << "ms, pad_cost 2 = " << pad_cost_2 << "ms\n";

            group_unhit_k.push_back(std::move(selected_k));
            group_unhit_v.push_back(std::move(selected_v));
        }

        auto t_end = high_resolution_clock::now();

        // 输出耗时
        double total_time = duration_cast<duration<double, milli>>(t_end - t_start).count();
        double step1 = duration_cast<duration<double, milli>>(t_get_unhit - t_start).count();
        double step2 = duration_cast<duration<double, milli>>(t_split - t_get_unhit).count();
        double step3 = step3_total;

        cout.precision(2);
        cout << fixed;
        cout << "[Timing] Total Time: " << total_time << " ms" << endl;
        cout << "[Timing] Step 1 (get_unhit_vector): " << step1 << " ms" << endl;
        cout << "[Timing] Step 2 (split_unhit): " << step2 << " ms" << endl;
        cout << "[Timing] Step 3 (pad_and_select): " << step3 << " ms" << endl;

        return {group_unhit_k, group_unhit_v, group_unhit_list};

    } catch (const std::exception& e) {
        std::cerr << "C++ Exception: " << e.what() << std::endl;
        throw;
    }
}



std::tuple<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<std::vector<float>>>> 
CPUCache::select_kv_vector_v2(
    const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
    const std::vector<std::vector<std::vector<float>>>& k_cache, // shape: (n, bh_total, d)
    const std::vector<std::vector<std::vector<float>>>& v_cache,
    const std::vector<int>& head_group
) {
    
    int max_unhit = prefetch_idx.size();
    int num_heads_in_class = head_group.size();
    int bh_total = k_cache[0].size();
    int d = k_cache[0][0].size();
    int n = k_cache.size();


    auto t_init_start = high_resolution_clock::now();

    // 输出 shape: [max_unhit][num_heads_in_class][d]
    std::vector<std::vector<std::vector<float>>> selected_k(max_unhit,
        std::vector<std::vector<float>>(num_heads_in_class, std::vector<float>(d, 0.0f)));
    std::vector<std::vector<std::vector<float>>> selected_v = selected_k; // same shape

    
    auto t_init_end = high_resolution_clock::now();

    
    auto t_sel_start = high_resolution_clock::now();
    for (int i = 0; i < max_unhit; ++i) {
        for (int j = 0; j < num_heads_in_class; ++j) {
            int token_idx = prefetch_idx[i][j];
            int head_idx = head_group[j];

            // 访问缓存：[token][bh][d]
            selected_k[i][j] = k_cache[token_idx][head_idx];
            selected_v[i][j] = v_cache[token_idx][head_idx];
        }
    }
    auto t_sel_end = high_resolution_clock::now();

    
    double step1 = duration_cast<duration<double, milli>>(t_init_end - t_init_start).count();
    double step2 = duration_cast<duration<double, milli>>(t_sel_end - t_sel_start).count();

    // cout << "################ get unhit select\n";
    // cout << "[Timing] Step 1 init: " << step1 << " ms" << endl;
    // cout << "[Timing] Step 2 select: " << step2 << " ms" << endl;
    // cout << "################ get unhit select end\n";

    return {selected_k, selected_v};
}


std::vector<std::vector<int>> CPUCache::pad_and_convert_unhits_vector_test(
    const std::vector<std::vector<int>>& pure_unhit_list) 
{
    if (pure_unhit_list.empty()) {
        return {{0}};  // 返回 shape = [1][1]
    }

    size_t batch_size = pure_unhit_list.size();
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

    // 初始化结果：[max_unhit_len][batch_size]
    std::vector<std::vector<int>> result(max_unhit_len, std::vector<int>(batch_size, 0));

    // 填充
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& unhit = pure_unhit_list[b];
        for (size_t t = 0; t < unhit.size(); ++t) {
            result[t][b] = unhit[t];
        }
    }

    return result;
}

// OpenMP并行版本
std::vector<std::vector<int>> CPUCache::pad_and_convert_unhits_vector(
    const std::vector<std::vector<int>>& pure_unhit_list)
{
    if (pure_unhit_list.empty()) {
        return {{0}};
    }

    size_t batch_size = pure_unhit_list.size();
    size_t max_unhit_len = 2;
    bool all_empty = true;

    // 串行计算 max_unhit_len（避免 omp 临界区）
    for (const auto& unhit : pure_unhit_list) {
        if (!unhit.empty()) {
            all_empty = false;
            max_unhit_len = std::max(max_unhit_len, unhit.size());
        }
    }

    if (all_empty) {
        max_unhit_len = 1;
    }

    // 初始化结果：[max_unhit_len][batch_size]
    std::vector<std::vector<int>> result(max_unhit_len, std::vector<int>(batch_size, 0));

    // 仅并行填充部分
    #pragma omp parallel for
    for (int b = 0; b < static_cast<int>(batch_size); ++b) {
        const auto& unhit = pure_unhit_list[b];
        for (size_t t = 0; t < unhit.size(); ++t) {
            result[t][b] = unhit[t];
        }
    }

    return result;
}


int CPUCache::select_kv_vector_v3(
    const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
    const std::vector<std::vector<std::vector<float>>>& k_cache, // shape: (n, bh_total, d)
    const std::vector<std::vector<std::vector<float>>>& v_cache,
    const std::vector<int>& head_group,

    std::vector<std::vector<std::vector<float>>>& selected_k,
    std::vector<std::vector<std::vector<float>>>& selected_v
) {
    
    int max_unhit = prefetch_idx.size();
    int num_heads_in_class = head_group.size();
    int bh_total = k_cache[0].size();
    int d = k_cache[0][0].size();
    int n = k_cache.size();

    auto t_init_start = high_resolution_clock::now();

    // 输出 shape: [max_unhit][num_heads_in_class][d]
    selected_k.resize(max_unhit, std::vector<std::vector<float>>(num_heads_in_class, std::vector<float>(d)));
    selected_v.resize(max_unhit, std::vector<std::vector<float>>(num_heads_in_class, std::vector<float>(d)));

    auto t_init_end = high_resolution_clock::now();
    
    auto t_sel_start = high_resolution_clock::now();
    for (int i = 0; i < max_unhit; ++i) {
        for (int j = 0; j < num_heads_in_class; ++j) {
            int token_idx = prefetch_idx[i][j];
            int head_idx = head_group[j];

            // 访问缓存：[token][bh][d]
            selected_k[i][j] = k_cache[token_idx][head_idx];
            selected_v[i][j] = v_cache[token_idx][head_idx];
        }
    }
    auto t_sel_end = high_resolution_clock::now();

    
    double step1 = duration_cast<duration<double, milli>>(t_init_end - t_init_start).count();
    double step2 = duration_cast<duration<double, milli>>(t_sel_end - t_sel_start).count();

    cout << "################ get unhit select\n";
    cout << "[Timing] Step 1 init: " << step1 << " ms" << endl;
    cout << "[Timing] Step 2 select: " << step2 << " ms" << endl;
    cout << "################ get unhit select end\n";

    return 0;
}


int CPUCache::select_kv_vector_v3_test(
    const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
    const std::vector<std::vector<std::vector<float>>>& k_cache, // shape: (n, bh_total, d)
    const std::vector<std::vector<std::vector<float>>>& v_cache,
    const std::vector<int>& head_group,

    std::vector<std::vector<std::vector<float>>>& selected_k,
    std::vector<std::vector<std::vector<float>>>& selected_v
) {
    
    int max_unhit = prefetch_idx.size();
    int num_heads_in_class = head_group.size();
    int bh_total = k_cache[0].size();
    int d = k_cache[0][0].size();
    int n = k_cache.size();

    auto t_init_start = high_resolution_clock::now();

    // 输出 shape: [max_unhit][num_heads_in_class][d]
    // std::vector<std::vector<std::vector<float>>> selected_k(max_unhit,
    //     std::vector<std::vector<float>>(num_heads_in_class, std::vector<float>(d, 0.0f)));
    // std::vector<std::vector<std::vector<float>>> selected_v = selected_k; // same shape

    selected_k.resize(max_unhit, std::vector<std::vector<float>>(num_heads_in_class, std::vector<float>(d)));
    selected_v.resize(max_unhit, std::vector<std::vector<float>>(num_heads_in_class, std::vector<float>(d)));

    auto t_init_end = high_resolution_clock::now();
    
    auto t_sel_start = high_resolution_clock::now();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < max_unhit; ++i) {
        for (int j = 0; j < num_heads_in_class; ++j) {
            int token_idx = prefetch_idx[i][j];
            int head_idx = head_group[j];

            // 访问缓存：[token][bh][d]
            selected_k[i][j] = k_cache[token_idx][head_idx];
            selected_v[i][j] = v_cache[token_idx][head_idx];
        }
    }
    auto t_sel_end = high_resolution_clock::now();

    
    double step1 = duration_cast<duration<double, milli>>(t_init_end - t_init_start).count();
    double step2 = duration_cast<duration<double, milli>>(t_sel_end - t_sel_start).count();

    cout << "################ get unhit select test\n";
    cout << "[Timing] Step 1 init: " << step1 << " ms" << endl;
    cout << "[Timing] Step 2 select: " << step2 << " ms" << endl;
    cout << "################ get unhit select test end\n";

    return 0;
}


/////////////////////////// 基于一维list的实现


std::tuple<
    std::vector<std::vector<float>>,
    std::vector<std::vector<float>>,
    std::vector<std::vector<std::vector<int>>>>
CPUCache::get_unhit_kv_list(
    const std::vector<std::vector<int>>& prefetch_idx, // [seq][bh_total]
    const std::vector<std::vector<std::vector<float>>>& all_keys,       // [n][bh_total][d]
    const std::vector<std::vector<std::vector<float>>>& all_values      // [n][bh_total][d]
) {
    std::vector<std::vector<float>> group_unhit_k;
    std::vector<std::vector<float>> group_unhit_v;

    group_unhit_k.reserve(group_num);
    group_unhit_v.reserve(group_num);

    try {
        // Step 1: 获取未命中的索引
        auto pure_unhit_list = get_unhit_vector(prefetch_idx);       // [batch][idx]

        // Step 2: 分组处理
        auto group_unhit_list = split_unhit(pure_unhit_list);

        // Step 3~4: 循环处理每个 group
        double step3_total = 0.0;

        for (int i = 0; i < group_num; ++i) {
            const auto& group_unhit = group_unhit_list[i];

            // Step 3: padding & 变换
            std::vector<std::vector<int>> padded_idx = pad_and_convert_unhits_vector_test(group_unhit);

            // Step 4: KV选择
            std::vector<float> selected_k;
            std::vector<float> selected_v;
            select_kv_list(padded_idx, all_keys, all_values, class_groups_[i], selected_k, selected_v);

            group_unhit_k.push_back(std::move(selected_k));
            group_unhit_v.push_back(std::move(selected_v));
        }

        return {group_unhit_k, group_unhit_v, group_unhit_list};

    } catch (const std::exception& e) {
        std::cerr << "C++ Exception: " << e.what() << std::endl;
        throw;
    }
}



std::tuple<
    std::vector<std::vector<float>>,
    std::vector<std::vector<float>>,
    std::vector<std::vector<std::vector<int>>>>
CPUCache::get_unhit_kv_list_test(
    const std::vector<std::vector<int>>& prefetch_idx, // [seq][bh_total]
    const std::vector<std::vector<std::vector<float>>>& all_keys,       // [n][bh_total][d]
    const std::vector<std::vector<std::vector<float>>>& all_values      // [n][bh_total][d]
) {
    std::vector<std::vector<float>> group_unhit_k;
    std::vector<std::vector<float>> group_unhit_v;

    group_unhit_k.reserve(group_num);
    group_unhit_v.reserve(group_num);

    try {
        auto t_start = high_resolution_clock::now();

        // Step 1: 获取未命中的索引
        auto pure_unhit_list = get_unhit_vector(prefetch_idx);       // [batch][idx]
        auto t_get_unhit = high_resolution_clock::now();

        // Step 2: 分组处理
        auto group_unhit_list = split_unhit(pure_unhit_list);
        auto t_split = high_resolution_clock::now();

        // Step 3~4: 循环处理每个 group
        double step3_total = 0.0;

        for (int i = 0; i < group_num; ++i) {
            const auto& group_unhit = group_unhit_list[i];

            std::cout << "group_unhit shape = (" << group_unhit.size() << ", " << group_unhit[0].size() << ")\n";

            // Step 3: padding & 变换

            auto t_pad_start_2 = high_resolution_clock::now();
            std::vector<std::vector<int>> padded_idx_2 = pad_and_convert_unhits_vector_test(group_unhit);
            auto t_pad_end_2 = high_resolution_clock::now();

            auto t_pad_start = high_resolution_clock::now();
            std::vector<std::vector<int>> padded_idx = pad_and_convert_unhits_vector(group_unhit);
            auto t_pad_end = high_resolution_clock::now();


            // Step 4: KV选择
            
            // 性能测试
            auto t_select_start_2 = high_resolution_clock::now();
            std::vector<float> selected_k_2;
            std::vector<float> selected_v_2;
            select_kv_list_test(padded_idx, all_keys, all_values, class_groups_[i], selected_k_2, selected_v_2);
            auto t_select_end_2 = high_resolution_clock::now();

            auto t_select_start = high_resolution_clock::now();
            std::vector<std::vector<std::vector<float>>> selected_k;
            std::vector<std::vector<std::vector<float>>> selected_v;
            select_kv_vector_v3(padded_idx, all_keys, all_values, class_groups_[i], selected_k, selected_v);
            auto t_select_end = high_resolution_clock::now();


            step3_total += duration_cast<duration<double, milli>>(t_select_end - t_pad_start).count();

            double pad_cost = duration_cast<duration<double, milli>>(t_pad_end - t_pad_start).count();
            double pad_cost_2 = duration_cast<duration<double, milli>>(t_pad_end_2 - t_pad_start_2).count();
            double select_cost = duration_cast<duration<double, milli>>(t_select_end - t_select_start).count();
            double select_cost_2 = duration_cast<duration<double, milli>>(t_select_end_2 - t_select_start_2).count();

            std::cout << "pad_cost = " << pad_cost << "ms, select_cost = " << select_cost << "ms\n";
            std::cout << "###################### select_cost 1 = " << select_cost << "ms, select_cost 2 = " << select_cost_2 << "ms\n";
            std::cout << "###################### pad_cost 1 = " << pad_cost << "ms, pad_cost 2 = " << pad_cost_2 << "ms\n";

            group_unhit_k.push_back(std::move(selected_k_2));
            group_unhit_v.push_back(std::move(selected_v_2));
        }

        auto t_end = high_resolution_clock::now();

        // 输出耗时
        double total_time = duration_cast<duration<double, milli>>(t_end - t_start).count();
        double step1 = duration_cast<duration<double, milli>>(t_get_unhit - t_start).count();
        double step2 = duration_cast<duration<double, milli>>(t_split - t_get_unhit).count();
        double step3 = step3_total;

        cout.precision(2);
        cout << fixed;
        cout << "[Timing] Total Time: " << total_time << " ms" << endl;
        cout << "[Timing] Step 1 (get_unhit_vector): " << step1 << " ms" << endl;
        cout << "[Timing] Step 2 (split_unhit): " << step2 << " ms" << endl;
        cout << "[Timing] Step 3 (pad_and_select): " << step3 << " ms" << endl;

        return {group_unhit_k, group_unhit_v, group_unhit_list};

    } catch (const std::exception& e) {
        std::cerr << "C++ Exception: " << e.what() << std::endl;
        throw;
    }
}



int CPUCache::select_kv_list_test(
    const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
    const std::vector<std::vector<std::vector<float>>>& k_cache, // shape: (n, bh_total, d)
    const std::vector<std::vector<std::vector<float>>>& v_cache,
    const std::vector<int>& head_group,

    std::vector<float>& selected_k,
    std::vector<float>& selected_v
) {
    
    int max_unhit = prefetch_idx.size();
    int num_heads_in_class = head_group.size();
    int bh_total = k_cache[0].size();
    int d = k_cache[0][0].size();
    int n = k_cache.size();

    auto t_init_start = high_resolution_clock::now();


    selected_k.reserve(max_unhit * num_heads_in_class * d);
    selected_v.reserve(max_unhit * num_heads_in_class * d);

    auto t_init_end = high_resolution_clock::now();
    
    auto t_sel_start = high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < max_unhit; ++i) {
        for (int j = 0; j < num_heads_in_class; ++j) {
            int token_idx = prefetch_idx[i][j];
            int head_idx = head_group[j];

            const auto& k_vec = k_cache[token_idx][head_idx];
            const auto& v_vec = v_cache[token_idx][head_idx];

            int offset = (i * num_heads_in_class + j)*d;

            // 访问缓存：[token][bh][d]
            std::copy(k_vec.begin(), k_vec.end(), selected_k.begin() + offset);
            std::copy(v_vec.begin(), v_vec.end(), selected_v.begin() + offset);
        }
    }
    auto t_sel_end = high_resolution_clock::now();

    
    double step1 = duration_cast<duration<double, milli>>(t_init_end - t_init_start).count();
    double step2 = duration_cast<duration<double, milli>>(t_sel_end - t_sel_start).count();

    cout << "################ List select test\n";
    cout << "[Timing] Step 1 init: " << step1 << " ms" << endl;
    cout << "[Timing] Step 2 select: " << step2 << " ms" << endl;
    cout << "################ List select test end\n";

    return 0;
}


int CPUCache::select_kv_list(
    const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
    const std::vector<std::vector<std::vector<float>>>& k_cache, // shape: (n, bh_total, d)
    const std::vector<std::vector<std::vector<float>>>& v_cache,
    const std::vector<int>& head_group,

    std::vector<float>& selected_k,
    std::vector<float>& selected_v
) {
    int max_unhit = prefetch_idx.size();
    int num_heads_in_class = head_group.size();
    int bh_total = k_cache[0].size();
    int d = k_cache[0][0].size();
    int n = k_cache.size();

    selected_k.reserve(max_unhit * num_heads_in_class * d);
    selected_v.reserve(max_unhit * num_heads_in_class * d);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < max_unhit; ++i) {
        for (int j = 0; j < num_heads_in_class; ++j) {
            int token_idx = prefetch_idx[i][j];
            int head_idx = head_group[j];

            const auto& k_vec = k_cache[token_idx][head_idx];
            const auto& v_vec = v_cache[token_idx][head_idx];

            int offset = (i * num_heads_in_class + j)*d;

            // 访问缓存：[token][bh][d]
            std::copy(k_vec.begin(), k_vec.end(), selected_k.begin() + offset);
            std::copy(v_vec.begin(), v_vec.end(), selected_v.begin() + offset);
        }
    }

    return 0;
}


///////////////////////////// 完全使用一维list
// void flatten_cache(
//     const std::vector<std::vector<std::vector<float>>>& cache,
//     std::vector<float>& flat_cache,
//     int n, int bh_total, int d
// ) {
//     flat_cache.resize(n * bh_total * d);
//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < bh_total; ++j) {
//             size_t offset = (i * bh_total + j) * d;
//             std::memcpy(flat_cache.data() + offset, cache[i][j].data(), d * sizeof(float));
//         }
//     }
// }




// int CPUCache::select_kv_list_v2(
//     const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
//     const std::vector<float>& k_cache, // shape: (n, bh_total, d)
//     const std::vector<float>& v_cache,
//     const int n,
//     const int bh_total,
//     const int d,
//     const std::vector<int>& head_group,

//     std::vector<float>& selected_k,
//     std::vector<float>& selected_v
// ) {
//     int max_unhit = prefetch_idx.size();
//     int num_heads_in_class = head_group.size();

//     selected_k.reserve(max_unhit * num_heads_in_class * d);
//     selected_v.reserve(max_unhit * num_heads_in_class * d);

//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < max_unhit; ++i) {
//         for (int j = 0; j < num_heads_in_class; ++j) {
//             int token_idx = prefetch_idx[i][j];
//             int head_idx = head_group[j];

//             size_t offset_flat = (token_idx * bh_total + head_idx) * d;

//             size_t offset_selected = (i * num_heads_in_class + j) * d;

//             std::memcpy(selected_k.data() + offset_selected, k_cache.data() + offset_flat, d * sizeof(float));
//             std::memcpy(selected_v.data() + offset_selected, v_cache.data() + offset_flat, d * sizeof(float));
//         }
//     }

//     return 0;
// }



// std::tuple<
//     std::vector<std::vector<float>>,
//     std::vector<std::vector<float>>,
//     std::vector<std::vector<std::vector<int>>>>
// CPUCache::get_unhit_kv_list_v2(
//     const std::vector<std::vector<int>>& prefetch_idx, // [seq][bh_total]
//     const std::vector<float>& all_keys,       // [n][bh_total][d]
//     const std::vector<float>& all_values      // [n][bh_total][d]
// ) {
//     std::vector<std::vector<float>> group_unhit_k;
//     std::vector<std::vector<float>> group_unhit_v;

//     group_unhit_k.reserve(group_num);
//     group_unhit_v.reserve(group_num);

//     try {
//         auto t_start = high_resolution_clock::now();

//         // Step 1: 获取未命中的索引
//         auto pure_unhit_list = get_unhit_vector(prefetch_idx);       // [batch][idx]
//         auto t_get_unhit = high_resolution_clock::now();

//         // Step 2: 分组处理
//         auto group_unhit_list = split_unhit(pure_unhit_list);
//         auto t_split = high_resolution_clock::now();

//         // Step 3~4: 循环处理每个 group
//         double step3_total = 0.0;

//         for (int i = 0; i < group_num; ++i) {
//             const auto& group_unhit = group_unhit_list[i];

//             std::cout << "group_unhit shape = (" << group_unhit.size() << ", " << group_unhit[0].size() << ")\n";

//             // Step 3: padding & 变换

//             auto t_pad_start_2 = high_resolution_clock::now();
//             std::vector<std::vector<int>> padded_idx_2 = pad_and_convert_unhits_vector_test(group_unhit);
//             auto t_pad_end_2 = high_resolution_clock::now();

//             auto t_pad_start = high_resolution_clock::now();
//             std::vector<std::vector<int>> padded_idx = pad_and_convert_unhits_vector(group_unhit);
//             auto t_pad_end = high_resolution_clock::now();


//             // Step 4: KV选择
            
//             // 性能测试
//             auto t_select_start_2 = high_resolution_clock::now();
//             std::vector<float> selected_k_2;
//             std::vector<float> selected_v_2;
//             select_kv_list_test(padded_idx, all_keys, all_values, class_groups_[i], selected_k_2, selected_v_2);
//             auto t_select_end_2 = high_resolution_clock::now();

//             auto t_select_start = high_resolution_clock::now();
//             std::vector<std::vector<std::vector<float>>> selected_k;
//             std::vector<std::vector<std::vector<float>>> selected_v;
//             select_kv_vector_v3(padded_idx, all_keys, all_values, class_groups_[i], selected_k, selected_v);
//             auto t_select_end = high_resolution_clock::now();


//             step3_total += duration_cast<duration<double, milli>>(t_select_end - t_pad_start).count();

//             double pad_cost = duration_cast<duration<double, milli>>(t_pad_end - t_pad_start).count();
//             double pad_cost_2 = duration_cast<duration<double, milli>>(t_pad_end_2 - t_pad_start_2).count();
//             double select_cost = duration_cast<duration<double, milli>>(t_select_end - t_select_start).count();
//             double select_cost_2 = duration_cast<duration<double, milli>>(t_select_end_2 - t_select_start_2).count();

//             std::cout << "pad_cost = " << pad_cost << "ms, select_cost = " << select_cost << "ms\n";
//             std::cout << "###################### select_cost 1 = " << select_cost << "ms, select_cost 2 = " << select_cost_2 << "ms\n";
//             std::cout << "###################### pad_cost 1 = " << pad_cost << "ms, pad_cost 2 = " << pad_cost_2 << "ms\n";

//             group_unhit_k.push_back(std::move(selected_k_2));
//             group_unhit_v.push_back(std::move(selected_v_2));
//         }

//         auto t_end = high_resolution_clock::now();

//         // 输出耗时
//         double total_time = duration_cast<duration<double, milli>>(t_end - t_start).count();
//         double step1 = duration_cast<duration<double, milli>>(t_get_unhit - t_start).count();
//         double step2 = duration_cast<duration<double, milli>>(t_split - t_get_unhit).count();
//         double step3 = step3_total;

//         cout.precision(2);
//         cout << fixed;
//         cout << "[Timing] Total Time: " << total_time << " ms" << endl;
//         cout << "[Timing] Step 1 (get_unhit_vector): " << step1 << " ms" << endl;
//         cout << "[Timing] Step 2 (split_unhit): " << step2 << " ms" << endl;
//         cout << "[Timing] Step 3 (pad_and_select): " << step3 << " ms" << endl;

//         return {group_unhit_k, group_unhit_v, group_unhit_list};

//     } catch (const std::exception& e) {
//         std::cerr << "C++ Exception: " << e.what() << std::endl;
//         throw;
//     }
// }


//////////// tensor based

std::vector<std::vector<int>> CPUCache::get_unhit_tensor2vec(const torch::Tensor& prefetch_idx_tensor) {
    int token_num = prefetch_idx_tensor.size(0);

    // Access raw pointer
    const int32_t* prefetch_ptr = prefetch_idx_tensor.data_ptr<int32_t>();

    std::vector<std::vector<std::vector<int>>> thread_local_unhits(omp_get_max_threads(), std::vector<std::vector<int>>(bh));

    #pragma omp parallel for
    for (int tid = 0; tid < token_num; tid++) {
        int thread_id = omp_get_thread_num();

        for (int i = 0; i < bh; i++) {
            int32_t cur_token = prefetch_ptr[tid * bh + i];
            const auto& bh_cache_set = cache_maps[i];

            if (bh_cache_set.find(static_cast<int>(cur_token)) == bh_cache_set.end()) {
                thread_local_unhits[thread_id][i].push_back(static_cast<int>(cur_token));
            }
        }
    }

    // Merge thread-local results
    std::vector<std::vector<int>> pure_unhit_list(bh);
    for (const auto& thread_unhits : thread_local_unhits) {
        for (int i = 0; i < bh; i++) {
            pure_unhit_list[i].insert(pure_unhit_list[i].end(), thread_unhits[i].begin(), thread_unhits[i].end());
        }
    }

    return pure_unhit_list;
}

std::tuple<torch::Tensor, torch::Tensor> CPUCache::select_kv_tensor(
    const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
    const torch::Tensor& k_cache,  // shape: (n, bh_total, d), dtype: float16
    const torch::Tensor& v_cache,  // shape: (n, bh_total, d), dtype: float16
    const std::vector<int>& head_group  // shape: [num_heads_in_class]
) {
    TORCH_CHECK(k_cache.dim() == 3 && v_cache.dim() == 3, "k/v_cache must be 3D tensors");
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16 && v_cache.dtype() == torch::kFloat16, "k/v_cache must be float16");

    int max_unhit = prefetch_idx.size();
    int num_heads_in_class = head_group.size();
    int d = k_cache.size(2);

    // 初始化输出 tensor
    torch::Tensor selected_k = torch::empty({max_unhit, num_heads_in_class, d}, k_cache.options());
    torch::Tensor selected_v = torch::empty({max_unhit, num_heads_in_class, d}, v_cache.options());

    auto t_init_start = high_resolution_clock::now();

    auto k_ptr = k_cache.accessor<at::Half, 3>();  // [n][bh][d]
    auto v_ptr = v_cache.accessor<at::Half, 3>();
    auto sel_k_ptr = selected_k.accessor<at::Half, 3>();
    auto sel_v_ptr = selected_v.accessor<at::Half, 3>();

    auto t_init_end = high_resolution_clock::now();
    auto t_sel_start = high_resolution_clock::now();

    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < max_unhit; ++i) {
        for (int j = 0; j < num_heads_in_class; ++j) {
            int token_idx = prefetch_idx[i][j];  // [n]
            int head_idx = head_group[j];        // [bh]

            for (int k = 0; k < d; ++k) {
                sel_k_ptr[i][j][k] = k_ptr[token_idx][head_idx][k];
                sel_v_ptr[i][j][k] = v_ptr[token_idx][head_idx][k];
            }
        }
    }

    auto t_sel_end = high_resolution_clock::now();

    double step1 = duration_cast<duration<double, milli>>(t_init_end - t_init_start).count();
    double step2 = duration_cast<duration<double, milli>>(t_sel_end - t_sel_start).count();

    cout << "################ get unhit select\n";
    cout << "[Timing] Step 1 init: " << step1 << " ms" << endl;
    cout << "[Timing] Step 2 select: " << step2 << " ms" << endl;
    cout << "################ get unhit select end\n";

    return {selected_k, selected_v};
}


std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> 
    CPUCache::get_unhit_kv_tensor(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values)
{
    // std::cout << "[c++] get_unhit_kv: start step 1" << std::endl;

    // 对每个group 分开处理
    std::vector<torch::Tensor> group_unhit_k;
    std::vector<torch::Tensor> group_unhit_v;
    group_unhit_k.reserve(group_num);
    group_unhit_v.reserve(group_num);

    // 原逻辑
    // Step 2: 获取未命中的索引列表
    auto pure_unhit_vec = get_unhit_tensor2vec(prefetch_idx);
    auto group_unhits = split_unhit(pure_unhit_vec);

    
    // std::cout << "[c++] get_unhit_kv: unhit get success" << std::endl;

    for (int i = 0; i < group_num; i ++) {
        auto tmp_unhit = group_unhits[i];

        // Step 3: 使用 pad_and_convert_unhits 填充未命中索引
        std::vector<std::vector<int>> unhit_idx_vec = pad_and_convert_unhits_vector_test(tmp_unhit);

        // Step 5: 选择未命中的 KV 缓存

        auto [un_cached_k, un_cached_v] = select_kv_tensor(unhit_idx_vec, keys, values, class_groups_[i]);

        // step 7: 添加到输出结果内
        group_unhit_k.push_back(un_cached_k);
        group_unhit_v.push_back(un_cached_v);
    }
    
    // std::cout << "[c++] get_unhit_kv: finish" << std::endl;

    
    return {group_unhit_k, group_unhit_v, group_unhits};
}
