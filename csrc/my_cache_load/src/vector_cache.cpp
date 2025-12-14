#include "cpu_cache.h"
// #include "cuda_ops.h"

#include <iostream>
#include <unordered_set>
#include <vector>
#include <torch/torch.h>
#include <omp.h>
#include <chrono>
#include <stdexcept>


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

    // 输出 shape: [max_unhit][num_heads_in_class][d]
    std::vector<std::vector<std::vector<float>>> selected_k(max_unhit,
        std::vector<std::vector<float>>(num_heads_in_class, std::vector<float>(d, 0.0f)));
    std::vector<std::vector<std::vector<float>>> selected_v = selected_k; // same shape

    for (int i = 0; i < max_unhit; ++i) {
        for (int j = 0; j < num_heads_in_class; ++j) {
            int token_idx = prefetch_idx[i][j];
            int head_idx = head_group[j];

            if (token_idx < 0 || token_idx >= n) {
                continue; // skip padding or out-of-bounds
            }

            int flat_idx = token_idx;  // 使用 token_idx 和 head_idx 索引
            int bh_idx = head_idx;

            // 访问缓存：[token][bh][d]
            const auto& k_vec = k_cache[flat_idx][bh_idx];
            const auto& v_vec = v_cache[flat_idx][bh_idx];

            selected_k[i][j] = k_vec;
            selected_v[i][j] = v_vec;
        }
    }

    return {selected_k, selected_v};
}


std::vector<std::vector<int>> CPUCache::pad_and_convert_unhits_vector(
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

