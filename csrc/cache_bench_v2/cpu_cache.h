#ifndef CPU_CACHE_H
#define CPU_CACHE_H

#include <vector>
#include <unordered_set>
#include <torch/torch.h>
#include <omp.h>  // OpenMP 并行加速

class CPUCache {

private:
    int cache_token_size;
    int bh;
    int head_dim;

    // 新增静态分类相关成员变量
    std::vector<int> head_classes_;  // 每个head的预定义类别
    std::unordered_map<int, std::vector<int>> class_groups_;  // 类别到head列表的映射

public:
    // 新增的成员变量
    torch::Tensor cache_keys;   // 关键缓存张量
    torch::Tensor cache_values; // 值缓存张量
    std::vector<std::unordered_set<int>> cache_maps;  // 高效哈希存储
    std::vector<int64_t> cur_cache_shape;

    CPUCache();

    /*
        prefetch_idx: shape (n', 1, bh)
        cache_shape: (n', bh, d)
        k_cache: Key cache (n, bh, d)
        v_cache: Value cache (n, bh, d)
    */

    CPUCache(int bh, const torch::Tensor& prefetch_idx, const std::vector<int64_t>& cache_shape);
    std::vector<std::vector<int>> get_unhit(const torch::Tensor& prefetch_idx);
    at::Tensor pad_and_convert_unhits(const std::vector<std::vector<int>>& pure_unhit_list);
    
    std::tuple<at::Tensor, at::Tensor> select_kv(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);
    
    // cache load
    // return gpu_k, gpu_v
    std::tuple<at::Tensor, at::Tensor, std::vector<std::vector<int>>> load_with_cached(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);
    
    // direct load
    // return gpu_k, gpu_v, cpu_k, cpu_v
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> direct_load(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);

    
    std::tuple<std::vector<std::vector<int>>, double> load_with_cached_test(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);
    double direct_load_test(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);
    
    std::vector<int64_t> show_cache_shape();

    // cache 更新
    int update_cache_map(const torch::Tensor& prefetch_idx);
    int update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);


    // class cache v2
    std::vector<int> classify_heads(const std::vector<std::vector<int>>& pure_unhit_list, const int idx_len);
    std::unordered_map<int, std::vector<int>> group_heads_by_class(const std::vector<int>& head_classes);
    torch::Tensor pad_class_unhits(const std::vector<std::vector<int>>& class_unhit_list, int max_unhit, int num_heads);
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<int>>> load_with_cached_v2(
        const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);
    std::tuple<torch::Tensor, torch::Tensor> select_kv_v2(
        const torch::Tensor& prefetch_idx,  // 形状: (max_unhit, num_heads_in_class)
        const torch::Tensor& k_cache,       // 形状: (n, bh_total, d)
        const torch::Tensor& v_cache,
        const std::vector<int>& head_group // 新增参数：当前组的头索引列表
    );
    
    std::vector<torch::Tensor> SplitGroupIdx(
        std::vector<std::vector<int>>& unhit_list, 
        std::unordered_map<int, std::vector<int>>& group_head_ids
    );
    
    void initialize_head_classes(const std::vector<int>& head_classes);

};

#endif  // CPU_CACHE_H
