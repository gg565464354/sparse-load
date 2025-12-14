#ifndef CPU_CACHE_H
#define CPU_CACHE_H

#include <vector>
#include <unordered_set>
#include <torch/torch.h>
#include <omp.h>  // OpenMP 并行加速

class CPUCache {
public:
    CPUCache(int bh, const torch::Tensor& prefetch_idx, const std::vector<int64_t>& cache_shape);
    std::vector<std::vector<int>> get_unhit(const torch::Tensor& prefetch_idx);
    at::Tensor pad_and_convert_unhits(const std::vector<std::vector<int>>& pure_unhit_list);
    
    std::tuple<at::Tensor, at::Tensor> select_kv(const torch::Tensor& prefetch_idx,
                                                 const torch::Tensor& k_cache,
                                                 const torch::Tensor& v_cache);
    
    std::tuple<std::vector<std::vector<int>>, double> load_with_cached(const torch::Tensor& prefetch_idx,
                                                                       const torch::Tensor& keys,
                                                                       const torch::Tensor& values);
    double direct_load(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);
    
    void _update_cache_map(const torch::Tensor& prefetch_idx);

private:
    // 新增的成员变量
    torch::Tensor cache_keys;   // 关键缓存张量
    torch::Tensor cache_values; // 值缓存张量
    
    int cache_token_size;
    int bh;
    int head_dim;
    std::vector<std::unordered_set<int>> cache_maps;  // 高效哈希存储
};

#endif  // CPU_CACHE_H




///////////////////////////////// pure version
// #include <vector>
// #include <unordered_set>
// #include <torch/torch.h>
// #include <iostream>

// class CPUCache {
// public:
//     CPUCache(int bh, const torch::Tensor& prefetch_idx, std::vector<int64_t> cache_shape)
//         : cache_token_size(cache_shape[0]), bh(bh), head_dim(cache_shape.back()) {
        
//         cache_keys = torch::empty(cache_shape, torch::kFloat).cpu().pin_memory();
//         cache_values = torch::empty(cache_shape, torch::kFloat).cpu().pin_memory();
        
//         cache_maps.resize(bh);
//         for (int i = 0; i < bh; ++i) {
//             for (int j = 0; j < cache_token_size; ++j) {
//                 cache_maps[i].insert(j);
//             }
//         }
//         _update_cache_map(prefetch_idx);
//     }

//     void _update_cache_map(const torch::Tensor& prefetch_idx) {
//         int bh = prefetch_idx.size(-1);
//         auto bh_index = prefetch_idx.permute({2, 1, 0}).view({bh, -1});

//         for (int i = 0; i < bh; ++i) {
//             cache_maps[i].clear();
//             auto bh_idx_list = bh_index[i].to(torch::kCPU).contiguous();
//             auto bh_idx_data = bh_idx_list.data_ptr<int>();
//             for (int j = 0; j < bh_idx_list.size(0); ++j) {
//                 cache_maps[i].insert(bh_idx_data[j]);
//             }
//         }
//     }

//     std::vector<std::vector<int>> get_unhit(const torch::Tensor& prefetch_idx) {
//         int token_num = prefetch_idx.size(0);
//         int bh = prefetch_idx.size(2);
//         std::vector<std::vector<int>> pure_unhit_list(bh);

//         for (int i = 0; i < bh; ++i) {
//             const auto& bh_cache_set = cache_maps[i];
//             auto prefetch_data = prefetch_idx.accessor<int, 3>();

//             for (int tid = 0; tid < token_num; ++tid) {
//                 int cur_token = prefetch_data[tid][0][i];
//                 if (bh_cache_set.find(cur_token) == bh_cache_set.end()) {
//                     pure_unhit_list[i].push_back(cur_token);
//                 }
//             }
//         }
//         return pure_unhit_list;
//     }

// private:
//     int cache_token_size;
//     int bh;
//     int head_dim;
//     torch::Tensor cache_keys;
//     torch::Tensor cache_values;
//     std::vector<std::unordered_set<int>> cache_maps;
// };


