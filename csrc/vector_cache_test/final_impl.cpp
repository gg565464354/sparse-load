#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <tuple>
#include <stdexcept>
#include <omp.h>


// std::vector版本的select_kv
void select_kv_vector(
    const std::vector<std::vector<std::vector<int>>>& prefetch_idx,
    const std::vector<std::vector<std::vector<float>>>& k_cache,
    const std::vector<std::vector<std::vector<float>>>& v_cache,
    const std::vector<int>& head_group,
    std::vector<std::vector<std::vector<float>>>& selected_k,
    std::vector<std::vector<std::vector<float>>>& selected_v
) {
    int unhit = prefetch_idx.size();
    int bh = head_group.size();
    int d = k_cache[0][0].size();

    selected_k.resize(unhit, std::vector<std::vector<float>>(bh, std::vector<float>(d)));
    selected_v.resize(unhit, std::vector<std::vector<float>>(bh, std::vector<float>(d)));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < unhit; ++i) {
        for (int h = 0; h < bh; ++h) {
            int idx = prefetch_idx[i][0][h];
            int head_id = head_group[h]
            for (int j = 0; j < d; ++j) {
                selected_k[i][h][j] = k_cache[idx][h][j];
                selected_v[i][h][j] = v_cache[idx][h][j];
            }
        }
    }

    
    // for (int i = 0; i < unhit; ++i) {
    //     for (int h = 0; h < bh; ++h) {
    //         int token_idx = prefetch_idx[i][0][h];
    //         // int head_idx = head_group[j];

    //         selected_k[i][h] = k_cache[token_idx][h];
    //         selected_v[i][h] = v_cache[token_idx][h];
    //     }
    // }
}


// std::vector版本的select_kv
void select_kv_vector(
    const std::vector<std::vector<std::vector<int>>>& prefetch_idx,
    const std::vector<std::vector<std::vector<float>>>& k_cache,
    const std::vector<std::vector<std::vector<float>>>& v_cache,
    std::vector<std::vector<std::vector<float>>>& selected_k,
    std::vector<std::vector<std::vector<float>>>& selected_v
) {
    int unhit = prefetch_idx.size();
    int bh = k_cache[0].size();
    int d = k_cache[0][0].size();

    selected_k.resize(unhit, std::vector<std::vector<float>>(bh, std::vector<float>(d)));
    selected_v.resize(unhit, std::vector<std::vector<float>>(bh, std::vector<float>(d)));
    
    for (int i = 0; i < unhit; ++i) {
        for (int h = 0; h < bh; ++h) {
            int token_idx = prefetch_idx[i][0][h];
            // int head_idx = head_group[j];

            selected_k[i][h] = k_cache[token_idx][h];
            selected_v[i][h] = v_cache[token_idx][h];
        }
    }

    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < unhit; ++i) {
    //     for (int h = 0; h < bh; ++h) {
    //         int idx = prefetch_idx[i][0][h];
    //         for (int j = 0; j < d; ++j) {
    //             selected_k[i][h][j] = k_cache[idx][h][j];
    //             selected_v[i][h][j] = v_cache[idx][h][j];
    //         }
    //     }
    // }
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
    // std::vector<std::vector<std::vector<float>>> selected_k(max_unhit,
    //     std::vector<std::vector<float>>(num_heads_in_class, std::vector<float>(d, 0.0f)));
    // std::vector<std::vector<std::vector<float>>> selected_v = selected_k; // same shape

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