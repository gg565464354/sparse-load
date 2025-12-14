#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <tuple>
#include <stdexcept>
#include <omp.h>

// 使用torch实现的select_kv函数
std::tuple<torch::Tensor, torch::Tensor> select_kv_torch(const torch::Tensor& prefetch_idx,
                                                        const torch::Tensor& k_cache,
                                                        const torch::Tensor& v_cache) {
    auto squeezed_idx = prefetch_idx.squeeze().to(k_cache.device());

    if (squeezed_idx.dim() == 1) {
        squeezed_idx = squeezed_idx.unsqueeze(0);  // Shape: (1, bh)
    }

    int n = k_cache.size(0);
    auto invalid_indices = torch::logical_or(squeezed_idx < 0, squeezed_idx >= n);
    if (invalid_indices.any().item<bool>()) {
        throw std::out_of_range("Indices out of range in prefetch_idx");
    }

    int bh = k_cache.size(1);
    auto arange_tensor = torch::arange(bh, torch::TensorOptions().dtype(torch::kInt).device(k_cache.device()));

    auto ind = (squeezed_idx * bh + arange_tensor).to(torch::kInt);

    auto flat_k_cache = k_cache.view({-1, k_cache.size(2)});
    auto flat_v_cache = v_cache.view({-1, v_cache.size(2)});

    auto selected_k = torch::embedding(flat_k_cache, ind, -1, /*padding_idx=*/-1);
    auto selected_v = torch::embedding(flat_v_cache, ind, -1, /*padding_idx=*/-1);

    return {selected_k, selected_v};
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
    
    // #pragma omp parallel
    // for (int i = 0; i < unhit; ++i) {
    //     for (int h = 0; h < bh; ++h) {
    //         int token_idx = prefetch_idx[i][0][h];
    //         // int head_idx = head_group[j];

    //         selected_k[i][h] = k_cache[token_idx][h];
    //         selected_v[i][h] = v_cache[token_idx][h];
    //     }
    // }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < unhit; ++i) {
        for (int h = 0; h < bh; ++h) {
            int idx = prefetch_idx[i][0][h];
            for (int j = 0; j < d; ++j) {
                selected_k[i][h][j] = k_cache[idx][h][j];
                selected_v[i][h][j] = v_cache[idx][h][j];
            }
        }
    }
}

int main() {
    torch::manual_seed(42);
    const int n = 2048;
    const int bh = 32;
    const int d = 128;
    const int repeats = 100;

    std::vector<int> test_max_unhit = {128, 256, 512, 768};

    for (int max_unhit : test_max_unhit) {
        std::cout << "\n>>> Testing max_unhit = " << max_unhit << std::endl;

        // ========== Torch 数据 ==========
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto k_cache = torch::rand({n, bh, d}, options);
        auto v_cache = torch::rand({n, bh, d}, options);
        auto prefetch_idx_tensor = torch::randint(0, n, {max_unhit, 1, bh}, torch::TensorOptions().dtype(torch::kInt32));

        std::cout << "prefetch_idx_tensor shape = " << prefetch_idx_tensor.sizes() << std::endl;

        // ========== Vector 数据 ==========
        std::vector<std::vector<std::vector<float>>> k_cache_vec(n, std::vector<std::vector<float>>(bh, std::vector<float>(d)));
        std::vector<std::vector<std::vector<float>>> v_cache_vec(n, std::vector<std::vector<float>>(bh, std::vector<float>(d)));
        std::vector<std::vector<std::vector<int>>> prefetch_idx_vec(max_unhit, std::vector<std::vector<int>>(1, std::vector<int>(bh)));

        // 将 torch tensor 转为 vector，k_cache, v_cache
        auto k_data = k_cache.accessor<float, 3>();
        auto v_data = v_cache.accessor<float, 3>();
        for (int i = 0; i < n; ++i)
            for (int h = 0; h < bh; ++h)
                for (int j = 0; j < d; ++j) {
                    k_cache_vec[i][h][j] = k_data[i][h][j];
                    v_cache_vec[i][h][j] = v_data[i][h][j];
                }

        // 将 prefetch_idx_tensor 转为三维vector (max_unhit, 1, bh)
        auto idx_data = prefetch_idx_tensor.accessor<int32_t, 3>();
        for (int i = 0; i < max_unhit; ++i)
            for (int j = 0; j < 1; ++j)
                for (int h = 0; h < bh; ++h) {
                    prefetch_idx_vec[i][j][h] = idx_data[i][j][h];
                }

        // =================== warm up =========
        for (int i = 0; i < 10; i ++)
            auto [t_k, t_v] = select_kv_torch(prefetch_idx_tensor, k_cache, v_cache);

        // ========== Torch 测试 ==========
        double torch_total_ms = 0.0;

        double vec_total_ms = 0.0;
        std::vector<std::vector<std::vector<float>>> sel_k, sel_v;

        for (int r = 0; r < repeats; ++r) {
            
            auto start2 = std::chrono::high_resolution_clock::now();

            select_kv_vector(prefetch_idx_vec, k_cache_vec, v_cache_vec, sel_k, sel_v);

            auto end2 = std::chrono::high_resolution_clock::now();
            vec_total_ms += std::chrono::duration<double, std::milli>(end2 - start2).count();
            

            auto start = std::chrono::high_resolution_clock::now();

            auto [selected_k, selected_v] = select_kv_torch(prefetch_idx_tensor, k_cache, v_cache);

            auto end = std::chrono::high_resolution_clock::now();
            torch_total_ms += std::chrono::duration<double, std::milli>(end - start).count();

        }
        std::cout << "Torch select_kv avg time: " << (torch_total_ms / repeats) << " ms" << std::endl;
        std::cout << "Vector select_kv avg time: " << (vec_total_ms / repeats) << " ms" << std::endl;

        // ========== Vector 测试 ==========
        // double vec_total_ms = 0.0;
        // std::vector<std::vector<std::vector<float>>> sel_k, sel_v;
        // for (int r = 0; r < repeats; ++r) {
        //     auto start = std::chrono::high_resolution_clock::now();

        //     select_kv_vector(prefetch_idx_vec, k_cache_vec, v_cache_vec, sel_k, sel_v);

        //     auto end = std::chrono::high_resolution_clock::now();
        //     vec_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        // }
        // std::cout << "Vector select_kv avg time: " << (vec_total_ms / repeats) << " ms" << std::endl;
    }

    return 0;
}


///////////////////////////////////////////// vector 测试
// #include <iostream>
// #include <vector>
// #include <tuple>
// #include <chrono>
// #include <random>
// #include <iomanip>

// // ---------------- 被测试函数 ----------------
// std::tuple<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<std::vector<float>>>>
// select_kv_vector_v2(
//     const std::vector<std::vector<int>>& prefetch_idx,
//     const std::vector<std::vector<std::vector<float>>>& k_cache,
//     const std::vector<std::vector<std::vector<float>>>& v_cache,
//     const std::vector<int>& head_group
// ) {
//     int max_unhit = prefetch_idx.size();
//     int num_heads_in_class = head_group.size();
//     int d = k_cache[0][0].size();
//     int n = k_cache.size();

//     std::vector<std::vector<std::vector<float>>> selected_k(max_unhit,
//         std::vector<std::vector<float>>(num_heads_in_class, std::vector<float>(d, 0.0f)));
//     std::vector<std::vector<std::vector<float>>> selected_v = selected_k;

//     for (int i = 0; i < max_unhit; ++i) {
//         for (int j = 0; j < num_heads_in_class; ++j) {
//             int token_idx = prefetch_idx[i][j];
//             int head_idx = head_group[j];

//             if (token_idx < 0 || token_idx >= n) {
//                 continue;
//             }

//             selected_k[i][j] = k_cache[token_idx][head_idx];
//             selected_v[i][j] = v_cache[token_idx][head_idx];
//         }
//     }

//     return {selected_k, selected_v};
// }

// std::tuple<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<std::vector<float>>>>
// select_kv_vector_v2_optimized(
//     const std::vector<std::vector<int>>& prefetch_idx,
//     const std::vector<std::vector<std::vector<float>>>& k_cache,
//     const std::vector<std::vector<std::vector<float>>>& v_cache,
//     const std::vector<int>& head_group
// ) {
//     int max_unhit = prefetch_idx.size();
//     int num_heads_in_class = head_group.size();
//     int d = k_cache[0][0].size();
//     int n = k_cache.size();

//     std::vector<std::vector<std::vector<float>>> selected_k(
//         max_unhit, std::vector<std::vector<float>>(num_heads_in_class, std::vector<float>(d)));

//     std::vector<std::vector<std::vector<float>>> selected_v = selected_k;

//     for (int i = 0; i < max_unhit; ++i) {
//         for (int j = 0; j < num_heads_in_class; ++j) {
//             int token_idx = prefetch_idx[i][j];

//             int head_idx = head_group[j];
//             selected_k[i][j] = k_cache[token_idx][head_idx];
//             selected_v[i][j] = v_cache[token_idx][head_idx];
//         }
//     }

//     // #pragma omp parallel for collapse(2)
//     // for (int i = 0; i < max_unhit; ++i) {
//     //     for (int h = 0; h < num_heads_in_class; ++h) {
//     //         int token_idx = prefetch_idx[i][h];
//     //         // int head_idx = head_group[h];
//     //         int head_idx = h;

//     //         // selected_k[i][head_idx] = k_cache[token_idx][head_idx];
//     //         // selected_v[i][head_idx] = v_cache[token_idx][head_idx];

//     //         for (int j = 0; j < d; ++j) {
//     //             selected_k[i][head_idx][j] = k_cache[token_idx][head_idx][j];
//     //             selected_v[i][head_idx][j] = v_cache[token_idx][head_idx][j];
//     //         }
//     //     }
//     // }

//     return {selected_k, selected_v};
// }

// // ---------------- 测试函数 ----------------
// void run_test(int max_unhit, int n, int bh_total, int d, int num_heads_in_class, int repeats,
//               const std::vector<std::vector<std::vector<float>>>& k_cache,
//               const std::vector<std::vector<std::vector<float>>>& v_cache) {

//     std::mt19937 rng(42);
//     std::uniform_int_distribution<int> dist_token(0, n - 1);

//     std::vector<std::vector<int>> prefetch_idx(max_unhit, std::vector<int>(num_heads_in_class));
//     for (int i = 0; i < max_unhit; ++i)
//         for (int j = 0; j < num_heads_in_class; ++j)
//             prefetch_idx[i][j] = dist_token(rng);

//     std::vector<int> head_group(num_heads_in_class);
//     for (int j = 0; j < num_heads_in_class; ++j)
//         head_group[j] = j;

//     double total_ms_orig = 0.0;
//     double total_ms_opt = 0.0;
    
//     for (int t = 0; t < 10; t ++) {
//         select_kv_vector_v2(prefetch_idx, k_cache, v_cache, head_group);
//     }

//     for (int r = 0; r < repeats; ++r) {
//         auto start1 = std::chrono::high_resolution_clock::now();
//         auto [k1, v1] = select_kv_vector_v2(prefetch_idx, k_cache, v_cache, head_group);
//         auto end1 = std::chrono::high_resolution_clock::now();
//         total_ms_orig += std::chrono::duration<double, std::milli>(end1 - start1).count();

//         auto start2 = std::chrono::high_resolution_clock::now();
//         auto [k2, v2] = select_kv_vector_v2_optimized(prefetch_idx, k_cache, v_cache, head_group);
//         auto end2 = std::chrono::high_resolution_clock::now();
//         total_ms_opt += std::chrono::duration<double, std::milli>(end2 - start2).count();
//     }

//     std::cout << std::setw(10) << max_unhit
//               << std::setw(20) << total_ms_orig / repeats
//               << std::setw(25) << total_ms_opt / repeats
//               << std::setw(15) << (total_ms_orig / total_ms_opt)
//               << "\n";
// }

// // ---------------- 主函数 ----------------
// int main() {
//     const int n = 2048;
//     const int bh_total = 32;
//     const int d = 128;
//     const int num_heads_in_class = 16;
//     const int repeats = 100;

//     std::mt19937 rng(42);
//     std::uniform_real_distribution<float> dist_float(0.0f, 1.0f);

//     std::vector<std::vector<std::vector<float>>> k_cache(n,
//         std::vector<std::vector<float>>(bh_total, std::vector<float>(d)));
//     std::vector<std::vector<std::vector<float>>> v_cache = k_cache;

//     for (int i = 0; i < n; ++i)
//         for (int j = 0; j < bh_total; ++j)
//             for (int k = 0; k < d; ++k)
//                 k_cache[i][j][k] = v_cache[i][j][k] = dist_float(rng);

//     std::vector<int> test_unhits = {128, 256, 512, 1024, 2048};

//     std::cout << std::setw(10) << "max_unhit"
//               << std::setw(20) << "Original Time (ms)"
//               << std::setw(25) << "Optimized Time (ms)"
//               << std::setw(15) << "Speedup"
//               << "\n";
//     std::cout << std::string(70, '-') << "\n";

//     for (int max_unhit : test_unhits) {
//         run_test(max_unhit, n, bh_total, d, num_heads_in_class, repeats, k_cache, v_cache);
//     }

//     return 0;
// }



////////////////////////////// pad unhit cost
// #include <iostream>
// #include <vector>
// #include <random>
// #include <chrono>
// #include <numeric>
// #include <omp.h>

// // 原始串行函数
// std::vector<std::vector<int>> pad_and_convert_unhits_vector(
//     const std::vector<std::vector<int>>& pure_unhit_list)
// {
//     if (pure_unhit_list.empty()) {
//         return {{0}};
//     }

//     size_t batch_size = pure_unhit_list.size();
//     size_t max_unhit_len = 2;
//     bool all_empty = true;

//     for (const auto& unhit : pure_unhit_list) {
//         if (!unhit.empty()) {
//             all_empty = false;
//             max_unhit_len = std::max(max_unhit_len, unhit.size());
//         }
//     }

//     if (all_empty) {
//         max_unhit_len = 1;
//     }

//     std::vector<std::vector<int>> result(max_unhit_len, std::vector<int>(batch_size, 0));

//     for (size_t b = 0; b < batch_size; ++b) {
//         const auto& unhit = pure_unhit_list[b];
//         for (size_t t = 0; t < unhit.size(); ++t) {
//             result[t][b] = unhit[t];
//         }
//     }

//     return result;
// }

// // 🔥 OpenMP并行版本
// std::vector<std::vector<int>> pad_and_convert_unhits_vector_v2(
//     const std::vector<std::vector<int>>& pure_unhit_list)
// {
//     if (pure_unhit_list.empty()) {
//         return {{0}};
//     }

//     size_t batch_size = pure_unhit_list.size();
//     size_t max_unhit_len = 2;
//     bool all_empty = true;

//     // 串行计算 max_unhit_len（避免 omp 临界区）
//     for (const auto& unhit : pure_unhit_list) {
//         if (!unhit.empty()) {
//             all_empty = false;
//             max_unhit_len = std::max(max_unhit_len, unhit.size());
//         }
//     }

//     if (all_empty) {
//         max_unhit_len = 1;
//     }

//     // 初始化结果：[max_unhit_len][batch_size]
//     std::vector<std::vector<int>> result(max_unhit_len, std::vector<int>(batch_size, 0));

//     // 仅并行填充部分
//     #pragma omp parallel for
//     for (int b = 0; b < static_cast<int>(batch_size); ++b) {
//         const auto& unhit = pure_unhit_list[b];
//         for (size_t t = 0; t < unhit.size(); ++t) {
//             result[t][b] = unhit[t];
//         }
//     }

//     return result;
// }



// // 随机数据生成器
// std::vector<std::vector<int>> generate_random_unhit_list(size_t batch_size, size_t max_len, int value_range = 1000) {
//     std::vector<std::vector<int>> result(batch_size);
//     std::mt19937 rng(42);
//     std::uniform_int_distribution<size_t> len_dist(0, max_len);
//     std::uniform_int_distribution<int> value_dist(1, value_range);

//     for (size_t i = 0; i < batch_size; ++i) {
//         size_t len = len_dist(rng);
//         result[i].resize(len);
//         for (size_t j = 0; j < len; ++j) {
//             result[i][j] = value_dist(rng);
//         }
//     }

//     return result;
// }

// // 性能测试函数
// template <typename Func>
// double benchmark(Func f, const std::vector<std::vector<int>>& input, int repeat = 100) {
//     std::vector<double> times;
//     times.reserve(repeat);

//     for (int i = 0; i < repeat; ++i) {
//         auto start = std::chrono::high_resolution_clock::now();
//         auto output = f(input);
//         auto end = std::chrono::high_resolution_clock::now();
//         double ms = std::chrono::duration<double, std::milli>(end - start).count();
//         times.push_back(ms);
//     }

//     double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
//     return avg;
// }

// int main() {
//     size_t batch_size = 4096;
//     size_t max_unhit_len = 128;
//     int repeat = 100;

//     auto input = generate_random_unhit_list(batch_size, max_unhit_len);

//     std::cout << "Testing original version...\n";
//     double t1 = benchmark(pad_and_convert_unhits_vector, input, repeat);
//     std::cout << "Average time (original): " << t1 << " ms\n\n";

//     std::cout << "Testing OpenMP version...\n";
//     double t2 = benchmark(pad_and_convert_unhits_vector_v2, input, repeat);
//     std::cout << "Average time (OpenMP): " << t2 << " ms\n";

//     return 0;
// }

////////////////////////////// pad unhit cost