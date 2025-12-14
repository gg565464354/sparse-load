// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <chrono>

// // 分组信息结构体
// struct GroupInfo {
//     std::vector<int64_t> head_indices;  // 组内包含的头索引
//     int max_kv_len;                     // 组内最大KV长度
// };

// // 标准注意力计算
// torch::Tensor standard_attention(
//     const torch::Tensor& query,
//     const torch::Tensor& key,
//     const torch::Tensor& value,
//     float scale_factor,
//     const torch::Tensor& mask = {}) 
// {
//     auto attn_scores = torch::matmul(query, key.transpose(-1, -2)) * scale_factor;
    
//     if (mask.defined()) {
//         attn_scores.masked_fill_(mask == 0, -1e9);
//     }
    
//     auto attn_weights = torch::softmax(attn_scores, -1);
//     return torch::matmul(attn_weights, value);
// }

// // 分组注意力计算
// torch::Tensor grouped_attention(
//     const torch::Tensor& query,
//     const torch::Tensor& key,
//     const torch::Tensor& value,
//     const std::vector<GroupInfo>& groups,
//     float scale_factor,
//     const torch::Tensor& mask = {}) 
// {
//     auto output = torch::zeros_like(query);
    
//     for (const auto& group : groups) {
//         auto heads = torch::tensor(group.head_indices, torch::kInt64);
//         const int max_len = group.max_kv_len;
        
//         // 提取组内参数
//         auto q = query.index_select(1, heads);
//         auto k = key.index_select(1, heads).narrow(2, 0, max_len);
//         auto v = value.index_select(1, heads).narrow(2, 0, max_len);
        
//         // 修复 mask 维度截取错误：原来是 narrow(2, ...)，应该是 narrow(3, ...)
//         auto m = mask.defined() ? mask.index_select(1, heads).narrow(3, 0, max_len) : torch::Tensor();
        
//         // 计算组内注意力
//         auto attn = standard_attention(q, k, v, scale_factor, m);
//         output.index_copy_(1, heads, attn);
//     }
    
//     return output;
// }

// // 生成测试数据
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<GroupInfo>> 
// generate_test_data(int batch=2, int num_heads=6, int seq_len=8, int dim=64) 
// {
//     std::vector<GroupInfo> groups = {
//         {{0, 1, 2}, 5},
//         {{3, 4, 5}, 3}
//     };
    
//     auto query = torch::randn({batch, num_heads, seq_len, dim});
//     auto key = torch::randn({batch, num_heads, 8, dim});
//     auto value = torch::randn({batch, num_heads, 8, dim});
    
//     for (auto& group : groups) {
//         for (int head : group.head_indices) {
//             key.slice(2, group.max_kv_len, 8).select(1, head).fill_(0);
//             value.slice(2, group.max_kv_len, 8).select(1, head).fill_(0);
//         }
//     }
    
//     return {query, key, value, groups};
// }

// // 验证测试
// void test_attention_equivalence() {
//     auto [query, key, value, groups] = generate_test_data();
//     const float scale = 1.0 / sqrt(query.size(3));
    
//     auto mask = torch::ones({query.size(0), query.size(1), query.size(2), key.size(2)});
//     for (const auto& group : groups) {
//         for (int head : group.head_indices) {
//             mask.slice(3, group.max_kv_len, key.size(2)).select(1, head).fill_(0);
//         }
//     }
    
//     auto start = std::chrono::high_resolution_clock::now();
//     auto output_std = standard_attention(query, key, value, scale, mask);
//     auto time_std = std::chrono::duration_cast<std::chrono::microseconds>(
//         std::chrono::high_resolution_clock::now() - start).count();
    
//     start = std::chrono::high_resolution_clock::now();
//     auto output_grp = grouped_attention(query, key, value, groups, scale, mask);
//     auto time_grp = std::chrono::duration_cast<std::chrono::microseconds>(
//         std::chrono::high_resolution_clock::now() - start).count();
    
//     auto diff = (output_std - output_grp).abs().max().item<float>();
//     std::cout << "最大差异: " << diff << "\n";
//     assert(diff < 1e-5);
    
//     std::cout << "=== 性能对比 ==="
//               << "\n标准注意力: " << time_std << " μs"
//               << "\n分组注意力: " << time_grp << " μs" 
//               << "\n加速比: " << static_cast<float>(time_std)/time_grp << "x\n";
// }

// int main() {
//     torch::manual_seed(42);
//     torch::cuda::manual_seed_all(42);
//     torch::globalContext().setDeterministicCuDNN(true);
    
//     test_attention_equivalence();
    
//     return 0;
// }
////////////////////////////////////////////////

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>

constexpr int NUM_RUNS = 10;

struct GroupInfo {
    std::vector<int64_t> head_indices;
    int max_kv_len;
};

torch::Tensor standard_attention(const torch::Tensor& query,
                                 const torch::Tensor& key,
                                 const torch::Tensor& value,
                                 float scale_factor,
                                 const torch::Tensor& mask = {}) {
    auto attn_scores = torch::matmul(query, key.transpose(-1, -2)) * scale_factor;
    if (mask.defined()) {
        attn_scores.masked_fill_(mask == 0, -1e9);
    }
    auto attn_weights = torch::softmax(attn_scores, -1);
    return torch::matmul(attn_weights, value);
}

// A group attention realization for grouped data
// torch::Tensor group_attention_core(
//     std::vector<torch::Tensor> group_q,
//     std::vector<torch::Tensor> group_k,
//     std::vector<torch::Tensor> group_v,
//     std::vector<torch::Tensor> group_heads,
//     torch::Tensor output
// ) {
//     std::cout << "group q shape" << group_q[0].sizes() << std::endl;
//     for (int id = 0; id < group_heads.size(); id ++) {
//         auto attn_out = torch::scaled_dot_product_attention(group_q[id], group_k[id], group_v[id], c10::nullopt, true); // 不用 mask
//         output.index_copy_(1, group_heads[id], attn_out);  // 按 head 写回原位
//     }

//     return output;
// }

torch::Tensor group_attention_core(
    const std::vector<torch::Tensor>& group_q,        // [B, h_i, Lq, D]
    const std::vector<torch::Tensor>& group_k,        // [B, h_i, Lk, D]
    const std::vector<torch::Tensor>& group_v,        // [B, h_i, Lk, D]
    const std::vector<torch::Tensor>& group_heads,    // 每组是 [h_i]，表示当前 group 中的 head 索引
    torch::Tensor output                              // [B, total_heads, Lq, D]
) {
    for (size_t i = 0; i < group_heads.size(); ++i) {
        auto& q = group_q[i]; // [B, h_i, Lq, D]
        auto& k = group_k[i]; // [B, h_i, Lk, D]
        auto& v = group_v[i]; // [B, h_i, Lk, D]
        auto& heads = group_heads[i]; // [h_i]

        // Flash Attention expects [B * H, L, D]
        int B = q.size(0);
        int H = q.size(1);
        int Lq = q.size(2);
        int D = q.size(3);

        auto q_reshape = q.reshape({B * H, Lq, D});
        auto k_reshape = k.reshape({B * H, k.size(2), D});
        auto v_reshape = v.reshape({B * H, v.size(2), D});

        // 执行 Flash Attention
        auto attn_out = torch::scaled_dot_product_attention(
            q_reshape, k_reshape, v_reshape, c10::nullopt, /*is_causal=*/true
        ); // [B * H, Lq, D]

        // Reshape 回原始 [B, H, Lq, D]
        auto attn_out_reshaped = attn_out.view({B, H, Lq, D});

        // 将输出写回 output 的对应 head index 位置
        for (int h = 0; h < heads.size(0); ++h) {
            int head_idx = heads[h].item<int>(); // 写入目标 head 索引
            // 写入每个 batch 中对应位置
            output.select(1, head_idx).copy_(attn_out_reshaped.select(1, h));
        }
    }

    return output;
}



std::tuple<torch::Tensor, long> groupwise_flash_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::vector<GroupInfo>& groups) {

    auto output = torch::zeros_like(query);

    auto group_k = std::vector<torch::Tensor>{};
    auto group_v = std::vector<torch::Tensor>{};
    auto group_q = std::vector<torch::Tensor>{};
    auto group_heads = std::vector<torch::Tensor>{};
    

    for (const auto& group : groups) {
        auto heads = torch::tensor(group.head_indices, torch::TensorOptions().dtype(torch::kLong).device(query.device()));
        const int max_kv_len = group.max_kv_len;

        auto q = query.index_select(1, heads);  // [B, H_g, Q, D]
        auto k = key.index_select(1, heads).narrow(2, 0, max_kv_len);  // [B, H_g, K, D]
        auto v = value.index_select(1, heads).narrow(2, 0, max_kv_len);  // [B, H_g, K, D]

        group_k.push_back(k);
        group_v.push_back(v);
        group_q.push_back(q);
        group_heads.push_back(heads);
    }

    // 开始计时
    torch::cuda::synchronize();
    auto start = std::chrono::high_resolution_clock::now();

    // 开始执行
    // int id = 0;
    // for (const auto& group:groups) {

    //     auto attn_out = torch::scaled_dot_product_attention(group_q[id], group_k[id], group_v[id], c10::nullopt, false); // 不用 mask
    //     output.index_copy_(1, group_heads[id], attn_out);  // 按 head 写回原位

    //     id += 1;
    // }
    group_attention_core(group_q, group_k, group_v, group_heads, output);

    // 结束计时并计算通信时间
    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    long communication_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();


    return {output, communication_time};
}


torch::Tensor grouped_attention(const torch::Tensor& query,
                                const torch::Tensor& key,
                                const torch::Tensor& value,
                                const std::vector<GroupInfo>& groups,
                                float scale_factor,
                                const torch::Tensor& mask = {}) {
    auto output = torch::zeros_like(query);

    for (const auto& group : groups) {
        auto heads = torch::tensor(group.head_indices, torch::TensorOptions().dtype(torch::kLong).device(query.device()));
        const int max_len = group.max_kv_len;

        auto q = query.index_select(1, heads);
        auto k = key.index_select(1, heads).narrow(2, 0, max_len);
        auto v = value.index_select(1, heads).narrow(2, 0, max_len);
        auto m = mask.defined() ? mask.index_select(1, heads).narrow(3, 0, max_len) : torch::Tensor();

        auto attn = standard_attention(q, k, v, scale_factor, m);
        output.index_copy_(1, heads, attn);
    }

    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<GroupInfo>>
generate_test_data(int batch=2, int num_heads=6, int seq_len=8, int dim=64, torch::Device device=torch::kCUDA) {
    std::vector<GroupInfo> groups = {
        {{}, seq_len},
        {{}, int(3*seq_len/4)}
    };
    int half_num_head = int(num_heads/2);
    for (int i = 0; i < half_num_head; i ++) {
        groups[0].head_indices.push_back(i);
        groups[1].head_indices.push_back(i + half_num_head);
        // groups[0].head_indices.push_back(2*i);
        // groups[1].head_indices.push_back(2*i + 1);
    }

    auto query = torch::randn({batch, num_heads, 1, dim}, device);
    auto key = torch::randn({batch, num_heads, seq_len, dim}, device);
    auto value = torch::randn({batch, num_heads, seq_len, dim}, device);

    for (auto& group : groups) {
        if (group.max_kv_len < seq_len) {
            for (int head : group.head_indices) {
                key.slice(2, group.max_kv_len, seq_len).select(1, head).fill_(0);
                value.slice(2, group.max_kv_len, seq_len).select(1, head).fill_(0);
            }
        }
    }

    return {query, key, value, groups};
}

torch::Tensor build_mask(int batch, int heads, int q_len, int kv_len,
                         const std::vector<GroupInfo>& groups,
                         torch::Device device) {
    auto mask = torch::ones({batch, heads, q_len, kv_len}, torch::kFloat).to(device);
    for (const auto& group : groups) {
        for (int head : group.head_indices) {
            mask.slice(3, group.max_kv_len, kv_len).select(1, head).fill_(0);
        }
    }
    return mask;
}

template <typename F>
int64_t benchmark(F func, int warmup=2, int runs=NUM_RUNS) {
    for (int i = 0; i < warmup; ++i) func();
    torch::cuda::synchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) func();
    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / runs;
}



void test_attention_equivalence() {
    torch::manual_seed(42);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    auto [query, key, value, groups] = generate_test_data(1, 64, 3000, 128, device);
    const float scale = 1.0 / sqrt(query.size(3));
    auto mask = build_mask(query.size(0), query.size(1), query.size(2), key.size(2), groups, device);

    // 转换 mask 以兼容 scaled_dot_product_attention（bool 类型）
    auto bool_mask = mask.to(torch::kBool);

    // 进行数据类型转换
    auto query_fp16 = query.to(torch::kCUDA, torch::kHalf);
    auto key_fp16 = key.to(torch::kCUDA, torch::kHalf);
    auto value_fp16 = value.to(torch::kCUDA, torch::kHalf);

    // 预热 + 多轮测试
    torch::Tensor out_flash, out_grouped;

    int64_t flash_time = benchmark([&]() {
        out_flash = torch::scaled_dot_product_attention(query_fp16, key_fp16, value_fp16, bool_mask, true);
    });

    std::cout << "standard success" << std::endl;

    
    long grouped_time;
    for (int i = 0; i < NUM_RUNS; i ++) {
        auto result = groupwise_flash_attention(query, key, value, groups);
        out_grouped = std::get<0>(result);  // 提取第一个元素 (torch::Tensor)
        grouped_time = std::get<1>(result); // 提取第二个元素 (long int)
    }
    grouped_time /= NUM_RUNS;

    // int64_t grouped_time = benchmark([&]() {
    //     out_grouped = torch::scaled_dot_product_attention(query_fp16, key_fp16, value_fp16, c10::nullopt, false);
    // });

    auto diff = (out_flash - out_grouped).abs().max().item<float>();
    std::cout << "最大差异: " << diff << "\n";
    // TORCH_CHECK(diff < 1e-4, "输出结果不一致");

    std::cout << "dtype: " << out_flash.dtype() << "\n";
    std::cout << "=== 多轮平均性能对比 ==="
              << "\nFlash/SDPA Attention 平均耗时: " << flash_time << " us"
              << "\nGrouped Attention 平均耗时:    " << grouped_time << " us"
              << "\n加速比 (Grouped vs Flash):    " << static_cast<float>(flash_time)/grouped_time << "x\n";
}

int main() {
    test_attention_equivalence();
    return 0;
}