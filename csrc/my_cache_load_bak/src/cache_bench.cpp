#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "cpu_cache.h"  // 确保此头文件包含 CPUCache 的定义
#include <random>
#include <ctime>

void benchmark_get_unhit(int token_num, int bh, int cache_size) {
    // 生成随机的 prefetch_idx 张量，值范围在 [0, cache_size + 100]
    auto prefetch_idx = torch::randint(0, cache_size + 100, {token_num, 1, bh}, torch::kInt);
    
    // 初始化 CPUCache
    std::vector<int64_t> cache_shape = {cache_size, bh, 64}; // 64 代表 head_dim，
    CPUCache cache(bh, prefetch_idx, cache_shape);
    
    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();
    auto unhit_tokens = cache.get_unhit(prefetch_idx);
    // 同步 GPU，确保所有操作已完成
    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end - start;
    std::cout << "Test Case (tokens: " << token_num << ", bh: " << bh << ", cache: " << cache_size << ") -> Time: " << duration.count() << " s" << std::endl;
    // std::cout << "padded_tensor shape(" << padded_tensor.size(0) << ", 1," << padded_tensor.size(1) << ")" << std::endl;
}


// 随机生成测试样例
std::vector<std::vector<int>> generate_random_unhits(int bh, int min_len, int max_len, int min_val, int max_val) {
    std::vector<std::vector<int>> unhit_list;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> len_dist(min_len, max_len);
    std::uniform_int_distribution<> val_dist(min_val, max_val);

    for (int i = 0; i < bh; ++i) {
        int len = len_dist(gen);  // 随机生成长度
        std::vector<int> unhit(len);
        for (int j = 0; j < len; ++j) {
            unhit[j] = val_dist(gen);  // 随机生成整数值
        }
        unhit_list.push_back(unhit);
    }

    return unhit_list;
}


int benchmark_pad_unhit(int bh, int min_len, int max_len, int min_val, int max_val) {
    // 生成随机的 unhit 数据
    std::vector<std::vector<int>> random_unhits = generate_random_unhits(bh, min_len, max_len, min_val, max_val);

    // 打印生成的随机数据
    // std::cout << "Generated random unhits:" << std::endl;
    // for (const auto& unhit : random_unhits) {
    //     for (int val : unhit) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // 创建 CPUCache 实例
    auto prefetch_idx = torch::randint(0, 100, {100, 1, bh}, torch::kInt);
    std::vector<int64_t> cache_shape = {100, bh, 64}; 
    CPUCache cpu_cache(bh, prefetch_idx, cache_shape);

    // 调用 pad_and_convert_unhits 方法进行测试
    auto start = std::chrono::high_resolution_clock::now();
    at::Tensor padded_tensor = cpu_cache.pad_and_convert_unhits(random_unhits);
    // 同步 GPU，确保所有操作已完成
    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // 打印填充后的 Tensor
    std::chrono::duration<double> duration = end - start;
    std::cout << "Test Case (bh: " << bh << ", min_len: " << min_len << ", max_len: " << max_len << ") -> Time: " << duration.count() << " s" << std::endl;

    return 0;
}

double calculate_average(const std::vector<double>& data) {
    // 检查输入是否为空
    if (data.empty()) {
        throw std::runtime_error("Error: Input vector is empty.");
    }

    // 使用 std::accumulate 计算总和
    double sum = std::accumulate(data.begin(), data.end(), 0.0);

    // 计算平均值
    double average = sum / data.size();

    return average;
}

int calculate_total_tokens(const std::vector<std::vector<int>>& pure_unhit_list) {
    // 初始化总 token 数量为 0
    int total_tokens = 0;

    // 遍历 pure_unhit_list 中的每个子向量
    for (const auto& unhit : pure_unhit_list) {
        // 累加当前子向量的大小
        total_tokens += unhit.size();
    }

    return total_tokens;
}


int bench_load_cache(int token_num, int cache_num, int bh, int test_cases) {
    // 创建示例数据
    std::vector<double> cache_cost = {};
    std::vector<double> direct_cost = {};
    std::vector<double> hit_rate_list = {};

    for (int i = 0; i < test_cases; i ++) {
        // int bh = 32;
        double same_rate = 0.7;
        int same_size = static_cast<int>(same_rate * cache_num);
        
        auto prefetch_idx = torch::randint(0, token_num, {cache_num, 1, bh}, torch::kInt);
        auto cache_idx = torch::randint(0, token_num, {cache_num, 1, bh}, torch::kInt);

        // cache_idx[:same_size, :, :] = prefetch_idx[:same_size, :, :]
        cache_idx.slice(/*dim=*/0, /*start=*/0, /*end=*/same_size) = 
            prefetch_idx.slice(/*dim=*/0, /*start=*/0, /*end=*/same_size);
        
        std::vector<int64_t> cache_shape = {cache_num, bh, 128}; 
        std::vector<int64_t> full_shape = {token_num, bh, 128}; 
        
        auto keys = torch::empty(full_shape, torch::dtype(torch::kFloat32));
        auto values = torch::empty(full_shape, torch::dtype(torch::kFloat32));

        // 初始化 CPUCache 对象
        CPUCache cache(bh, cache_idx, cache_shape);

        // 单独测试select_kv
        // auto [un_cached_k, un_cached_v] = cache.select_kv(prefetch_idx, keys, values);
        // std::cout << "prefetch_idx shape: " << prefetch_idx.sizes() << std::endl;
        // std::cout << "un_cached_k shape: " << un_cached_k.sizes() << std::endl;

        // 调用 load_with_cached 方法
        // std::cout << "#### direct load" << std::endl;
        // auto tmp_tim = cache.direct_load(prefetch_idx, keys, values);

        // std::cout << "#### load with cache" << std::endl;
        auto [pure_unhit_list, cache_time] = cache.load_with_cached_test(prefetch_idx, keys, values);
        
        // std::cout << "#### direct load" << std::endl;
        auto direct_time = cache.direct_load_test(prefetch_idx, keys, values);
        
        if (i != 0) {  
            cache_cost.push_back(cache_time);
            direct_cost.push_back(direct_time);

            auto unhit_num = calculate_total_tokens(pure_unhit_list);
            auto all_num = cache_num * bh;
            double hit_rate = double(all_num - unhit_num)/all_num;
            hit_rate_list.push_back(hit_rate);
            // std::cout << "all_num = " << all_num << std::endl;
            // std::cout << "unhit_num = " << unhit_num << std::endl;
            // std::cout << "hit_rate = " << (all_num - unhit_num)/all_num << std::endl;
        }

        // std::cout << "load success" << std::endl;

        // 打印结果
        // std::cout << "Pure unhit list size: " << pure_unhit_list.size() << std::endl;
        // std::cout << "Communication time: " << communication_time << " ms" << std::endl;
    }
    auto avg_cache_cost = calculate_average(cache_cost);
    auto avg_direct_cost = calculate_average(direct_cost); 
    auto avg_hit_rate = calculate_average(hit_rate_list); 

    std::cout << "(" << token_num << ", " << cache_num << ", " << bh << ") : avg_direct_cost=" << avg_direct_cost 
        << "\t avg_cache_cost=" << avg_cache_cost << "\t avg_hit_rate=" << avg_hit_rate << std::endl;

    return 0;
}


int bench_load_cache_v2(int token_num, int cache_num, int bh, int test_cases) {
    // 创建示例数据
    std::vector<double> cache_cost = {};
    std::vector<double> direct_cost = {};
    std::vector<double> hit_rate_list = {};

    for (int i = 0; i < test_cases; i ++) {
        // int bh = 32;
        double same_rate = 0.7;
        int same_size = static_cast<int>(same_rate * cache_num);
        
        auto prefetch_idx = torch::randint(0, token_num, {cache_num, 1, bh}, torch::kInt);
        auto cache_idx = torch::randint(0, token_num, {cache_num, 1, bh}, torch::kInt);

        // cache_idx[:same_size, :, :] = prefetch_idx[:same_size, :, :]
        cache_idx.slice(/*dim=*/0, /*start=*/0, /*end=*/same_size) = 
            prefetch_idx.slice(/*dim=*/0, /*start=*/0, /*end=*/same_size);
        
        std::vector<int64_t> cache_shape = {cache_num, bh, 128}; 
        std::vector<int64_t> full_shape = {token_num, bh, 128}; 
        
        auto keys = torch::empty(full_shape, torch::dtype(torch::kFloat32));
        auto values = torch::empty(full_shape, torch::dtype(torch::kFloat32));

        // 初始化 CPUCache 对象
        CPUCache cache(bh, cache_idx, cache_shape);

        // 初始化分类
        // 将前30%的head设为高命中类别（0），中间50%为中命中（1），后20%为低命中（2）
        std::vector<int> head_classes(bh);
        for (int i = 0; i < bh; ++i) {
            if (i < 0.3 * bh) head_classes[i] = 0;
            else if (i < 0.8 * bh) head_classes[i] = 1;
            else head_classes[i] = 2;
        }

        cache.initialize_head_classes(head_classes);

        // 单独测试select_kv
        // auto [un_cached_k, un_cached_v] = cache.select_kv(prefetch_idx, keys, values);
        // std::cout << "prefetch_idx shape: " << prefetch_idx.sizes() << std::endl;
        // std::cout << "un_cached_k shape: " << un_cached_k.sizes() << std::endl;

        // 调用 load_with_cached 方法
        // std::cout << "#### direct load" << std::endl;
        // auto tmp_tim = cache.direct_load(prefetch_idx, keys, values);

        // std::cout << "#### load with cache" << std::endl;
        auto [final_k, final_v, pure_unhit_list] = cache.load_with_cached_v2(prefetch_idx, keys, values);
        
        // std::cout << "#### direct load" << std::endl;
        // auto direct_time = cache.direct_load_test(prefetch_idx, keys, values);
        
        // if (i != 0) {  
        //     cache_cost.push_back(cache_time);
        //     direct_cost.push_back(direct_time);

        //     auto unhit_num = calculate_total_tokens(pure_unhit_list);
        //     auto all_num = cache_num * bh;
        //     double hit_rate = double(all_num - unhit_num)/all_num;
        //     hit_rate_list.push_back(hit_rate);
        //     // std::cout << "all_num = " << all_num << std::endl;
        //     // std::cout << "unhit_num = " << unhit_num << std::endl;
        //     // std::cout << "hit_rate = " << (all_num - unhit_num)/all_num << std::endl;
        // }

        // std::cout << "load success" << std::endl;

        // 打印结果
        // std::cout << "Pure unhit list size: " << pure_unhit_list.size() << std::endl;
        // std::cout << "Communication time: " << communication_time << " ms" << std::endl;
    }
    auto avg_cache_cost = calculate_average(cache_cost);
    auto avg_direct_cost = calculate_average(direct_cost); 
    auto avg_hit_rate = calculate_average(hit_rate_list); 

    std::cout << "(" << token_num << ", " << cache_num << ", " << bh << ") : avg_direct_cost=" << avg_direct_cost 
        << "\t avg_cache_cost=" << avg_cache_cost << "\t avg_hit_rate=" << avg_hit_rate << std::endl;

    return 0;
}

void test_select_kv_performance() {
    // 初始化参数
    const int n = 1000;     // 总token数
    const int bh_total = 64; // 总头数
    const int d = 128;       // 特征维度
    const int test_runs = 100;

    // 创建缓存实例
    CPUCache cache;

    // 生成测试数据 ------------------------------------------------------------
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    
    // 原始方法测试数据（统一填充）
    auto original_idx = torch::cat({
        torch::randint(0, n, {50, bh_total}),  // 50个有效索引
        torch::full({50, bh_total}, -1)        // 50个填充
    }, 0).to(torch::kInt);                     // Shape: (100, 64)

    // 改进方法测试数据（分2组）
    const std::vector<std::vector<int>> head_groups = {
        {0, 1, 2, 3},       // 第1组：4个头（假设高未命中）
        {4, 5, 6, 7, 8, 9}  // 第2组：6个头（假设低未命中）
    };
    
    std::vector<torch::Tensor> grouped_indices;
    for (const auto& group : head_groups) {
        const int group_size = group.size();
        auto idx = torch::cat({
            torch::randint(0, n, {30, group_size}), // 30个有效
            torch::full({30, group_size}, -1)       // 30个填充
        }, 0).to(torch::kInt);                      // Shape: (60, group_size)
        grouped_indices.push_back(idx);
    }

    // 创建KV缓存
    auto k_cache = torch::randn({n, bh_total, d}, options);
    auto v_cache = torch::randn({n, bh_total, d}, options);

    // 原始版本测试 -----------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_runs; ++i) {
        auto [k, v] = cache.select_kv(original_idx, k_cache, v_cache);
        if (k.device().is_cuda()) torch::cuda::synchronize();
    }
    auto duration_original = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    // 改进版本测试 -----------------------------------------------------------
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_runs; ++i) {
        std::vector<torch::Tensor> all_k, all_v;
        for (size_t g = 0; g < head_groups.size(); ++g) {
            auto [k, v] = cache.select_kv_v2(
                grouped_indices[g], 
                k_cache,
                v_cache,
                head_groups[g]
            );
            all_k.push_back(k);
            all_v.push_back(v);
        }
        
        // 合并结果（按头顺序）
        auto final_k = torch::cat(all_k, 1);
        auto final_v = torch::cat(all_v, 1);
        if (final_k.device().is_cuda()) torch::cuda::synchronize();
    }
    auto duration_v2 = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    // 结果验证（可选）-------------------------------------------------------
    auto [k_orig, v_orig] = cache.select_kv(original_idx, k_cache, v_cache);
    std::vector<torch::Tensor> v2_results;
    for (size_t g = 0; g < head_groups.size(); ++g) {
        auto [k, v] = cache.select_kv_v2(grouped_indices[g], k_cache, v_cache, head_groups[g]);
        v2_results.push_back(k);
    }
    auto k_v2 = torch::cat(v2_results, 1);

    // 比较有效数据部分
    auto valid_mask = (original_idx != -1);
    assert(torch::allclose(
        k_orig.index({valid_mask}),
        k_v2.index({valid_mask}),
        1e-5, 1e-8
    ));

    // 性能输出 -------------------------------------------------------------
    std::cout << "=== Performance Comparison ===" << std::endl;
    std::cout << "Original select_kv: " << duration_original << " ms\n";
    std::cout << "Improved select_kv_v2: " << duration_v2 << " ms\n";
    std::cout << "Speedup Ratio: " 
              << static_cast<float>(duration_original)/duration_v2 << "x\n";
    std::cout << "Validation Passed: Results are consistent\n";
}


int main_test() {
    // 测试不同参数下的 get_unhit 执行时间
    // std::cout << "######## get unhit cost" << std::endl;
    // std::vector<std::tuple<int, int, int>> test_cases = {
    //     // {128, 8, 1024},
    //     // {256, 16, 2048},
    //     // {512, 32, 4096},
    //     {1024, 64, 8192},
    //     {4096, 64, 8192},
    //     {8192, 64, 8192}
    // };

    // for (const auto& [tokens, bh, cache] : test_cases) {
    //     benchmark_get_unhit(tokens, bh, cache);
    // }

    // std::cout << "######## pad unhit cost" << std::endl;

    // std::vector<std::tuple<int, int, int, int, int>> test_cases2 = {
    //     {64, 512, 1024, 0, 9},
    //     {64, 512, 1024, 0, 9},
    //     {64, 512, 1024, 0, 9},
    //     {64, 1024, 2048, 0, 9},
    //     {64, 2048, 4096, 0, 9},
    // };
    // for (const auto& [bh, min_len, max_len, min_val, max_val] : test_cases2) {
    //     benchmark_pad_unhit(bh, min_len, max_len, min_val, max_val);
    // }

    // // test load cache cost
    // int test_num = 6;
    // std::vector<std::tuple<int, int, int, int>> test_cases3 = {
    //     // {2048, 1024, 64, test_num},
    //     // {4096, 2048, 64, test_num},
    //     // {8192, 4096, 64, test_num},
    //     // {10240, 5120, 64, test_num},
    //     {512, 256, 28, test_num},
    //     {1024, 512, 28, test_num}
    // };
    // for (const auto& [token_num, cache_num, bh, test_num] : test_cases3) {
    //     bench_load_cache_v2(token_num, cache_num, bh, test_num);
    // }
    // bench_load_cache(8192, 4096, 64, 21);

    test_select_kv_performance();
    // std::cout << "test";

    return 0;
}   