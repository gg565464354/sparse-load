#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "cpu_cache.h"  // 确保此头文件包含 CPUCache 的定义
#include <random>
#include <ctime>
#include <cuda_runtime.h>

// #include <torch/csrc/autograd/profiler.h>
// #include <torch/csrc/autograd/profiler_kineto.h>
// #include <fstream>

// void benchmark_get_unhit(int token_num, int bh, int cache_size) {
//     // 生成随机的 prefetch_idx 张量，值范围在 [0, cache_size + 100]
//     auto prefetch_idx = torch::randint(0, cache_size + 100, {token_num, 1, bh}, torch::kInt);
    
//     // 初始化 CPUCache
//     std::vector<int64_t> cache_shape = {cache_size, bh, 64}; // 64 代表 head_dim，
//     CPUCache cache(bh, prefetch_idx, cache_shape);
    
//     // 计时开始
//     auto start = std::chrono::high_resolution_clock::now();
//     auto unhit_tokens = cache.get_unhit(prefetch_idx);
//     // 同步 GPU，确保所有操作已完成
//     torch::cuda::synchronize();
//     auto end = std::chrono::high_resolution_clock::now();
    
//     std::chrono::duration<double> duration = end - start;
//     std::cout << "Test Case (tokens: " << token_num << ", bh: " << bh << ", cache: " << cache_size << ") -> Time: " << duration.count() << " s" << std::endl;
//     // std::cout << "padded_tensor shape(" << padded_tensor.size(0) << ", 1," << padded_tensor.size(1) << ")" << std::endl;
// }


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


// int benchmark_pad_unhit(int bh, int min_len, int max_len, int min_val, int max_val) {
//     // 生成随机的 unhit 数据
//     std::vector<std::vector<int>> random_unhits = generate_random_unhits(bh, min_len, max_len, min_val, max_val);

//     // 打印生成的随机数据
//     // std::cout << "Generated random unhits:" << std::endl;
//     // for (const auto& unhit : random_unhits) {
//     //     for (int val : unhit) {
//     //         std::cout << val << " ";
//     //     }
//     //     std::cout << std::endl;
//     // }

//     // 创建 CPUCache 实例
//     auto prefetch_idx = torch::randint(0, 100, {100, 1, bh}, torch::kInt);
//     std::vector<int64_t> cache_shape = {100, bh, 64}; 
//     CPUCache cpu_cache(bh, prefetch_idx, cache_shape);

//     // 调用 pad_and_convert_unhits 方法进行测试
//     auto start = std::chrono::high_resolution_clock::now();
//     at::Tensor padded_tensor = cpu_cache.pad_and_convert_unhits(random_unhits);
//     // 同步 GPU，确保所有操作已完成
//     torch::cuda::synchronize();
//     auto end = std::chrono::high_resolution_clock::now();

//     // 打印填充后的 Tensor
//     std::chrono::duration<double> duration = end - start;
//     std::cout << "Test Case (bh: " << bh << ", min_len: " << min_len << ", max_len: " << max_len << ") -> Time: " << duration.count() << " s" << std::endl;

//     return 0;
// }

double calculate_average(const std::vector<double>& data) {
    // 检查输入是否为空
    if (data.empty()) {
        throw std::runtime_error("Qin Error: Input vector is empty.");
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

std::vector<std::vector<int>> global_class_group_ids = {
    {27, 44, 52, 56},
    {8, 11, 13, 14, 15, 17, 21, 26, 28, 29, 32, 34, 39, 43, 45, 46, 47, 49, 55, 62},
    {2, 3, 4, 6, 7, 12, 16, 18, 19, 20, 22, 23, 25, 30, 31, 33, 36, 38, 41, 42, 48, 50, 51, 53, 58, 60, 61, 63},
    {0, 1, 5, 9, 10, 24, 35, 37, 40, 54, 57, 59}
};


int bench_load_cache(int token_num, int cache_num, int bh, int test_cases) {
    // 创建示例数据
    std::vector<double> cache_cost = {};
    std::vector<double> cache_cost_2 = {};
    std::vector<double> direct_cost = {};
    std::vector<double> hit_rate_list = {};

    for (int i = 0; i < test_cases; i ++) {
        // int bh = 32;
        double same_rate = 0.4;
        int same_size = static_cast<int>(same_rate * cache_num);
        int same_bh = int(bh/2);
        
        auto prefetch_idx = torch::randint(0, token_num, {cache_num, 1, bh}, torch::kInt);
        auto cache_idx = torch::randint(0, token_num, {cache_num, 1, bh}, torch::kInt);

        // cache_idx[:same_size, :same_bh, :] = prefetch_idx[:same_size, :same_bh, :]
        cache_idx.slice(/*dim=*/0, /*start=*/0, /*end=*/same_size).slice(1, 0, same_bh) = 
            prefetch_idx.slice(/*dim=*/0, /*start=*/0, /*end=*/same_size).slice(1, 0, same_bh);
        
        std::vector<int64_t> cache_shape = {cache_num, bh, 128}; 
        std::vector<int64_t> full_shape = {token_num, bh, 128}; 
        
        auto keys = torch::empty(full_shape, torch::dtype(torch::kFloat16));
        auto values = torch::empty(full_shape, torch::dtype(torch::kFloat16));

        // TODO: 初始化class
        int step_len = int(bh/4);
        std::vector<std::vector<int>> class_group_ids{{}, {}, {}, {}};
        for (int q = 0; q < step_len; q ++) {
            class_group_ids[0].push_back(q);
            class_group_ids[1].push_back(q+step_len);
            class_group_ids[2].push_back(q+2*step_len);
            class_group_ids[3].push_back(q+3*step_len);
        }

        // std::cout << "class_group_ids[0] = " << class_group_ids[0] << std::endl;
        // std::cout << "class_group_ids[1] = " << class_group_ids[1] << std::endl;
        
        // 初始化 CPUCache 对象
        CPUCache cache(bh, cache_idx, cache_shape, global_class_group_ids);
        CPUCache cache2(bh, cache_idx, cache_shape, global_class_group_ids);

        // 调用 load_with_cached 方法
        // std::cout << "#### direct load" << std::endl;
        auto tmp_tim = cache.direct_load(prefetch_idx, keys, values);

        auto [pure_unhit_list, cache_time] = cache.load_with_cached_test(prefetch_idx, keys, values);

        auto [pure_unhit_list_2, cache2_time] = cache2.load_with_cached_test_v2(prefetch_idx, keys, values);        
        
        auto direct_time = cache.direct_load_test(prefetch_idx, keys, values);
        
        if (i != 0) {  
            cache_cost.push_back(cache_time);
            cache_cost_2.push_back(cache2_time);
            direct_cost.push_back(direct_time);

            auto unhit_num = calculate_total_tokens(pure_unhit_list);
            auto all_num = cache_num * bh;
            double hit_rate = double(all_num - unhit_num)/all_num;
            hit_rate_list.push_back(hit_rate);
            // std::cout << "all_num = " << all_num << std::endl;
            // std::cout << "unhit_num = " << unhit_num << std::endl;
            // std::cout << "hit_rate = " << (all_num - unhit_num)/all_num << std::endl;
        }

        // 打印结果
        // std::cout << "Pure unhit list size: " << pure_unhit_list.size() << std::endl;
        // std::cout << "Communication time: " << communication_time << " ms" << std::endl;
    }

    auto avg_cache_cost = calculate_average(cache_cost);
    auto avg_cache_cost_2 = calculate_average(cache_cost_2);
    auto avg_direct_cost = calculate_average(direct_cost); 
    auto avg_hit_rate = calculate_average(hit_rate_list); 

    std::cout << "(" << token_num << ", " << cache_num << ", " << bh << ") : avg_direct_cost=" << avg_direct_cost 
        << "\t avg_cache_cost=" << avg_cache_cost   << "\t avg_cache_cost_2=" << avg_cache_cost_2 
        << "\t avg_hit_rate=" << avg_hit_rate << std::endl;

    return 0;
}


int bench_update_cache(int token_num, int cache_num, int bh, int test_cases) {
    // 创建示例数据

    double same_rate = 0.7;
    int same_size = static_cast<int>(same_rate * cache_num);
    int same_bh = int(bh/2);
    
    auto cache_idx = torch::randint(0, token_num, {cache_num, 1, bh}, torch::kInt);

    std::vector<int64_t> cache_shape = {cache_num, bh, 128}; 
    std::vector<int64_t> full_shape = {token_num, bh, 128}; 
    
    auto keys = torch::empty(full_shape, torch::dtype(torch::kFloat32));
    auto values = torch::empty(full_shape, torch::dtype(torch::kFloat32));

    // TODO: 初始化class
    int step_len = int(bh/4);
    std::vector<std::vector<int>> class_group_ids = global_class_group_ids;

    // 初始化 CPUCache 对象
    CPUCache cache1(bh, cache_idx, cache_shape, class_group_ids);
    CPUCache cache2(bh, cache_idx, cache_shape, class_group_ids);

    
    std::vector<double> update_v1_cost = {};
    std::vector<double> update_v2_cost = {};
    for (int i = 0; i < test_cases; i ++) {
        auto cache_idx = torch::randint(0, token_num, {cache_num, 1, bh}, torch::kInt);

        // torch::cuda::synchronize();
        // auto start1 = std::chrono::high_resolution_clock::now();
        // auto [gpu_k, gpu_v, cpu_k, cpu_v] = cache1.direct_load(cache_idx, keys, values);
        // // cache1.update_cache(cache_idx, cpu_k, cpu_v);
        // torch::cuda::synchronize();
        // auto end1 = std::chrono::high_resolution_clock::now();

        
        torch::cuda::synchronize();
        auto start1 = std::chrono::high_resolution_clock::now();
        auto [cpu_k, cpu_v] = cache1.generate_update_cache(cache_idx, keys, values);
        // cache1.update_cache(cache_idx, cpu_k, cpu_v);
        torch::cuda::synchronize();
        auto end1 = std::chrono::high_resolution_clock::now();


        torch::cuda::synchronize();
        auto start2 = std::chrono::high_resolution_clock::now();
        auto [group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v] = cache2.generate_cache(cache_idx, keys, values);
        cache2.update_cache(cache_idx, group_cpu_k, group_cpu_v);
        torch::cuda::synchronize();
        auto end2 = std::chrono::high_resolution_clock::now();

        update_v1_cost.push_back((std::chrono::duration_cast<std::chrono::milliseconds>(end1-start1)).count());
        update_v2_cost.push_back((std::chrono::duration_cast<std::chrono::milliseconds>(end2-start2)).count());
    }

    auto avg_v1_cost = calculate_average(update_v1_cost);
    auto avg_v2_cost = calculate_average(update_v2_cost); 

    std::cout << "(" << token_num << ", " << cache_num << ", " << bh << ") : without upload =" << avg_v1_cost 
        << "\t avg_v2_cost=" << avg_v2_cost << std::endl;

    return 0;
}



int bench_load_cache_profile(int token_num, int cache_num, int bh, int test_cases=1) {
    // 创建示例数据
    std::vector<double> cache_cost = {};
    std::vector<double> cache_cost_2 = {};
    std::vector<double> direct_cost = {};
    std::vector<double> hit_rate_list = {};

    for (int i = 0; i < test_cases; i ++) {
        // int bh = 32;
        double same_rate = 0.4;
        int same_size = static_cast<int>(same_rate * cache_num);
        int same_bh = int(bh/2);
        
        auto prefetch_idx = torch::randint(0, token_num, {cache_num, 1, bh}, torch::kInt);
        auto cache_idx = torch::randint(0, token_num, {cache_num, 1, bh}, torch::kInt);

        // cache_idx[:same_size, :same_bh, :] = prefetch_idx[:same_size, :same_bh, :]
        cache_idx.slice(/*dim=*/0, /*start=*/0, /*end=*/same_size).slice(1, 0, same_bh) = 
            prefetch_idx.slice(/*dim=*/0, /*start=*/0, /*end=*/same_size).slice(1, 0, same_bh);
        
        std::vector<int64_t> cache_shape = {cache_num, bh, 128}; 
        std::vector<int64_t> full_shape = {token_num, bh, 128}; 
        
        auto keys = torch::empty(full_shape, torch::dtype(torch::kFloat16));
        auto values = torch::empty(full_shape, torch::dtype(torch::kFloat16));

        // TODO: 初始化class
        int step_len = int(bh/4);
        std::vector<std::vector<int>> class_group_ids{{}, {}, {}, {}};
        for (int q = 0; q < step_len; q ++) {
            class_group_ids[0].push_back(q);
            class_group_ids[1].push_back(q+step_len);
            class_group_ids[2].push_back(q+2*step_len);
            class_group_ids[3].push_back(q+3*step_len);
        }

        // std::cout << "class_group_ids[0] = " << class_group_ids[0] << std::endl;
        // std::cout << "class_group_ids[1] = " << class_group_ids[1] << std::endl;
        
        // 初始化 CPUCache 对象
        CPUCache cache(bh, cache_idx, cache_shape, global_class_group_ids);
        CPUCache cache2(bh, cache_idx, cache_shape, global_class_group_ids);

        // 调用 load_with_cached 方法
        // std::cout << "#### direct load" << std::endl;
        auto tmp_tim = cache.direct_load(prefetch_idx, keys, values);


        // 配置 profiler
        // using namespace torch::autograd::profiler;
        // ProfilerConfig config{
        //     ProfilerState::KINETO,
        //     /* report_input_shapes */ false,
        //     /* profile_memory */ false,
        //     /* with_stack */ false,
        //     /* with_flops */ false
        // };

        // std::set<torch::profiler::impl::ActivityType> activities = {
        //     torch::profiler::impl::ActivityType::CPU,
        //     torch::profiler::impl::ActivityType::CUDA
        // };

        // // 开始 profile
        // enableProfiler(config, activities, {});


        auto [pure_unhit_list, cache_time] = cache.load_with_cached_test_v3(prefetch_idx, keys, values);
        // torch::cuda::synchronize();

        auto [pure_unhit_list_2, cache2_time] = cache2.load_with_cached_test_v2(prefetch_idx, keys, values);  
        // // torch::cuda::synchronize();  
        
        auto direct_time = cache.direct_load_test(prefetch_idx, keys, values);

        // // 结束prifile
        // torch::cuda::synchronize();
        // auto trace = disableProfiler();

        // 保存为 Chrome Trace JSON
        // std::ofstream out("trace_output.json");
        // out << trace->json();  // 注意 json() 是 shared_ptr 方法
        // out.close();
        
        if (i != 0) {  
            cache_cost.push_back(cache_time);
            cache_cost_2.push_back(cache2_time);
            direct_cost.push_back(direct_time);

            auto unhit_num = calculate_total_tokens(pure_unhit_list);
            auto all_num = cache_num * bh;
            double hit_rate = double(all_num - unhit_num)/all_num;
            hit_rate_list.push_back(hit_rate);
            // std::cout << "all_num = " << all_num << std::endl;
            // std::cout << "unhit_num = " << unhit_num << std::endl;
            // std::cout << "hit_rate = " << (all_num - unhit_num)/all_num << std::endl;
        }

        // 打印结果
        // std::cout << "Pure unhit list size: " << pure_unhit_list.size() << std::endl;
        // std::cout << "Communication time: " << communication_time << " ms" << std::endl;
    }

    auto avg_cache_cost = calculate_average(cache_cost);
    auto avg_cache_cost_2 = calculate_average(cache_cost_2);
    auto avg_direct_cost = calculate_average(direct_cost); 
    auto avg_hit_rate = calculate_average(hit_rate_list); 

    std::cout << "(" << token_num << ", " << cache_num << ", " << bh << ") : avg_direct_cost=" << avg_direct_cost 
        << "\t avg_cache_cost=" << avg_cache_cost   << "\t avg_cache_cost_2=" << avg_cache_cost_2 
        << "\t avg_hit_rate=" << avg_hit_rate << std::endl;

    return 0;
}


int main() {
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

    // test load cache cost
    int test_num = 6;
    std::vector<std::tuple<int, int, int, int>> test_cases3 = {
        // {2048, 1024, 64, test_num},
        // {4096, 2048, 64, test_num},
        {8192, 4096, 64, test_num},
        // {10240, 5120, 64, test_num},
    };
    // for (const auto& [token_num, cache_num, bh, test_num] : test_cases3) {
    //     bench_load_cache(token_num, cache_num, bh, test_num);
    // }
    
    for (const auto& [token_num, cache_num, bh, test_num] : test_cases3) {
        bench_load_cache_profile(token_num, cache_num, bh, test_num);
    }

    // for (const auto& [token_num, cache_num, bh, test_num] : test_cases3) {
    //     bench_update_cache(token_num, cache_num, bh, test_num);
    // }
    // bench_load_cache(8192, 4096, 64, 21);

    return 0;
}   