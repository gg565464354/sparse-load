#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "cpu_cache.h"  // 确保此头文件包含 CPUCache 的定义

void benchmark_get_unhit(int token_num, int bh, int cache_size) {
    // 生成随机的 prefetch_idx 张量，值范围在 [0, cache_size + 100]
    auto prefetch_idx = torch::randint(0, cache_size + 100, {token_num, 1, bh}, torch::kInt);
    
    // 初始化 CPUCache
    std::vector<int64_t> cache_shape = {cache_size, bh, 64}; // 64 代表 head_dim，
    CPUCache cache(bh, prefetch_idx, cache_shape);
    
    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();
    auto unhit_tokens = cache.get_unhit(prefetch_idx);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end - start;
    std::cout << "Test Case (tokens: " << token_num << ", bh: " << bh << ", cache: " << cache_size << ") -> Time: " << duration.count() << " s" << std::endl;
}

int main() {
    // 测试不同参数下的 get_unhit 执行时间
    // std::vector<std::tuple<int, int, int>> test_cases = {
    //     {128, 8, 1024},
    //     {256, 16, 2048},
    //     {512, 32, 4096},
    //     {1024, 64, 8192}
    // };

    // for (const auto& [tokens, bh, cache] : test_cases) {
    //     benchmark_get_unhit(tokens, bh, cache);
    // }

    std::vector<std::tuple<int, int, int>> test_cases = {
        {128, 64, 1024},
        {256, 64, 2048},
        {512, 64, 4096},
        {1024, 64, 8192}
    };

    for (const auto& [tokens, bh, cache] : test_cases) {
        benchmark_get_unhit(tokens, bh, cache);
    }

    
    return 0;
}   