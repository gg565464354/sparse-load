#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "cpu_cache.h"  // 确保此头文件包含 CPUCache 的定义
#include <random>
#include <ctime>

void benchmark_get_unhit(int token_num, int bh, int cache_size);

// 随机生成测试样例
std::vector<std::vector<int>> generate_random_unhits(int bh, int min_len, int max_len, int min_val, int max_val);

int benchmark_pad_unhit(int bh, int min_len, int max_len, int min_val, int max_val);

double calculate_average(const std::vector<double>& data);

int calculate_total_tokens(const std::vector<std::vector<int>>& pure_unhit_list);


int bench_load_cache(int token_num, int cache_num, int bh, int test_cases);

int main_test();