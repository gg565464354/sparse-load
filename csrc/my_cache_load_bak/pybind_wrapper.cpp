#include <torch/extension.h>
// #include "src/cache_bench.h"
#include "src/cache_api.h"

// 假设你的函数签名如下
// int main_test();

// common cache api
int init_cache(int bh, const torch::Tensor& cache_idx, const std::vector<int64_t>& cache_shape,  const std::vector<std::vector<int>>& class_group_ids); // return 0-success, 1-fail
int update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache); // return 0-success, 1-uninitial cache
int update_cache_idx(const torch::Tensor& prefetch_idx); // return 0-success, 1-uninitial cache
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<std::vector<std::vector<int>>>> cache_load(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_load(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);
std::vector<int64_t> show_cache_shape();

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    generate_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);
int update_group_cache(const torch::Tensor& prefetch_idx, const std::vector<torch::Tensor>& k_cache, const std::vector<torch::Tensor>& v_cache);

// void process_data(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch::init();
    
    // m.def("main_test", &main_test, "A simple test for Cache Module");
    m.def("init_cache", &init_cache, "Init a CPU Cache");
    m.def("update_cache", &update_cache, "Update a CPU Cache, including cached kv and cached idx");
    m.def("update_cache_idx", &update_cache_idx, "Update a CPU Cache, only update cached idx");
    m.def("cache_load", &cache_load, "Load using CPU Cache");
    m.def("direct_load", &direct_load, "Direct Load without CPU Cache");
    m.def("show_cache_shape", &show_cache_shape, "show_cache_shape");

    // update cache
    m.def("generate_cache", &generate_cache, "generate_cache");
    m.def("update_group_cache", &update_group_cache, "update_group_cache");
}