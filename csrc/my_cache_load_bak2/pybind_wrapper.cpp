#include <torch/extension.h>
// #include "src/cache_bench.h"
#include "src/cache_api.h"
#include "src/cpu_cache.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

// 假设你的函数签名如下
// int main_test();

// common cache api
int init_cache(int bh, const torch::Tensor& cache_idx, const std::vector<int64_t>& cache_shape,  const std::vector<std::vector<int>>& class_group_ids); // return 0-success, 1-fail
int update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache); // return 0-success, 1-uninitial cache
int update_cache_idx(const torch::Tensor& prefetch_idx); // return 0-success, 1-uninitial cache
std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<std::vector<std::vector<int>>>> cache_load(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_load(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);
std::vector<int64_t> show_cache_shape();

// std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<std::vector<std::vector<int>>>> cache_load_v2(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);

// api for python load
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_cached_kv();
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> get_unhit_kv(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);

// api for cache update
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    generate_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);
int update_group_cache(const torch::Tensor& prefetch_idx, const std::vector<torch::Tensor>& k_cache, const std::vector<torch::Tensor>& v_cache);

// static api for unhit
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> 
    static_get_unhit_kv(CPUCache& cache, const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);

// vector class api


// void process_data(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch::init();
    
    // m.def("main_test", &main_test, "A simple test for Cache Module");
    m.def("init_cache", &init_cache, "Init a CPU Cache");
    m.def("update_cache", &update_cache, "Update a CPU Cache, including cached kv and cached idx");
    m.def("update_cache_idx", &update_cache_idx, "Update a CPU Cache, only update cached idx");
    m.def("cache_load", &cache_load, "Load using CPU Cache", py::call_guard<py::gil_scoped_release>());
    // m.def("cache_load_v2", &cache_load_v2, "Load using CPU Cache version 2");
    m.def("direct_load", &direct_load, "Direct Load without CPU Cache", py::call_guard<py::gil_scoped_release>());
    m.def("show_cache_shape", &show_cache_shape, "show_cache_shape");

    // update cache
    m.def("generate_cache", &generate_cache, "generate_cache");
    m.def("update_group_cache", &update_group_cache, "update_group_cache");

    // python load
    m.def("get_cached_kv", &get_cached_kv, "get_cached_kv", py::call_guard<py::gil_scoped_release>());
    m.def("get_unhit_kv", &get_unhit_kv, "get_unhit_kv", py::call_guard<py::gil_scoped_release>());

    // static api for unhit
    m.def("static_get_unhit_kv", &static_get_unhit_kv, "static_get_unhit_kv", py::call_guard<py::gil_scoped_release>());

    // 打包成类
    py::class_<CPUCache>(m, "CPUCache")
        .def(py::init<>())  // 默认构造函数
        .def(py::init<int, const torch::Tensor&, const std::vector<int64_t>&, const std::vector<int>&>())  // 带参数的构造函数
        .def(py::init<int, const torch::Tensor&, const std::vector<int64_t>&, const std::vector<std::vector<int>>&>())  // 另一个带参数的构造函数
        .def("show_cache_shape", &CPUCache::show_cache_shape)
        // .def("classify_heads", &CPUCache::classify_heads)
        // .def("group_heads_by_class", &CPUCache::group_heads_by_class)
        // .def("pad_class_unhits", &CPUCache::pad_class_unhits)
        // .def("SplitGroupIdx", &CPUCache::SplitGroupIdx)
        // .def("update_cache_map", &CPUCache::update_cache_map)
        // .def("update_cache", static_cast<int (CPUCache::*)(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)>(&CPUCache::update_cache))
        .def("update_cache", &CPUCache::update_cache)
        .def("update_group_cache", &CPUCache::update_group_cache)
        .def("update_cache_v2", &CPUCache::update_cache_v2)
        .def("SplitIdx", &CPUCache::SplitIdx)
        .def("generate_cache", &CPUCache::generate_cache)
        .def("generate_update_cache", &CPUCache::generate_update_cache)
        .def("load_with_cached", &CPUCache::load_with_cached)
        // .def("load_with_cached_test", &CPUCache::load_with_cached_test)
        // .def("split_unhit", &CPUCache::split_unhit)
        // .def("initialize_head_classes", static_cast<void (CPUCache::*)(const std::vector<int>&)>(&CPUCache::initialize_head_classes))
        // .def("initialize_head_classes", static_cast<void (CPUCache::*)(const std::vector<std::vector<int>> &)>(&CPUCache::initialize_head_classes))
        // .def("get_unhit", &CPUCache::get_unhit)
        // .def("pad_and_convert_unhits", &CPUCache::pad_and_convert_unhits)
        .def("select_kv_v2", &CPUCache::select_kv_v2)
        .def("select_kv", &CPUCache::select_kv)
        .def("direct_load", &CPUCache::direct_load)
        .def("direct_load_test", &CPUCache::direct_load_test)
        
        // async cpu compute api
        .def("get_cached_kv", &CPUCache::get_cached_kv, py::call_guard<py::gil_scoped_release>())
        .def("get_unhit_kv", &CPUCache::get_unhit_kv, py::call_guard<py::gil_scoped_release>())
        .def("asyn_update_cache", &CPUCache::asyn_update_cache, py::call_guard<py::gil_scoped_release>())

        // static api 
        .def("select_kv_vector_v2", &CPUCache::select_kv_vector_v2, py::call_guard<py::gil_scoped_release>())
        .def("get_unhit_kv_vector", &CPUCache::get_unhit_kv_vector, py::call_guard<py::gil_scoped_release>())
        .def("get_unhit_vector", &CPUCache::get_unhit_vector, py::call_guard<py::gil_scoped_release>());

}