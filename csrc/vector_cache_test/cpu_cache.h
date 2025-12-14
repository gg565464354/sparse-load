#ifndef CPU_CACHE_H
#define CPU_CACHE_H

#include <vector>
#include <unordered_set>
#include <torch/torch.h>
#include <omp.h>  // OpenMP 并行加速

class CPUCache {

public:
    int cache_token_size;
    int bh;
    int head_dim;
    int head_num;

    // 新增静态分类相关成员变量
    int group_num; // cache 的 type 类型
    std::vector<int> head_classes_;  // 每个head的预定义类别
    std::unordered_map<int, std::vector<int>> class_groups_;  // 类别到head列表的映射
    std::unordered_map<int, torch::Tensor> class_groups_tensor_;  // 类别到head列表的映射
    torch::Tensor combined_group_ids_tensor;

    /*
        prefetch_idx: shape (n', 1, bh)
        cache_shape: (n', bh, d)
        k_cache: Key cache (n, bh, d)
        v_cache: Value cache (n, bh, d)
    */

// public:
    // 新增的成员变量
    std::vector<torch::Tensor> cache_keys;   // 关键缓存张量
    std::vector<torch::Tensor> cache_values; // 值缓存张量
    std::vector<std::unordered_set<int>> cache_maps;  // 高效哈希存储
    std::vector<int64_t> cur_cache_shape;

    CPUCache();
    std::vector<int64_t> show_cache_shape();
    
    // cache load
    // return gpu_k, gpu_v

    // class cache v2
    std::vector<int> classify_heads(const std::vector<std::vector<int>>& pure_unhit_list, const int idx_len);
    std::unordered_map<int, std::vector<int>> group_heads_by_class(const std::vector<int>& head_classes);
    torch::Tensor pad_class_unhits(const std::vector<std::vector<int>>& class_unhit_list, int max_unhit, int num_heads);

    std::vector<torch::Tensor> SplitGroupIdx(
        std::vector<std::vector<int>>& unhit_list, 
        std::unordered_map<int, std::vector<int>>& group_head_ids
    );
    

    /////////////////////////////////////////////////////////// 需要更新的函数
    /* 
    当前技术方案：
        忽略batch_size，直接以bh管理所有的head
        prefetch_idx 还是作为一个整体，获取unhit的时候直接获取整体
        cache 部分划分不同 group
        select kv 也要分 group 进行
        最终 cache load 的结果也是生成划分group的数据
    */

    // 初始化函数，直接划分kv，设置多个cache实例
    CPUCache(int bh, const torch::Tensor& prefetch_idx, const std::vector<int64_t>& cache_shape, const std::vector<int>& head_classes);
    CPUCache(int _bh, const torch::Tensor& prefetch_idx, const std::vector<int64_t>& cache_shape, const std::vector<std::vector<int>>& class_group_ids);

    // split cache

    // cache map 更新，需要考虑根据head group 划分，需要考虑batch size
    int update_cache_map(const torch::Tensor& prefetch_idx);

    // 更新cache内容，需要根据head group划分新的k_cache 和 v_cache
    int update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);
    int update_cache_v2(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);
    int update_group_cache(const torch::Tensor& prefetch_idx, const std::vector<torch::Tensor>& group_cached_k, const std::vector<torch::Tensor>& group_cached_v);
    // int update_cache_with_group_data(const std::vector<torch::Tensor>& group_idx, const std::vector<torch::Tensor>& group_cached_k, const std::vector<torch::Tensor>& group_cached_v);
    
    std::vector<torch::Tensor> SplitIdx(const torch::Tensor& prefetch_idx);
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> 
        generate_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> 
        generate_update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);
    int asyn_update_cache(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);

    // direct load（直接用原来的就行）

    // cache load：对不同group的head分开处理，然后分别加载关键kv，然后聚合成为多个tensor作为输出，按照顺序
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> load_with_cached(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);

    std::tuple<std::vector<std::vector<int>>, double> load_with_cached_test(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);

    // std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> load_with_cached_v2(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);

    std::vector<std::vector<std::vector<int>>> split_unhit(std::vector<std::vector<int>> pure_unhit_list);

    // use python for transfer
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_cached_kv();
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> get_unhit_kv(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);


    ///////////////////////////////////////////////////////////////// 不需要更新的函数
    // 似乎可以直接用着，但初始化的时候需要调用
    void initialize_head_classes(const std::vector<int>& head_classes);
    void initialize_head_classes(const std::vector<std::vector<int>>& class_group_ids);

    // 判断命中率，可以直接
    std::vector<std::vector<int>> get_unhit(const torch::Tensor& prefetch_idx);
    torch::Tensor pad_and_convert_unhits(const std::vector<std::vector<int>>& pure_unhit_list);

    // 可以直接用来选取不同group的关键token
    std::tuple<torch::Tensor, torch::Tensor> select_kv_v2(
        const torch::Tensor& prefetch_idx,  // 形状: (max_unhit, num_heads_in_class)
        const torch::Tensor& k_cache,       // 形状: (n, bh_total, d)
        const torch::Tensor& v_cache,
        const std::vector<int>& head_group // 新增参数：当前组的头索引列表
    );

    // 用来实现直接加载
    std::tuple<torch::Tensor, torch::Tensor> select_kv(const torch::Tensor& prefetch_idx, const torch::Tensor& k_cache, const torch::Tensor& v_cache);
    
    // 直接加载，就不用改了
    // return gpu_k, gpu_v, cpu_k, cpu_v
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_load(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);
    double direct_load_test(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);


    
    ////////////////////// 基于vector的实现
    std::tuple<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<std::vector<float>>>> 
    select_kv_vector_v2(
        const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
        const std::vector<std::vector<std::vector<float>>>& k_cache, // shape: (n, bh_total, d)
        const std::vector<std::vector<std::vector<float>>>& v_cache,
        const std::vector<int>& head_group
    );
    
    int select_kv_vector_v3(
        const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
        const std::vector<std::vector<std::vector<float>>>& k_cache, // shape: (n, bh_total, d)
        const std::vector<std::vector<std::vector<float>>>& v_cache,
        const std::vector<int>& head_group,
        
        std::vector<std::vector<std::vector<float>>>& selected_k,
        std::vector<std::vector<std::vector<float>>>& selected_v
    );

    int select_kv_vector_v3_test(
        const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
        const std::vector<std::vector<std::vector<float>>>& k_cache, // shape: (n, bh_total, d)
        const std::vector<std::vector<std::vector<float>>>& v_cache,
        const std::vector<int>& head_group,
        
        std::vector<std::vector<std::vector<float>>>& selected_k,
        std::vector<std::vector<std::vector<float>>>& selected_v
    );

    std::tuple<
        std::vector<std::vector<std::vector<std::vector<float>>>>,
        std::vector<std::vector<std::vector<std::vector<float>>>>,
        std::vector<std::vector<std::vector<int>>>>
    get_unhit_kv_vector(
        const std::vector<std::vector<int>>& prefetch_idx, // [seq][bh_total]
        const std::vector<std::vector<std::vector<float>>>& all_keys,       // [n][bh_total][d]
        const std::vector<std::vector<std::vector<float>>>& all_values      // [n][bh_total][d]
    );
    
    std::tuple<
        std::vector<std::vector<std::vector<std::vector<float>>>>,
        std::vector<std::vector<std::vector<std::vector<float>>>>,
        std::vector<std::vector<std::vector<int>>>>
    get_unhit_kv_vector_test(
        const std::vector<std::vector<int>>& prefetch_idx, // [seq][bh_total]
        const std::vector<std::vector<std::vector<float>>>& all_keys,       // [n][bh_total][d]
        const std::vector<std::vector<std::vector<float>>>& all_values      // [n][bh_total][d]
    );

    std::vector<std::vector<int>> get_unhit_vector(const std::vector<std::vector<int>>& prefetch_idx_vec);
    std::vector<std::vector<int>> pad_and_convert_unhits_vector(const std::vector<std::vector<int>>& pure_unhit_list);
    std::vector<std::vector<int>> pad_and_convert_unhits_vector_test(const std::vector<std::vector<int>>& pure_unhit_list);

    
    ////////////////////// 基于一维list的实现
    std::tuple<
        std::vector<std::vector<float>>,
        std::vector<std::vector<float>>,
        std::vector<std::vector<std::vector<int>>>>
    get_unhit_kv_list(
        const std::vector<std::vector<int>>& prefetch_idx, // [seq][bh_total]
        const std::vector<std::vector<std::vector<float>>>& all_keys,       // [n][bh_total][d]
        const std::vector<std::vector<std::vector<float>>>& all_values      // [n][bh_total][d]
    );

    std::tuple<
        std::vector<std::vector<float>>,
        std::vector<std::vector<float>>,
        std::vector<std::vector<std::vector<int>>>>
    get_unhit_kv_list_test(
        const std::vector<std::vector<int>>& prefetch_idx, // [seq][bh_total]
        const std::vector<std::vector<std::vector<float>>>& all_keys,       // [n][bh_total][d]
        const std::vector<std::vector<std::vector<float>>>& all_values      // [n][bh_total][d]
    );

    int select_kv_list_test(
        const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
        const std::vector<std::vector<std::vector<float>>>& k_cache, // shape: (n, bh_total, d)
        const std::vector<std::vector<std::vector<float>>>& v_cache,
        const std::vector<int>& head_group,

        std::vector<float>& selected_k,
        std::vector<float>& selected_v
    );

    
    int select_kv_list(
        const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
        const std::vector<std::vector<std::vector<float>>>& k_cache, // shape: (n, bh_total, d)
        const std::vector<std::vector<std::vector<float>>>& v_cache,
        const std::vector<int>& head_group,

        std::vector<float>& selected_k,
        std::vector<float>& selected_v
    );


    ////// Tensor case
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<std::vector<std::vector<int>>>> 
    get_unhit_kv_tensor(const torch::Tensor& prefetch_idx, const torch::Tensor& keys, const torch::Tensor& values);

    std::tuple<torch::Tensor, torch::Tensor> select_kv_tensor(
        const std::vector<std::vector<int>>& prefetch_idx,  // shape: (max_unhit, num_heads_in_class)
        const torch::Tensor& k_cache,  // shape: (n, bh_total, d), dtype: float16
        const torch::Tensor& v_cache,  // shape: (n, bh_total, d), dtype: float16
        const std::vector<int>& head_group  // shape: [num_heads_in_class]
    );

    std::vector<std::vector<int>> get_unhit_tensor2vec(const torch::Tensor& prefetch_idx_tensor);
};

#endif  // CPU_CACHE_H
