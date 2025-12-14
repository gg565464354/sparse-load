import numpy as np
import numba
import xxhash
import time
import random

# 生成测试数据
def generate_data(n_cache=1000000, n_queries=100000):
    cache_indices = random.sample(range(n_cache * 10), n_cache)  # 生成稀疏索引
    query_indices = random.sample(range(n_cache * 10), n_queries)  # 生成查询索引
    return cache_indices, query_indices

# 方案 1: NumPy + Numba 并行二分查找
class CacheNumba:
    def __init__(self, index_list):
        self.index_table = np.array(sorted(index_list), dtype=np.int32)  # 预排序数组

    @staticmethod
    @numba.njit(parallel=True)
    def numba_search(index_table, indices):
        hits = 0
        for i in numba.prange(len(indices)):
            idx = np.searchsorted(index_table, indices[i])
            if idx < len(index_table) and index_table[idx] == indices[i]:
                hits += 1
        return hits

    def get_hit_rate(self, indices):
        hits = self.numba_search(self.index_table, np.array(indices, dtype=np.int32))
        return hits, None, None  # 直接返回 None 代替未命中列表



# 方案 2: Python 内置 set（比 dict 查询更快）
class CacheSet:
    def __init__(self, index_list):
        self.index_table = set(index_list)  # 使用 Python 内置 set

    def get_hit_rate(self, indices):
        hits = 0
        cached_ids_dict = {}
        un_cached_ids = []

        for idx in indices:
            if idx in self.index_table:
                hits += 1
                cached_ids_dict[idx] = True
            else:
                un_cached_ids.append(idx)

        return hits, cached_ids_dict, un_cached_ids

# 方案 3: xxHash 哈希表
class CacheXXHash:
    def __init__(self, index_list):
        self.index_table = {xxhash.xxh64_intdigest(idx.to_bytes(8, byteorder="little")): True for idx in index_list}

    def get_hit_rate(self, indices):
        hits = 0
        cached_ids_dict = {}
        un_cached_ids = []

        for idx in indices:
            hash_val = xxhash.xxh64_intdigest(idx.to_bytes(8, byteorder="little"))
            if hash_val in self.index_table:
                hits += 1
                cached_ids_dict[idx] = True
            else:
                un_cached_ids.append(idx)

        return hits, cached_ids_dict, un_cached_ids

# 运行基准测试
def benchmark_cache(cache_class, cache_indices, query_indices):
    cache = cache_class(cache_indices)
    start_time = time.time()
    hits, _, _ = cache.get_hit_rate(query_indices)
    elapsed_time = time.time() - start_time
    elapsed_time = elapsed_time*1000
    print(f"{cache_class.__name__}: Hits = {hits}, Time = {elapsed_time:.6f} ms")

if __name__ == "__main__":
    cache_indices, query_indices = generate_data(n_cache=4000, n_queries=3000)
    
    benchmark_cache(CacheNumba, cache_indices, query_indices)
    benchmark_cache(CacheSet, cache_indices, query_indices)
    benchmark_cache(CacheXXHash, cache_indices, query_indices)
