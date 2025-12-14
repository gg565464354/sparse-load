import torch
import torch.nn.functional as F
import my_cache_load._C as _C
from concurrent.futures import ThreadPoolExecutor
import time
from torch.profiler import profile, record_function, ProfilerActivity

class CacheManager:
    def __init__(self, basic_group_head_ids, layer_head_num):
        """
        初始化 CacheManager，使用字典保存不同 layer_id 对应的缓存数据。
        """
        self._caches = {}  # key: layer_id, value: cache data
        self._basic_group_head_ids = basic_group_head_ids # key: layer_id, value: layer group head ids

        print("init self._basic_group_head_ids = ", self._basic_group_head_ids)
        self.layer_head_numbers = layer_head_num
        self.executor = ThreadPoolExecutor()  # 可复用线程池
    

    def init_basic_group_head_ids(self, layer_group_head_ids):
        for l in range(len(layer_group_head_ids)):
            self._basic_group_head_ids[l] = layer_group_head_ids[l]
    

    def add_cache(self, layer_id, batch_size, head_num, sparse_len, hidden_size):
        """
        添加或更新指定 layer_id 的缓存数据。
        
        :param layer_id: 缓存标识符（字符串）
        """
        if layer_id in self._caches:
            raise ValueError("Python ERROR! [add_cache] layer id is used!")
            return 1

        new_cache = self.create_cache_instance(layer_id, batch_size, head_num, sparse_len, hidden_size)

        self._caches[layer_id] = new_cache
        return 0

    def remove_cache(self, layer_id):
        """
        移除指定 layer_id 的缓存。
        
        :param layer_id: 缓存标识符
        """
        if layer_id in self._caches:
            del self._caches[layer_id]

    def has_cache(self, layer_id) -> bool:
        """
        检查是否存在指定 layer_id 的缓存。
        
        :param layer_id: 缓存标识符
        :return: 布尔值
        """
        return layer_id in self._caches

    def clear_caches(self):
        """
        清空所有缓存。
        """
        self._caches.clear()

    def cache_count(self) -> int:
        """
        返回当前缓存的数量。
        
        :return: 整数
        """
        return len(self._caches)


    def cpu_cache_load_asyn(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v):
        cur_cache = self._caches[layer_id]
        group_cached_k, group_cached_v = cur_cache.get_cached_kv()

        group_final_k = []
        group_final_v = []

        # Step 1: 提前异步提交 CPU 计算任务
        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_unhit = self.executor.submit(cur_cache.get_unhit_kv, prefetch_idx, all_k, all_v)

        # Step 2: 同时开始 cached_k/v 的 GPU 传输（异步）
        with torch.cuda.stream(transfer_stream):
            group_cached_gpu_k = [k.cuda(non_blocking=True) for k in group_cached_k]
            group_cached_gpu_v = [v.cuda(non_blocking=True) for v in group_cached_v]

        # Step 3: 等待 CPU 计算结束
        group_unhit_k, group_unhit_v, group_unhit = future_unhit.result()

        # Step 4: unhit_k/v -> GPU 传输 + 拼接（仍在 transfer_stream）
        with torch.cuda.stream(transfer_stream):
            unhit_gpu_k = [k.cuda(non_blocking=True) for k in group_unhit_k]
            unhit_gpu_v = [v.cuda(non_blocking=True) for v in group_unhit_v]

            for i in range(len(group_cached_k)):
                final_k = torch.cat([group_cached_gpu_k[i], unhit_gpu_k[i]], dim=0)
                final_v = torch.cat([group_cached_gpu_v[i], unhit_gpu_v[i]], dim=0)
                group_final_k.append(final_k)
                group_final_v.append(final_v)

        transfer_stream.synchronize()

        return group_final_k, group_final_v, group_unhit

    def gpu_cache_load_asyn(self, transfer_stream, prefetch_idx, all_k, all_v, group_cached_gpu_k, group_cached_gpu_v):
        cur_cache = self._caches[layer_id]

        group_final_k = []
        group_final_v = []

        # Step 1: 提前异步提交 CPU 计算任务
        group_unhit_k, group_unhit_v, group_unhit = cur_cache.get_unhit_kv(prefetch_idx, all_k, all_v)

        # Step 2: unhit_k/v -> GPU 传输 + 拼接（仍在 transfer_stream）
        with torch.cuda.stream(transfer_stream):
            unhit_gpu_k = [k.cuda(non_blocking=True) for k in group_unhit_k]
            unhit_gpu_v = [v.cuda(non_blocking=True) for v in group_unhit_v]

            for i in range(len(group_cached_gpu_k)):
                final_k = torch.cat([group_cached_gpu_k[i], unhit_gpu_k[i]], dim=0)
                final_v = torch.cat([group_cached_gpu_v[i], unhit_gpu_v[i]], dim=0)
                group_final_k.append(final_k)
                group_final_v.append(final_v)

        transfer_stream.synchronize()

        return group_final_k, group_final_v, group_unhit

    def load_and_update_cache(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v):
        group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = _C.generate_cache(cur_idx_list[i], all_k, all_v)

        stat = self._caches[layer_id].update_group_cache(cur_idx_list[i], group_cpu_k, group_cpu_v)
        if stat != 0:
            raise ValueError("Python ERROR! [load_and_update_cache] Update cache Fail!")

        return group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v
    
    async def update_cpu_cache_asyn(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v):
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            self.executor,
            self._cache[layer_id].asyn_update_cache,
            prefetch_idx,
            keys,
            values
        )

        return result

    def generate_class_group_ids(self, layer_id, batch_size):
        """
        Require static offline head group
        """
        print("layer id = ", layer_id, type(layer_id))
        print("self._basic_group_head_ids[layer_id] = ", self._basic_group_head_ids[layer_id])

        cur_basic_head_ids = self._basic_group_head_ids[layer_id]
        cur_head_num = self.layer_head_numbers
        final_head_ids = [[] for g in range(len(cur_basic_head_ids))]
        
        for b in range(batch_size): # 每个batch的head id 依次添加
            for g in range(len(cur_basic_head_ids)): # 遍历每个group
                for head_id in cur_basic_head_ids[g]: # 添加所有的head id，最大值为 bh*cur_head_num
                    final_head_ids[g].append(head_id + b*cur_head_num)

        return final_head_ids


    # 初始化c++模块中的cache
    def create_cache_instance(self, layer_id, batch_size, head_num, sparse_len, hidden_size):
        """ Initial the cache
        Call for every new batch of requests
        Building cache for every decoder layer.
        """
        if len(self._basic_group_head_ids) == 0:
            raise ValueError("Python ERROR! [create_cache_instance] basic_group_ids was not initialed!")

        # 构造 class group id
        new_class_group_id = self.generate_class_group_ids(layer_id, batch_size)
        
        cache_shape = (batch_size*head_num, sparse_len, hidden_size)
        cache_idx = torch.randint(low=0, high=N, size=(sparse_len, 1, batch_size*head_num)).to(torch.int32)

        # 初始化一个cache
        cache = _C.CPUCache(bh, cache_idx, cache_shape, new_class_group_id)
        
        return cache


def process_idx(layer_idx):
    # Step 1: 调整维度顺序，变为 (layer_num, N_p, bh)
    tensor_permuted = layer_idx.permute(0, 2, 1)

    # Step 2: 添加新维度，变为 (layer_num, N_p, 1, bh)
    tensor_unsqueezed = tensor_permuted.unsqueeze(2)

    # Step 3: 拆分张量，得到 layer_num 个形状为 (N_p, 1, bh) 的张量
    tensors_split = tensor_unsqueezed.unbind(dim=0)

    return tensors_split


N = 7800
N_p = 7800//2
d = 128
batch_size = 1
head_num = 64
bh = batch_size * head_num
# bh = 64

idx_shape = (N_p, 1, bh)
kv_shape = (N, bh, d)
cache_shape = (N_p, bh, d)

all_k = torch.randint(low=0, high=10000, size=kv_shape).cpu()
all_v = all_k + 1

class_group_ids = [
    [27, 44, 52, 56],
    [8, 11, 13, 14, 15, 17, 21, 26, 28, 29, 32, 34, 39, 43, 45, 46, 47, 49, 55, 62],
    [2, 3, 4, 6, 7, 12, 16, 18, 19, 20, 22, 23, 25, 30, 31, 33, 36, 38, 41, 42, 48, 50, 51, 53, 58, 60, 61, 63],
    [0, 1, 5, 9, 10, 24, 35, 37, 40, 54, 57, 59]
]

class_group_ids = [
    [0, 1, 5, 9, 10, 24, 35, 37, 40, 54, 57, 59],
    [2, 3, 4, 6, 7, 12, 16, 18, 19, 20, 22, 23, 25, 30, 31, 33, 36, 38, 41, 42, 48, 50, 51, 53, 58, 60, 61, 63],
    [8, 11, 13, 14, 15, 17, 21, 26, 28, 29, 32, 34, 39, 43, 45, 46, 47, 49, 55, 62],
    [27, 44, 52, 56],
]

# 初始化cache manager
Global_Cache_Manager = CacheManager({0:class_group_ids}, head_num)
Global_Cache_Manager.add_cache(0, batch_size, head_num, N_p, d)


# cache_idx = torch.randint(low=0, high=N, size=idx_shape).to(torch.int32)
# _C.init_cache(bh, cache_idx, cache_shape, class_group_ids)


transfer_stream = torch.cuda.Stream()


for layer_id in range(1):
    print(f"####################### layer #{layer_id}")

    # layer_id = 0
    data_name = f"./tmp/test2_layer{layer_id}.pt"
    layer_idx = torch.load(data_name)
    decode_idxs = process_idx(layer_idx) 


    cur_idx_list = decode_idxs
    test_num = len(cur_idx_list)
    # test_num = 4
    K = 4

    all_unhits = []

    cache_load_cost = []
    update_cost = []
    direct_load_cost = []

    for i in range(test_num):
        if i % K == 0:
            # 直接加载并且更新cache
            torch.cuda.synchronize()
            start_u = time.time()

            group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = Global_Cache_Manager.load_and_update_cache(layer_id, transfer_stream, cur_idx_list[i], all_k, all_v)
            # stat = _C.update_group_cache(cur_idx_list[i], group_cpu_k, group_cpu_v)
            
            # all_unhits.append([[]])
            
            torch.cuda.synchronize()
            end_u = time.time()
            update_cost.append(end_u-start_u)

        else:
            torch.cuda.synchronize()
            start1 = time.time()
            
            # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            # with profile(activities=activities, with_stack=True) as prof:

            # gpu_k, gpu_v, unhit_list = _C.cache_load(cur_idx_list[i], all_k, all_v)
            # gpu_k, gpu_v, unhit_list = cache_load_asyn(transfer_stream, cur_idx_list[i], all_k, all_v)
            # gpu cache load
            gpu_k, gpu_v, unhit_list = Global_Cache_Manager.cpu_cache_load_asyn(layer_id, transfer_stream, cur_idx_list[i], all_k, all_v)

            # cpu cache load
            # gpu_k, gpu_v, unhit_list = Global_Cache_Manager.gpu_cache_load_asyn(transfer_stream, cur_idx_list[i], all_k, all_v, group_gpu_k, group_gpu_v)
            
            torch.cuda.synchronize()
            # prof.export_chrome_trace(f"./trace_manager_cpu_cache_{i}.jsonl")

            end1 = time.time()
            cache_load_cost.append(end1-start1)
            all_unhits.append(unhit_list)

    print("cache load cost = ", sum(cache_load_cost)/len(cache_load_cost))

### gpu cache test end

    for i in range(test_num):
        # cur_idx = torch.randint(low=0, high=N, size=idx_shape).to(torch.int32)
        # cur_idx[:same_N] = prefetch_idx[same_N]
        torch.cuda.synchronize()
        start2 = time.time()

        gpu_k, gpu_v, cpu_k, cpu_v = _C.direct_load(cur_idx_list[i], all_k, all_v)

        torch.cuda.synchronize()
        end2 = time.time()

        direct_load_cost.append(end2-start2)

    print("direct load cost = ", sum(direct_load_cost)/len(direct_load_cost))

    print("cache update cost = ", sum(update_cost)/len(update_cost))

    # print("direct load list = ", direct_load_cost)
    # print("cache_load_cost list = ", cache_load_cost)
    # print("update_cost list = ", update_cost)


################# profile 测试