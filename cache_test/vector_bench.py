import torch
import torch.nn.functional as F
import my_cache_load._C as _C
from concurrent.futures import ThreadPoolExecutor

import time
import json

from torch.profiler import profile, record_function, ProfilerActivity


torch.cuda.init()

def process_idx(layer_idx):
    # Step 1: 调整维度顺序，变为 (layer_num, N_p, bh)
    tensor_permuted = layer_idx.permute(0, 2, 1)

    # Step 2: 添加新维度，变为 (layer_num, N_p, 1, bh)
    tensor_unsqueezed = tensor_permuted.unsqueeze(2)

    # Step 3: 拆分张量，得到 layer_num 个形状为 (N_p, 1, bh) 的张量
    tensors_split = tensor_unsqueezed.unbind(dim=0)

    return tensors_split

def get_unhit_num(unhit_list):
    unhit_num = 0
    max_num = 0
    for u in unhit_list:
        l = len(u)
        unhit_num += l
        max_num = max(l, max_num)
    return unhit_num, max_num


class CacheManager:
    def __init__(self, basic_group_head_ids, layer_head_num, update_pred=4):
        """
        初始化 CacheManager，使用字典保存不同 layer_id 对应的缓存数据。
        """
        self._caches = {}  # key: layer_id, value: cache data
        self._update_recode = {}
        self._update_pred = update_pred
        
        self._basic_group_head_ids = basic_group_head_ids # key: layer_id, value: layer group head ids
        self._cur_group_ids = {}

        # print("init self._basic_group_head_ids = ", self._basic_group_head_ids)
        self.layer_head_number = layer_head_num
        self.executor = ThreadPoolExecutor()  # 可复用线程池
    

    def init_basic_group_head_ids(self, layer_group_head_ids):
        for l in range(len(layer_group_head_ids)):
            self._basic_group_head_ids[l] = layer_group_head_ids[l]

    def add_cache(self, device, layer_id, batch_size, head_num, sparse_len, hidden_size):
        """
        添加或更新指定 layer_id 的缓存数据。
        
        :param layer_id: 缓存标识符（字符串）
        """
        if layer_id in self._caches:
            raise ValueError("Python ERROR! [add_cache] layer id is used!")
            return 1

        new_cache, new_class_group_id = self.create_cache_instance(device, layer_id, batch_size, head_num, sparse_len, hidden_size)

        self._caches[layer_id] = new_cache
        self._cur_group_ids[layer_id] = new_class_group_id
        self._update_recode[layer_id] = 0
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

    def get_layer_group_id(self, layer_id):
        return self._cur_group_ids[layer_id]

    def cpu_cache_load_asyn_test(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype):

        cur_cache = self._caches[layer_id]
        group_cached_k, group_cached_v = cur_cache.get_cached_kv()

        group_final_k = []
        group_final_v = []

        
        torch.cuda.synchronize()
        start_0 = time.time()

        prefetch_idx_vector = prefetch_idx.squeeze(1).tolist()
        # k_vector = all_k.tolist()
        # v_vector = all_v.tolist()

        k_flat_vector = all_k.view(-1).tolist()
        v_flat_vector = all_v.view(-1).tolist()

        seq = prefetch_idx.shape[0]
        n = all_k.shape[0]
        bh_total = all_k.shape[1]
        d = all_k.shape[2]

        
        torch.cuda.synchronize()
        end_0 = time.time()

        print(f"[Timing] tensor to list: {(end_0 - start_0) * 1000:.2f} ms")
        
        torch.cuda.synchronize()
        start_1 = time.time()
        
        # future_unhit = self.executor.submit(cur_cache.get_unhit_kv_vector, prefetch_idx.squeeze(1).tolist(), all_k.tolist(), all_v.tolist())
        # future_unhit = self.executor.submit(cur_cache.get_unhit_kv_list, prefetch_idx_vector, k_vector, v_vector)
        # group_unhit_k_vector, group_unhit_v_vector, group_unhit = future_unhit.result()


        # group_unhit_k_vector, group_unhit_v_vector, group_unhit = cur_cache.get_unhit_kv_list(prefetch_idx_vector, k_vector, v_vector)
        # group_unhit_k_vector, group_unhit_v_vector, group_unhit = cur_cache.get_unhit_kv_list_test(prefetch_idx_vector, k_vector, v_vector)
        group_unhit_k_vector, group_unhit_v_vector, group_unhit = cur_cache.get_unhit_kv_tensor(prefetch_idx, all_k, all_v)
        
        # group_unhit_k_vector, group_unhit_v_vector, group_unhit = cur_cache.get_unhit_kv_list_test(prefetch_idx_vector, k_flat_vector, v_flat_vector, seq, n, bh_total, d)
        
        
        torch.cuda.synchronize()
        end_1 = time.time()


        # Step 3: 等待 CPU 计算结束
        # group_unhit_k, group_unhit_v, group_unhit = future_unhit.result()
        
        torch.cuda.synchronize()
        start_2 = time.time()
        try:
            group_unhit_k = [torch.tensor(unhit_k, dtype=dtype) for unhit_k in group_unhit_k_vector]
            group_unhit_v = [torch.tensor(unhit_v, dtype=dtype) for unhit_v in group_unhit_v_vector]

        except Exception as e:
            print(f"future_unhit error: {e}")
        
        torch.cuda.synchronize()
        end_2 = time.time()

        
        # Step 2: 并行开始 cached_k/v 的 GPU 传输（异步）
        torch.cuda.synchronize()
        start_3 = time.time()
        with torch.cuda.stream(transfer_stream):
            group_cached_gpu_k = [k.to(dtype).cuda(non_blocking=True) for k in group_cached_k]
            group_cached_gpu_v = [v.to(dtype).cuda(non_blocking=True) for v in group_cached_v]
        
        
        transfer_stream.synchronize() 
        torch.cuda.synchronize()
        end_3 = time.time()


        # Step 4: unhit_k/v -> GPU 传输 + 拼接（仍在 transfer_stream）
        torch.cuda.synchronize()
        start_4 = time.time()

        with torch.cuda.stream(transfer_stream):
            unhit_gpu_k = [k.cuda(non_blocking=True) for k in group_unhit_k]
            unhit_gpu_v = [v.cuda(non_blocking=True) for v in group_unhit_v]

            for i in range(len(group_cached_k)):
                final_k = torch.cat([group_cached_gpu_k[i], unhit_gpu_k[i]], dim=0)
                final_v = torch.cat([group_cached_gpu_v[i], unhit_gpu_v[i]], dim=0)
                group_final_k.append(final_k)
                group_final_v.append(final_v)
            
        
        transfer_stream.synchronize() 
        torch.cuda.synchronize()
        end_4 = time.time()

        
        # print("CacheManager cpu_cache_load_asyn: T2")
        # transfer_stream.synchronize()        
        # print("CacheManager cpu_cache_load_asyn: Success")


        print(f"[Timing] Total function time: {(end_4 - start_1) * 1000:.2f} ms")
        print(f"[Timing] Unhit CPU processing time: {(end_1 - start_1) * 1000:.2f} ms")
        print(f"[Timing] Unhit GPU dtype transfer time: {(end_2 - start_2):.2f} ms")
        print(f"[Timing] Cached GPU transfer time: {(end_3 - start_3) * 1000:.2f} ms")
        print(f"[Timing] Unhit GPU transfer and concat time: {(end_4 - start_4) * 1000:.2f} ms")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        return group_final_k, group_final_v, group_unhit
    
    def cpu_cache_load_asyn(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype, cpu_cache_load_asyn):

        cur_cache = self._caches[layer_id]
        group_cached_k, group_cached_v = cur_cache.get_cached_kv()

        group_final_k = []
        group_final_v = []

        
        torch.cuda.synchronize()
        start_0 = time.time()

        prefetch_idx_vector = prefetch_idx.squeeze(1).tolist()
        # k_vector = all_k.tolist()
        # v_vector = all_v.tolist()

        k_flat_vector = all_k.view(-1).tolist()
        v_flat_vector = all_v.view(-1).tolist()

        seq = prefetch_idx.shape[0]
        n = all_k.shape[0]
        bh_total = all_k.shape[1]
        d = all_k.shape[2]

        
        torch.cuda.synchronize()
        end_0 = time.time()

        print(f"[Timing] tensor to list: {(end_0 - start_0) * 1000:.2f} ms")
        
        torch.cuda.synchronize()
        start_1 = time.time()
        
        # future_unhit = self.executor.submit(cur_cache.get_unhit_kv_vector, prefetch_idx.squeeze(1).tolist(), all_k.tolist(), all_v.tolist())
        # future_unhit = self.executor.submit(cur_cache.get_unhit_kv_list, prefetch_idx_vector, k_vector, v_vector)
        # group_unhit_k_vector, group_unhit_v_vector, group_unhit = future_unhit.result()


        # group_unhit_k_vector, group_unhit_v_vector, group_unhit = cur_cache.get_unhit_kv_list(prefetch_idx_vector, k_vector, v_vector)
        # group_unhit_k_vector, group_unhit_v_vector, group_unhit = cur_cache.get_unhit_kv_list_test(prefetch_idx_vector, k_vector, v_vector)
        # group_unhit_k_vector, group_unhit_v_vector, group_unhit = cur_cache.get_unhit_kv_tensor(prefetch_idx, all_k, all_v)

        ### final
        # group_unhit_k_vector, group_unhit_v_vector, group_unhit = cur_cache.get_unhit_kv_tensor(prefetch_idx, all_k, all_v)
        
        # group_unhit_k_vector, group_unhit_v_vector, group_unhit = cur_cache.get_unhit_kv_list_test(prefetch_idx_vector, k_flat_vector, v_flat_vector, seq, n, bh_total, d)
        
        ################################### new tensor
        empty_group_keys = []
        empty_group_values = []
        n = all_k.shape[0]
        d = all_k.shape[-1]
        for i in range(len(class_group_ids)):
            tmp_key = torch.empty((n, len(class_group_ids[i]), d), dtype=all_k.dtype, device=all_k.device)
            tmp_value = torch.empty((n, len(class_group_ids[i]), d), dtype=all_k.dtype, device=all_k.device)

            empty_group_keys.append(tmp_key)
            empty_group_values.append(tmp_value)
        
        if prefetch_idx.ndim == 3:
            prefetch_idx_int = prefetch_idx.squeeze(1).to(torch.int32)
        else:
            prefetch_idx_int = prefetch_idx.to(torch.int32)

        print("prefetch_idx_int shape = ", prefetch_idx_int.shape)


        group_unhit = cur_cache.get_unhit_kv_tensor_v3(prefetch_idx_int, all_k.clone(), all_v.clone(), empty_group_keys, empty_group_values)
        group_unhit_k = []
        group_unhit_v = []
        for i in range(len(group_unhit)):
            tmp_unhit_len = len(group_unhit[i])
            tmp_unhit_k = empty_group_keys[i][:tmp_unhit_len, :, :].contiguous()
            tmp_unhit_v = empty_group_values[i][:tmp_unhit_len, :, :].contiguous()
            group_unhit_k.append(tmp_unhit_k)
            group_unhit_v.append(tmp_unhit_v)
        

        ################################### new tensor
        
        torch.cuda.synchronize()
        end_1 = time.time()


        # Step 3: 等待 CPU 计算结束
        # group_unhit_k, group_unhit_v, group_unhit = future_unhit.result()
        
        torch.cuda.synchronize()
        start_2 = time.time()
        # try:
        #     group_unhit_k = [torch.tensor(unhit_k, dtype=dtype) for unhit_k in group_unhit_k_vector]
        #     group_unhit_v = [torch.tensor(unhit_v, dtype=dtype) for unhit_v in group_unhit_v_vector]

        # except Exception as e:
        #     print(f"future_unhit error: {e}")
        print("test")
        
        torch.cuda.synchronize()
        end_2 = time.time()

        
        # Step 2: 并行开始 cached_k/v 的 GPU 传输（异步）
        torch.cuda.synchronize()
        start_3 = time.time()
        with torch.cuda.stream(transfer_stream):
            group_cached_gpu_k = [k.to(dtype).cuda(non_blocking=True) for k in group_cached_k]
            group_cached_gpu_v = [v.to(dtype).cuda(non_blocking=True) for v in group_cached_v]
        
        
        transfer_stream.synchronize() 
        torch.cuda.synchronize()
        end_3 = time.time()


        # Step 4: unhit_k/v -> GPU 传输 + 拼接（仍在 transfer_stream）
        torch.cuda.synchronize()
        start_4 = time.time()

        with torch.cuda.stream(transfer_stream):
            unhit_gpu_k = [k.cuda(non_blocking=True) for k in group_unhit_k]
            unhit_gpu_v = [v.cuda(non_blocking=True) for v in group_unhit_v]

            for i in range(len(group_cached_k)):
                final_k = torch.cat([group_cached_gpu_k[i], unhit_gpu_k[i]], dim=0)
                final_v = torch.cat([group_cached_gpu_v[i], unhit_gpu_v[i]], dim=0)
                group_final_k.append(final_k)
                group_final_v.append(final_v)
            
        
        transfer_stream.synchronize() 
        torch.cuda.synchronize()
        end_4 = time.time()

        
        # print("CacheManager cpu_cache_load_asyn: T2")
        # transfer_stream.synchronize()        
        # print("CacheManager cpu_cache_load_asyn: Success")


        print(f"[Timing] Total function time: {(end_4 - start_1) * 1000:.2f} ms")
        print(f"[Timing] Unhit CPU processing time: {(end_1 - start_1) * 1000:.2f} ms")
        print(f"[Timing] Unhit GPU dtype transfer time: {(end_2 - start_2):.2f} ms")
        print(f"[Timing] Cached GPU transfer time: {(end_3 - start_3) * 1000:.2f} ms")
        print(f"[Timing] Unhit GPU transfer and concat time: {(end_4 - start_4) * 1000:.2f} ms")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

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
    
    # TODO: change to be a real async function can be called
    async def update_cpu_cache_asyn(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v):
        if self._update_pred[layer_id] < self._update_pred:
            return 0, None
        self.update_pred[layer_id] = 0
        
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            self.executor,
            self._cache[layer_id].asyn_update_cache,
            prefetch_idx,
            keys,
            values
        )

        return 1, result

    def generate_class_group_ids(self, layer_id, batch_size):
        """
        Require static offline head group
        """
        # print("layer id = ", layer_id, type(layer_id))
        # print("self._basic_group_head_ids[layer_id] = ", self._basic_group_head_ids[layer_id])

        cur_basic_head_ids = self._basic_group_head_ids[layer_id]
        cur_head_num = self.layer_head_number
        final_head_ids = [[] for g in range(len(cur_basic_head_ids))]
        
        for b in range(batch_size): # 每个batch的head id 依次添加
            for g in range(len(cur_basic_head_ids)): # 遍历每个group
                for head_id in cur_basic_head_ids[g]: # 添加所有的head id，最大值为 bh*cur_head_num
                    final_head_ids[g].append(head_id + b*cur_head_num)

        return final_head_ids


    # 初始化c++模块中的cache
    def create_cache_instance(self, device, layer_id, batch_size, head_num, sparse_len, hidden_size=128):
        """ Initial the cache
        Call for every new batch of requests
        Building cache for every decoder layer.
        """
        if len(self._basic_group_head_ids) == 0:
            raise ValueError("Python ERROR! [create_cache_instance] basic_group_ids was not initialed!")

        # 构造 class group id
        new_class_group_id = self.generate_class_group_ids(layer_id, batch_size)
        new_class_group_id_tensor = [
            torch.tensor(head_id_list, dtype=torch.int32).to(device) for head_id_list in new_class_group_id    
        ]
        
        bh = batch_size*head_num
        cache_shape = (bh, sparse_len, hidden_size)
        cache_idx = torch.randint(low=0, high=sparse_len, size=(sparse_len, 1, batch_size*head_num)).to(torch.int32)

        # 初始化一个cache
        cache = _C.CPUCache(bh, cache_idx, cache_shape, new_class_group_id)
        
        return cache, new_class_group_id_tensor


##################################### 传统方法

def select_kv(prefetch_idx, k_cache, v_cache):
    """Selects and aggregates critical KV caches using speculated indices

    On the decoding stage, aggregates the critical KV caches corresponding to
    the speculated prefetch index using embedding function.

    Args:
        prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
        k_cache: Key cache (n, bh, d)
        v_cache: Value cache (n, bh, d)

    Returns:
        selected_k: selected key cache (n', bh, d)
        selected_v: selected value cache (n', bh, d)
    """

    prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
    ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1])[None, :]
    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
    return selected_k, selected_v





# 创建cache instance

class_group_ids = [
    [27, 44, 52, 56],
    [8, 11, 13, 14, 15, 17, 21, 26, 28, 29, 32, 34, 39, 43, 45, 46, 47, 49, 55, 62],
    [2, 3, 4, 6, 7, 12, 16, 18, 19, 20, 22, 23, 25, 30, 31, 33, 36, 38, 41, 42, 48, 50, 51, 53, 58, 60, 61, 63],
    [0, 1, 5, 9, 10, 24, 35, 37, 40, 54, 57, 59]
]

basic_group_head_ids = {
    0: class_group_ids
}


N = 7800
N_p = 7800//2
d = 128
bh = 64

idx_shape = (N_p, 1, bh)
kv_shape = (N, bh, d)
cache_shape = (N_p, bh, d)

all_k = torch.randint(low=0, high=10000, size=kv_shape).cpu().to(torch.float16)
all_v = all_k + 1
cache_idx = torch.randint(low=0, high=N, size=idx_shape).to(torch.int32)


transfer_stream = torch.cuda.Stream()


###### Cache Init
cmanager = CacheManager(basic_group_head_ids, 1)
cmanager.add_cache(
    device='cuda:0',
    layer_id=0, 
    batch_size=1, 
    head_num=bh, 
    sparse_len=N_p, 
    hidden_size=d
)

print("Init success")

#### load test data
data_name = f"./tmp/test2_layer{0}.pt"
layer_idx = torch.load(data_name)
decode_idxs = process_idx(layer_idx) 

cur_idx_list = decode_idxs
test_num = len(cur_idx_list)

# test
test_num = 4

K = 4

all_unhits = []

cache_load_cost = []
update_cost = []
direct_load_cost = []

#### cpud load test 

for i in range(test_num):
    if i % K == 0:
        # 直接加载并且更新cache
        torch.cuda.synchronize()
        start_u = time.time()

        group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = cmanager.load_and_update_cache(0, transfer_stream, cur_idx_list[i], all_k, all_v)
        # stat = _C.update_group_cache(cur_idx_list[i], group_cpu_k, group_cpu_v)
        
        all_unhits.append([[]])
        
        torch.cuda.synchronize()
        end_u = time.time()
        update_cost.append(end_u-start_u)

    else:
        torch.cuda.synchronize()
        start1 = time.time()

        gpu_k, gpu_v, unhit_list = cmanager.cpu_cache_load_asyn(0, transfer_stream, cur_idx_list[i], all_k, all_v, all_k.dtype,  class_group_ids)
        
        torch.cuda.synchronize()
        end1 = time.time()
        cache_load_cost.append(end1-start1)
        all_unhits.append(unhit_list)

print("cache load cost = ", sum(cache_load_cost)/len(cache_load_cost))


### direct load cost
for i in range(test_num):
    torch.cuda.synchronize()
    start2 = time.time()

    gpu_k, gpu_v, cpu_k, cpu_v = _C.direct_load(cur_idx_list[i], all_k, all_v)

    torch.cuda.synchronize()
    end2 = time.time()

    direct_load_cost.append(end2-start2)

print("direct load cost = ", sum(direct_load_cost)/len(direct_load_cost))


unhit_records = []
all_require = bh*N_p

for i in range(len(all_unhits)):
    unhit = all_unhits[i]
    sum_cnt, max_cnt = get_unhit_num(unhit)
    unhit_records.append(sum_cnt)

    print(f"avg_unhit_rate = {sum_cnt/all_require}; max_unhit_rate = {max_cnt/N_p};")