import torch
import torch.nn.functional as F
import my_cache_load._C as _C
from concurrent.futures import ThreadPoolExecutor

import time
import json

from torch.profiler import profile, record_function, ProfilerActivity


class CacheManager:
    def __init__(self, basic_group_head_ids, layer_head_num, update_pred=4):
        """
        初始化 CacheManager，使用字典保存不同 layer_id 对应的缓存数据。
        """
        self._caches = {}  # key: layer_id, value: cache data
        self._update_recode = {}
        self._update_pred = update_pred
        
        self._basic_group_head_ids = basic_group_head_ids # key: layer_id, value: layer group head ids
        self._cur_group_ids = {} # e.g. {1:[[1,2],[3,4],[5,6]]}

        # print("init self._basic_group_head_ids = ", self._basic_group_head_ids)
        self.layer_head_number = layer_head_num
        self.executor = ThreadPoolExecutor()  # 可复用线程池

        self._layer_tmp_key = None
        self._layer_tmp_value = None
    

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
        self._update_recode[layer_id] = self._update_pred+1 # 第一次load必须更新
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


    def cpu_cache_load_asyn(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype):
        cur_cache = self._caches[layer_id]
        cur_group_ids = self._cur_group_ids[layer_id]
        group_cached_k, group_cached_v = cur_cache.get_cached_kv()

        group_final_k = []
        group_final_v = []
        
        # print("CacheManager cpu_cache_load_asyn: T0")

        # step 0: 初始化结果空间
        empty_group_keys = []
        empty_group_values = []
        n = all_k.shape[0]
        d = all_k.shape[-1]

        # # 生成候选区间
        for i in range(len(cur_group_ids)):
            tmp_key = torch.empty((n, len(cur_group_ids[i]), d), dtype=all_k.dtype, device=all_k.device)
            tmp_value = torch.empty((n, len(cur_group_ids[i]), d), dtype=all_k.dtype, device=all_k.device)

            empty_group_keys.append(tmp_key)
            empty_group_values.append(tmp_value)
        prefetch_idx_int = prefetch_idx.squeeze(1).to(torch.int32)
        

        # Step 1: 提前异步提交 CPU 计算任务
        future_unhit = self.executor.submit(cur_cache.get_unhit_kv_tensor_v3, prefetch_idx_int, all_k, all_v, empty_group_keys, empty_group_values)

        # Step 2: 并行开始 cached_k/v 的 GPU 传输（异步）
        with torch.cuda.stream(transfer_stream):
            group_cached_gpu_k = [k.to(dtype).cuda(non_blocking=True) for k in group_cached_k]
            group_cached_gpu_v = [v.to(dtype).cuda(non_blocking=True) for v in group_cached_v]
        
        # print("CacheManager cpu_cache_load_asyn: tensor gpu load")

        # Step 3: 等待 CPU 计算结束
        try:
            group_unhit = future_unhit.result()
        except Exception as e:
            print(f"future_unhit error: {e}")

        
        group_unhit_k = []
        group_unhit_v = []
        for i in range(len(group_unhit)):
            tmp_unhit_len = len(group_unhit[i])
            tmp_unhit_k = empty_group_keys[i][:tmp_unhit_len, :, :]
            tmp_unhit_v = empty_group_values[i][:tmp_unhit_len, :, :]
            group_unhit_k.append(tmp_unhit_k)
            group_unhit_v.append(tmp_unhit_v)
        

        # print("CacheManager cpu_cache_load_asyn: T1")

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

    def load_and_update_cache(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype):
        group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = self._caches[layer_id].generate_cache(prefetch_idx, all_k, all_v)

        print(f"[load_and_update_cache] len group_gpu_k = {len(group_gpu_k)}")

        stat = self._caches[layer_id].update_group_cache(prefetch_idx, group_cpu_k, group_cpu_v)
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

    # 统一的更新和加载接口
    def unified_load_api(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype):
        # 判断是否需要更新
        if self._update_recode[layer_id] >= self._update_pred:
            # 直接更新cache
            self._update_recode[layer_id] = 0
            group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = self.load_and_update_cache(layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype)
            return (group_gpu_k, group_gpu_v, None)
        else:
            self._update_recode[layer_id] += 1
            group_final_k, group_final_v, group_unhit = self.cpu_cache_load_asyn(layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype)
            return (group_final_k, group_final_v, group_unhit)




torch.cuda.init()

def process_idx(layer_idx):
    # Step 1: 调整维度顺序，变为 (layer_num, N_p, bh)
    tensor_permuted = layer_idx.permute(0, 2, 1)

    # Step 2: 添加新维度，变为 (layer_num, N_p, 1, bh)
    tensor_unsqueezed = tensor_permuted.unsqueeze(2)

    # Step 3: 拆分张量，得到 layer_num 个形状为 (N_p, 1, bh) 的张量
    tensors_split = tensor_unsqueezed.unbind(dim=0)

    return tensors_split

def get_group_unhit_rate(group_list, token_len):
    unhit_num = 0
    max_num = 0

    group_max_unhit_rate = []
    group_avg_unhit_rate = []
    for unhit_list in group_list:
        cur_unhit_rate = []
        for u in unhit_list:
            l = len(u)
            cur_unhit_rate.append(l/token_len)
        max_unhit_rate = max(cur_unhit_rate)
        avg_unhit_rate = sum(cur_unhit_rate)/len(cur_unhit_rate)

        # print(f"max_unhit_rate = {max_unhit_rate}, avg_unhit_rate = {avg_unhit_rate}")

        group_max_unhit_rate.append(max_unhit_rate)
        group_avg_unhit_rate.append(avg_unhit_rate)
        
    return group_max_unhit_rate, group_avg_unhit_rate


############################## begin run


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


K = 4


###### Cache Init
cmanager = CacheManager(basic_group_head_ids, 1, K)
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
test_num = 100
all_unhits = []

cache_load_cost = []
update_cost = []
direct_load_cost = []

##### warm up
for i in range(test_num):
    gpu_k, gpu_v, cpu_k, cpu_v = _C.direct_load(cur_idx_list[i], all_k, all_v)

#### cpu load 

for i in range(test_num):
    torch.cuda.synchronize()
    start1 = time.time()

    gpu_k, gpu_v, unhit_list = cmanager.unified_load_api(0, transfer_stream, cur_idx_list[i], all_k, all_v, all_k.dtype)
    
    torch.cuda.synchronize()
    end1 = time.time()
    cache_load_cost.append(end1-start1)
    if unhit_list != None:
        all_unhits.append(unhit_list)

print("[old] all cache load cost = ", sum(cache_load_cost)/len(cache_load_cost))

update_cost = []
pure_cost = []
for i in range(len(cache_load_cost)):
    if i % K == 0:
        update_cost.append(cache_load_cost[i])
    else:
        pure_cost.append(cache_load_cost[i])

# print("[old] update cache cost = ", sum(update_cost)/len(update_cost))
# print("[old] pure load cost = ", sum(pure_cost)/len(pure_cost))



#### cpu load test 
# for i in range(test_num):
#     torch.cuda.synchronize()
#     start1 = time.time()

#     gpu_k, gpu_v, unhit_list = cmanager.unified_load_api_test(0, transfer_stream, cur_idx_list[i], all_k, all_v, all_k.dtype)
    
#     torch.cuda.synchronize()
#     end1 = time.time()
#     cache_load_cost.append(end1-start1)
#     if unhit_list != None:
#         all_unhits.append(unhit_list)

# print("[test] all cache load cost = ", sum(cache_load_cost)/len(cache_load_cost), "all = ", cache_load_cost)

# update_cost = []
# pure_cost = []
# for i in range(len(cache_load_cost)):
#     if i % K == 0:
#         update_cost.append(cache_load_cost[i])
#     else:
#         pure_cost.append(cache_load_cost[i])

# print("[test] update cache cost = ", sum(update_cost)/len(update_cost))
# print("[test] pure load cost = ", sum(pure_cost)/len(pure_cost))

### direct load cost
for i in range(test_num):
    torch.cuda.synchronize()
    start2 = time.time()

    gpu_k, gpu_v, cpu_k, cpu_v = _C.direct_load(cur_idx_list[i], all_k, all_v)

    torch.cuda.synchronize()
    end2 = time.time()

    direct_load_cost.append(end2-start2)

print("[direct] direct load cost = ", sum(direct_load_cost)/len(direct_load_cost))


for i in range(len(all_unhits)):
    unhit = all_unhits[i]
    max_rates, avg_rates = get_group_unhit_rate(unhit, N_p)

    # print(f"###########\n avg_rates={avg_rates} \n max_rates={max_rates}")