import torch
import torch.nn.functional as F
import my_cache_load._C as _C
from concurrent.futures import ThreadPoolExecutor

import sys


# 原函数定义（确保可运行）
def select_kv(prefetch_idx, k_cache, v_cache):
    prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
    ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1])[None, :].to(k_cache.device)
    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
    return selected_k, selected_v


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

        # unhit kv pinned memory
        self._pinned_unhit_k_list = []
        self._pinned_unhit_v_list = []
        self._pinned_n_p = 0

        # pinned space for cached tensor
        self._pinned_cached_k_list = {}
        self._pinned_cached_v_list = {}
        # self._pinned_cache_shape = {}

        # gpu cache
        self._use_gpu_cache = {} # should be initial in create
        self._gpu_cached_group_kv = {}

        # data ana
        self.cnt = 0
    

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

        # 在这里决定他是否要使用gpu cache
        # self._use_gpu_cache[layer_id] = True

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


    def cpu_cache_load_asyn(self, layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v, dtype):
        cur_cache = self._caches[layer_id]
        cur_group_ids = self._cur_group_ids[layer_id]
        group_cached_k, group_cached_v = cur_cache.get_cached_kv()

        group_final_k = []
        group_final_v = []
        
        # print("CacheManager cpu_cache_load_asyn: T0")


        # step 0: 初始化结果空间
        # empty_group_keys = []
        # empty_group_values = []
        n_p = prefetch_idx.shape[0]
        d = all_k.shape[-1]

        # # generate pinned space for unhit_kv
        # tensor_shape = (n_p, len(cur_group_ids[0]), d)
        if (self._pinned_unhit_k_list == []) or (self._pinned_n_p < n_p):
            if self._pinned_unhit_k_list != []:
                # for i in range(len(self._pinned_unhit_k_list)):
                del self._pinned_unhit_k_list
                del self._pinned_unhit_v_list
                self._pinned_unhit_k_list = []
                self._pinned_unhit_v_list = []
            
            for i in range(len(cur_group_ids)):
                cur_l = len(cur_group_ids[i])
                tmp_key = torch.empty((n_p, cur_l, d), dtype=all_k.dtype, device=all_k.device, pin_memory=True)
                tmp_value = torch.empty((n_p, cur_l, d), dtype=all_k.dtype, device=all_k.device, pin_memory=True)
                
                self._pinned_unhit_k_list.append(tmp_key)
                self._pinned_unhit_v_list.append(tmp_value)

        prefetch_idx_int = prefetch_idx.squeeze(1).to(torch.int32)
        pad_idx_list = prefetch_idx[0][0].tolist()
        

        # Step 1: 提前异步提交 CPU 计算任务
        # future_unhit = self.executor.submit(cur_cache.get_unhit_kv_tensor_v3, prefetch_idx_int, all_k, all_v, self._pinned_unhit_k_list, self._pinned_unhit_v_list)

        future_unhit = self.executor.submit(cur_cache.get_unhit_kv_tensor_v4, prefetch_idx_int, pad_idx_list, all_k, all_v, self._pinned_unhit_k_list, self._pinned_unhit_v_list)

        # Step 2: 并行开始 cached_k/v 的 GPU 传输（异步）
        with torch.cuda.stream(transfer_stream):
            group_cached_gpu_k = [k.cuda(non_blocking=True) for k in group_cached_k]
            group_cached_gpu_v = [v.cuda(non_blocking=True) for v in group_cached_v]
        
        # Step 3: 等待 CPU 计算结束
        try:
            group_unhit = future_unhit.result()
        except Exception as e:
            print(f"future_unhit error: {e}")

        
        group_unhit_k = []
        group_unhit_v = []
        for i in range(len(group_unhit)):
            # tmp_unhit_len = len(group_unhit[i][0])
            tmp_unhit_len = max([len(unhit) for unhit in group_unhit[i]])
            tmp_unhit_k = self._pinned_unhit_k_list[i][:tmp_unhit_len, :, :]
            tmp_unhit_v = self._pinned_unhit_v_list[i][:tmp_unhit_len, :, :]
            group_unhit_k.append(tmp_unhit_k)
            group_unhit_v.append(tmp_unhit_v)
        

        
        # print("################################# cpu_load Layer #", layer_id)
        # print("prefetch_idx shape =", prefetch_idx.shape)
        # # group_shape = [k.shape for k in group_cached_k]
        # # print("cached kv shape = ", group_shape)
        # group_unhit_shape = [unhit.shape for unhit in group_unhit_k]
        # print("group_unhit_shape = ", group_unhit_shape)
        

        # print("CacheManager cpu_cache_load_asyn: T1")

        # Step 4: unhit_k/v -> GPU 传输 + 拼接（仍在 transfer_stream）
        with torch.cuda.stream(transfer_stream):
            unhit_gpu_k = [k.cuda(non_blocking=True) for k in group_unhit_k]
            unhit_gpu_v = [v.cuda(non_blocking=True) for v in group_unhit_v]

            for i in range(len(group_cached_k)):
                group_final_k.append((group_cached_gpu_k[i], unhit_gpu_k[i]))
                group_final_v.append((group_cached_gpu_v[i], unhit_gpu_v[i]))

        # transfer_stream.synchronize()
        
        # print("################################# cpu_load")

        return group_final_k, group_final_v, group_unhit


    def gpu_cache_load_asyn(self, layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v):
        cur_cache = self._caches[layer_id]
        cur_group_ids = self._cur_group_ids[layer_id]

        group_cached_gpu_k, group_cached_gpu_v = self._gpu_cached_group_kv[layer_id]

        # result dtype = [(gpu_cached_k, gpu_cached_v)] (group_num)
        group_final_k = []
        group_final_v = []

        # step 0: 初始化结果空间
        n_p = prefetch_idx.shape[0]
        d = all_k.shape[-1]

        # # generate pinned space for unhit_kv
        # tensor_shape = (n_p, len(cur_group_ids[0]), d)
        if (self._pinned_unhit_k_list == []) or (self._pinned_n_p < n_p):
            if self._pinned_unhit_k_list != []:
                # for i in range(len(self._pinned_unhit_k_list)):
                del self._pinned_unhit_k_list
                del self._pinned_unhit_v_list
                self._pinned_unhit_k_list = []
                self._pinned_unhit_v_list = []
            
            for i in range(len(cur_group_ids)):
                cur_l = len(cur_group_ids[i])
                tmp_key = torch.empty((n_p, cur_l, d), dtype=all_k.dtype, device=all_k.device, pin_memory=True)
                tmp_value = torch.empty((n_p, cur_l, d), dtype=all_k.dtype, device=all_k.device, pin_memory=True)
                
                self._pinned_unhit_k_list.append(tmp_key)
                self._pinned_unhit_v_list.append(tmp_value)

        prefetch_idx_int = prefetch_idx.squeeze(1).to(torch.int32)

        # Step 1: 获取未命中KV
        group_unhit = cur_cache.get_unhit_kv_tensor_v3(
            prefetch_idx_int, 
            all_k, 
            all_v, 
            self._pinned_unhit_k_list, 
            self._pinned_unhit_v_list
        )

        # Step 2: 获取传输完成的结果
        group_unhit_k = []
        group_unhit_v = []
        for i in range(len(group_unhit)):
            tmp_unhit_len = len(group_unhit[i][0])
            tmp_unhit_k = self._pinned_unhit_k_list[i][:tmp_unhit_len, :, :]
            tmp_unhit_v = self._pinned_unhit_v_list[i][:tmp_unhit_len, :, :]
            group_unhit_k.append(tmp_unhit_k)
            group_unhit_v.append(tmp_unhit_v)

        # Step 3: unhit_k/v -> GPU 传输 + 收集结果
        with torch.cuda.stream(transfer_stream):
            unhit_gpu_k = [k.cuda(non_blocking=True) for k in group_unhit_k]
            unhit_gpu_v = [v.cuda(non_blocking=True) for v in group_unhit_v]

            for i in range(len(group_cached_gpu_k)):
                # final_k = torch.cat([group_cached_gpu_k[i], unhit_gpu_k[i]], dim=0)
                # final_v = torch.cat([group_cached_gpu_v[i], unhit_gpu_v[i]], dim=0)
                group_final_k.append((group_cached_gpu_k[i], unhit_gpu_k[i]))
                group_final_v.append((group_cached_gpu_k[i], unhit_gpu_k[i]))

        # transfer_stream.synchronize()

        return group_final_k, group_final_v, group_unhit
    
    
    def load_and_update_gpu_cached(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v):
        # print(f"Layer #{layer_id} use gpu cache")
        cur_group_ids = self._cur_group_ids[layer_id]

        # step 1: 如果预留的pinned tensor不够大, 创建新的pinned tensor
        if (layer_id not in self._pinned_cached_k_list) or (self._pinned_cached_k_list[layer_id][0].shape[0] < prefetch_idx.shape[0]):
            if layer_id in self._pinned_cached_k_list:
                del self._pinned_cached_k_list[layer_id]
                del self._pinned_cached_v_list[layer_id]
            
            cached_len = int(1.2*prefetch_idx.shape[0])
            # cached_len = 1000

            # change the pinned into group shapes
            self._pinned_cached_k_list[layer_id] = []
            self._pinned_cached_v_list[layer_id] = []
            for i in range(len(cur_group_ids)):
                tmp_cached_bh = len(cur_group_ids[i])
                tmp_cached_shape = [cached_len, tmp_cached_bh, all_k.shape[2]]
                self._pinned_cached_k_list[layer_id].append(torch.empty(tmp_cached_shape, dtype=all_k.dtype, device=all_k.device, pin_memory=True))
                self._pinned_cached_v_list[layer_id].append(torch.empty(tmp_cached_shape, dtype=all_k.dtype, device=all_k.device, pin_memory=True))
                
            # self._pinned_cache_shape[layer_id] = (prefetch_idx.shape[0], tmp_cached_bh, all_k.shape[2])

        
        # step 2: 将pinned tensor切分成group大小
        cur_cached_len = prefetch_idx.shape[0]

        final_cpu_k = []
        final_cpu_v = []
        bh_start = 0
        for i in range(len(cur_group_ids)):
            group_len = len(cur_group_ids[i])

            cur_pinned_cached_k = self._pinned_cached_k_list[layer_id][i]
            cur_pinned_cached_v = self._pinned_cached_v_list[layer_id][i]

            final_cpu_k.append(cur_pinned_cached_k[:cur_cached_len, :, :])
            final_cpu_v.append(cur_pinned_cached_v[:cur_cached_len, :, :])
            bh_start += group_len

        # Step 3: 获取新的kv数据
        if len(self._cur_group_ids[layer_id]) == 1:
            selected_k, selected_v = select_kv(prefetch_idx, all_k, all_v)
            group_cpu_k = [selected_k]
            group_cpu_v = [selected_v]
        else:
            group_cpu_k, group_cpu_v = self._caches[layer_id].generate_update_cache(prefetch_idx, all_k, all_v)

        # Step 4: 将数据拷贝到pinned_tensor中
        for i in range(len(cur_group_ids)):
            final_cpu_k[i].copy_(group_cpu_k[i])
            final_cpu_v[i].copy_(group_cpu_v[i])

        # Step 5: 提前cpu cache更新任务
        # stat = self._caches[layer_id].update_group_cache(prefetch_idx, group_cpu_k, group_cpu_v)
        cur_cache = self._caches[layer_id]
        future_unhit = self.executor.submit(cur_cache.update_group_cache, prefetch_idx, final_cpu_k, final_cpu_v)
        
        # Step 6: 并行开始 KV 的 GPU 传输（异步）
        with torch.cuda.stream(transfer_stream):
            group_gpu_k = [k.cuda(non_blocking=True) for k in final_cpu_k]
            group_gpu_v = [v.cuda(non_blocking=True) for v in final_cpu_v]
        
        # Step 7: 等待 CPU 计算结束
        try:
            stat = future_unhit.result()
            if stat != 0:
                raise ValueError("Python ERROR! [load_and_update_cache] Update cache Fail!")
        except Exception as e:
            print(f"future_unhit error: {e}")

        # 更新gpu cache
        if layer_id in self._gpu_cached_group_kv:
            del self._gpu_cached_group_kv[layer_id]
        self._gpu_cached_group_kv[layer_id] = (group_gpu_k, group_gpu_v)

        return group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v


    def load_and_update_cpu_cache(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype):

        cur_group_ids = self._cur_group_ids[layer_id]

        # step 1: 如果预留的pinned tensor不够大, 创建新的pinned tensor
        if (layer_id not in self._pinned_cached_k_list) or (self._pinned_cached_k_list[layer_id][0].shape[0] < prefetch_idx.shape[0]):
            if layer_id in self._pinned_cached_k_list:
                del self._pinned_cached_k_list[layer_id]
                del self._pinned_cached_v_list[layer_id]
            
            # cached_len = int(1.2*prefetch_idx.shape[0])
            cached_len = prefetch_idx.shape[0]
            # cached_len = 1000

            # cached_len = all_k.shape[0]
            # cached_shape = [cached_len, all_k.shape[1], all_k.shape[2]]

            # change the pinned into group shapes
            self._pinned_cached_k_list[layer_id] = []
            self._pinned_cached_v_list[layer_id] = []
            for i in range(len(cur_group_ids)):
                tmp_cached_bh = len(cur_group_ids[i])
                tmp_cached_shape = [cached_len, tmp_cached_bh, all_k.shape[2]]
                self._pinned_cached_k_list[layer_id].append(torch.empty(tmp_cached_shape, dtype=all_k.dtype, device=all_k.device, pin_memory=True))
                self._pinned_cached_v_list[layer_id].append(torch.empty(tmp_cached_shape, dtype=all_k.dtype, device=all_k.device, pin_memory=True))

            # self._pinned_cache_shape[layer_id] = (prefetch_idx.shape[0], all_k.shape[1], all_k.shape[2])

        # step 2: 将pinned tensor切分成group大小
        cur_cached_len = prefetch_idx.shape[0]

        final_cpu_k = []
        final_cpu_v = []
        bh_start = 0
        for i in range(len(cur_group_ids)):
            group_len = len(cur_group_ids[i])

            cur_pinned_cached_k = self._pinned_cached_k_list[layer_id][i]
            cur_pinned_cached_v = self._pinned_cached_v_list[layer_id][i]

            final_cpu_k.append(cur_pinned_cached_k[:cur_cached_len, :, :])
            final_cpu_v.append(cur_pinned_cached_v[:cur_cached_len, :, :])
            bh_start += group_len

        # Step 3: 获取新的kv数据
        if len(self._cur_group_ids[layer_id]) == 1:
            selected_k, selected_v = select_kv(prefetch_idx, all_k, all_v)
            group_cpu_k = [selected_k]
            group_cpu_v = [selected_v]
        else:
            group_cpu_k, group_cpu_v = self._caches[layer_id].generate_update_cache(prefetch_idx, all_k, all_v)

        # Step 4: 将数据拷贝到pinned_tensor中
        for i in range(len(cur_group_ids)):
            final_cpu_k[i].copy_(group_cpu_k[i])
            final_cpu_v[i].copy_(group_cpu_v[i])
        
        # torch.cuda.synchronize() 

        # Step 5: 提前cpu cache更新任务
        # stat = self._caches[layer_id].update_group_cache(prefetch_idx, group_cpu_k, group_cpu_v)
        cur_cache = self._caches[layer_id]
        future_unhit = self.executor.submit(cur_cache.update_group_cache, prefetch_idx, final_cpu_k, final_cpu_v)

        # Step 6: 并行开始 KV 的 GPU 传输（异步）
        with torch.cuda.stream(transfer_stream):
            group_gpu_k = [k.cuda(non_blocking=True) for k in final_cpu_k]
            group_gpu_v = [v.cuda(non_blocking=True) for v in final_cpu_v]


        # tmp 保存kv并观察是否一致
        # torch.save(final_cpu_k[0], f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/keys_b16_l{layer_id}.pt")
        # torch.save(final_cpu_v[0], f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/values_b16_l{layer_id}.pt")
        # sys.exit(0)
        
        
        # Step 7: 等待 CPU 计算结束
        try:
            stat = future_unhit.result()
            if stat != 0:
                raise ValueError("Python ERROR! [load_and_update_cache] Update cache Fail!")
        except Exception as e:
            print(f"future_unhit error: {e}")

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

    # def direct_load(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v):
    #     cur_group_ids = self._cur_group_ids[layer_id]

    #     # step 1: 如果预留的pinned tensor不够大, 创建新的pinned tensor
    #     if (layer_id not in self._pinned_cached_k_list) or (self._pinned_cached_k_list[layer_id][0].shape[0] < prefetch_idx.shape[0]):
    #         if layer_id in self._pinned_cached_k_list:
    #             del self._pinned_cached_k_list[layer_id]
    #             del self._pinned_cached_v_list[layer_id]
            
    #         # cached_len = int(1.2*prefetch_idx.shape[0])
    #         cached_len = 1000

    #         # cached_len = all_k.shape[0]
    #         # cached_shape = [cached_len, all_k.shape[1], all_k.shape[2]]

    #         # change the pinned into group shapes
    #         self._pinned_cached_k_list[layer_id] = []
    #         self._pinned_cached_v_list[layer_id] = []
    #         for i in range(len(cur_group_ids)):
    #             tmp_cached_bh = len(cur_group_ids[i])
    #             tmp_cached_shape = [cached_len, tmp_cached_bh, all_k.shape[2]]
    #             self._pinned_cached_k_list[layer_id].append(torch.empty(tmp_cached_shape, dtype=all_k.dtype, device=all_k.device, pin_memory=True))
    #             self._pinned_cached_v_list[layer_id].append(torch.empty(tmp_cached_shape, dtype=all_k.dtype, device=all_k.device, pin_memory=True))

    #         # self._pinned_cache_shape[layer_id] = (prefetch_idx.shape[0], all_k.shape[1], all_k.shape[2])

    #     # step 2: 将pinned tensor切分成group大小
    #     cur_cached_len = prefetch_idx.shape[0]

    #     final_cpu_k = []
    #     final_cpu_v = []
    #     bh_start = 0
    #     for i in range(len(cur_group_ids)):
    #         group_len = len(cur_group_ids[i])

    #         cur_pinned_cached_k = self._pinned_cached_k_list[layer_id][i]
    #         cur_pinned_cached_v = self._pinned_cached_v_list[layer_id][i]

    #         final_cpu_k.append(cur_pinned_cached_k[:cur_cached_len, :, :])
    #         final_cpu_v.append(cur_pinned_cached_v[:cur_cached_len, :, :])
    #         bh_start += group_len

    #     # Step 3: 获取新的kv数据
    #     # if len(self._cur_group_ids[layer_id]) == 1:
    #     #     selected_k, selected_v = select_kv(prefetch_idx, all_k, all_v)
    #     #     group_cpu_k = [selected_k]
    #     #     group_cpu_v = [selected_v]
    #     # else:
    #     group_cpu_k, group_cpu_v = self._caches[layer_id].generate_update_cache(prefetch_idx, all_k, all_v)

    #     # Step 4: 将数据拷贝到pinned_tensor中
    #     for i in range(len(cur_group_ids)):
    #         final_cpu_k[i].copy_(group_cpu_k[i])
    #         final_cpu_v[i].copy_(group_cpu_v[i])

    #     # Step 5: 提前cpu cache更新任务
    #     # stat = self._caches[layer_id].update_group_cache(prefetch_idx, group_cpu_k, group_cpu_v)
    #     cur_cache = self._caches[layer_id]

    #     # Step 6: 并行开始 KV 的 GPU 传输（异步）
    #     with torch.cuda.stream(transfer_stream):
    #         group_gpu_k = [k.cuda(non_blocking=True) for k in final_cpu_k]
    #         group_gpu_v = [v.cuda(non_blocking=True) for v in final_cpu_v]
        
    #     return group_gpu_k, group_gpu_v



    # 统一的更新和加载接口
    def unified_load_api(self, layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v, dtype):
        # record prefetch_idx
        # path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_l{layer_id}_{cnt}.pt"
        # self.cnt += 1
        # torch.save(prefetch_idx, path)
        # print("layer id = ", layer_id)


        
        # if layer_id in self._pinned_cache_shape:
        #     cur_cached_cnt = self._pinned_cache_shape[layer_id][0]
        # else:
        #     cur_cached_cnt = 0
        # cur_cached_pred = int(1.1 * cur_cached_cnt)

        # print(f"layer_di={layer_id}, prefetch_idx.shape[0]={prefetch_idx.shape[0]}, cur_cached_pred={cur_cached_cnt}")
        # print(f"layer_di={layer_id}, prefetch_idx.shape[0]={prefetch_idx.shape[0]}, pad_idx.shape={pad_idx.shape}")

        
        # if int(layer_id) in [20]:
        # print(f"[CacheManager] layer #{layer_id} prefetch_idx shape = ", prefetch_idx.shape)

        # if prefetch_idx.shape[0] == 1000:
        #     group_gpu_k, group_gpu_v = self.direct_load(layer_id, transfer_stream, prefetch_idx, all_k, all_v)
        #     return (group_gpu_k, group_gpu_v, None)

        # el
        if layer_id in self._use_gpu_cache:
            # 判断是否需要更新
            if (self._update_recode[layer_id] >= self._update_pred):
            # if (self._update_recode[layer_id] >= self._update_pred) or (prefetch_idx.shape[0] > cur_cached_pred):
                # 直接更新cache
                self._update_recode[layer_id] = 0
                group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = self.load_and_update_gpu_cached(layer_id, transfer_stream, prefetch_idx, all_k, all_v)
                return (group_gpu_k, group_gpu_v, None)
            else:
                self._update_recode[layer_id] += 1
                group_final_k, group_final_v, group_unhit = self.gpu_cache_load_asyn(layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v)
                return (group_final_k, group_final_v, group_unhit)
        else:

            # 判断是否需要更新
            if (self._update_recode[layer_id] >= self._update_pred):
            # if (self._update_recode[layer_id] >= self._update_pred) or (prefetch_idx.shape[0] > cur_cached_pred):
                # 直接更新cache
                self._update_recode[layer_id] = 1
                group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = self.load_and_update_cpu_cache(layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype)
                return (group_gpu_k, group_gpu_v, None)
            else:
                self._update_recode[layer_id] += 1
                group_final_k, group_final_v, group_unhit = self.cpu_cache_load_asyn(layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v, dtype)
                
                return (group_final_k, group_final_v, group_unhit)



N = 20
N_p = 5
d = 4
batch_size = 2
head_num = 4
bh = batch_size * head_num
# bh = 64

idx_shape = (N_p, 1, bh)
kv_shape = (N, bh, d)
cache_shape = (N_p, bh, d)

all_k = torch.randint(low=0, high=10000, size=kv_shape).float().half().cpu()
all_v = all_k + 1

class_group_ids = [[i for i in range(bh)]]

# 初始化cache manager
Global_Cache_Manager = CacheManager({0:class_group_ids}, head_num)
Global_Cache_Manager.add_cache("cpu", 0, batch_size, head_num, N_p, d)

# 生成prefetch idx
prefetch_idx = torch.randint(0, N, (N_p, 1, bh))

######### 开始进行测试
cur_cache  = Global_Cache_Manager._caches[0]


# 普通获取kv
selected_k, selected_v = select_kv(prefetch_idx, all_k, all_v)

# cache获取KV
prefetch_idx_list = prefetch_idx.squeeze(1).tolist()

empty_k = torch.zeros_like(selected_k)
empty_v = torch.zeros_like(selected_v)

cur_cache.select_kv_tensor_v3(
    prefetch_idx_list, 
    all_k, 
    all_v, 
    [i for i in range(bh)],
    empty_k, 
    empty_v
)

print("success")

print("select_k == empty_k?", torch.equal(selected_k, empty_k))