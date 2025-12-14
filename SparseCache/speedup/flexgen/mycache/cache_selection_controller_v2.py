import torch
import torch.nn.functional as F
import my_cache_load._C as _C
from concurrent.futures import ThreadPoolExecutor

import sys

class CacheManager:
    def __init__(self, basic_group_head_ids, layer_head_num, update_pred=4):
        """
        初始化 CacheManager，使用字典保存不同 layer_id 对应的缓存数据。
        """
        self._caches = {}  # key: layer_id, value: cache data
        self._update_recode = {}

        # tmp change 
        update_pred = 9
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
        self.gpu_cache_pred = 2
        self.cpu_cache_pred = 1.8
        self._max_cpu_cached_len = 0
        self._max_gpu_cached_len = 0
        # self._max_cached_len = 0
        # self._pinned_cache_shape = {}

        # gpu cache
        self._use_gpu_cache = {} # should be initial in create
        self._gpu_cached_group_kv = {}

        # data ana
        self.cnt = 0

        # update according to max cache len
        self._require_update = {}
        self.max_unhit_rate = 0.4 # max 40% unhit kv
        # self.

    

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
        self._require_update[layer_id] = True # 第一次load必须更新

        # 在这里决定他是否要使用gpu cache
        self._use_gpu_cache[layer_id] = True

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
    
    def get_layer_cache(self, layer_id):
        return self._caches[layer_id]

    def cpu_cache_load_asyn(self, layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v, dtype):
        cur_cache = self._caches[layer_id]
        cur_group_ids = self._cur_group_ids[layer_id]
        group_cached_k, group_cached_v = cur_cache.get_cached_kv()

        group_final_k = []
        group_final_v = []
        
        # print("CacheManager cpu_cache_load_asyn: T0", flush=True)


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
        
        # print("CacheManager cpu_cache_load_asyn: T1 init success", flush=True)


        # Step 1: 提前异步提交 CPU 计算任务
        # future_unhit = self.executor.submit(cur_cache.get_unhit_kv_tensor_v3, prefetch_idx_int, all_k, all_v, self._pinned_unhit_k_list, self._pinned_unhit_v_list)

        future_unhit = self.executor.submit(cur_cache.get_unhit_kv_tensor_v5, prefetch_idx_int, pad_idx_list, all_k, all_v, self._pinned_unhit_k_list, self._pinned_unhit_v_list)
        
        # version 6, with max unhit limit
        # max_unhit_limit = int(self.max_unhit_rate * n_p) 
        # future_unhit = self.executor.submit(cur_cache.get_unhit_kv_tensor_v6, prefetch_idx_int, pad_idx_list, all_k, all_v, max_unhit_limit, self._pinned_unhit_k_list, self._pinned_unhit_v_list)

        # Step 2: 并行开始 cached_k/v 的 GPU 传输（异步）
        with torch.cuda.stream(transfer_stream):
            group_cached_gpu_k = [k.cuda(non_blocking=True) for k in group_cached_k]
            group_cached_gpu_v = [v.cuda(non_blocking=True) for v in group_cached_v]
        
        # transfer_stream.synchronize()
        # print("CacheManager cpu_cache_load_asyn: GPU transfer success", flush=True)

        # Step 3: 等待 CPU 计算结束
        try:
            group_unhit = future_unhit.result()
        except Exception as e:
            print(f"future_unhit error: {e}")

    
        # print("CacheManager cpu_cache_load_asyn: unhit get success", flush=True)

        ### group unhit
        # unhit_cmp = []
        # prefetch_idx_cmp = []

        # # for unhit cmp
        # for i in range(32):
        #     unhit1 = group_unhit[0][i]
        #     unhit2 = group_unhit[0][i+32]
        
        #     if len(unhit1) != len(unhit2):
        #         unhit_cmp.append(False)
        #         continue
                
        #     has = [(id in unhit2) for id in unhit1]
        #     if False in has:
        #         unhit_cmp.append(False)
        #         continue
            
        #     unhit_cmp.append(True)
        
        # # for prefetch id cmp
        # prefetch_idx_T = prefetch_idx_int.T
        # for i in range(32):
        #     pidx_h1 = prefetch_idx_T[i]
        #     pidx_h2 = prefetch_idx_T[i+32]
        #     prefetch_idx_cmp.append(torch.equal(pidx_h1, pidx_h2))
        
        # print("CacheManage: unhit_cmp = ", unhit_cmp)
        # print("CacheManage: prefetch_idx_cmp = ", prefetch_idx_cmp)

        
        group_unhit_k = []
        group_unhit_v = []
        for i in range(len(group_unhit)):
            # tmp_unhit_len = len(group_unhit[i][0])
            if isinstance(group_unhit[i][0], list):
                tmp_unhit_len = max([len(unhit) for unhit in group_unhit[i]])
            else:
                tmp_unhit_len = max(group_unhit[i])

            tmp_unhit_k = self._pinned_unhit_k_list[i][:tmp_unhit_len, :, :]
            tmp_unhit_v = self._pinned_unhit_v_list[i][:tmp_unhit_len, :, :]
            group_unhit_k.append(tmp_unhit_k)
            group_unhit_v.append(tmp_unhit_v)
        

        # print("tmp flush", flush=True)
        print("################################# cpu_load Layer #", layer_id)
        
        print("prefetch_idx shape =", prefetch_idx.shape)
        # group_shape = [k.shape for k in group_cached_k]
        # print("cached kv shape = ", group_shape)
        group_unhit_shape = [unhit.shape for unhit in group_unhit_k]
        print("group_unhit_shape = ", group_unhit_shape)
        

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

    def cpu_cache_load_asyn_v2(self, layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v, dtype):
        cur_cache = self._caches[layer_id]
        cur_group_ids = self._cur_group_ids[layer_id]
        group_cached_k, group_cached_v = cur_cache.get_cached_kv()

        group_final_k = []
        group_final_v = []
        
        # print("CacheManager cpu_cache_load_asyn: T0", flush=True)


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
        
        # print("CacheManager cpu_cache_load_asyn: T1 init success", flush=True)


        # Step 1: 提前异步提交 CPU 计算任务
        # future_unhit = self.executor.submit(cur_cache.get_unhit_kv_tensor_v3, prefetch_idx_int, all_k, all_v, self._pinned_unhit_k_list, self._pinned_unhit_v_list)

        future_unhit = self.executor.submit(cur_cache.get_unhit_kv_tensor_v5, prefetch_idx_int, pad_idx_list, all_k, all_v, self._pinned_unhit_k_list, self._pinned_unhit_v_list)
        
        # version 6, with max unhit limit
        # max_unhit_limit = int(self.max_unhit_rate * n_p) 
        # future_unhit = self.executor.submit(cur_cache.get_unhit_kv_tensor_v6, prefetch_idx_int, pad_idx_list, all_k, all_v, max_unhit_limit, self._pinned_unhit_k_list, self._pinned_unhit_v_list)

        # Step 2: 并行开始 cached_k/v 的 GPU 传输（异步）
        with torch.cuda.stream(transfer_stream):
            group_cached_gpu_k = [k.cuda(non_blocking=True) for k in group_cached_k]
            group_cached_gpu_v = [v.cuda(non_blocking=True) for v in group_cached_v]
        
        # transfer_stream.synchronize()
        # print("CacheManager cpu_cache_load_asyn: GPU transfer success", flush=True)

        # Step 3: 等待 CPU 计算结束
        try:
            group_unhit = future_unhit.result()
        except Exception as e:
            print(f"future_unhit error: {e}")

    
        # print("CacheManager cpu_cache_load_asyn: unhit get success", flush=True)

        ### group unhit
        # unhit_cmp = []
        # prefetch_idx_cmp = []

        # # for unhit cmp
        # for i in range(32):
        #     unhit1 = group_unhit[0][i]
        #     unhit2 = group_unhit[0][i+32]
        
        #     if len(unhit1) != len(unhit2):
        #         unhit_cmp.append(False)
        #         continue
                
        #     has = [(id in unhit2) for id in unhit1]
        #     if False in has:
        #         unhit_cmp.append(False)
        #         continue
            
        #     unhit_cmp.append(True)
        
        # # for prefetch id cmp
        # prefetch_idx_T = prefetch_idx_int.T
        # for i in range(32):
        #     pidx_h1 = prefetch_idx_T[i]
        #     pidx_h2 = prefetch_idx_T[i+32]
        #     prefetch_idx_cmp.append(torch.equal(pidx_h1, pidx_h2))
        
        # print("CacheManage: unhit_cmp = ", unhit_cmp)
        # print("CacheManage: prefetch_idx_cmp = ", prefetch_idx_cmp)

        
        group_unhit_k = []
        group_unhit_v = []
        for i in range(len(group_unhit)):
            # tmp_unhit_len = len(group_unhit[i][0])
            if isinstance(group_unhit[i][0], list):
                tmp_unhit_len = max([len(unhit) for unhit in group_unhit[i]])
            else:
                tmp_unhit_len = max(group_unhit[i])

            tmp_unhit_k = self._pinned_unhit_k_list[i][:tmp_unhit_len, :, :]
            tmp_unhit_v = self._pinned_unhit_v_list[i][:tmp_unhit_len, :, :]
            group_unhit_k.append(tmp_unhit_k)
            group_unhit_v.append(tmp_unhit_v)
        

        # print("tmp flush", flush=True)
        print("################################# cpu_load Layer #", layer_id)
        
        print("prefetch_idx shape =", prefetch_idx.shape)
        group_shape = [k.shape for k in group_cached_k]
        print("cached kv shape = ", group_shape)
        group_unhit_shape = [unhit.shape for unhit in group_unhit_k]
        print("group_unhit_shape = ", group_unhit_shape)
        

        # print("CacheManager cpu_cache_load_asyn: T1")

        # Step 4: unhit_k/v -> GPU 传输 + 拼接（仍在 transfer_stream）
        with torch.cuda.stream(transfer_stream):
            unhit_gpu_k = [k.cuda(non_blocking=True) for k in group_unhit_k]
            unhit_gpu_v = [v.cuda(non_blocking=True) for v in group_unhit_v]

            for i in range(len(group_cached_k)):
                group_final_k.append((group_cached_gpu_k[i], unhit_gpu_k[i]))
                group_final_v.append((group_cached_gpu_v[i], unhit_gpu_v[i]))

        # transfer_stream.synchronize()

        ################################################## 
        # Final step: 开始更新cpu cache
        # data: group_cached_k, group_unhit_k
        cur_cache_len = group_cached_k[0].shape[0]
        max_unhit_len = max([unhit_k.shape[0] for unhit_k in group_unhit_k])

        # 如果剩余cache空间足够，更新cache
        if self._max_cpu_cached_len >= (cur_cache_len + max_unhit_len):
            new_group_cached_k = []
            new_group_cached_v = []
            # new_group_unhit = []
            for i in range(len(group_cached_k)):
                cur_pinned_cached_k = self._pinned_cached_k_list[layer_id][i]
                cur_pinned_cached_v = self._pinned_cached_v_list[layer_id][i]

                cur_unhit_len = group_unhit_k[i].shape[0]
                cur_final_len = cur_cache_len + cur_unhit_len
                cur_pad_len = cur_unhit_len

                # cur_final_len = min(self._max_cached_len, cur_final_len)
                # cur_pad_len = cur_final_len - cur_cache_len

                cur_pinned_cached_k[cur_cache_len:cur_final_len, :, :] = group_unhit_k[i][:cur_pad_len, :, :]
                cur_pinned_cached_v[cur_cache_len:cur_final_len, :, :] = group_unhit_v[i][:cur_pad_len, :, :]

                new_group_cached_k.append(cur_pinned_cached_k[:cur_final_len])
                new_group_cached_v.append(cur_pinned_cached_v[:cur_final_len])
                # new_group_unhit.append([unhit[:cur_pad_len] for unhit in group_unhit[i]])

            # TODO: 完善direct_update_cache_map_with_unhit的功能
            # cur_cache.direct_update_cache_map_with_unhit(new_group_unhit[0])
            # cur_cache.direct_update_cache_map_with_unhit(group_unhit[0])
            cur_cache.direct_update_cache_map_with_group_unhit(group_unhit)
            cur_cache.direct_update_cache(new_group_cached_k, new_group_cached_v)
        else:
            # 剩余空间不够
            self._require_update[layer_id] = True
        
        # print("################################# cpu_load")

        return group_final_k, group_final_v, group_unhit


    def gpu_cache_load_asyn_v3(self, layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v):
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


        # Step 1: 获取未命中KV
        prefetch_idx_int = prefetch_idx.squeeze(1).to(torch.int32)
        pad_idx_list = prefetch_idx[0][0].tolist()
        group_unhit = cur_cache.get_unhit_kv_tensor_v5(
            prefetch_idx_int, 
            pad_idx_list,
            all_k, 
            all_v, 
            self._pinned_unhit_k_list, 
            self._pinned_unhit_v_list
        )

        # update cache map with unhit
        # cur_cache.direct_update_cache_map_with_unhit(group_unhit[0])
        cur_cache.direct_update_cache_map_with_group_unhit(group_unhit)


        # Step 2: 获取传输完成的结果
        group_unhit_k = []
        group_unhit_v = []
        for i in range(len(group_unhit)):
            # tmp_unhit_len = len(group_unhit[i][0])
            if isinstance(group_unhit[i][0], list):
                tmp_unhit_len = max([len(unhit) for unhit in group_unhit[i]])
            else:
                tmp_unhit_len = max(group_unhit[i])

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
        
        print("tmp flush", flush=True)
        print("################################# gpu_load Layer #", layer_id)
        print("prefetch_idx shape =", prefetch_idx.shape)
        group_cache_shape = [unhit.shape for unhit in group_cached_gpu_k]
        print("group_cache_shape = ", group_cache_shape)
        group_unhit_shape = [unhit.shape for unhit in group_unhit_k]
        print("group_unhit_shape = ", group_unhit_shape)

        # step4: 判断是否需要更新
        cur_cache_len = group_cached_gpu_k[0].shape[0]
        max_unhit_len = max([unhit_k.shape[0] for unhit_k in group_unhit_k])
        # 如果剩余cache空间足够，更新cache
        if self._max_gpu_cached_len < (cur_cache_len + max_unhit_len):
            print(f"require update = True, _max_cached_len={self._max_gpu_cached_len} cur_cache_len={cur_cache_len} max_unhit_len={max_unhit_len}", flush = True)
            self._require_update[layer_id] = True


        return group_final_k, group_final_v, group_unhit

    
    def gpu_cache_load_asyn_v4(self, layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v):
        cur_cache = self._caches[layer_id]
        cur_group_ids = self._cur_group_ids[layer_id]

        group_cached_gpu_k, group_cached_gpu_v = self._gpu_cached_group_kv[layer_id]

        group_final_k = []
        group_final_v = []

        # step 0: 初始化结果空间
        n_p = prefetch_idx.shape[0]
        d = all_k.shape[-1]

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

        # step 1: 获取unhit id
        prefetch_idx_int = prefetch_idx.squeeze(1).to(torch.int32)
        pad_idx_list = prefetch_idx[0][0].tolist()
        group_unhits, group_pad_idxs = cur_cache.get_one_group_unhit_kv_ids(prefetch_idx_int, pad_idx_list)

        # step 2: 逐个group获取kv
        
        group_final_k = []
        group_final_v = []

        group_max_unhit_len = []
        
        with torch.cuda.stream(transfer_stream):
            for group_id in range(len(cur_group_ids)):

                # 获取当前group的unhit id
                unhit_len_list = cur_cache.get_one_group_unhit_kv_v1(
                    group_unhits[group_id], 
                    group_pad_idxs[group_id], 
                    group_id,
                    all_k,
                    all_v,
                    self._pinned_unhit_k_list[group_id],
                    self._pinned_unhit_v_list[group_id])

                max_unhit_len = max(unhit_len_list)
                group_max_unhit_len.append(max_unhit_len)

                # 截取需要的tensor
                tmp_unhit_k = self._pinned_unhit_k_list[group_id][:max_unhit_len, :]
                tmp_unhit_v = self._pinned_unhit_v_list[group_id][:max_unhit_len, :]

                # 传输并记录tensor
                group_final_k.append((group_cached_gpu_k[group_id], tmp_unhit_k.cuda(non_blocking=True)))
                group_final_v.append((group_cached_gpu_k[group_id], tmp_unhit_v.cuda(non_blocking=True)))
    

        print("tmp flush", flush=True)
        print("################################# gpu_load Layer #", layer_id)
        print("prefetch_idx shape =", prefetch_idx.shape)
        group_cache_shape = [unhit.shape for unhit in group_cached_gpu_k]
        print("group_cache_shape = ", group_cache_shape)
        # group_unhit_shape = [unhit.shape for unhit in group_unhit_k]
        print("group_max_unhit_len = ", group_max_unhit_len)

        # step4: 判断是否需要更新
        cur_cache_len = max([gpu_k.shape[0] for gpu_k in group_cached_gpu_k])
        max_unhit_len = max(group_max_unhit_len)
        # 如果剩余cache空间足够，更新cache
        if self._max_gpu_cached_len < (cur_cache_len + max_unhit_len):
            print(f"require update = True, _max_cached_len={self._max_gpu_cached_len} cur_cache_len={cur_cache_len} max_unhit_len={max_unhit_len}", flush = True)
            self._require_update[layer_id] = True


        return group_final_k, group_final_v, group_unhits
 

    def gpu_cache_load_asyn_v2(self, layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v):
        cur_cache = self._caches[layer_id]
        cur_group_ids = self._cur_group_ids[layer_id]

        group_cached_gpu_k, group_cached_gpu_v = self._gpu_cached_group_kv[layer_id]

        group_final_k = []
        group_final_v = []

        # step 0: 初始化结果空间
        n_p = prefetch_idx.shape[0]
        d = all_k.shape[-1]

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

        # step 1: 获取unhit id
        prefetch_idx_int = prefetch_idx.squeeze(1).to(torch.int32)
        pad_idx_list = prefetch_idx[0][0].tolist()
        group_unhits, group_pad_idxs = cur_cache.get_one_group_unhit_kv_ids(prefetch_idx_int, pad_idx_list)

        # step 2: 逐个group获取kv
        
        group_final_k = []
        group_final_v = []
        
        with torch.cuda.stream(transfer_stream):
            for group_id in range(len(cur_group_ids)):

                # 获取当前group的unhit id
                unhit_len_list = cur_cache.get_one_group_unhit_kv_v1(
                    group_unhits[group_id], 
                    group_pad_idxs[group_id], 
                    group_id,
                    all_k,
                    all_v,
                    self._pinned_unhit_k_list[group_id],
                    self._pinned_unhit_v_list[group_id])

                max_unhit_len = max(unhit_len_list)

                # 截取需要的tensor
                tmp_unhit_k = self._pinned_unhit_k_list[group_id][:max_unhit_len, :]
                tmp_unhit_v = self._pinned_unhit_v_list[group_id][:max_unhit_len, :]

                # 传输并记录tensor
                group_final_k.append((group_cached_gpu_k[group_id], tmp_unhit_k.cuda(non_blocking=True)))
                group_final_v.append((group_cached_gpu_k[group_id], tmp_unhit_v.cuda(non_blocking=True)))
        
        print("################################# gpu_load Layer #", layer_id)
        
        print("prefetch_idx shape =", prefetch_idx.shape)
        # group_shape = [k.shape for k in group_cached_k]
        # print("cached kv shape = ", group_shape)
        group_unhit_shape = [unhit[1].shape for unhit in group_final_k]
        print("group_unhit_shape = ", group_unhit_shape)

        return group_final_k, group_final_v, group_unhits


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


        # Step 1: 获取未命中KV
        prefetch_idx_int = prefetch_idx.squeeze(1).to(torch.int32)
        pad_idx_list = prefetch_idx[0][0].tolist()
        group_unhit = cur_cache.get_unhit_kv_tensor_v5(
            prefetch_idx_int, 
            pad_idx_list,
            all_k, 
            all_v, 
            self._pinned_unhit_k_list, 
            self._pinned_unhit_v_list
        )

        # max_unhit_limit = int(self.max_unhit_rate * n_p) 
        # cur_cache.get_unhit_kv_tensor_v6(
        #     prefetch_idx_int, 
        #     pad_idx_list, 
        #     all_k, 
        #     all_v, 
        #     max_unhit_limit, 
        #     self._pinned_unhit_k_list, 
        #     self._pinned_unhit_v_list
        # )


        # Step 2: 获取传输完成的结果
        group_unhit_k = []
        group_unhit_v = []
        for i in range(len(group_unhit)):
            # tmp_unhit_len = len(group_unhit[i][0])
            if isinstance(group_unhit[i][0], list):
                tmp_unhit_len = max([len(unhit) for unhit in group_unhit[i]])
            else:
                tmp_unhit_len = max(group_unhit[i])

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
            
            cached_len = int(self.gpu_cache_pred * prefetch_idx.shape[0])
            self._max_gpu_cached_len = cached_len
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

        # print("Cache Manager load_and_update_cpu_cache #1", flush=True)

        # step 1: 如果预留的pinned tensor不够大, 创建新的pinned tensor
        if (layer_id not in self._pinned_cached_k_list) or (self._pinned_cached_k_list[layer_id][0].shape[0] < prefetch_idx.shape[0]):
            if layer_id in self._pinned_cached_k_list:
                del self._pinned_cached_k_list[layer_id]
                del self._pinned_cached_v_list[layer_id]
            
            # cached_len = int(1.2*prefetch_idx.shape[0])
            cached_len = int(prefetch_idx.shape[0] * self.cpu_cache_pred) # version 3 预留两倍cache大小
            self._max_cpu_cached_len = cached_len

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

        # print("Cache Manager load_and_update_cpu_cache #2", flush=True)

        # Step 3: 获取新的kv数据
        if len(self._cur_group_ids[layer_id]) == 1:
            selected_k, selected_v = select_kv(prefetch_idx, all_k, all_v)
            group_cpu_k = [selected_k]
            group_cpu_v = [selected_v]
        else:
            group_cpu_k, group_cpu_v = self._caches[layer_id].generate_update_cache(prefetch_idx, all_k, all_v)

        # print("Cache Manager load_and_update_cpu_cache #3", flush=True)
        # Step 4: 将数据拷贝到pinned_tensor中
        for i in range(len(cur_group_ids)):
            final_cpu_k[i].copy_(group_cpu_k[i])
            final_cpu_v[i].copy_(group_cpu_v[i])
        
        # print("Cache Manager load_and_update_cpu_cache #4", flush=True)
        # torch.cuda.synchronize() 

        # Step 5: 提前cpu cache更新任务
        # stat = self._caches[layer_id].update_group_cache(prefetch_idx, group_cpu_k, group_cpu_v)
        cur_cache = self._caches[layer_id]
        future_unhit = self.executor.submit(cur_cache.update_group_cache, prefetch_idx, final_cpu_k, final_cpu_v)

        # print("Cache Manager load_and_update_cpu_cache #5", flush=True)
        # Step 6: 并行开始 KV 的 GPU 传输（异步）
        with torch.cuda.stream(transfer_stream):
            group_gpu_k = [k.cuda(non_blocking=True) for k in final_cpu_k]
            group_gpu_v = [v.cuda(non_blocking=True) for v in final_cpu_v]


        # tmp 保存kv并观察是否一致
        # torch.save(final_cpu_k[0], f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/keys_b16_l{layer_id}.pt")
        # torch.save(final_cpu_v[0], f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/values_b16_l{layer_id}.pt")
        # sys.exit(0)
        
        # print("Cache Manager load_and_update_cpu_cache #6", flush=True)
        
        # Step 7: 等待 CPU 计算结束
        try:
            stat = future_unhit.result()
            if stat != 0:
                raise ValueError("Python ERROR! [load_and_update_cache] Update cache Fail!")
        except Exception as e:
            print(f"future_unhit error: {e}")
        
        # print("Cache Manager load_and_update_cpu_cache #7", flush=True)

        return group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v
    

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

    def update_gpu_cache_with_new_tensor(self, layer_id, new_k_cache, new_v_cache):
        '''
            Directly update cache.
            Be called before attention.
            The cache map is update when get unhit.
        '''
        if layer_id not in self._use_gpu_cache:
            return

        # cur_cache = self._caches[layer_id]
        # cur_cache.direct_update_cache(new_k_cache, new_v_cache)
        self._gpu_cached_group_kv[layer_id] = (new_k_cache, new_v_cache)



    # 统一的更新和加载接口
    def unified_load_api(self, layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v, dtype):
        # record prefetch_idx
        # path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_l{layer_id}_{cnt}.pt"
        # self.cnt += 1
        # torch.save(prefetch_idx, path)
        # print("CacheManager layer id = #", layer_id, flush=True)

        if layer_id in self._use_gpu_cache:
            # 判断是否需要更新
            # 固定步长更新
            # if (self._update_recode[layer_id] >= self._update_pred):
            # 自适应更新
            if self._require_update[layer_id]:
                # 直接更新cache
                self._update_recode[layer_id] = 0
                self._require_update[layer_id] = False
                group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = self.load_and_update_gpu_cached(layer_id, transfer_stream, prefetch_idx, all_k, all_v)
                return (group_gpu_k, group_gpu_v, None)
            else:
                self._update_recode[layer_id] += 1
                
                # group_final_k, group_final_v, group_unhit = self.gpu_cache_load_asyn(layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v)
                # group_final_k, group_final_v, group_unhit = self.gpu_cache_load_asyn_v2(layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v)
                
                # version 3
                group_final_k, group_final_v, group_unhit = self.gpu_cache_load_asyn_v3(layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v)
                # version 4 Fail
                # group_final_k, group_final_v, group_unhit = self.gpu_cache_load_asyn_v4(layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v)
                
                return (group_final_k, group_final_v, group_unhit)
        else:
            # 判断是否需要更新
            # 固定步长更新
            # if (self._update_recode[layer_id] >= self._update_pred):
            # 自适应更新
            if self._require_update[layer_id]:
                # 直接更新cache
                self._update_recode[layer_id] = 1
                self._require_update[layer_id] = False
                group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = self.load_and_update_cpu_cache(layer_id, transfer_stream, prefetch_idx, all_k, all_v, dtype)
                
                # print(f"CacheManager layer id = #{layer_id} load_and_update_cpu_cache suc", flush=True)
                return (group_gpu_k, group_gpu_v, None)
            else:
                
                # print(f"CacheManager layer id = #{layer_id} cpu_cache_load_asyn begin", flush=True)
                self._update_recode[layer_id] += 1
                # group_final_k, group_final_v, group_unhit = self.cpu_cache_load_asyn(layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v, dtype)
                
                group_final_k, group_final_v, group_unhit = self.cpu_cache_load_asyn_v2(layer_id, transfer_stream, prefetch_idx, pad_idx, all_k, all_v, dtype)
                
                # print(f"CacheManager layer id = #{layer_id} cpu_cache_load_asyn success", flush=True)
                return (group_final_k, group_final_v, group_unhit)


        

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


def speculate_attention(hidden, p_w_q, p_k_c, n_head, alpha, max_num_kv):
    """Speculates the indices of the critical KV caches of next attention layer.

    On the decoding stage, by using the hidden states (layer i), partial query
    weight (layer i+1), and partial key cache (layer i+1), speculates the
    attention score of the next layer. After that, counts the number of
    critical tokens and gets the indcies of the top-k KV cache tokens with high
    attention scores.

    Args:
        hidden: Hidden states of layer i (b, 1, D)
        p_w_q: Partial query weight (D', D)
        p_k_c: Partial key cache (n, bh, d')

        Note that bh * d' == D'

    Returns:
        prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
    """
    b = hidden.shape[0]
    p_q = F.linear(hidden, p_w_q, bias=None)
    p_q = p_q.view(b, 1, n_head, -1)
    p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)

    p_attn = torch.bmm(p_q, p_k_c.permute(1, 2, 0))
    max_ = torch.max(p_attn, dim=-1)[0]
    # thr_ = (max_ - alpha).unsqueeze(-1).repeat(1, 1, p_attn.shape[-1])
    # count = torch.where(
    #     p_attn > thr_, torch.ones_like(p_attn), torch.zeros_like(p_attn)
    # )
    # mean = torch.mean(torch.sum(count, dim=-1)).item()
    # prefetch_idx = torch.topk(
    #     p_attn.permute(2, 1, 0), min(int(mean), max_num_kv), dim=0
    # )[1]
    
    prefetch_idx = torch.topk(
        p_attn.permute(2, 1, 0), max_num_kv, dim=0
    )[1]

    return prefetch_idx

# old version 
# def speculate_attention(hidden, p_w_q, p_k_c, n_head, alpha, max_num_kv):
#     """Speculates the indices of the critical KV caches of next attention layer.

#     On the decoding stage, by using the hidden states (layer i), partial query
#     weight (layer i+1), and partial key cache (layer i+1), speculates the
#     attention score of the next layer. After that, counts the number of
#     critical tokens and gets the indcies of the top-k KV cache tokens with high
#     attention scores.

#     Args:
#         hidden: Hidden states of layer i (b, 1, D)
#         p_w_q: Partial query weight (D', D)
#         p_k_c: Partial key cache (n, bh, d')

#         Note that bh * d' == D'

#     Returns:
#         prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
#     """
#     b = hidden.shape[0]
#     p_q = F.linear(hidden, p_w_q, bias=None)
#     p_q = p_q.view(b, 1, n_head, -1)
#     p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)

#     p_attn = torch.bmm(p_q, p_k_c.permute(1, 2, 0))
#     max_ = torch.max(p_attn, dim=-1)[0]
#     thr_ = (max_ - alpha).unsqueeze(-1).repeat(1, 1, p_attn.shape[-1])
#     count = torch.where(
#         p_attn > thr_, torch.ones_like(p_attn), torch.zeros_like(p_attn)
#     )
#     mean = torch.mean(torch.sum(count, dim=-1)).item()
#     prefetch_idx = torch.topk(
#         p_attn.permute(2, 1, 0), min(int(mean), max_num_kv), dim=0
#     )[1]

#     return prefetch_idx

