import torch
import torch.nn.functional as F
import time

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

def select_kv_one_bh(prefetch_idx, target_bh_id, k_cache, v_cache):
    bh_idx = prefetch_idx[:, target_bh_id:target_bh_id+1]

    ind = bh_idx * k_cache.shape[1] + torch.tensor([[target_bh_id]])
    
    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))

    return selected_k, selected_v


class CPUCache:
    '''
        Create when a new batch reach.
    '''

    def __init__(self, bh, prefetch_idx, cache_shape):
        self.cache_token_size = cache_shape[0]
        self.bh = bh
        self.head_dim = cache_shape[-1]
        self.cache_keys = torch.empty(cache_shape).cpu().pin_memory()
        self.cache_values = torch.empty(cache_shape).cpu().pin_memory()

        # 创建索引表记录 key 和 value 的存储位置
        self.cache_maps = [set([i for i in range(0, self.cache_token_size)]) for i in range(bh)]
        self._update_cache_map(prefetch_idx)
        

    def _update_cache_map(self, prefetch_idx):
        bh = prefetch_idx.shape[-1]
        bh_index = prefetch_idx.permute(2, 1, 0).view(bh, -1)

        for i in range(bh):
            bh_idx_list = bh_index[i].tolist()
            self.cache_maps[i] = set(bh_idx_list)
    
    def _update_cache_map_one_bh(self, prefetch_idx, bh):
        bh_index = prefetch_idx.permute(2, 1, 0)
        idx_list = prefetch_idx[bh][0].tolist()
        self.cache_maps[bh] = set(idx_list)

    def get_unhit(self, prefetch_idx):
        '''
            generate the unhit kv id according to the cache_maps
            prefetch_idx: shape (n', 1, bh)
        '''
        token_num, _, bh = prefetch_idx.shape
        pure_unhit_list = [[] for i in range(bh)]
        
        for i in range(bh):
            bh_cache_set = self.cache_maps[i]

            for tid in range(token_num):
                cur_token = prefetch_idx[tid][0][i]
                # print(f"#get_unhit cur_token = {type(cur_token)}")
                
                if int(cur_token) not in bh_cache_set:
                    pure_unhit_list[i].append(cur_token)

        # pad the cache
        return pure_unhit_list
    
    def pad_unhit(self, pure_unhit_list, draft_idx):
        for i in range(self.bh):
            for t in range(len(pure_unhit_list[i])):
                draft_idx[t][0][i] = pure_unhit_list[i][t]
        
        return draft_idx
    

    def update_cache_one_bh(self, target_bh_id, prefetch_idx, all_keys, all_values):
        torch.cuda.synchronize()
        # 把需要的数据传输到gpu上
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        self._update_cache_map_one_bh(prefetch_idx, target_bh_id)
        bh_select_k, bh_select_v = select_kv_one_bh(prefetch_idx, target_bh_id, all_keys, all_values) 
        self.cache_keys[:, target_bh_id:target_bh_id+1] = bh_select_k
        self.cache_values[:, target_bh_id:target_bh_id+1] = bh_select_v
        
        # 计算通信时间
        end_event.record()
        torch.cuda.synchronize()
        update_time = start_event.elapsed_time(end_event)

        return update_time
    

    def load_with_cached(self, prefetch_idx, keys, values):
        # get unhit
        pure_unhit_list = self.get_unhit(prefetch_idx)
        max_unhit_len = max([len(unhit) for unhit in pure_unhit_list])

        # pad unhit
        unhit_slots = torch.empty([max_unhit_len, 1, prefetch_idx.shape[2]]).int()
        un_cached_idx = self.pad_unhit(pure_unhit_list, unhit_slots)

        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        gpu_cached_k = self.cache_keys.cuda()
        gpu_cached_v = self.cache_values.cuda()
        
        # begin load
        un_cached_k, un_cached_v = select_kv(un_cached_idx, keys, values)
        print(f"un cache idx = {un_cached_idx.shape}")
        gpu_uncached_k = un_cached_k.cuda()
        gpu_uncached_v = un_cached_v.cuda()
        
        # print(f"gpu_uncached_k shape = {gpu_uncached_k.shape}, gpu_cached_v shape = {gpu_cached_v.shape}")
        final_k = torch.concat((gpu_cached_k, gpu_uncached_k), dim=0)
        final_v = torch.concat((gpu_cached_v, gpu_uncached_v), dim=0)

        # 计算通信时间
        end_event.record()
        torch.cuda.synchronize()
        communication_time = start_event.elapsed_time(end_event)
        
        print(f"unhit index shape={unhit_slots.shape}, tensor={gpu_uncached_k.shape}, time cost = {communication_time}")

        return pure_unhit_list, communication_time

    def direct_load(self, prefetch_idx, keys, values):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        select_k, select_v = select_kv(prefetch_idx, keys, values)
        gpu_select_k = select_k.cuda()
        gpu_select_v = select_v.cuda()

        
        # 计算通信时间
        end_event.record()
        torch.cuda.synchronize()
        communication_time = start_event.elapsed_time(end_event)

        print(f"direct index shape={prefetch_idx.shape}, tensor={gpu_select_k.shape}, time cost = {communication_time}")

        return communication_time


############## my test case
DEVICE = "cpu"
def rand_tensor(shape, a, b):
    # 生成服从标准正态分布的张量
    tensor = torch.randn(shape, device=DEVICE)

    # 将张量缩放到 [a, b] 范围并取整
    scaled_tensor = (b - a) * (tensor - tensor.min()) / (tensor.max() - tensor.min()) + a
    integer_tensor = torch.round(scaled_tensor).int()  # 四舍五入取整
    # print("rand_tensor.shape = ", integer_tensor.shape)

    return integer_tensor

def generate_random_index(bh, select_num, N):
    if select_num > N:
        raise ValueError("select_num cannot be greater than N.")
    
    # 初始化结果张量
    result = torch.empty((bh, select_num), dtype=torch.int64)
    
    for i in range(bh):
        # 生成 [1, N] 的随机排列，并取前 select_num 个数
        random_perm = torch.randperm(N)  # [1, N]
        result[i] = random_perm[:select_num]
    
    need_idx = result.permute(1, 0).unsqueeze(1)
    
    return need_idx


if __name__ == "__main__":
    print("test")

    ##################################### 纯粹load test
    N = 4096
    # N = 8192
    # N = 8
    N = 10240
    N_prime = int(N*0.5)
    same_N = int(N*0.35)

    hit_rate = 0.8
    N_cached = int(N_prime*hit_rate)
    N_un_cached = int(N_prime*(1-hit_rate))
    bh = 64
    # bh = 4
    d = 128
    head_dim = d

    #
    # TEST_CASE = 10
    TEST_CASE = 1
    full_load_cost = []
    cache_load_cost = []
    hit_rates = []


    for i in range(TEST_CASE):
        cached_idx = generate_random_index(bh, N_prime, N) 
        prefetch_idx = generate_random_index(bh, N_prime, N)  # (n=5, bh=2, d=3)

        prefetch_idx[:same_N] = cached_idx[:same_N]

        # un_cached_idx = rand_tensor((N_un_cached, 1, bh), 0, N-1)
        # cached_k = torch.randn((N_cached, bh, d), device=DEVICE)
        # cached_v = cached_k + 0.1

        all_keys = torch.randn((N, bh, d), device=DEVICE)
        all_values = all_keys + 0.1  # 让 v_cache 和 k_cache 不同

        # 创建CPU Cache
        cache_instance = CPUCache(bh, cached_idx, (N_prime, bh, head_dim))

        # direct load
        direct_cost = cache_instance.direct_load(prefetch_idx, all_keys, all_values)

        # cache load
        unhit_list, cache_cost = cache_instance.load_with_cached(prefetch_idx, all_keys, all_values)


        # get hit rate
        all_tokens = N_prime*bh 
        unhit_len_list = [len(unhit) for unhit in unhit_list]
        unhit_len = sum(unhit_len_list)
        hit_rate = (all_tokens - unhit_len)/all_tokens
        print(f"unhit_len_list = {unhit_len_list}")

        # record 
        full_load_cost.append(direct_cost)
        cache_load_cost.append(cache_cost)
        hit_rates.append(hit_rate)
        print(f"cache_cost={cache_cost} direct_cost={direct_cost} hit_test={hit_rate}")

        

    avg_full_cost = sum(full_load_cost)/TEST_CASE
    avg_cache_cost = sum(cache_load_cost)/TEST_CASE
    avg_hit_rates = sum(hit_rates)/TEST_CASE
    print("avg_full_cost = ", avg_full_cost)
    print("avg_cache_cost = ", avg_cache_cost)
    print("avg_hit_rates = ", avg_hit_rates)



# if __name__ == "__main__":
#     print("test")

#     ##################################### 纯粹load test
#     N = 4096
#     N = 8192
#     # N = 10240
#     N_prime = int(N*0.5)

#     hit_rate = 0.8
#     N_cached = int(N_prime*hit_rate)
#     N_un_cached = int(N_prime*(1-hit_rate))
#     bh = 64
#     d = 128

#     #
#     TEST_CASE = 10
#     full_cost = []
#     cache_cost = []
#     full_cmp_cost = []
#     cache_cmp_cost = []

#     for i in range(TEST_CASE):
#         cached_idx = rand_tensor((N_prime, 1, bh), 0, N_prime-1) 
#         prefetch_idx = rand_tensor((N_prime, 1, bh), 0, N_prime-1)  # (n=5, bh=2, d=3)
#         un_cached_idx = rand_tensor((N_un_cached, 1, bh), 0, N_prime-1)
#         cached_k = torch.randn((N_cached, bh, d), device=DEVICE)
#         cached_v = cached_k + 0.1

#         k_cache = torch.randn((N_prime, bh, d), device=DEVICE)
#         v_cache = k_cache + 0.1  # 让 v_cache 和 k_cache 不同


#         torch.cuda.synchronize()
#         start1 = time.time()
#         selected_k, selected_v = select_kv(prefetch_idx, k_cache, v_cache)
#         gpu_tensor_k = selected_k.cuda()
#         gpu_tensor_v = selected_v.cuda()
#         torch.cuda.synchronize()
#         end1 = time.time()
#         full_cost.append(end1 - start1)
#         # full_cmp_cost.append(tcost)
        

#         torch.cuda.synchronize()
#         start2 = time.time()
#         un_cached_k, un_cached_v = select_kv(un_cached_idx, k_cache, v_cache)
#         gpu_uncached_k = un_cached_k.cuda()
#         gpu_uncached_v = un_cached_v.cuda()

#         gpu_cached_k = cached_k.cuda()
#         gpu_cached_v = cached_v.cuda()

#         final_k = torch.concat((gpu_cached_k, gpu_uncached_k), dim=0)
#         final_v = torch.concat((gpu_cached_v, gpu_uncached_v), dim=0)

#         torch.cuda.synchronize()
#         end2 = time.time()
#         cache_cost.append(end2 - start2)
#         # cache_cmp_cost.append(tcost)

#     avg_full_cost = sum(full_cost)/TEST_CASE
#     avg_cache_cost = sum(cache_cost)/TEST_CASE
#     print("avg_full_cost = ", avg_full_cost)
#     print("avg_cache_cost = ", avg_cache_cost)

