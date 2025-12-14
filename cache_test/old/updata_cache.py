import torch
import numpy as np
import numba
        

class GPULRUCache:
    def __init__(self, token_size, head_dim):
        self.cache_capacity = token_size
        self.head_dim = head_dim
        self.cache_keys = torch.empty(self.cache_capacity, head_dim).to("cuda")
        self.cache_values = torch.empty(self.cache_capacity, head_dim).to("cuda")

        # 创建索引表记录 key 和 value 的存储位置
        self.index_table = {id: idx for idx, id in enumerate(range(0, token_size))}
        # self.index_table = set(range(0, token_size))
        
    
    def update_cache(self, cached_ids_dict, un_cached_ids, keys, values):
        # self.index_table = {}

        # 把需要的数据传输到gpu上
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        not_cached_keys = keys[un_cached_ids].contiguous().cuda()
        not_cached_values = values[un_cached_ids].contiguous().cuda()
        
        # 计算通信时间
        end_event.record()
        torch.cuda.synchronize()
        communication_time = start_event.elapsed_time(end_event)

        # find miss ids
        start_update_event = torch.cuda.Event(enable_timing=True)
        end_update_event = torch.cuda.Event(enable_timing=True)
        start_update_event.record()

        missed_cache_indices = [id for idx, id in enumerate(self.index_table.keys()) if id not in cached_ids_dict]
        replace_num = min(len(missed_cache_indices), len(un_cached_ids))

        # update the cache
        for i in range(replace_num):
            miss_id = missed_cache_indices[i]
            pos = self.index_table[miss_id]

            self.cache_keys[pos] = keys[i]
            self.cache_values[pos] = values[i]

            del self.index_table[miss_id]
            self.index_table[un_cached_ids[i]] = pos
        
        # 计算更新时间
        end_update_event.record()
        torch.cuda.synchronize()
        update_time = start_update_event.elapsed_time(end_update_event)

        return communication_time, update_time

    def get_hit_rate(self, indices, keys, values):
        assert keys.shape[1] == self.head_dim
        assert values.shape[1] == self.head_dim

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        hits = 0
        cached_ids_dict = {}
        un_cached_ids = []
        for id in indices:
            if id in self.index_table:
                hits += 1
                cached_ids_dict[id] = True
            else:
                un_cached_ids.append(id)
        
        
        # 计算通信时间
        end_event.record()
        torch.cuda.synchronize()
        hit_cost = start_event.elapsed_time(end_event)

        return hits, cached_ids_dict, un_cached_ids, hit_cost
    
    def direct_communication_cost(self, indices, keys, values):
        # 把需要的数据传输到gpu上
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        not_cached_keys = keys[indices].contiguous().cuda()
        not_cached_values = values[indices].contiguous().cuda()
        
        # 计算通信时间
        end_event.record()
        torch.cuda.synchronize()
        communication_time = start_event.elapsed_time(end_event)

        return communication_time

class NewGPUCache:
    def __init__(self, token_size, head_dim):
        self.cache_capacity = token_size
        self.head_dim = head_dim
        self.cache_keys = torch.empty(self.cache_capacity, head_dim).to("cuda")
        self.cache_values = torch.empty(self.cache_capacity, head_dim).to("cuda")

        # 创建索引表记录 key 和 value 的存储位置
        # self.index_table = {id: idx for idx, id in enumerate(range(0, token_size))}
        self.index_table = set(range(0, token_size))
    
    def update_cache(self, new_index_list, keys, values):

        torch.cuda.synchronize()
        # 把需要的数据传输到gpu上
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        
        self.index_table = set(new_index_list)

        self.cache_keys = keys[new_index_list].contiguous().cuda()
        self.cache_values = values[new_index_list].contiguous().cuda()
        
        # 计算通信时间
        end_event.record()
        torch.cuda.synchronize()
        update_time = start_event.elapsed_time(end_event)

        return update_time
    
    def load_un_cached(self, un_cached_ids, keys, values):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        not_cached_keys = keys[un_cached_ids].contiguous().cuda()
        not_cached_values = values[un_cached_ids].contiguous().cuda()

        final_keys = torch.concat([self.cache_keys, not_cached_keys])
        final_values = torch.concat([self.cache_values, not_cached_values])
        
        # 计算通信时间
        end_event.record()
        torch.cuda.synchronize()
        communication_time = start_event.elapsed_time(end_event)

        return communication_time
    
    def new_load_un_cached(self, un_cached_ids, keys, values):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        not_cached_keys = keys[un_cached_ids].contiguous().pin_memory()
        not_cached_values = values[un_cached_ids].contiguous().pin_memory()

        # 异步拷贝到 GPU
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            not_cached_keys = not_cached_keys.to(device='cuda', non_blocking=True)
            not_cached_values = not_cached_values.to(device='cuda', non_blocking=True)
        
        num_cached = self.cache_keys.shape[0]  # 旧缓存大小
        num_new = not_cached_keys.shape[0]  # 新数据大小
        total_size = num_cached + num_new  # 总大小

        # **直接声明足够大的 Tensor**
        new_cache_keys = torch.empty((total_size, self.cache_keys.shape[1]), dtype=self.cache_keys.dtype, device='cuda')
        new_cache_values = torch.empty((total_size, self.cache_values.shape[1]), dtype=self.cache_values.dtype, device='cuda')

        # **拷贝数据**
        new_cache_keys[:num_cached] = self.cache_keys  # 先拷贝旧缓存
        new_cache_keys[num_cached:] = not_cached_keys  # 再拷贝新数据

        new_cache_values[:num_cached] = self.cache_values
        new_cache_values[num_cached:] = not_cached_values

        
        # 计算通信时间
        end_event.record()
        torch.cuda.synchronize()
        communication_time = start_event.elapsed_time(end_event)

        return communication_time
        

    def get_hit_rate(self, indices, keys, values):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        hits = 0
        # cached_ids_dict = {}
        un_cached_ids = []
        
        for idx in indices:
            if idx in self.index_table:
                hits += 1
            else:
                un_cached_ids.append(idx)
        
        # 计算通信时间
        end_event.record()
        torch.cuda.synchronize()
        hit_cost = start_event.elapsed_time(end_event)

        return hits, None, un_cached_ids, hit_cost
    
    def direct_communication_cost(self, indices, keys, values):
        torch.cuda.synchronize()
        # 把需要的数据传输到gpu上
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        not_cached_keys = keys[indices].contiguous().cuda()
        not_cached_values = values[indices].contiguous().cuda()
        
        # 计算通信时间
        end_event.record()
        torch.cuda.synchronize()
        communication_time = start_event.elapsed_time(end_event)

        return communication_time



# 更新显存kv cache的开销
layer_num = 80
head_num = 64
group_num = 8
token_len = 8000
output_len = 149

K = 8000
k = int(K/2)

batch_size = 1
page_cnt = k
head_dim = 128


# 初始化固定大小的数据池
all_key_ids = torch.arange(K).cpu()
all_keys = torch.randn(K, head_dim).cpu()
all_values = torch.randn(K, head_dim).cpu()

# ############ 直接topk的测试
# 对真实数据进行处理
tar_name = "7.7k"
output_name = 150
result_file_name = f"./result/win_{tar_name}_k20_win4.jsonl"
# result_file_name = f"./result/cache_{tar_name}_top20_all_t150.jsonl"
for layer_id in range(0, layer_num):
    # for head_id in range(0, head_num):
    for group_id in range(0, group_num):
    # for head_id in [10]:
        # for topk_rate in [0.5, 0.4, 0.3]:
        topk_rate = 0.2
        cache_rate = 0.2
        ############ 开始测试缓存能力
        # topk_rate = 0.5
        topk = int(topk_rate*token_len)
        # cache_rate = 0.5
        M = int(token_len*cache_rate) # 缓存大小


        # 获取topk index
        topk_list = []
        wei_list = []

        hit_rate_list = []
        win_hit_rate_list = []
        time_cost_list = []

        # get group indices
        group_indices = []
        for h in range(8):
            head_id = 8*group_id + h
            indices_name = f"./tmp_file/{tar_name}/idx_l{layer_id}_h{head_id}_out{output_name}_nosm.pt"
            indices = torch.load(indices_name, weights_only=True).to(torch.int).cpu().tolist()
            group_indices.append(indices)

        # topk = 100
        # generate real topk indices
        token_indices = []
        for i in range(output_len):
            tmp_indices_list = [idx[i][:topk] for idx in group_indices]
            # merged_list = list(set().union(*tmp_indices_list))

            tmp_set = set()
            for idx in tmp_indices_list:
                for id in idx:
                    if id not in tmp_set:
                        tmp_set.add(int(id))
            merged_list = list(tmp_set)
            token_indices.append(merged_list)

        idx_lens = [len(l) for l in token_indices]
        avg_len = sum(idx_lens)/output_len

        # 构建cache
        cache_len = int(token_len*cache_rate) # 缓存大小
        cache_len = int(avg_len)
        M = cache_len # 缓存大小
        
        GPU_Cache = NewGPUCache(M, head_dim)

        # for i in range(1, output_len):
        for i in range(output_len-20, output_len):
            ### 加载数据
            indices_arr = token_indices[i]
            token_idx = torch.tensor(indices_arr)

            #### test gpu cache
            sampled_key_ids = token_idx

            # direct load
            direct_cost = GPU_Cache.direct_communication_cost(indices_arr, all_keys, all_values)

            hits, cached_ids_dict, un_cached_ids, hit_cost = GPU_Cache.get_hit_rate(indices_arr, all_keys,  all_values)
            comm_cost = GPU_Cache.new_load_un_cached(un_cached_ids, all_keys, all_values)
            udpate_cost = GPU_Cache.update_cache(indices_arr, all_keys, all_values)
            
            hit_rate = hits/len(indices_arr)
            all_cost = hit_cost + comm_cost # + udpate_cost
            cost_rate = all_cost/direct_cost

            print(f"Layer #{layer_id} Group #{group_id}:  hit_rates={hit_rate} \t cost_rate={cost_rate}")
            print(f"direct={direct_cost} cache={all_cost}  {(hit_cost, comm_cost, udpate_cost)}")
            
    #     break
    # break

