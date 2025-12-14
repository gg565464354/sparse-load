import torch
import time
from collections import OrderedDict
import json
from collections import Counter
import heapq

class GPULRUCache:
    def __init__(self, batch_size, head_num, page_cnt, head_dim):
        self.batch_size = batch_size
        self.head_num = head_num
        self.page_cnt = page_cnt
        self.head_dim = head_dim
        self.cache_capacity = int(page_cnt * batch_size * head_num)
        
        # 在 GPU 上分配固定大小的缓存空间
        # print(type(self.cache_capacity), type(head_dim))
        
        self.cache_keys = torch.empty(self.cache_capacity, head_dim).to("cuda")
        self.cache_values = torch.empty(self.cache_capacity, head_dim).to("cuda")
        self.cache = OrderedDict()
        
        # 创建索引表记录 key 和 value 的存储位置
        self.index_table = {}
        self.current_index = 0

    def access(self, key_ids, keys, values):
        """
        访问一批 key-value，并缓存热门数据
        :param key_ids: (k,) 维度的 key ID 张量
        :param keys: (k, head_dim) 维度的 key 张量 (在 CPU 上)
        :param values: (k, head_dim) 维度的 value 张量 (在 CPU 上)
        """
        # print("keys.shape[1] = ", keys.shape )

        assert keys.shape[1] == self.head_dim
        assert values.shape[1] == self.head_dim
        
        k = key_ids.shape[0]
        hits = 0
        
        # 记录通信时间
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        not_cached_ids = []
        not_cached_keys = []
        not_cached_values = []

        for i in range(k):
            key_id = int(key_ids[i])

            # print("key id shape = ", key_ids.shape)
            # print("k = ", k)
            
            if key_id in self.cache:
                # 命中缓存，移动到末尾表示最近使用
                hits += 1
                self.cache.move_to_end(key_id)
            else:
                # 未命中缓存，收集未缓存数据
                not_cached_ids.append(key_id)
                not_cached_keys.append(keys[i])
                not_cached_values.append(values[i])
        
        if not_cached_keys:
            # 将未缓存的 key-value 批量传输到 GPU
            not_cached_keys = torch.stack(not_cached_keys).cuda()
            not_cached_values = torch.stack(not_cached_values).cuda()
            
            for i in range(len(not_cached_keys)):
                key_id = not_cached_ids[i]
                
                if len(self.cache) >= self.cache_capacity:
                    # 缓存满了，移除最久未使用的项
                    evicted_id, _ = self.cache.popitem(last=False)
                    evicted_index = self.index_table.pop(evicted_id)
                else:
                    evicted_index = self.current_index
                    self.current_index = (self.current_index + 1) % self.cache_capacity
                
                # 保存新 key-value 到 GPU 缓存和索引表
                self.cache_keys[evicted_index] = not_cached_keys[i]
                self.cache_values[evicted_index] = not_cached_values[i]
                self.cache[key_id] = True
                self.index_table[key_id] = evicted_index
        
        end_event.record()
        
        # 计算通信时间
        torch.cuda.synchronize()
        communication_time = start_event.elapsed_time(end_event)
        
        return hits, communication_time


def quest_indices_extend(indices, page_num, max_len):
    '''
        把quest算法识别出来的indices重新拓展成原始indices
    '''
    final_indices = []

    for i in indices:
        for j in range(0, 4):
            cur_id = i*page_num + j
            if cur_id < max_len:
                final_indices.append(cur_id)

    return final_indices

def list_to_dict(arr):
    dic = {}

    for a in arr:
        dic[int(a)] = True

    return dic

def get_window_cache(M, indices):
    # 统计所有数组中数字的频率
    counter = Counter()
    for nums in indices:
        counter.update(nums)

    # 使用堆找出前 k 个最高频数字
    arr = [num for num, _ in heapq.nlargest(M, counter.items(), key=lambda x: x[1])]
    return list_to_dict(arr)

# 测试

layer_num = 80
head_num = 64
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
    for head_id in range(0, head_num):
    # for head_id in [10]:
        # for topk_rate in [0.5, 0.4, 0.3]:
        for topk_rate in [0.2]:
            cache_rate = 0.2
            ############ 开始测试缓存能力
            # topk_rate = 0.5
            topk = int(topk_rate*token_len)

            # cache_rate = 0.5
            M = int(token_len*cache_rate) # 缓存大小
            # M = topk # 缓存大小
            gpu_cache = GPULRUCache(1, 1, M, head_dim)

            # 获取topk index
            topk_list = []
            wei_list = []

            hit_rate_list = []
            win_hit_rate_list = []
            time_cost_list = []

            
            indices_name = f"./tmp_file/{tar_name}/idx_l{layer_id}_h{head_id}_out{output_name}_nosm.pt"
            indices = torch.load(indices_name, weights_only=True).cpu()

            cache_pred = 4
            cache_indices = []
            cache_num = output_len//cache_pred

            #
            cache_indices.append(list_to_dict(indices[0].tolist()))
            for i in range(cache_num):
                start = i*cache_pred
                end = start + cache_pred
                win_cache = get_window_cache(M, indices[start:end])
                cache_indices.append(list_to_dict(win_cache))

            for i in range(1, output_len):
                ### 加载数据
                token_indices = indices[i]
                indices_arr = indices[i].tolist()

                ##################### get topk 
                # topk_list.append(indices_arr[:topk])

                #### test gpu cache
                sampled_key_ids = token_indices[:topk]
                sampled_keys = all_keys[token_indices]
                sampled_values = all_values[token_indices]

                # 直接使用上一回合内容作为cache
                cur_cache = list_to_dict(indices[i-1].tolist()[:topk])
                hits = 0
                for id in sampled_key_ids:
                    if int(id) in cur_cache:
                        hits += 1
                hit_rate_list.append(hits)

                # 使用observe windows里的内容作为cache
                win_id = int(i / cache_pred)
                win_cache = cache_indices[win_id]
                win_hits = 0

                # print(f"win_cache  = {win_cache}")
                for id in sampled_key_ids:
                    if int(id) in win_cache:
                        win_hits += 1
                win_hit_rate_list.append(win_hits)

                # print(f"token #{i} win = {win_hits} \t old = {hits} \t all = {M}")


            avg_hit = sum(hit_rate_list)/(output_len-1)
            avg_hit_rate = avg_hit/M
            # avg_time_cost = sum(time_cost_list)/output_len

            avg_win_rate = (sum(win_hit_rate_list)/(output_len-1))/M
            
            print(f"######### layer id = {layer_id}, head id = {head_id}")            
            hit_json = {
                "layer_id": layer_id,
                "head_id": head_id,
                "hit": avg_hit_rate,
                "win_hit": avg_win_rate,
                # "time": avg_time_cost,
            }

            print("hit json = ", hit_json)
            print("hit_rate_list = ", [hit_cnt/M for hit_cnt in hit_rate_list])
            print("win_hit_rate_list = ", [hit_cnt/M for hit_cnt in win_hit_rate_list])

            # 写入 JSONL 文件
            with open(result_file_name, "a", encoding="utf-8") as file:
                file.write(json.dumps(hit_json, ensure_ascii=False) + "\n")
    #     break
    # break

############### group attention

# tar_name = "7.3k"
# output_name = 150
# result_file_name = f"./result/cache_{tar_name}_k20_c30_nosm.jsonl"
# # result_file_name = f"./result/cache_{tar_name}_top20_all_t150.jsonl"
# for layer_id in range(0, layer_num):
#     for head_id in range(0, head_num):
#     # for head_id in [10]:
#         # for topk_rate in [0.5, 0.4, 0.3]:
#         topk_rate = 0.3
#         cache_rate = 0.3
#         ############ 开始测试缓存能力
#         # topk_rate = 0.5
#         topk = int(topk_rate*token_len)

#         # cache_rate = 0.5
#         M = int(token_len*cache_rate) # 缓存大小
#         # M = topk # 缓存大小
#         gpu_cache = GPULRUCache(1, 1, M, head_dim)

#         # 获取topk index
#         topk_list = []
#         wei_list = []

#         hit_rate_list = []
#         time_cost_list = []

        
#         indices_name = f"./tmp_file/{tar_name}/idx_l{layer_id}_h{head_id}_out{output_name}_nosm.pt"
#         indices = torch.load(indices_name, weights_only=True).cpu()

#         for i in range(output_len):
#             ### 加载数据
#             token_indices = indices[i]
#             indices_arr = indices[i].tolist()

#             ##################### get topk 
#             # topk_list.append(indices_arr[:topk])

#             #### test gpu cache
#             sampled_key_ids = token_indices[:topk]
#             sampled_keys = all_keys[token_indices]
#             sampled_values = all_values[token_indices]

#             hits, time_cost = gpu_cache.access(sampled_key_ids, all_keys, all_values)
#             hit_rate_list.append(hits)
#             time_cost_list.append(time_cost)

#             if i > 0:
#                 tmp_hits = 0
#                 for id in sampled_key_ids:
#                     if id in indices[i-1].tolist()[:topk]:
#                         tmp_hits += 1
#                 print(f"cur hits = {hits} / {tmp_hits}")

#         avg_hit = sum(hit_rate_list)/output_len
#         avg_hit_rate = avg_hit/topk
#         avg_time_cost = sum(time_cost_list)/output_len
        
#         print(f"######### layer id = {layer_id}, head id = {head_id}")            
#         hit_json = {
#             "layer_id": layer_id,
#             "head_id": head_id,
#             "hit": avg_hit_rate,
#             "time": avg_time_cost,
#         }

#         print("hit json = ", hit_json)
#         print("hit_rate_list = ", [hit_cnt/topk for hit_cnt in hit_rate_list])

#         # 写入 JSONL 文件
#         # with open(result_file_name, "a", encoding="utf-8") as file:
#         #     file.write(json.dumps(hit_json, ensure_ascii=False) + "\n")

#         break
#     break



############### quest 的topk测试

# 对真实数据进行处理
# QUEST_PAGE_NUM = 4
# result_file_name = "quest_top50_p4_v1.jsonl"
# for layer_id in range(0, layer_num):
#     for head_id in range(0, head_num):
#     # for head_id in [10]:
#         # for topk_rate in [0.5, 0.4, 0.3]:
#         for topk_rate in [0.5]:
#             ############ 开始测试缓存能力
#             # topk_rate = 0.5
#             topk = int(topk_rate*token_len)

#             # cache_rate = 0.5
#             # M = int(topk*cache_rate) # 缓存大小
#             M = topk # 缓存大小
#             gpu_cache = GPULRUCache(1, 1, M, head_dim)

#             # 获取topk index
#             topk_list = []
#             wei_list = []

#             hit_rate_list = []
#             time_cost_list = []

#             for i in range(token_len):
#                 ### 加载数据
#                 indices_name = f"./tmp_file/indices_l{layer_id}_h{head_id}_t{i}.pt"
#                 indices = torch.load(indices_name).cpu()

#                 indices_arr = indices.tolist()
#                 # quest change
#                 indices_arr = quest_indices_extend(indices_arr, QUEST_PAGE_NUM, token_len)

#                 ##################### get topk 
#                 # topk_list.append(indices_arr[:topk])

#                 #### test gpu cache
#                 sampled_key_ids = indices[:topk]
#                 sampled_keys = all_keys[indices]
#                 sampled_values = all_values[indices]

#                 hits, time_cost = gpu_cache.access(sampled_key_ids, sampled_keys, sampled_values)
#                 hit_rate_list.append(hits)
#                 time_cost_list.append(time_cost)

#             avg_hit = sum(hit_rate_list)/token_len
#             avg_hit_rate = avg_hit/topk
#             avg_time_cost = sum(time_cost_list)/token_len
            
#             print(f"######### layer id = {layer_id}, head id = {head_id}")            
#             hit_json = {
#                 "layer_id": layer_id,
#                 "head_id": head_id,
#                 "hit": avg_hit_rate,
#                 "time": avg_time_cost,
#             }

#             print("hit json = ", hit_json)

#             # 写入 JSONL 文件
#             with open(result_file_name, "a", encoding="utf-8") as file:
#                 file.write(json.dumps(hit_json, ensure_ascii=False) + "\n")
        # break
    # break
