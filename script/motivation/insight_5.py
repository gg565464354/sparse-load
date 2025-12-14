import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random

def get_lru_cache(his_data, cache_size):
    access_history = []
    for d in his_data:
        access_history.extend(d)
    counter = Counter(access_history)
    most_common_ids = [id_ for id_, _ in counter.most_common(cache_size)]

    return most_common_ids


def get_avg(data):
    return sum(data)/len(data)

def get_hit_rate(cache, tar):
    tar_len = len(tar)
    hit_num = 0
    for i in tar:
        if i in cache:
            hit_num += 1
    unhit = tar_len-hit_num
    return unhit, hit_num/tar_len

def get_unhit_id_incre(cache, tar):
    tar_len = len(tar)
    unhit = []
    hit_cnt = 0

    for i in tar:
        if i in cache:
            hit_cnt += 1
        else:
            unhit.append(i)

    return unhit, hit_cnt/tar_len

def in_update_cache(cache, unhit):
    for i in unhit:
        cache[i] = True
    return cache


def heu_cache_rebuild(pre_idx):
    cache = [in_update_cache({}, head_idx) for head_idx in pre_idx]
    return cache


token_num = 20
layer_num = 30
head_num = 32
all_cnt = 570
layer_idx = [[] for i in range(30)]
layer_idx_list = [[] for i in range(30)]

sparse_num = 200
cache_size_pred = 500

for i in range(all_cnt):
    lid = i % layer_num
    # cur_path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_l{i}.pt"
    cur_path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_inf_b1_l{i}.pt"
    cur_pidx = torch.load(cur_path).cpu()

    cur_pidx_reshape = cur_pidx.squeeze(1).T

    cur_pidx_reshape = cur_pidx_reshape[:, :sparse_num]

    layer_idx[lid].append(cur_pidx_reshape)
    layer_idx_list[lid].append(cur_pidx_reshape.tolist())

for i in range(layer_num):
    layer_shape = [idx.shape for idx in layer_idx[i]]
    print(f"#{i} {layer_shape}")


def LRU_Cache(cache_pred, layer_idx_list):    
    layer_hit_rates = [[] for i in range(layer_num)]
    layer_unhit_nums = [[] for i in range(layer_num)]
    layer_pidx_len = [[] for i in range(layer_num)]

    token_len = len(layer_idx_list[0])
    # cache_pred = 8

    for i in range(layer_num):
        cur_layer_hit_rate = layer_hit_rates[i]
        cur_layer_pidx_list = layer_idx_list[i]
        
        cur_cache = [cur_layer_pidx_list[0][h] for h in range(head_num)]
        
        for t in range(0, len(cur_layer_pidx_list)):
            
            cur_token_hit_rates = []
            
            for h in range(head_num):    
                # cached_id = t - (t % cache_pred)
                # cached_id = max(0, t - 1)
                if t % cache_pred == 0:
                    if t < cache_pred:
                        cur_cache[h] = cur_layer_pidx_list[0][h]
                    else:
                        # print(f"len(cur_layer_pidx_list) = {len(cur_layer_pidx_list)} t={t} cache_pred={cache_pred}")
                        his_data = [cur_layer_pidx_list[his_id][h] for his_id in range(t)]
                        cur_cache[h] = get_lru_cache(his_data, cache_size_pred)

                # begin compute hit_rate
                cache_head_idx = cur_cache[h]
                cur_head_idx = cur_layer_pidx_list[t][h]

                unhit_list, hit_rate = get_hit_rate(cache_head_idx, cur_head_idx)

                cur_token_hit_rates.append(hit_rate)
            
            min_layer_rate = min(cur_token_hit_rates) # 取最小命中率作为该layer的命中率
            avg_layer_rate = sum(cur_token_hit_rates)/len(cur_token_hit_rates) # 取最小命中率作为该layer的命中率

            cur_layer_hit_rate.append(min_layer_rate)
        
    print("############## LRU Pool hit rates")
    lru_hit_rate = []
    for l in range(layer_num):
        layer_vag_rate = sum(layer_hit_rates[l])/len(layer_hit_rates[l])
        print(f"Layer #{l} hit_rates = {layer_hit_rates[l]} avg = {layer_vag_rate}")
        lru_hit_rate.append(layer_vag_rate)
    
    return lru_hit_rate


####################### LRU Cache 命中率

# layer_hit_rates = [[] for i in range(layer_num)]
# layer_unhit_nums = [[] for i in range(layer_num)]
# layer_pidx_len = [[] for i in range(layer_num)]

# token_len = len(layer_idx_list[0])
# cache_pred = 8

# for i in range(layer_num):
#     cur_layer_hit_rate = layer_hit_rates[i]
#     cur_layer_pidx_list = layer_idx_list[i]
    
#     cur_cache = [cur_layer_pidx_list[0][h] for h in range(head_num)]
    
#     for t in range(0, len(cur_layer_pidx_list)):
        
#         cur_token_hit_rates = []
        
#         for h in range(head_num):    
#             # cached_id = t - (t % cache_pred)
#             # cached_id = max(0, t - 1)
#             if t % cache_pred == 0:
#                 if t < cache_pred:
#                     cur_cache[h] = cur_layer_pidx_list[0][h]
#                 else:
#                     # print(f"len(cur_layer_pidx_list) = {len(cur_layer_pidx_list)} t={t} cache_pred={cache_pred}")
#                     his_data = [cur_layer_pidx_list[t-his_id][h] for his_id in range(cache_pred)]
#                     cur_cache[h] = get_lru_cache(his_data, cache_size_pred)

#             # begin compute hit_rate
#             cache_head_idx = cur_cache[h]
#             cur_head_idx = cur_layer_pidx_list[t][h]

#             unhit_list, hit_rate = get_hit_rate(cache_head_idx, cur_head_idx)

#             cur_token_hit_rates.append(hit_rate)
        
#         min_layer_rate = min(cur_token_hit_rates) # 取最小命中率作为该layer的命中率
#         avg_layer_rate = sum(cur_token_hit_rates)/len(cur_token_hit_rates) # 取最小命中率作为该layer的命中率

#         cur_layer_hit_rate.append(min_layer_rate)
    
# print("############## LRU Pool hit rates")
# lru_hit_rate = []
# for l in range(layer_num):
#     layer_vag_rate = sum(layer_hit_rates[l])/len(layer_hit_rates[l])
#     print(f"Layer #{l} hit_rates = {layer_hit_rates[l]} avg = {layer_vag_rate}")
#     lru_hit_rate.append(layer_vag_rate)


lru_hit_rate_p8 = LRU_Cache(8, layer_idx_list)
lru_hit_rate_p4 = LRU_Cache(4, layer_idx_list)
lru_hit_rate_p2 = LRU_Cache(2, layer_idx_list)

####################### 完全 Incremental Cache

in_layer_hit_rates = [[] for i in range(layer_num)]
in_all_layer_cache_size = []

token_len = len(layer_idx_list[0])
cache_pred = 4

for i in range(layer_num):
    in_cur_layer_hit_rate = in_layer_hit_rates[i]
    in_cur_layer_pidx_list = layer_idx_list[i]
    cur_cache = [in_update_cache({}, in_cur_layer_pidx_list[0][h]) for h in range(head_num)]
    
    # 记录cache大小
    cur_layer_cache_size = []
    cur_layer_cache_size.append([len(cur_cache[h]) for h in range(head_num)])
    
    for t in range(0, len(in_cur_layer_pidx_list)):
        
        in_cur_token_hit_rates = []
        
        for h in range(head_num):    
            cached_id = t - (t % cache_pred)
            # cached_id = max(0, t - 1)

            cache_head_idx = cur_cache[h]
            cur_head_idx = in_cur_layer_pidx_list[t][h]

            unhit_list, hit_rate = get_unhit_id_incre(cache_head_idx, cur_head_idx)

            in_update_cache(cur_cache[h], unhit_list)

            in_cur_token_hit_rates.append(hit_rate)
        
        min_layer_rate = min(in_cur_token_hit_rates) # 取最小命中率作为该layer的命中率
        avg_layer_rate = sum(in_cur_token_hit_rates)/len(in_cur_token_hit_rates) # 取最小命中率作为该layer的命中率

        in_cur_layer_hit_rate.append(min_layer_rate)
        cur_layer_cache_size.append([len(cur_cache[h]) for h in range(head_num)])

    in_all_layer_cache_size.append(cur_layer_cache_size)
    

# process the cache size
# all_layer_max_cache_size = [[max(token_cache_size) for token_cache_size in in_all_layer_cache_size[l]] for l in range(layer_num)]
all_layer_max_cache_size = [[get_avg(token_cache_size) for token_cache_size in in_all_layer_cache_size[l]] for l in range(layer_num)]

print("############## Incremental Pool hit rates")
incre_pool_hit_rates = []
for l in range(layer_num):
    # print(f"Layer #{l} hit_rates = {in_layer_hit_rates[l]} avg = {layer_vag_rate}")
    layer_vag_rate = sum(in_layer_hit_rates[l])/len(in_layer_hit_rates[l])
    # print(f"Layer #{l} hit_rates = {in_layer_hit_rates[l]} avg = {layer_vag_rate}")
    # print(f"in_all_layer_cache_size[{l}] = {all_layer_max_cache_size[l]}")
    # print(f"in_all_layer_cache_size[{l}] = {in_all_layer_cache_size[l]}")
    incre_pool_hit_rates.append(layer_vag_rate)



########################################  启发式渐进Cache


in_heu_layer_hit_rates = [[] for i in range(layer_num)]
in_heu_all_layer_cache_size = []

token_len = len(layer_idx_list[0])

for i in range(layer_num):
    # in_heu_cur_layer_hit_rate = in_heu_layer_hit_rates[i]

    in_heu_cur_layer_pidx_list = layer_idx_list[i]

    cur_cache = [in_update_cache({}, in_heu_cur_layer_pidx_list[0][h]) for h in range(head_num)]


    # 记录cache大小
    cur_layer_cache_size = []
    cur_layer_cache_size.append([len(cur_cache[h]) for h in range(head_num)])
    
    for t in range(0, len(in_heu_cur_layer_pidx_list)):
        
        in_heu_cur_token_hit_rates = []
        tmp_cache_size = [len(cur_cache[h]) for h in range(head_num)]
        # 如果大小超过了限制，更新Cache
        if max(tmp_cache_size) > cache_size_pred:
            print(f"update at Layer #{i} Token #{t}")
            cur_cache = heu_cache_rebuild(in_heu_cur_layer_pidx_list[t])
            # tmp_cache_size = [len(cur_cache[h]) for h in range(head_num)] 
            # in_heu_cur_token_hit_rates = [1 for h in range(head_num)]
        else: 
            for h in range(head_num):
                cache_head_idx = cur_cache[h]
                cur_head_idx = in_heu_cur_layer_pidx_list[t][h]

                unhit_list, hit_rate = get_unhit_id_incre(cache_head_idx, cur_head_idx)

                in_update_cache(cur_cache[h], unhit_list)

                in_heu_cur_token_hit_rates.append(hit_rate)

            print(f"layer #{i} token #{t} = {in_heu_cur_token_hit_rates}")
            avg_hit_rate = get_avg(in_heu_cur_token_hit_rates)
            min_hit_rate = min(in_heu_cur_token_hit_rates)
            in_heu_layer_hit_rates[i].append(min_hit_rate)

        # process cache rate
        tmp_cache_size = [len(cur_cache[h]) for h in range(head_num)]
        cur_layer_cache_size.append(tmp_cache_size)

    in_heu_all_layer_cache_size.append(cur_layer_cache_size)


heu_all_layer_max_cache_size = [[max(token_cache_size) for token_cache_size in in_heu_all_layer_cache_size[l]] for l in range(layer_num)]
# heu_all_layer_max_cache_size = [[get_avg(token_cache_size) for token_cache_size in in_heu_all_layer_cache_size[l]] for l in range(layer_num)]


print("############## Incremental Pool hit rates")
incre_heu_pool_hit_rates = []
for l in range(layer_num):
    # print("in_heu_layer_hit_rates[l] = ", in_heu_layer_hit_rates[l])
    layer_vag_rate = sum(in_heu_layer_hit_rates[l])/len(in_heu_layer_hit_rates[l])
    incre_heu_pool_hit_rates.append(layer_vag_rate)


#################################### 开始画图

print("############## Hit Rate RESULT")
print(f"lru_hit_rate_p8 = {lru_hit_rate_p8}")
print(f"lru_hit_rate_p4 = {lru_hit_rate_p4}")
# print(f"incre_pool_hit_rates = {incre_pool_hit_rates}")


idx = [i+2 for i in range(len(incre_pool_hit_rates))]
plt.figure(figsize=(10, 6))
plt.plot(idx, lru_hit_rate_p2, label='LRU Cache K=2', marker='s')
plt.plot(idx, lru_hit_rate_p4, label='LRU Cache K=4', marker='s')
# plt.plot(idx, incre_pool_hit_rates, label='Increment Cache', marker='*')
plt.plot(idx, incre_heu_pool_hit_rates, label='Increment Cache', marker='^')
plt.xlabel('Layer ID')
plt.ylabel('Hit Rate')
# plt.title('Discrete vs. Continue Transfer Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("insight_5.png")


# print("############## Cache Size RESULT")

# # idx = [i+2 for i in range(len(incre_pool_hit_rates))]
# nidx = [i for i in range(token_len)]
# base_cache_size = [500 for i in range(token_len)]
# plt.figure(figsize=(10, 6))

# plt.plot(nidx, base_cache_size, label=f'Layer #{l+2}', marker='s')
# # for l in [4, 5, 6]:
# for l in range(layer_num):
#     plt.plot(nidx, all_layer_max_cache_size[l], label=f'In. Layer #{l+2}', marker='s')
#     # plt.plot(nidx, heu_all_layer_max_cache_size[l], label=f'Heu. Layer #{l+2}', marker='^')

# # 设置X轴上的刻度位置和显示格式
# plt.xticks(np.arange(0, token_len, 1))
# # plt.xlim(0, token_len)

# plt.xlabel('Token ID')
# plt.ylabel('Cache Size')
# # plt.title('Discrete vs. Continue Transfer Time')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("insight_6.png")
