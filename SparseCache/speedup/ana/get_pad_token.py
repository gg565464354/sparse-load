import torch

def get_hit_rate(cache, tar):
    tar_len = len(tar)
    hit_num = 0
    for i in tar:
        if i in cache:
            hit_num += 1
    unhit = tar_len-hit_num
    return unhit, hit_num/tar_len

token_num = 9
layer_num = 30
head_num = 32
all_cnt = 270
layer_idx = [[] for i in range(30)]
layer_idx_list = [[] for i in range(30)]

for i in range(all_cnt):
    lid = i % layer_num
    cur_path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data_infin/pidx_l{i}.pt"
    cur_pidx = torch.load(cur_path).cpu()

    cur_pidx_reshape = cur_pidx.squeeze(1).T

    layer_idx[lid].append(cur_pidx_reshape)
    layer_idx_list[lid].append(cur_pidx_reshape.tolist())

for i in range(layer_num):
    layer_shape = [idx.shape for idx in layer_idx[i]]
    # print(f"#{i} {layer_shape}")


t_record = [0 for _ in range(2010)]

for l in range(layer_num):
    for h in range(head_num):
        for t in range(9):
            pidx_len = len(layer_idx_list[l][t][h])
            for i in range(pidx_len):
                cur_id = int(layer_idx_list[l][t][h][i])
                # print("cur_id")
                t_record[cur_id] += 1000 - i

# 获取最小值
require_record = t_record[:2000]
min_value = min(require_record)
min_indices = [i for i, v in enumerate(require_record) if v == min_value]

# 获取最小值的第一个索引
min_index = require_record.index(min_value)

print(f"最小值是 {min_value}，索引是 {min_indices}")


# # 计算命中率
# layer_hit_rates = [[] for i in range(layer_num)]
# layer_unhit_nums = [[] for i in range(layer_num)]
# layer_pidx_len = [[] for i in range(layer_num)]

# for i in range(layer_num):
#     cur_layer_hit_rate = layer_hit_rates[i]
#     cur_layer_unhit_nums = layer_unhit_nums[i]
#     cur_layer_pidx_len = layer_pidx_len[i]

#     cur_layer_pidx_list = layer_idx_list[i]
    
#     for h in range(head_num):    
#         cur_head_hit_rates = []
#         cur_head_unhit_nums = []
#         cur_head_pidx_len = []
#         for t in range(0, len(cur_layer_pidx_list)):
#             # cached_id = t - (t % 4)
#             cached_id = max(0, t - 1)
#             cache_head_idx = cur_layer_pidx_list[cached_id][h]
#             cur_head_idx = cur_layer_pidx_list[t][h]

#             unhit, hit_rate = get_hit_rate(cache_head_idx, cur_head_idx)
#             # print("cache_head_idx = ", len(cache_head_idx))

#             cur_head_hit_rates.append(hit_rate)
#             cur_head_unhit_nums.append(unhit)
#             cur_head_pidx_len.append(len(cur_head_idx))

#             break
        
#         cur_layer_hit_rate.append(cur_head_hit_rates)
#         cur_layer_unhit_nums.append(cur_head_unhit_nums)
#         cur_layer_pidx_len.append(cur_head_pidx_len)
#         break
#     break
# print("############## hit rates")
# for l in range(layer_num):
#     for h in range(head_num):
#         print(f"Layer #{l} Head #{h} hit_rates = {layer_hit_rates[l][h]}")
            

# print("############## unhit nums")
# for l in range(layer_num):
#     for h in range(head_num):
#         show = [f"{layer_unhit_nums[l][h][t]}/{layer_pidx_len[l][h][t]}" for t in range(token_num)]
#         print(f"Layer #{l} Head #{h} unhit_nums = {show}")





