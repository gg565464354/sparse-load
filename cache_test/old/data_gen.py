import torch
import json

import torch.nn.functional as F
from collections import defaultdict, OrderedDict
import random

############ 数据处理
def get_topk_keys(keys, query, topk):
    
    scores = torch.matmul(query, keys.transpose(-2, -1))
    # scores = F.softmax(scores, dim=-1)

    topk_weight, topk_key_indices = torch.topk(scores, topk, dim=-1)
    # topk_weight, topk_key_indices = torch.sort(scores, descending=True)
    
    return topk_weight, topk_key_indices

def trans_topp_keys(weis, indices, topp):
    sum_wei = 0
    
    for i in range(len(weis)):
        sum_wei += weis[i]
        if sum_wei > topp:
            return weis[:i], indices[:i]
    return weis, indices


############ 初始化参数
layer_num = 80
head_num = 64 
kv_group_num = 8

layer_id = 20
head_id = 10
query_id = 1000
# token_len = 2654


########### 临时测试



################### common topk测试
# query begin and end

output_len = 150

for l in range(layer_num):
    tar_name = "7.7k"
    dir = f"/NVME1/projects/qin/test_model/tmp_file/input_{tar_name}"
    key_name = f"{dir}/keys_l{l}_t1.pt"
    query_name = f"{dir}/querys_l{l}_t1.pt"
    
    key_tensor = torch.load(key_name, weights_only=False).to("cuda:0")
    # query_tensor = torch.load(query_name).to("cuda:0")

    for h in range(head_num):
        ### 加载数据
        print(f"l{l}_h{h} Shape = {key_tensor.shape}")

        ### get the keys
        keys = key_tensor[0][h//kv_group_num]
        # querys = query_tensor[0][h]


        ############ 开始测试缓存能力
        # topk_rate = 0.5
        token_len = key_tensor.shape[-2]
        topk = token_len

        # print("token len = ", token_len)

        # 获取topk index
        topk_list = []
        all_indices = []
        for t in range(2, output_len+1):
            cur_query = torch.load(f"{dir}/querys_l{l}_t{t}.pt", weights_only=False).to("cuda:0")
            head_query = cur_query[0][h][0]

            topk_weight, indices = get_topk_keys(keys, head_query, topk)
            all_indices.append(indices)
            print(f"l{l} h{h} = ", indices)

        indices = torch.stack(all_indices, dim=0)
        indices_name = f"./tmp_file/{tar_name}/idx_l{l}_h{h}_out{output_len}_nosm.pt"
        torch.save(indices, indices_name)   

        print("indices", indices.shape)

    #     break
    # break


##################### quest topk

# def split_and_extract_max_min(tensors):
#     n = tensors.shape[0]
#     remainder = n % 4
    
#     # 如果不能整除4，填充最后一个 key
#     if remainder != 0:
#         padding = 4 - remainder
#         last_key = tensors[-1].unsqueeze(0).repeat(padding, 1)
#         tensors = torch.cat([tensors, last_key], dim=0)
    
#     k = tensors.shape[0] // 4
#     tensors = tensors.view(k, 4, 128)
    
#     # 每个 (4, 128) 中，沿 dim=0 取最大值和最小值
#     max_keys = tensors.max(dim=1).values
#     min_keys = tensors.min(dim=1).values
    
#     return max_keys, min_keys


# def calculate_quest_topk_indices(query, max_keys, min_keys, topk_ratio):
#     # 计算 query 与 max_keys、min_keys 的得分
#     max_scores = torch.matmul(query, max_keys.T)
#     min_scores = torch.matmul(query, min_keys.T)
    
#     # 取每组中的较大得分
#     final_scores = torch.max(max_scores, min_scores)
    
#     # 选取 top-k/2 索引
#     k = max_keys.shape[0]
#     topk = int(k * topk_ratio)
#     top_scores, top_indices = torch.topk(final_scores, topk)
    
#     return top_scores, top_indices




# # 测试
# for l in range(layer_num):
#     for h in range(head_num):
#         ### 加载数据
#         dir = "key_states_2.6k"
#         key_name = f"./{dir}/key_states_l{l}.pt"
#         query_name = f"./{dir}/query_states_l{l}.pt"

#         key_tensor = torch.load(key_name).to("cuda:0")
#         query_tensor = torch.load(query_name).to("cuda:0")

#         print(f"l{l}_h{h} Shape = {key_tensor.shape}")

#         ### get the keys
#         keys = key_tensor[0][h]
#         querys = query_tensor[0][h]

#         # get max and min for quest
#         max_keys, min_keys = split_and_extract_max_min(keys)


#         ############ 开始测试缓存能力
#         topk_rate = 0.5
#         token_len = key_tensor.shape[-2]
#         # topk = token_len

#         # print("token len = ", token_len)

#         # 获取topk index
#         topk_list = []
#         for i in range(token_len):
#             # topk_weight, indices = calculate_quest_topk_indices(querys[i], max_keys, min_keys, topk_rate)
#             topk_weight, indices = calculate_quest_topk_indices(querys[i], max_keys, min_keys, 1)

#             # print("indices = ", indices)

#             # wei_name = f"./quest_file/wei_l{l}_h{h}_t{i}.pt"
#             indices_name = f"./quest_file/indices_l{l}_h{h}_t{i}.pt"
#             # torch.save(topk_weight, wei_name)
#             torch.save(indices, indices_name)  
#     #         break
    #     break
    # break
