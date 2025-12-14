import my_cache_load._C as _C
import torch
import sys
import time
import json

import concurrent.futures


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

# shape = _C.show_cache_shape()
# print("shape = ", shape)


def cache_load_asyn(transfer_stream, prefetch_idx, all_k, all_v):
    group_cached_k, group_cached_v = _C.get_cached_kv()

    group_final_k = []
    group_final_v = []

    # Step 1: 提前异步提交 CPU 计算任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_unhit = executor.submit(_C.get_unhit_kv, prefetch_idx, all_k, all_v)

        # Step 2: 同时开始 cached_k/v 的 GPU 传输（异步）
        with torch.cuda.stream(transfer_stream):
            group_cached_gpu_k = [k.cuda(non_blocking=True) for k in group_cached_k]
            group_cached_gpu_v = [v.cuda(non_blocking=True) for v in group_cached_v]

        # Step 3: 等待 CPU 计算结束
        group_unhit_k, group_unhit_v, group_unhit = future_unhit.result()

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


def unhit_load_asyn(transfer_stream, prefetch_idx, all_k, all_v, group_cached_gpu_k, group_cached_gpu_v):
    group_final_k = []
    group_final_v = []

    # Step 1: 提前异步提交 CPU 计算任务
    group_unhit_k, group_unhit_v, group_unhit = _C.get_unhit_kv(prefetch_idx, all_k, all_v)

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



N = 7800
N_p = 7800//2
d = 128
bh = 64
# bh = 64

idx_shape = (N_p, 1, bh)
kv_shape = (N, bh, d)
cache_shape = (N_p, bh, d)

all_k = torch.randint(low=0, high=10000, size=kv_shape).cpu()
all_v = all_k + 1

class_group_ids = [
    [27, 44, 52, 56],
    [8, 11, 13, 14, 15, 17, 21, 26, 28, 29, 32, 34, 39, 43, 45, 46, 47, 49, 55, 62],
    [2, 3, 4, 6, 7, 12, 16, 18, 19, 20, 22, 23, 25, 30, 31, 33, 36, 38, 41, 42, 48, 50, 51, 53, 58, 60, 61, 63],
    [0, 1, 5, 9, 10, 24, 35, 37, 40, 54, 57, 59]
]


cache_idx = torch.randint(low=0, high=N, size=idx_shape).to(torch.int32)
_C.init_cache(bh, cache_idx, cache_shape, class_group_ids)

transfer_stream = torch.cuda.Stream()


for layer_id in range(1):
    print(f"####################### layer #{layer_id}")

    # layer_id = 0
    data_name = f"./tmp/test2_layer{layer_id}.pt"
    layer_idx = torch.load(data_name)
    decode_idxs = process_idx(layer_idx) 

    # for one_idx in decode_idxs:
    #     print("one_idx shape = ", one_idx.shape)

    cur_idx_list = decode_idxs
    test_num = len(cur_idx_list)
    print("test num = ", test_num)
    K = 4

    all_unhits = []

    cache_load_cost = []
    direct_load_cost = []

    for i in range(test_num):
        torch.cuda.synchronize()
        start1 = time.time()

        if i % K == 0:
        # if i == 0:
            # 直接加载并且更新cache
            group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = _C.generate_cache(cur_idx_list[i], all_k, all_v)
            stat = _C.update_group_cache(cur_idx_list[i], group_cpu_k, group_cpu_v)
            if stat != 0:
                print("ERROR! update cache fail! stat = ", stat)
                sys.exit(0)
            
            all_unhits.append([[]])
        else:
            # gpu_k, gpu_v, unhit_list = _C.cache_load(cur_idx_list[i], all_k, all_v)

            # cpu cache load
            # gpu_k, gpu_v, unhit_list = cache_load_asyn(transfer_stream, cur_idx_list[i], all_k, all_v)

            # gpu cache load
            gpu_k, gpu_v, unhit_list = unhit_load_asyn(transfer_stream, cur_idx_list[i], all_k, all_v, group_gpu_k, group_gpu_v)
            
            all_unhits.append(unhit_list)
            
        torch.cuda.synchronize()
        end1 = time.time()
        cache_load_cost.append(end1-start1)

    print("cache load cost = ", sum(cache_load_cost)/len(cache_load_cost))
    # print("cache_load_cost = ", cache_load_cost)


    torch.cuda.synchronize()
    start2 = time.time()
    for i in range(test_num):
        # cur_idx = torch.randint(low=0, high=N, size=idx_shape).to(torch.int32)
        # cur_idx[:same_N] = prefetch_idx[same_N]
        
        gpu_k, gpu_v, cpu_k, cpu_v = _C.direct_load(cur_idx_list[i], all_k, all_v)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    end2 = time.time()
    print("direct load cost = ", (end2-start2)/test_num)


# 
# unhit_records = []
# all_require = bh*N_p

# for i in range(len(all_unhits)):
#     unhit = all_unhits[i]
#     sum_cnt, max_cnt = get_unhit_num(unhit)
#     unhit_records.append(sum_cnt)

#     print(f"avg_unhit_rate = {sum_cnt/all_require}; max_unhit_rate = {max_cnt/N_p}; load_cost = {cache_load_cost[i]}")
