import my_cache_load._C as _C
import torch
import sys
import time

torch.cuda.init()

_C.main_test()

# shape = _C.show_cache_shape()
# print("shape = ", shape)


# N = 2048
# N_p = int(N//2)
# d = 128
# bh = 8*32
# bh = 64

# idx_shape = (N_p, 1, bh)
# kv_shape = (N, bh, d)
# cache_shape = (N_p, bh, d)

# all_k = torch.randint(low=0, high=10000, size=kv_shape).cpu()
# all_v = all_k + 1

# cache_idx = torch.randint(low=0, high=N, size=idx_shape).to(torch.int32)
# prefetch_idx = torch.randint(low=0, high=N, size=idx_shape).to(torch.int32)

# _C.init_cache(bh, cache_idx, cache_shape)
# cur_shape = _C.show_cache_shape()
# print("shape = ", cur_shape)


# TEST_CASE = 20
# same_rate = 0.5
# same_N = int(N_p*same_rate)

# cur_idx_list = []
# for i in range(TEST_CASE):
#     cur_idx = torch.randint(low=0, high=N, size=idx_shape).to(torch.int32)
#     cur_idx[:same_N] = prefetch_idx[same_N]
#     cur_idx_list.append(cur_idx)


# gpu_k, gpu_v, cpu_k, cpu_v = _C.direct_load(prefetch_idx, all_k, all_v)
# stat = _C.update_cache(prefetch_idx, cpu_k, cpu_v)
# # print("shape = ", cpu_k.shape)
# # print("stat = ", stat)

# K = 4
# torch.cuda.synchronize()
# start1 = time.time()
# for i in range(TEST_CASE):
#     if i % K == 0:
#     # if i == 0:
#         # 直接加载并且更新cache
#         gpu_k, gpu_v, cpu_k, cpu_v = _C.direct_load(cur_idx_list[i], all_k, all_v)
#         stat = _C.update_cache(cur_idx_list[i], cpu_k, cpu_v)
#         if stat != 0:
#             print("ERROR! update cache fail! stat = ", stat)
#             sys.exit(0)
#     else:
#         gpu_k, gpu_v, unhit_list = _C.cache_load(cur_idx_list[i], all_k, all_v)
#     torch.cuda.synchronize()

#         # get hit rate
# torch.cuda.synchronize()
# end1 = time.time()
# print("cache load cost = ", end1-start1)


# torch.cuda.synchronize()
# start2 = time.time()
# for i in range(TEST_CASE):
#     # cur_idx = torch.randint(low=0, high=N, size=idx_shape).to(torch.int32)
#     # cur_idx[:same_N] = prefetch_idx[same_N]
    
#     gpu_k, gpu_v, cpu_k, cpu_v = _C.direct_load(cur_idx_list[i], all_k, all_v)
#     torch.cuda.synchronize()

# torch.cuda.synchronize()
# end2 = time.time()
# print("direct load cost = ", end2-start2)

