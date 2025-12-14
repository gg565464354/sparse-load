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

layer_idx_1 = [[] for i in range(30)]
layer_idx_list_1 = [[] for i in range(30)]

for i in range(all_cnt):
    lid = i % layer_num
    # my cache
    cur_path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_b16_l{i}.pt"
    cur_path_1 = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_b1_l{i}.pt"

    # infinigen
    cur_path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_inf_b16_l{i}.pt"
    cur_path_1 = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_inf_b1_l{i}.pt"
    # cur_path = f"/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/pidx_inf_l{i}.pt"
    
    cur_pidx = torch.load(cur_path).cpu()
    cur_pidx_1 = torch.load(cur_path_1).cpu()

    cur_pidx_reshape = cur_pidx.squeeze(1).T
    cur_pidx_reshape_1 = cur_pidx_1.squeeze(1).T

    layer_idx[lid].append(cur_pidx_reshape)
    layer_idx_list[lid].append(cur_pidx_reshape.tolist())

    
    layer_idx_1[lid].append(cur_pidx_reshape_1)
    layer_idx_list_1[lid].append(cur_pidx_reshape_1.tolist())



# for i in range(layer_num):
#     layer_shape = [idx.shape for idx in layer_idx[i]]
#     print(f"#{i} {layer_shape}")
    
#     layer_shape_1 = [idx.shape for idx in layer_idx_1[i]]
#     print(f"#{i} {layer_shape_1}")


# 比较

for token_id in range(8):
    layer_id = 10
    # token_id = 6
    pidx = layer_idx[layer_id][token_id]
    pidx_1 = layer_idx_1[layer_id][token_id]

    # hid = 4

    # print("完全一致? ", torch.equal(pidx[hid], pidx_1[hid]))

    # print("pidx[0][-20:] = ", pidx[hid][-20:])
    # print("pidx_1[0][-20:] = ", pidx_1[hid][-20:])

    ############################ 测试不同batch 的id差异
    print(f"################### layer {layer_id}")
    cur_cnt = []
    for i in range(32):
        hid = i
        h_pidx = pidx[32+hid]
        h_pidx_1 = pidx_1[hid]
        cnt = 0
        for j in range(len(h_pidx)):
            if h_pidx[j] not in h_pidx_1:
                cnt += 1
        cur_cnt.append(cnt)

    print(f"token #{token_id} cnt = {cur_cnt}")

    ############################ 测试不同batch 的id差异
    # print("同一个请求不同batch id 的 pidx 相似性") # 结论,不同batch size的内容是一致的
    # print(f"################### layer {layer_id}")
    # cur_cnt = []
    # for i in range(31):
    #     hid = i
    #     h_pidx = pidx[64+hid]
    #     h_pidx_1 = pidx[hid]
    #     cnt = 0
    #     for j in range(len(h_pidx)):
    #         if h_pidx[j] not in h_pidx_1:
    #             cnt += 1
    #     cur_cnt.append(cnt)

    # print(f"token #{token_id} cnt = {cur_cnt}")




# 验证不同请求的prefetch_idx是否相似
# a1 = layer_idx[0][0]
# print("a1.shape = ", a1.shape)

# pidx_1 = a1[:32, :]
# pidx_2 = a1[32:64, :]
# pidx_3 = a1[64:96, :]

# print("完全一致? ", torch.equal(pidx_3, pidx_2))


# k_path = "/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/keys_b16_l2.pt"
# v_path = "/NVME1/projects/qin/InfiniGen-main/speedup/tmp_data/values_b16_l2.pt"
# keys = torch.load(k_path)
# values = torch.load(v_path)

# print("k shape = ", keys.shape)
# print("v shape = ", values.shape)

# k1 = keys[:, :32, :]
# k2 = keys[:, 32:64, :]
# k2 = keys[:, -32:, :]


# print("keys 完全一致? ", torch.equal(k1, k2))


# v1 = values[:, :32, :]
# v2 = values[:, 32:64, :]
# v3 = values[:, 64:96, :]

# print("values 完全一致? ", torch.equal(v1, v2))