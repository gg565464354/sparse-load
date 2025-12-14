import json
import torch

def get_real_id(layer_id, head_id, token_id):
    layer_num = 80
    head_num = 64
    decode_num = 100
    
    final = layer_id*head_num*decode_num + head_id*decode_num + token_id
    return final

# 读取json文件并且解析

data_path = "/NVME1/projects/qin/test_model/output_100.jsonl"

head_token_idx = []
all_jsons = []

# idx_dict = {}
# for hid in range(28*28):
#     tmp_head_dict = {}
#     for tid in range(113):
#         tmp_head_dict[tid] = []
#     idx_dict[hid] = tmp_head_dict



# 打开并读取 .jsonl 文件
all_token_len = 0
with open(data_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 去掉行尾的换行符并解析 JSON
        data_dict = json.loads(line.strip())

        if data_dict["testid"] <= 0:
            continue
            
        if data_dict["testid"] == 2:
            break
        
        # 打印或处理字典对象
        # print(data_dict)
        all_jsons.append(data_dict)
        head_token_idx.append(data_dict['keyindex'])

        # print(f"testid={data_dict['testid']} headid={data_dict['headid']} tokenid={data_dict['tokenid']}")
        
        all_token_len = len(data_dict['keyindex'])

print("num = ", len(all_jsons))

layer_num = 1
head_num = 64
decode_num = 100

# 构造测试数据
# 一个layer的所有head，在113个token的更新频率
N = all_token_len
N_p = int(N*0.5)
print("all token len = ", all_token_len)

every_layer_idx = []
for l in range(layer_num):
    layer_idx_for_all_token = []
    for t in range(decode_num):
        head_idx_for_a_token = []
        for h in range(head_num):
            real_id = get_real_id(l, h, t)
            head_idx_for_a_token.append(head_token_idx[real_id][:N_p])
    
        layer_idx_for_all_token.append(head_idx_for_a_token)
    
    layer_tensor = torch.tensor(layer_idx_for_all_token, dtype=torch.int)
    print(f"layer #{l} shape = ", layer_tensor.shape)
    every_layer_idx.append(layer_tensor)
        
# show shape
for l in range(len(every_layer_idx)):
    print("layer_idx = ", every_layer_idx[l].shape)

    torch.save(every_layer_idx[l], f"./tmp/test2_layer{l}.pt")