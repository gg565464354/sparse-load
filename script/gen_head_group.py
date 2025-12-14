'''Design for opt 6.7b'''

import json
import torch

LAYER_NUM = 32
HEAD_NUM = 32


def read_jsonl_file(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除行末的换行符并解析JSON格式的字符串
            json_object = json.loads(line.strip())
            # 对解析后的JSON对象进行处理
            # print(json_object)
            result.append(json_object)
    
    return result


# 使用函数读取JSONL文件
json_result = read_jsonl_file('./opt_categorized_heads.jsonl')
# json_result = read_jsonl_file('./opt_one_group_heads.jsonl')

all_layer_group_ids = {}

for l in range(LAYER_NUM):
    layer_group_ids = [[], [], [], []]
    for h in range(HEAD_NUM):
        id = l*LAYER_NUM + h
        cur_jsonl = json_result[id]

        # four group
        # if cur_jsonl["category"] == "< 0.7":
        #     layer_group_ids[0].append(h)
        # elif cur_jsonl["category"] == ">= 0.7 and < 0.8":
        #     layer_group_ids[1].append(h)
        # elif cur_jsonl["category"] == ">= 0.8 and < 0.9":
        #     layer_group_ids[2].append(h)
        # elif cur_jsonl["category"] == ">= 0.9":
        #     layer_group_ids[3].append(h)

        # one group
        layer_group_ids[0].append(h)
    
    final_group_ids = []

    # 对特殊的分组进行处理，防止部分分组head过少
    for group in layer_group_ids:
        if len(group) != 0:
            final_group_ids.append(group)
    
    all_layer_group_ids[l] = final_group_ids


# 打印最终结果
# for i in range(LAYER_NUM):

with open('opt_6.7b_one_group_ids.json', 'w', encoding='utf-8') as file:
    json.dump(all_layer_group_ids, file, ensure_ascii=False, indent=4)