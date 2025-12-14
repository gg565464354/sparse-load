import re
import json

log_path = "test.log"
out_path = "/NVME1/projects/qin/sparse-load/script/opt_6.7b_2_continue_group.json"

# 读取日志文件内容
with open(log_path, "r", encoding="utf-8") as f:  # 替换为你的日志文件名
    lines = f.readlines()

unhit_lens_lists = []

# 提取包含 unhit_lens = [...] 的行并解析
for line in lines:
    if "unhit_lens" in line:
        match = re.search(r"\[([^\]]+)\]", line)
        if match:
            nums_str = match.group(1)
            nums = [int(n.strip()) for n in nums_str.split(',')]
            unhit_lens_lists.append(nums)

# 输出结果
# for i, lst in enumerate(unhit_lens_lists):
#     print(f"unhit_lens[{i}] =", lst)

# # 可选：保存为 JSON 文件
# import json
# with open("parsed_unhit_lens.json", "w", encoding="utf-8") as f:
#     json.dump(unhit_lens_lists, f, indent=2)


LAYER_NUM = 30
HEAD_NUM = 32

high_unhit_record =  [[0 for i in range(HEAD_NUM)] for j in range(LAYER_NUM)]
max_unhit_record = 150

for i, lst in enumerate(unhit_lens_lists):
    # print(f"unhit_lens[{i}] =", lst)
    layer_id = i % LAYER_NUM
    for j, unhit in enumerate(lst):
        if unhit > max_unhit_record:
            high_unhit_record[layer_id][j] = 1

# 
final_group_map = {}
high_head_num = 0
for i in range(LAYER_NUM):
    low_group = []
    high_group = []
    for j in range(HEAD_NUM):
        if high_unhit_record[i][j] == 1:
            high_group.append(j)
            high_head_num += 1
        else:
            low_group.append(j)

    if high_group == []:
        final_group_map[i+2] = [low_group]
    else:
        # final_group_map[i+2] = [high_group, low_group]
        final_group_map[i+2] = [low_group, high_group]

### version 2 粗暴划分相同大小的分组
final_group_map = {}

# 手动补齐 layer 0, 1
final_group_map[0] = [[i for i in range(HEAD_NUM)]]
final_group_map[1] = [[i for i in range(HEAD_NUM)]]

# 其他layer根据 group数量二分
group_num = 2
for i in range(LAYER_NUM):
    cur_layer_id = i + 2
    group_head_num = int(HEAD_NUM // 2)

    layer_head_list = []
    for g in range(group_num):
        cur_group_head_list = []
        for h in range(group_head_num):
            cur_group_head_list.append(h + g*group_head_num)
        layer_head_list.append(cur_group_head_list)
    final_group_map[cur_layer_id] = layer_head_list
### version 2 粗暴划分相同大小的分组



print("high_head_num = ", high_head_num)
print(final_group_map)

with open(out_path, 'w', encoding='utf-8') as file:
    json.dump(final_group_map, file, ensure_ascii=False, indent=4)
