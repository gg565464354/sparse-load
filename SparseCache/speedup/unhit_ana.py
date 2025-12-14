import re
import json

log_path = "run_log_b32.txt"
log_path = "run_log_b16.txt"
out_path = "/NVME1/projects/qin/sparse-load/script/opt_6.7b_2_group_ids.json"

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


LAYER_NUM = 30
HEAD_NUM = 32

high_unhit_record =  [[0 for i in range(HEAD_NUM)] for j in range(LAYER_NUM)]
max_unhit_record = 150

for i, lst in enumerate(unhit_lens_lists):
    # print(f"unhit_lens[{i}] =", lst)
    layer_id = i % HEAD_NUM


for i in range(len(unhit_lens_lists)):
    first = 0
    end = 32
    cur_unhit_len = unhit_lens_lists[i]

    same = []
    for j in range(HEAD_NUM):
        if cur_unhit_len[j + first] == cur_unhit_len[j+end]:
            same.append(0)
        else:
            same.append(1)
    diff_num = sum(same)
    
    print(f"unhit #{i} = {diff_num}")




