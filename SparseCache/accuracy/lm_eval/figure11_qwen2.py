import json
import os

with open("figure11-config.json") as f:
    config = json.load(f)
os.system("mkdir -p results")

shots = 5
partial = 0.2
capacity = 1.0

# Prepare dataset - 为Qwen2-1.5B-Instruct准备任务
for task in ["rte", "piqa"]:
    cmd = []
    cmd.append("python -u generate_task_data.py")
    cmd.append(f"--output-file results/{task}-{shots}.jsonl")
    cmd.append(f"--task-name {task}")
    cmd.append(f"--num-fewshot {shots}")
    cmd = ' '.join(cmd)
    os.system(cmd)

## Baseline
print("="*10+" Full cache " + "="*10)
# 使用Qwen2-1.5B-Instruct模型
qwen2_dir = "/root/model/Qwen2-1.5B-Instruct"
tasks = ["rte", "piqa"]
for task in tasks:
    cmd = []
    cmd.append("bash full_cache.sh")
    cmd.append(task)
    cmd.append(qwen2_dir)
    cmd.append("qwen2")  # 模型类型改为qwen2
    cmd.append(str(shots))
    cmd = ' '.join(cmd)
    print(cmd)
    os.system(cmd)
    print("-------------------------------------------")

# InfiniGen
print("="*10+" InfiniGen " + "="*10)
# 使用Qwen2-1.5B-Instruct模型
qwen2_dir = "/root/model/Qwen2-1.5B-Instruct"
tasks = ["rte", "piqa"]
for task in tasks:
    for retain_ratio in range(4):
        alpha, budget = config["Qwen2-1.5B-Instruct"][task][retain_ratio]  # 从配置中读取Qwen2参数
        cmd = []
        cmd.append("bash ours.sh")
        cmd.append(task)
        cmd.append(qwen2_dir)
        cmd.append(qwen2_dir)
        cmd.append("qwen2")  # 模型类型改为qwen2
        cmd.append(str(shots))
        cmd.append(str(partial))
        cmd.append(str(alpha))
        cmd.append(str(capacity))
        cmd.append(str(budget))
        cmd = ' '.join(cmd)
        print(cmd)
        os.system(cmd)
        print("-------------------------------------------")

## H2O
print("="*10+" H2O " + "="*10)
# 使用Qwen2-1.5B-Instruct模型
qwen2_dir = "/root/model/Qwen2-1.5B-Instruct"
tasks = ["rte", "piqa"]
for task in tasks:
    for ratio in [0.25, 0.125, 0.0625, 0.03125]:
        cmd = []
        cmd.append("bash h2o.sh")
        cmd.append(task)
        cmd.append(qwen2_dir)
        cmd.append("qwen2")  # 模型类型改为qwen2
        cmd.append(str(shots))
        cmd.append(str(ratio)) # heavy_ratio
        cmd.append(str(ratio)) # recent_ratio
        cmd = ' '.join(cmd)
        print(cmd)
        os.system(cmd)
        print("-------------------------------------------")

## Quant.
print("="*10+" Quantization " + "="*10)
# 使用Qwen2-1.5B-Instruct模型
qwen2_dir = "/root/model/Qwen2-1.5B-Instruct"
tasks = ["rte", "piqa"]
for task in tasks:
    for qbits in [8, 4, 2, 1]:
        cmd = []
        cmd.append("bash quant.sh")
        cmd.append(task)
        cmd.append(qwen2_dir)
        cmd.append("qwen2")  # 模型类型改为qwen2
        cmd.append(str(shots))
        cmd.append(str(qbits))
        cmd = ' '.join(cmd)
        print(cmd)
        os.system(cmd)
        print("-------------------------------------------")
