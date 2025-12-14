import json
import os

with open("figure11-config.json") as f:
    config = json.load(f)
os.system("mkdir -p results")

shots = 5
partial = 0.2
capacity = 1.0

# Prepare dataset
# for task in ["rte"]:
#     cmd = []
#     cmd.append("python -u generate_task_data.py")
#     cmd.append(f"--output-file results/{task}-{shots}.jsonl")
#     cmd.append(f"--task-name {task}")
#     cmd.append(f"--num-fewshot {shots}")
#     cmd = ' '.join(cmd)
#     os.system(cmd)

# InfiniGen
print("="*10+" InfiniGen " + "="*10)
# OPT
for size in ["6.7b"]:
    if size == "6.7b":
        tasks = ["rte"]
    # elif size == "13b":
    #     tasks = ["winogrande", "openbookqa"]
    # elif size == "30b":
    #     tasks = ["copa", "openbookqa"]
    for task in tasks:
        for retain_ratio in range(4):
            alpha, budget = config[f"opt-{size}"][task][retain_ratio]
            cmd = []
            cmd.append("bash ours.sh")
            cmd.append(task)
            cmd.append(f"/workspace/SparseCache/accuracy/setup/opt_model/opt-6.7b")
            cmd.append(f"facebook/opt-{size}")
            cmd.append("opt")
            cmd.append(str(shots))
            cmd.append(str(partial))
            cmd.append(str(alpha))
            cmd.append(str(capacity))
            cmd.append(str(budget))
            cmd = ' '.join(cmd)
            print(cmd)
            os.system(cmd)
            print("-------------------------------------------")

