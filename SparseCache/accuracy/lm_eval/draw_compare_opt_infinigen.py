import json
import re
import os
import matplotlib.pyplot as plt

# 绝对路径
base = "/workspace/SparseCache/accuracy/lm_eval"

# 稀疏比例
sparsities = [0.1, 0.2, 0.3, 0.4, 0.5]

# 文件路径模板
vanilla_fp = f"{base}/evaluation_rte-5-opt-6.7b-vanilla.json"
topk_tpl = f"{base}/evaluation_rte-5-opt-6.7b-gqacache-topk-eager-{{:.1f}}.json"
dynk_tpl = f"{base}/evaluation_rte-5-opt-6.7b-gqacache-dynamic_k-eager-{{:.1f}}.json"

# 五个infinigen文件路径（忽略alpha和capacity，按budget分组）
infinigen_files = [
    ("0.2", f"{base}/evaluation_rte-5-opt-6.7b-infinigen-2-0.2-0.75.json"),
    ("0.2", f"{base}/evaluation_rte-5-opt-6.7b-infinigen-4-0.2-0.75.json"),
    ("0.4", f"{base}/evaluation_rte-5-opt-6.7b-infinigen-5-0.4-0.75.json"),
    ("0.6", f"{base}/evaluation_rte-5-opt-6.7b-infinigen-7-0.6-0.75.json"),
    ("0.7", f"{base}/evaluation_rte-5-opt-6.7b-infinigen-7-0.7-0.75.json")
]

_acc_pattern = re.compile(r'"acc,none"\s*:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)')

def read_acc(fp):
    # 优先标准 JSON
    try:
        with open(fp, "r") as f:
            return json.load(f)["rte"]["acc,none"]
    except Exception:
        # 回退：从文本中正则提取 acc,none
        try:
            with open(fp, "r") as f:
                text = f.read(2_000_000)  # 读前 2MB 足够覆盖头部指标
            m = _acc_pattern.search(text)
            if not m:
                raise ValueError(f"未在文件中找到 acc,none: {fp}")
            return float(m.group(1))
        except Exception as e:
            raise RuntimeError(f"解析失败: {fp} -> {e}")

# 读取 vanilla 的 acc 作为水平线
vanilla_acc = read_acc(vanilla_fp)

# 读取 topk-eager 的 acc
topk_accs = [read_acc(topk_tpl.format(s)) for s in sparsities]

# 读取 dynamic_k-eager 的 acc
dynk_accs = [read_acc(dynk_tpl.format(s)) for s in sparsities]

# 读取所有infinigen文件的acc，按budget分组
infinigen_accs = {}
for budget, filepath in infinigen_files:
    try:
        acc = read_acc(filepath)
        if budget not in infinigen_accs:
            infinigen_accs[budget] = []
        infinigen_accs[budget].append(acc)
        print(f"Infinigen budget={budget}: {acc}")
    except Exception as e:
        print(f"读取 budget={budget} 失败: {e}")

# 计算每个budget组合的平均值
infinigen_avg_accs = {}
for budget, accs in infinigen_accs.items():
    avg_acc = sum(accs) / len(accs)
    infinigen_avg_accs[budget] = avg_acc
    print(f"Budget={budget} 平均准确度: {avg_acc:.4f} (共{len(accs)}个样本)")

# 绘图
plt.figure(figsize=(10,6))
plt.plot(sparsities, [vanilla_acc]*len(sparsities), label="Vanilla", linestyle="--", color="gray", linewidth=2)
plt.plot(sparsities, topk_accs, marker="o", label="TopK-Eager", linewidth=2)
plt.plot(sparsities, dynk_accs, marker="s", label="Cache-DynamicK-Eager", linewidth=2)

# 添加所有infinigen数据作为水平线（按budget分组）
colors = ["red", "blue", "green", "orange", "purple"]
markers = ["^", "v", "<", ">", "D"]
for i, (budget, avg_acc) in enumerate(infinigen_avg_accs.items()):
    label = f"Infinigen (budget={budget})"
    plt.plot(sparsities, [avg_acc]*len(sparsities), 
             marker=markers[i], label=label, color=colors[i], linewidth=2)

plt.xlabel("sparsity_ratio")
plt.ylabel("accuracy")
plt.title("RTE: OPT-6.7B sparsity_ratio vs accuracy")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
from matplotlib.ticker import MultipleLocator

ax = plt.gca()
ax.set_xlim(0.1, 0.5)
ax.xaxis.set_major_locator(MultipleLocator(0.1))
plt.savefig("rte_opt67b_infinigen_sparsity_vs_acc.png", dpi=200, bbox_inches='tight')

# 控制台打印核对
print("\n=== 结果汇总 ===")
print("Vanilla:", vanilla_acc)
print("TopK-Eager:", dict(zip(sparsities, topk_accs)))
print("DynamicK-Eager:", dict(zip(sparsities, dynk_accs)))
print("Infinigen结果（忽略alpha和capacity，按budget分组）:")
for budget, avg_acc in infinigen_avg_accs.items():
    print(f"  budget={budget}: {avg_acc:.4f}")