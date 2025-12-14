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

# 绘图
plt.figure(figsize=(6,4))
plt.plot(sparsities, [vanilla_acc]*len(sparsities), label="Vanilla", linestyle="--", color="gray")
plt.plot(sparsities, topk_accs, marker="o", label="TopK-Eager")
plt.plot(sparsities, dynk_accs, marker="s", label="Cache-DynamicK-Eager")

plt.xlabel("sparsity_ratio")
plt.ylabel("accuracy")
plt.title("RTE: OPT-6.7B sparsity_ratio vs accuracy")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
from matplotlib.ticker import MultipleLocator

ax = plt.gca()
ax.set_xlim(0.1, 0.5)
# ax.set_ylim(0.4, 0.6)  # 添加这行来设置y轴范围
ax.set_ylim(0.4, 0.7)  # 添加这行来设置y轴范围
ax.xaxis.set_major_locator(MultipleLocator(0.1))
plt.savefig("rte_opt67b_sparsity_vs_acc.png", dpi=200)

# 控制台打印核对
print("Vanilla:", vanilla_acc)
print("TopK-Eager:", dict(zip(sparsities, topk_accs)))
print("DynamicK-Eager:", dict(zip(sparsities, dynk_accs)))