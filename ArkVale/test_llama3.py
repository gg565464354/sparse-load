import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arkvale import adapter

# 1. 设置你的模型路径
# 将 "..." 替换为你的本地模型路径
path = "/share/models/Meta-Llama-3.1-8B" 

# 2. 设置设备和数据类型
dev = torch.device("cuda:0")
dtype = torch.float16

print(f"正在从 {path} 加载模型...")
# 3. 加载模型和分词器
model = (
    AutoModelForCausalLM
    .from_pretrained(path, torch_dtype=dtype, device_map=dev, attn_implementation="sdpa")
    .eval()
)
tokenizer = AutoTokenizer.from_pretrained(path)

print("模型加载完成，正在启用 ArkVale 引擎...")
# 4. 启用 ArkVale 引擎
# page_budgets 控制了 KV Cache 在 GPU 上的大小
# 如果设置为 None，则不启用“驱逐-召回”机制，相当于全量缓存
adapter.enable_arkvale(
    model, 
    dtype=dtype, 
    device=dev, 
    page_size=32,
    page_budgets=4096 // 32, # 示例值：设置 4096 token 的 GPU 缓存预算
    page_topks=32,
    n_max_bytes=40 * (1 << 30),      # GPU 内存池上限 (40 GB)
    n_max_cpu_bytes=80 * (1 << 30),  # CPU 内存池上限 (80 GB)
)
print("ArkVale 引擎已启用。")

# 5. 准备输入并进行推理
prompt = "中国的首都是哪里？"
inputs = tokenizer(prompt, return_tensors="pt").to(dev)

print(f"\n输入Prompt: {prompt}")
print("正在生成回答...")

# 生成文本
outputs = model.generate(**inputs, max_new_tokens=50, use_cache=True)

# 解码并打印结果
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n模型输出:\n", response_text)