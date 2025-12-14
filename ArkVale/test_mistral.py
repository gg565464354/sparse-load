import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arkvale import adapter

# 1. 设置你的模型路径
# 将路径修改为你的 Mistral-7B 模型所在的文件夹
path = "/share/models/Mistral-7B-Instruct-v0.2" # <--- 修改这里

# 2. 设置设备和数据类型
dev = torch.device("cuda:0")
dtype = torch.float16

print(f"正在从 {path} 加载模型...")
# 3. 加载模型和分词器
# 注意：移除了 attn_implementation 参数以兼容旧版 transformers
model = (
    AutoModelForCausalLM
    .from_pretrained(path, torch_dtype=dtype, device_map=dev, attn_implementation="sdpa")
    .eval()
)
tokenizer = AutoTokenizer.from_pretrained(path)

print("模型加载完成，正在启用 ArkVale 引擎...")
# 4. 启用 ArkVale 引擎
adapter.enable_arkvale(
    model, 
    dtype=dtype, 
    device=dev, 
    page_size=32,
    page_budgets=4096 // 32,
    page_topks=32,
    n_max_bytes=40 * (1 << 30),
    n_max_cpu_bytes=30 * (1 << 30),
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