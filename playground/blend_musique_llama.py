import os, sys, json
import numpy as np
import torch
import time
# 确保使用本地 transformers 库
local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, local_lib_path)
# 确保优先使用本地 transformers 而不是 site-packages
if 'transformers' in sys.modules:
    del sys.modules['transformers']

# 临时禁用 torchvision 以避免版本冲突
sys.modules['torchvision'] = None
sys.modules['torchvision.transforms'] = None

print(f"Using transformers from: {__import__('transformers').__file__}")
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path
from itertools import chain

eval_dataset = load_dataset("/workspace/CacheBlend/inputs/musique_s.json")

# 使用自定义的 LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained(
    "/root/model/LLaMA-2-7B-32K",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/root/model/LLaMA-2-7B-32K", use_fast=False)

# 注入 skewing_matrix 与 partial_weight_q（确保不触发SparseCache逻辑）
try:
    skew_path = "/workspace/SparseCache/accuracy/setup/skewing_matrix/LLaMA-2-7B-32K.pt"
    weights_dir = "/workspace/SparseCache/accuracy/setup/weights/LLaMA-2-7B-32K_0.2"
    A = torch.load(skew_path, map_location="cpu")

    partial_weight_ratio = 0.2
    alpha = 5.0
    capacity = 1.0
    budget = 0.2

    device = model.device
    dtype = next(model.parameters()).dtype

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.partial_weight_ratio = partial_weight_ratio
        attn.partial_weight_q = torch.load(
            os.path.join(weights_dir, f"partial_weight_q_{layer_idx}.pt"),
            map_location="cpu"
        ).to(device=device, dtype=dtype)
        attn.alpha = alpha
        attn.capacity = capacity
        attn.budget = budget
        skew = A[layer_idx] if isinstance(A, (list, tuple)) else A[layer_idx]
        attn.skewing_matrix = skew.to(device=device, dtype=dtype)
        
        # 关键：确保previous_hidden_states为None，这样就不会触发SparseCache逻辑
        attn.previous_hidden_states = None
        
    print("Successfully injected weights/skewing_matrix (SparseCache disabled)")
except Exception as e:
    print(f"Warning: failed to inject weights/skewing_matrix: {e}")

# 设置pad token如果没有的话
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 添加一个简单的测试来验证模型
print("Testing model with a simple prompt...")
simple_test = "The capital of France is"
test_ids = tokenizer.encode(simple_test, return_tensors="pt").to(model.device)
with torch.no_grad():
    test_output = model.generate(
        test_ids,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
test_generated = test_output[0][test_ids.shape[1]:]
test_result = tokenizer.decode(test_generated, skip_special_tokens=True)
print(f"Simple test result: {repr(test_result)}")
print("=" * 50)

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

ttft_full = []
f1_full = []

# 修复parse_generation函数（临时解决方案）
def safe_parse_generation(s):
    s = s.lstrip('\n').split('\n')[0].strip()
    if not s:  # 如果字符串为空
        return ""
    
    words = s.split()
    if not words:  # 如果没有单词
        return ""
        
    if s.startswith("Yes") or s.startswith("yes"):
        return "Yes"
    elif words[0].startswith("No") or words[0].startswith("no"):
        return "No"
    return s

# 只测试第一个样本
for i, ex in enumerate(eval_dataset):
    if i >= 1:  # 只测试一个样本
        break
        
    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)

    # 简化prompt格式，更接近原始风格
    input_text = prefix_prompt
    
    # 添加所有文档
    for doc in doc_prompts:
        input_text += doc
    
    # 添加问题
    input_text += q_prompt
    
    print(f"Input prompt length: {len(input_text)} chars")
    print(f"Question: {ex['question']}")
    
    # 直接tokenize
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    print(f"Input tokens: {input_ids.shape[1]}")
    
    # 使用transformers生成
    with torch.no_grad():
        start_time = time.time()
        
        output = model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        first_token_time = time.time()
        
        generated_ids = output[0][input_ids.shape[1]:]
        print(f"Generated token IDs: {generated_ids.tolist()}")
        print(f"Number of generated tokens: {len(generated_ids)}")
        
        if len(generated_ids) > 0:
            for idx, token_id in enumerate(generated_ids):
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                print(f"Token {idx}: ID={token_id}, Text={repr(token_text)}")
        
        res = tokenizer.decode(generated_ids, skip_special_tokens=True)
        ttft = first_token_time - start_time
    
    print(f"Raw generation: {repr(res)}")
    parsed_res = safe_parse_generation(res)
    print(f"Parsed generation: {repr(parsed_res)}")
    print(f"TTFT with full prefill: {ttft}")
    print(f"Expected answers: {answers}")
    break

print("Test completed.")
