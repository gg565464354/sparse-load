import torch
import json
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path
from itertools import chain

eval_dataset = load_dataset("/workspace/CacheBlend/inputs/musique_s.json")

# 使用transformers加载模型
model = AutoModelForCausalLM.from_pretrained(
    "/root/model/LLaMA-2-7B-32K",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/root/model/LLaMA-2-7B-32K", use_fast=False)

# 设置pad token如果没有的话
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

for ex in eval_dataset:
    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)

    # 简化prompt格式，更接近原始风格
    # 构造输入，类似原始代码的逻辑
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
        # 记录开始时间
        start_time = time.time()
        
        # 生成文本 - 调整参数
        output = model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=False,  # 贪婪解码
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=None,  # 移除temperature参数
            top_p=None,       # 移除top_p参数
        )
        
        # 记录第一个token生成时间（近似TTFT）
        first_token_time = time.time()
        
        # 解码生成的文本
        generated_ids = output[0][input_ids.shape[1]:]  # 只取新生成的部分
        res = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 计算总生成时间作为TTFT的近似
        ttft = first_token_time - start_time
    
    print(f"Raw generation: {repr(res)}")
    
    # 使用修复的解析函数
    parsed_res = safe_parse_generation(res)
    print(f"Parsed generation: {repr(parsed_res)}")
    
    print(f"TTFT with full prefill: {ttft}")
    ttft_full.append(ttft)
    
    # 使用修复的解析结果计算F1
    if parsed_res:  # 只有当解析结果非空时才计算F1
        f1 = max([compute_f1(parsed_res, answer, tokenizer) for answer in answers])
    else:
        f1 = 0.0  # 如果生成为空，F1为0
        
    f1_full.append(f1)
    print(f"Expected answers: {answers}")
    print(f"F1 score: {f1}")
    print("------------")

print("---------------Result Summary---------------------")
print(f"TTFT with full prefill: {np.mean(ttft_full)}")
print(f"F1 with full prefill: {np.mean(f1_full)}")
