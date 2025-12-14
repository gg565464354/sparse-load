# 移除vLLM相关导入，改用HuggingFace Transformers
import torch
import json
import numpy as np
import os 
import sys
local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, local_lib_path)
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path
import time

eval_dataset = load_dataset("/workspace/CacheBlend/inputs/musique_s.json")

# 使用HuggingFace Transformers加载模型
model = AutoModelForCausalLM.from_pretrained(
    "/root/model/Qwen2-1.5B-Instruct", 
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/root/model/Qwen2-1.5B-Instruct",
    use_fast=True,
    trust_remote_code=True
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

ttft_blend = []
ttft_full = []
f1_blend = []
f1_full = []

for ex in eval_dataset:
    answers = ex["answers"]
    docs_text = "".join([f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in ex["ctxs"]])
    question = ex["question"].rstrip("?") + "?"
    messages = [
        {"role": "system", "content": prefix_prompt},
        {"role": "user", "content": f"{docs_text}{query_prompt}{question}"}
    ]
    
    # 使用tokenizer的chat_template格式化消息
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = {"input_ids": input_ids.to(model.device)}
    
    # 生成设置
    generation_config = {
        "max_new_tokens": 32,
        "temperature": 0.0,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # 记录开始时间
    start_time = time.time()
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=32,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.0,
            repetition_penalty=1.0,
        )
    
    # 计算生成时间
    generation_time = time.time() - start_time
    
    # 解码生成的文本（只取新生成的部分）
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_len:]
    res = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    print(f"Normal generation: {res}")
    print(f"Generation time: {generation_time:.4f} seconds")
    
    # 计算F1分数
    f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
    f1_full.append(f1)
    print("------------")

print("---------------Result Summary---------------------")
print(f"F1 with full prefill: {np.mean(f1_full)}")
