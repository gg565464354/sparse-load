import sys
import os
import torch
import time
import json
import numpy as np
local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, local_lib_path)
from transformers import AutoTokenizer, AutoModelForCausalLM
import collections
import string
import re
from rouge_score import rouge_scorer

# 添加 libs 路径
local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, local_lib_path)

def load_dataset(dataset_path):
    """加载测试数据集"""
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)

def normalize_answer(s):
    """标准化答案格式"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def parse_generation(s):
    """解析生成的文本"""
    s = s.lstrip('\n').split('\n')[0]
    return s

def compute_rl(pred, gold):
    """计算 Rouge-L 分数"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL = scorer.score(gold, pred)['rougeL'].fmeasure
    return rougeL

def build_fewshot_prompt(example):
    """构建few-shot提示，参考blend_samsum.py"""
    q = "\n\n"+example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    q_prompt = f"{q}"
    return doc_prompts, q_prompt

def build_samsum_prompt(example, system_prompt="Summarize the dialogue into a few short sentences. "):
    """构建SamSum数据集的提示"""
    # 对于SamSum数据集，question字段通常包含对话内容
    dialogue = example["question"]
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"dialogue:\n{dialogue}\n\n Please generate the summarization："}
    ]
    
    return messages

def test_model_accuracy(model_path, dataset_path):
    """测试模型精度"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
        # padding_strategy="least_important"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 加载数据集
    eval_dataset = load_dataset(dataset_path)
    
    rouge_scores = []
    generation_times = []
    
    print(f"开始测试，样本数量: {len(eval_dataset)}")
    
    for i, example in enumerate(eval_dataset):
        if i % 10 == 0:
            print(f"正在处理第 {i+1} 个样本...")
        
        # 构建提示
        messages = build_samsum_prompt(example)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 生成答案并计时
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,  # samsum需要更长的生成长度
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0
            )
        
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        generation_time = end_time - start_time
        generation_times.append(generation_time)
        
        # 解析生成的答案
        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # 处理生成的文本（参考blend_samsum.py）
        response = parse_generation(response)
        
        # 计算 Rouge-L 分数
        answers = example["answers"]
        rouge_l = max([compute_rl(response, answer) for answer in answers])
        rouge_scores.append(rouge_l)
        
        print(f"样本 {i+1}: 预测摘要='{response[:50]}...', 标准摘要={[ans[:30]+'...' for ans in answers]}, Rouge-L={rouge_l:.3f}, 时间={generation_time:.3f}s")
    
    # 输出结果统计
    print("\n=============== 测试结果 ===============")
    print(f"平均 Rouge-L 分数: {np.mean(rouge_scores):.4f}")
    print(f"Rouge-L 分数标准差: {np.std(rouge_scores):.4f}")
    print(f"平均生成时间: {np.mean(generation_times):.4f}s")
    print(f"生成时间标准差: {np.std(generation_times):.4f}s")
    print(f"Rouge-L > 0.3 的样本比例: {np.mean([rl > 0.3 for rl in rouge_scores]):.2%}")
    print(f"Rouge-L > 0.5 的样本比例: {np.mean([rl > 0.5 for rl in rouge_scores]):.2%}")
    
    return {
        'rouge_scores': rouge_scores,
        'generation_times': generation_times,
        'mean_rouge': np.mean(rouge_scores),
        'mean_time': np.mean(generation_times)
    }

def main():
    model_path = "/root/model/Qwen2-1.5B-Instruct"
    dataset_path = "/workspace/CacheBlend/inputs/samsum.json"  # 需要确认路径
    
    results = test_model_accuracy(model_path, dataset_path)
    
    # 保存结果
    with open('samsum_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("结果已保存到 samsum_accuracy_results.json")

if __name__ == "__main__":
    main() 