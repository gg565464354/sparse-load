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
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no"):
        s = "No"
    return s

def compute_f1(a_pred, a_gold, tokenizer):
    """计算 F1 分数"""
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

<<<<<<< Updated upstream
def build_qa_prompt(example, system_prompt="Answer briefly in 5 words or less."):
    """构建问答提示 - 使用英文简化版本"""
    q = example["question"]
    # 限制上下文长度，只取前2个文档的前200个字符
    docs_text = "\n\n".join([f"{ctx['title']}\n{ctx['text'][:200]}" for ctx in example["ctxs"][:2]])
    
    # 使用更简单的英文提示，更明确的格式
    prompt = f"Context: {docs_text}\n\nQuestion: {q}\nAnswer:"
    
    return prompt
=======
def build_qa_prompt(example, system_prompt="请根据给定的上下文简洁回答问题，答案控制在5个词以内。"):
    """构建问答提示"""
    q = example["question"]
    docs_text = "\n\n".join([f"{ctx['title']}\n{ctx['text']}" for ctx in example["ctxs"]])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"上下文：\n{docs_text}\n\n问题：{q}"}
    ]
    
    return messages
>>>>>>> Stashed changes

def test_model_accuracy(model_path, dataset_path):
    """测试模型精度"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
<<<<<<< Updated upstream
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 为OPT模型设置pad token（非常重要！）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
=======
        # use_heavy_hitter_cache=False,
        padding_strategy="least_important",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
>>>>>>> Stashed changes
    # 加载数据集
    eval_dataset = load_dataset(dataset_path)
    
    f1_scores = []
    generation_times = []
    
    print(f"开始测试，样本数量: {len(eval_dataset)}")
<<<<<<< Updated upstream
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model vocab size: {model.config.vocab_size}")
=======
>>>>>>> Stashed changes
    
    for i, example in enumerate(eval_dataset):
        if i % 10 == 0:
            print(f"正在处理第 {i+1} 个样本...")
        
<<<<<<< Updated upstream
        try:
            # 构建提示 - 使用英文简化版本
            prompt = build_qa_prompt(example)
            
            # 添加长度限制和更安全的tokenization
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512  # 进一步减少最大长度
            ).to(device)
            
            # 检查token ID是否在有效范围内
            max_token_id = inputs["input_ids"].max().item()
            if max_token_id >= tokenizer.vocab_size:
                print(f"警告：发现超出范围的token ID: {max_token_id}, vocab_size: {tokenizer.vocab_size}")
                continue
            
            # 生成答案并计时
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,  # 严格限制到10个token
                    min_new_tokens=1,   # 至少生成1个token
                    do_sample=False,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # 移除stop_strings，改用后处理
                )
            
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            generation_time = end_time - start_time
            generation_times.append(generation_time)
            
            # 解析生成的答案
            response_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # 进一步清理响应，只取第一个句子/段落
            response = response.split('\n')[0].strip()
            response = response.split('.')[0].strip()
            response = response.split('Question:')[0].strip()  # 添加更多停止条件
            response = response.split('Context:')[0].strip()
            
            # 限制词数（以空格分割）
            words = response.split()
            if len(words) > 5:
                response = ' '.join(words[:5])
            
            # 计算 F1 分数
            answers = example["answers"]
            f1 = max([compute_f1(response, answer, tokenizer) for answer in answers])
            f1_scores.append(f1)
            
            print(f"样本 {i+1}: 预测答案='{response}', 标准答案={answers}, F1={f1:.3f}, 时间={generation_time:.3f}s")
            
        except Exception as e:
            print(f"样本 {i+1} 处理失败: {e}")
            continue
=======
        # 构建提示
        messages = build_qa_prompt(example)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 生成答案并计时
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
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
        
        # 计算 F1 分数
        answers = example["answers"]
        f1 = max([compute_f1(response, answer, tokenizer) for answer in answers])
        f1_scores.append(f1)
        
        print(f"样本 {i+1}: 预测答案='{response}', 标准答案={answers}, F1={f1:.3f}, 时间={generation_time:.3f}s")
>>>>>>> Stashed changes
    
    # 输出结果统计
    print("\n=============== 测试结果 ===============")
    print(f"平均 F1 分数: {np.mean(f1_scores):.4f}")
    print(f"F1 分数标准差: {np.std(f1_scores):.4f}")
    print(f"平均生成时间: {np.mean(generation_times):.4f}s")
    print(f"生成时间标准差: {np.std(generation_times):.4f}s")
    print(f"F1 > 0.5 的样本比例: {np.mean([f1 > 0.5 for f1 in f1_scores]):.2%}")
    
    return {
        'f1_scores': f1_scores,
        'generation_times': generation_times,
        'mean_f1': np.mean(f1_scores),
        'mean_time': np.mean(generation_times)
    }

def main():
    model_path = "/root/model/opt-6.7b"
    dataset_path = "/workspace/CacheBlend/inputs/musique_s.json"  # 需要确认路径
    
    results = test_model_accuracy(model_path, dataset_path)
    
    # 保存结果
    with open('accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("结果已保存到 accuracy_results.json")

if __name__ == "__main__":
    main()