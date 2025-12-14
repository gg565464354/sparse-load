import sys
import os
import torch
import time
import json
import numpy as np
import argparse
import math
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

def build_samsum_prompt(example, system_prompt="You are a helpful assistant."):
    """构建SamSum数据集的提示"""
    # 直接使用question字段，它已经包含了"Dialogue: [内容]\nSummary: "的格式
    question_text = example["question"]
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_text}
    ]
    
    return messages

def test_model_accuracy(model_path, dataset_path, args):
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
    
    # 添加InfiniGen配置逻辑
    if args.ours:
        print("启用InfiniGen优化...")
        if args.skewing_matrix_path is not None:
            print(f"加载skewing matrix: {args.skewing_matrix_path}")
            A = torch.load(args.skewing_matrix_path)
        
        # 获取模型的数据类型
        model_dtype = next(model.parameters()).dtype
        print(f"模型数据类型: {model_dtype}")
        
        for layer in range(len(model.model.layers)):
            model.model.layers[layer].self_attn.partial_weight_ratio = args.partial_weight_ratio
            
            # 加载partial_weight_q并确保数据类型匹配
            partial_weight_q = torch.load(args.partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
            model.model.layers[layer].self_attn.partial_weight_q = partial_weight_q.to(dtype=model_dtype)
            
            model.model.layers[layer].self_attn.alpha = args.alpha
            model.model.layers[layer].self_attn.capacity = args.capacity
            model.model.layers[layer].self_attn.budget = args.budget
            
            if args.skewing_matrix_path is not None:
                # 确保skewing_matrix的数据类型与模型一致
                skewing_matrix = A[layer].to(dtype=model_dtype)
                model.model.layers[layer].self_attn.skewing_matrix = skewing_matrix
        
        print(f"InfiniGen配置: partial_weight_ratio={args.partial_weight_ratio}, alpha={args.alpha}, capacity={args.capacity}, budget={args.budget}")
    
    # 加载数据集
    eval_dataset = load_dataset(dataset_path)
    
    rouge_scores = []
    generation_times = []
    density = []
    
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
        
        # 收集density数据
        if args.ours:
            density.append(model.get_density())
        
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
        
        # 清理previous_hidden_states
        if args.ours:
            for layer in model.model.layers:
                layer.self_attn.previous_hidden_states = None
    
    # 计算density统计
    if args.ours and density:
        avg_density = sum(density) / len(density) * 100
        retain_ratio = (1 - math.sqrt(1 - avg_density / 100)) * 100
        print(f"\n平均density: {avg_density:.2f}%")
        print(f"retain ratio: {retain_ratio:.2f}%")
    
    # 输出结果统计
    print("\n=============== 测试结果 ===============")
    print(f"平均 Rouge-L 分数: {np.mean(rouge_scores):.4f}")
    print(f"Rouge-L 分数标准差: {np.std(rouge_scores):.4f}")
    print(f"平均生成时间: {np.mean(generation_times):.4f}s")
    print(f"生成时间标准差: {np.std(generation_times):.4f}s")
    print(f"Rouge-L > 0.3 的样本比例: {np.mean([rl > 0.3 for rl in rouge_scores]):.2%}")
    print(f"Rouge-L > 0.5 的样本比例: {np.mean([rl > 0.5 for rl in rouge_scores]):.2%}")
    
    result_dict = {
        'rouge_scores': rouge_scores,
        'generation_times': generation_times,
        'mean_rouge': np.mean(rouge_scores),
        'mean_time': np.mean(generation_times)
    }
    
    if args.ours and density:
        result_dict['density'] = density
        result_dict['avg_density'] = avg_density
        result_dict['retain_ratio'] = retain_ratio
    
    return result_dict

def main():
    parser = argparse.ArgumentParser(description='Qwen2 模型在SamSum数据集上的准确性测试')
    
    # 基本参数
    parser.add_argument('--model-path', type=str, default="/root/model/Qwen2-1.5B-Instruct", help='模型路径')
    parser.add_argument('--dataset-path', type=str, default="/workspace/CacheBlend/inputs/samsum.json", help='数据集路径')
    parser.add_argument('--output-path', type=str, default='samsum_accuracy_results.json', help='结果输出路径')
    
    # InfiniGen参数
    parser.add_argument('--ours', action='store_true', help='启用InfiniGen优化')
    parser.add_argument("--partial_weight_ratio", type=float, default=0.1, help='部分权重比例')
    parser.add_argument("--partial_weight_path", type=str, help='部分权重路径')
    parser.add_argument("--skewing_matrix_path", type=str, help='偏斜矩阵路径')
    parser.add_argument("--alpha", type=float, default=5, help='alpha参数')
    parser.add_argument("--capacity", type=float, default=1.0, help='容量参数')
    parser.add_argument("--budget", type=float, default=0.2, help='预算参数')
    
    args = parser.parse_args()
    
    # 如果启用InfiniGen但没有提供必要路径，给出提示
    if args.ours and args.partial_weight_path is None:
        print("错误：启用InfiniGen时必须提供 --partial_weight_path")
        return
    
    print(f"模型路径: {args.model_path}")
    print(f"数据集路径: {args.dataset_path}")
    print(f"InfiniGen优化: {'启用' if args.ours else '禁用'}")
    
    results = test_model_accuracy(args.model_path, args.dataset_path, args)
    
    # 保存结果
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"结果已保存到 {args.output_path}")

if __name__ == "__main__":
    main() 