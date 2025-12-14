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
    
    # 检查字符串是否为空或没有单词
    if not s or not s.split():
        return s
    
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif s.split()[0].startswith("No") or s.split()[0].startswith("no"):
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

def build_qa_prompt(example, system_prompt="请根据给定的上下文简洁回答问题，答案控制在5个词以内。"):
    """构建问答提示"""
    q = example["question"]
    docs_text = "\n\n".join([f"{ctx['title']}\n{ctx['text']}" for ctx in example["ctxs"]])
    
    # 直接构建prompt字符串，不使用chat template
    prompt = f"{system_prompt}\n\n上下文：\n{docs_text}\n\n问题：{q}\n\n答案："
    
    return prompt

def test_model_accuracy(model_path, dataset_path):
    """测试模型精度"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 加载模型（使用fp16节省显存）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,  # 🚨 关键：使用fp16
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # ===== 优化的InfiniGen配置 =====
    print("启用InfiniGen优化...")
    
    skewing_matrix_path = "/workspace/SparseCache/accuracy/setup/skewing_matrix/Llama-2-7b-hf.pt"
    partial_weight_path = "/workspace/SparseCache/accuracy/setup/weights/Llama-2-7b-hf_0.2"
    
    # 🚨 不要一次性加载所有skewing matrix
    model_dtype = torch.float16  # 强制使用fp16
    
    print("开始逐层配置InfiniGen参数...")
    for layer in range(len(model.model.layers)):
        if layer % 8 == 0:
            print(f"配置第 {layer} 层...")
            
        model.model.layers[layer].self_attn.partial_weight_ratio = 0.2
        
        # 加载partial_weight_q
        try:
            partial_weight_q = torch.load(
                f"{partial_weight_path}/partial_weight_q_{layer}.pt",
                map_location='cpu'
            ).to(device='cuda', dtype=model_dtype)
            model.model.layers[layer].self_attn.partial_weight_q = partial_weight_q
        except Exception as e:
            print(f"警告：层 {layer} partial_weight_q 加载失败: {e}")
            continue
        
        # 单独加载当前层的skewing matrix
        try:
            A_full = torch.load(skewing_matrix_path, map_location='cpu')
            skewing_matrix = A_full[layer].to(device='cuda', dtype=model_dtype)
            model.model.layers[layer].self_attn.skewing_matrix = skewing_matrix
            del A_full  # 立即释放
        except Exception as e:
            print(f"警告：层 {layer} skewing_matrix 加载失败: {e}")
            continue
        
        model.model.layers[layer].self_attn.alpha = 5
        model.model.layers[layer].self_attn.capacity = 1.0
        model.model.layers[layer].self_attn.budget = 0.2
        
        # 每配置8层清理一次缓存
        if (layer + 1) % 8 == 0:
            torch.cuda.empty_cache()
    
    print("InfiniGen配置完成")
    torch.cuda.empty_cache()
    
    # 加载数据集
    eval_dataset = load_dataset(dataset_path)
    
    f1_scores = []
    generation_times = []
    
    print(f"开始测试，样本数量: {len(eval_dataset)}")
    
    for i, example in enumerate(eval_dataset):
        if i % 10 == 0:
            print(f"正在处理第 {i+1} 个样本...")
        
        # 构建提示
        prompt = build_qa_prompt(example)
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
    model_path = "/root/model/Llama-2-7b-hf"
    dataset_path = "/workspace/CacheBlend/inputs/musique_s.json"  # 需要确认路径
    
    results = test_model_accuracy(model_path, dataset_path)
    
    # 保存结果
    with open('accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("结果已保存到 accuracy_results.json")

if __name__ == "__main__":
    main()