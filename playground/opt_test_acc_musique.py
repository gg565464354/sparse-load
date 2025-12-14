import sys
<<<<<<< Updated upstream
<<<<<<< Updated upstream
import json
import torch
import numpy as np
from pathlib import Path
from vllm import LLM, SamplingParams

# --- System Path Setup ---
# Add the custom transformers library path to ensure the correct version is used.
# This is necessary as per the request to use a specific local installation.
<<<<<<< Updated upstream
sys.path.insert(0, "/workspace/playground/libs")
=======
sys.path.insert(0, "/workspace/playground/libs/transformers")
>>>>>>> Stashed changes
from transformers import AutoTokenizer

# --- Utility Functions ---
# These helper functions are based on the 'utils' from the reference script.

def load_dataset(path: str) -> list:
    """Loads a dataset from a JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 在 {path} 未找到数据集文件")
        # Return an empty list to prevent crashing, and allow the program to exit gracefully.
        return []
    except json.JSONDecodeError:
        print(f"错误: 无法解析 {path} 的JSON文件")
        return []

def normalize_question(s: str) -> str:
    """Normalizes a question string by lowercasing and removing punctuation."""
    import re
    import string
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_f1(prediction: str, ground_truth: str, tokenizer) -> float:
    """Computes the F1 score between a prediction and a ground truth answer."""
    pred_tokens = tokenizer.encode(normalize_question(prediction), add_special_tokens=False)
    truth_tokens = tokenizer.encode(normalize_question(ground_truth), add_special_tokens=False)
    
    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0
        
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    if not common_tokens:
        return 0.0
        
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

def build_qa_prompt(passages_str: str, question_str: str, query_prompt: str, prefix_prompt: str) -> str:
    """
    Constructs a complete question-answering prompt from a data example.
    """
    # Combine the prefix, passages, and the question into a single prompt string.
    return f"{prefix_prompt}{passages_str}{query_prompt}\n{question_str}"

# --- Main Evaluation Script ---

def main():
    """
    Main function to run the model evaluation.
    """
    # --- Configuration ---
    model_path = "/root/model/opt-6.7b"
    dataset_path = "/workspace/CacheBlend/inputs/musique_s.json"
    
    print("--- 开始评测 ---")
    print(f"模型: {model_path}")
    print(f"数据集: {dataset_path}")

    # --- Load Dataset ---
    eval_dataset = load_dataset(dataset_path)
    if not eval_dataset:
        print("无法在没有有效数据集的情况下继续评测。正在退出。")
        return

    # --- Initialize Model and Tokenizer ---
    print("\n正在加载模型和分词器...")
    max_model_len = 0
    try:
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.5,
            trust_remote_code=True # Required for some models
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm.set_tokenizer(tokenizer)
        max_model_len = llm.llm_engine.model_config.max_model_len
        print(f"模型和分词器加载成功。模型最大长度: {max_model_len} tokens。")
    except Exception as e:
        print(f"加载模型或分词器时出错: {e}")
        return

    # --- Prompting Strategy ---
    prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be concise.\n\nPassages:\n"
    query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be concise.\n\nQuestion:"

    # --- Evaluation Loop ---
    ttfts = []
    f1_scores = []
    
    print(f"\n开始对 {len(eval_dataset)} 个样本进行评测...")
    for i, ex in enumerate(eval_dataset):
        print(f"\n------------ 样本 {i+1}/{len(eval_dataset)} ------------")
        
        answers = ex.get("answers")
        if not answers:
            print(f"警告: 样本 {i+1} 未找到答案。正在跳过。")
            continue
            
        sampling_params = SamplingParams(temperature=0, max_tokens=32)

        # 修复: 实现上下文截断逻辑以避免跳过过长的样本
        try:
            passages_str = "\n".join([p['paragraph_text'] for p in ex['context']])
        except (KeyError, TypeError):
            passages_str = str(ex.get('context', ''))
        
        question_str = ex.get('question', '')

        # 计算提示中固定部分的token长度
        prefix_tokens = tokenizer.encode(prefix_prompt, add_special_tokens=False)
        query_tokens = tokenizer.encode(f"{query_prompt}\n{question_str}", add_special_tokens=False)
        
        # 为生成答案和特殊token留出空间
        max_passage_tokens = max_model_len - len(prefix_tokens) - len(query_tokens) - sampling_params.max_tokens - 5
        
        passage_tokens = tokenizer.encode(passages_str, add_special_tokens=False)
        
        if len(passage_tokens) > max_passage_tokens:
            print(f"警告: 样本 {i+1} 的上下文过长 ({len(passage_tokens)} tokens)。将截断至 {max_passage_tokens} tokens。")
            passage_tokens = passage_tokens[:max_passage_tokens]
            passages_str = tokenizer.decode(passage_tokens)

        full_prompt = build_qa_prompt(passages_str, question_str, query_prompt, prefix_prompt)
            
        try:
            output = llm.generate([full_prompt], sampling_params)
            
            if not output or not output[0].outputs or output[0].metrics is None:
                print(f"错误: 样本 {i+1} 生成失败。输出无效或不完整。")
                ttfts.append(-1)
                f1_scores.append(0.0)
                continue

            # --- 处理并存储结果 ---
            result = output[0]
            generated_text = result.outputs[0].text.strip()
            
            # 计算首个token生成时间 (TTFT).
            ttft = result.metrics.first_token_time - result.metrics.first_scheduled_time
            ttfts.append(ttft)
            
            # 计算F1分数
            f1 = max([compute_f1(generated_text, answer, tokenizer) for answer in answers])
            f1_scores.append(f1)
            
            # 打印模型输出以供调试
            print(f"问题: {ex.get('question', 'N/A')}")
            print(f"模型生成答案: {generated_text}")
            print(f"正确答案: {answers}")
            print(f"F1 分数: {f1:.4f}")
            print(f"TTFT: {ttft:.4f} 秒")

        except Exception as e:
            print(f"在为样本 {i+1} 生成时发生意外错误: {e}")
            ttfts.append(-1)
            f1_scores.append(0.0)

    # --- Final Summary ---
    print("\n--------------- 结果总结 ---------------------")
    if ttfts:
        valid_ttfts = [t for t in ttfts if t >= 0]
        if valid_ttfts:
            print(f"平均 TTFT: {np.mean(valid_ttfts):.4f} 秒")
        else:
            print("平均 TTFT: N/A (所有生成都失败或被跳过)")
    else:
        print("平均 TTFT: N/A (没有样本被处理)")

    if f1_scores:
        # 为更准确的平均值，排除跳过/失败样本的分数
        valid_f1s = [f for i, f in enumerate(f1_scores) if ttfts[i] >= 0]
        if valid_f1s:
             print(f"平均 F1 分数 (基于成功运行): {np.mean(valid_f1s):.4f}")
        else:
            print("平均 F1 分数: N/A (所有生成都失败或被跳过)")
    else:
        print("平均 F1 分数: N/A (没有样本被处理)")
    print("----------------------------------------------------")

if __name__ == "__main__":
    main()
=======
=======
>>>>>>> Stashed changes
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

def build_qa_prompt(example, query_prompt):
    """构建问答提示，仿照 blend_musique.py 的方式"""
    q = example["question"]
    doc_prompts = []
    
    # 为每个上下文创建文档提示
    for ctx in example["ctxs"]:
        doc_text = f"{ctx['title']}\n{ctx['text']}"
        doc_prompts.append(doc_text)
    
    # 构建问题提示
    q_prompt = f"{query_prompt} {q}"
    
    return doc_prompts, q_prompt

def test_model_accuracy(model_path, dataset_path, max_length=1800):
    """测试模型精度"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
        # use_heavy_hitter_cache=False,
        padding_strategy="least_important",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 定义提示模板，仿照 blend_musique.py
    prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
    query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
    
    # 加载数据集
    eval_dataset = load_dataset(dataset_path)
    
    f1_scores = []
    generation_times = []
    skipped_samples = 0
    
    print(f"开始测试，样本数量: {len(eval_dataset)}")
    print(f"最大序列长度限制: {max_length}")
    
    for i, example in enumerate(eval_dataset):
        if i % 10 == 0:
            print(f"正在处理第 {i+1} 个样本...")
        
        # 构建提示，不使用 chat_template
        doc_prompts, q_prompt = build_qa_prompt(example, query_prompt)
        
        # 构建完整的输入提示
        full_prompt = prefix_prompt
        for doc in doc_prompts:
            full_prompt += doc + "\n\n"
        full_prompt += q_prompt
        
        # 检查输入长度，如果过长则截断
        inputs = tokenizer(full_prompt, return_tensors="pt")
        input_length = inputs["input_ids"].shape[1]
        
        if input_length > max_length:
            print(f"样本 {i+1}: 输入长度 {input_length} 超过限制 {max_length}，进行截断")
            # 截断输入
            inputs = tokenizer(
                full_prompt, 
                return_tensors="pt", 
                max_length=max_length, 
                truncation=True
            )
            input_length = inputs["input_ids"].shape[1]
        
        # 如果截断后仍然过长，跳过这个样本
        if input_length > max_length:
            print(f"样本 {i+1}: 截断后仍然过长，跳过")
            skipped_samples += 1
            continue
            
        inputs = inputs.to(device)
        
        # 生成答案并计时
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    attention_mask=inputs.get("attention_mask", None)
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
            
            print(f"样本 {i+1}: 预测答案='{response}', 标准答案={answers}, F1={f1:.3f}, 时间={generation_time:.3f}s, 输入长度={input_length}")
            
        except Exception as e:
            print(f"样本 {i+1}: 生成失败 - {str(e)}")
            skipped_samples += 1
            continue
    
    # 输出结果统计
    print("\n=============== 测试结果 ===============")
    print(f"总样本数: {len(eval_dataset)}")
    print(f"跳过样本数: {skipped_samples}")
    print(f"成功处理样本数: {len(f1_scores)}")
    print(f"平均 F1 分数: {np.mean(f1_scores):.4f}")
    print(f"F1 分数标准差: {np.std(f1_scores):.4f}")
    print(f"平均生成时间: {np.mean(generation_times):.4f}s")
    print(f"生成时间标准差: {np.std(generation_times):.4f}s")
    print(f"F1 > 0.5 的样本比例: {np.mean([f1 > 0.5 for f1 in f1_scores]):.2%}")
    
    return {
        'f1_scores': f1_scores,
        'generation_times': generation_times,
        'mean_f1': np.mean(f1_scores),
        'mean_time': np.mean(generation_times),
        'skipped_samples': skipped_samples,
        'total_samples': len(eval_dataset)
    }

def main():
    model_path = "/root/model/opt-6.7b"
    dataset_path = "/workspace/CacheBlend/inputs/musique_s.json"
    
    # 设置更保守的最大长度，留出生成空间
    max_length = 1800  # OPT-6.7B 的最大长度是 2048，留出248个token用于生成
    
    results = test_model_accuracy(model_path, dataset_path, max_length)
    
    # 保存结果
    with open('opt_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("结果已保存到 opt_accuracy_results.json")

if __name__ == "__main__":
<<<<<<< Updated upstream
    main()
>>>>>>> Stashed changes
=======
    main()
>>>>>>> Stashed changes
