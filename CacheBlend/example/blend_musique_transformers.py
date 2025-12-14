import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os

# --- 模拟工具函数 ---
# 假设存在 utils.py 或者为了可运行性，我们在这里定义模拟函数。
# 请确保根据您的实际文件结构调整 'utils' 导入或使用这些模拟函数。

def load_dataset(path):
    """
    从文件路径加载真实的 JSON 或 JSONL 数据集。
    """
    print(f"Attempting to load REAL data from file: {path}")
    if not os.path.exists(path):
        print(f"Error: File not found at path: {path}")
        return []

    try:
        # 尝试作为标准JSON文件（一个包含所有样本的列表）读取
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} records from JSON file.")
        return data
    except json.JSONDecodeError:
        # 如果标准JSON失败，尝试作为JSON Lines (JSONL) 文件读取（每行一个JSON对象）
        print("Standard JSON loading failed. Trying JSONL format...")
        data = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip(): # 避免空行
                        data.append(json.loads(line))
            print(f"Successfully loaded {len(data)} records from JSONL file.")
            return data
        except Exception as e:
            print(f"Error loading JSONL file {path}: {e}")
            return []
    except Exception as e:
        print(f"An unexpected error occurred while reading {path}: {e}")
        return []

def build_qa_prompt(ex, query_prompt_template):
    """
    根据 Musique 样本构建文档提示和问题提示。
    使用正确的键名 'ctxs' 和 'text'。
    """
    
    # 1. 使用正确的键名 "ctxs" 获取上下文列表。
    #    使用 .get() 增加鲁棒性，防止某些样本缺少 "ctxs" 键。
    context_list = ex.get("ctxs", [])
    
    # 2. 遍历列表时，使用正确的键名 "text" 获取每个段落的具体内容。
    #    同样使用 .get() 防止内部结构不一致。
    full_context = "\n".join([p.get("text", "") for p in context_list])
    
    # 3. 构建最终的问题提示部分
    #    确保 'question' 键名是正确的（根据您的JSON样本，它是正确的）。
    question_text = ex.get("question", "")
    q_prompt = f"{query_prompt_template} {question_text}"
    
    # doc_prompts 现在包含一个包含所有上下文的字符串列表（如果需要分块处理，逻辑会更复杂）
    doc_prompts = [full_context] 
    
    return doc_prompts, q_prompt
def normalize_question(question):
    # 简单的标准化函数示例
    return question.strip()

def compute_f1(prediction, ground_truth, tokenizer=None):
    """
    计算 F1 分数的简化实现。
    """
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    common_tokens = set(pred_tokens) & set(gt_tokens)
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    if not common_tokens:
        return 0.0
        
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
# --- 模拟工具函数结束 ---


class TransformersBlend:
    """
    使用 Hugging Face Transformers 实现 KV 缓存管理和生成。
    """
    def __init__(self, model_path, device="cuda"):
        print(f"Initializing model from: {model_path}")
        
        # 首先尝试加载 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                print("Setting pad_token = eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            raise

        # 尝试多种方式加载模型
        try:
            # 方法1: 使用 AutoModelForCausalLM，修复参数名
            print("Attempting to load with AutoModelForCausalLM...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,  # 保持原有参数名，某些版本仍支持
                device_map="auto",
                trust_remote_code=True  # 添加这个参数
            )
        except Exception as e1:
            print(f"AutoModel failed: {e1}")
            try:
                # 方法2: 使用新的参数名
                print("Trying with 'dtype' parameter...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e2:
                print(f"New parameter format failed: {e2}")
                try:
                    # 方法3: 显式使用 OPTForCausalLM
                    print("Trying with explicit OPTForCausalLM...")
                    from transformers import OPTForCausalLM
                    self.model = OPTForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                except Exception as e3:
                    print(f"Explicit OPT model failed: {e3}")
                    raise Exception(f"All model loading attempts failed. Last error: {e3}")

        self.model.eval()
        # 获取模型实际所在的设备，以确保张量在正确的设备间移动
        self.device = next(self.model.parameters()).device
        print(f"Model loaded successfully on device: {self.device}")


def main():
    # --- 配置 ---
    # !!重要!!:请将此路径修改为您本地存储的 opt-6.7b 模型路径
    model_path = "/share/models/opt-6.7b" 
    dataset_path = "/workspace/CacheBlend/inputs/musique_s.json"
    max_examples_to_run = 100 # 限制运行的样本数量以进行快速测试

    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        print("Please update 'model_path' variable in the code to point to your model directory.")
        # return # 在实际运行中，如果路径错误应在此处停止

    # --- 初始化模型 ---
    # 使用 try-except 块来捕获可能的加载错误（例如路径问题或内存不足）
    try:
        blend_model = TransformersBlend(model_path)
    except Exception as e:
        print(f"Failed to initialize model '{model_path}'. Error: {e}")
        return

    # --- 加载数据集 ---
    eval_dataset = load_dataset(dataset_path)
    
    # --- 提示词定义 ---
    prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words.\nPassages:\n"
    query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
    
    # --- 结果记录 ---
    results = {
        "ttft_blend": [], "f1_blend": [],
        "ttft_full": [], "f1_full": []
    }
    
    # ==================== 特殊 Token 定义 (针对 OPT 优化) ====================
    # 对于标准的基础版 OPT 模型，我们不使用 Llama/Mistral 等模型特有的指令 token ID。
    # 我们只在序列开头添加标准的 BOS token。
    
    # 1. s_start_full: 包含序列开始符 (BOS) 和前缀提示词。
    #    使用 add_special_tokens=False 来避免 tokenizer 自动添加额外的 BOS/EOS。
    s_start_full_prompt_ids = blend_model.tokenizer.encode(prefix_prompt, add_special_tokens=False)
    # 手动添加 BOS token ID (Beginning of Sequence)
    s_start_full = [blend_model.tokenizer.bos_token_id] + s_start_full_prompt_ids

    # 2. s_start: 块之间的分隔符。对于 OPT，留空即可，依靠提示中的换行符。
    s_start = [] 
    
    # 3. s_end: 问题之后、答案之前的分隔符。留空。
    s_end = []
    # =======================================================================

    print(f"\nStarting evaluation loop (max {max_examples_to_run} examples)...")
    print(f"Using BOS token ID: {blend_model.tokenizer.bos_token_id}, EOS token ID: {blend_model.tokenizer.eos_token_id}")

    for ex_index, ex in enumerate(eval_dataset):
        # if ex_index >= max_examples_to_run:
        #     print(f"\nReached maximum examples limit ({max_examples_to_run}). Stopping evaluation.")
        #     break
            
        answers = ex["answers"]
        # build_qa_prompt 返回 [文档块列表], 问题字符串
        doc_prompts, q_prompt_text = build_qa_prompt(ex, query_prompt)
        
        # --- 1. 编码输入序列 ---
        # 编码文档块。使用 add_special_tokens=False 避免在中间插入BOS/EOS。
        # 注意：原代码的 build_qa_prompt 逻辑需要确认 doc_prompts 是如何组织的。
        # 假设 doc_prompts 是一个字符串列表，每个元素是一个文档块。
        doc_chunk_ids_list = [blend_model.tokenizer.encode(doc, add_special_tokens=False) for doc in doc_prompts]
        q_ids = blend_model.tokenizer.encode(q_prompt_text, add_special_tokens=False)
        
        # --- 2. 构建完整输入序列 ---
        # 序列结构: [BOS + prefix] + [doc1] + [doc2] + ... + [question]
        full_sequence_ids = list(s_start_full) # 复制 s_start_full
        
        prefix_len = len(full_sequence_ids) # 记录前缀结束位置（不含文档）
        
        for chunk_ids in doc_chunk_ids_list:
            full_sequence_ids.extend(s_start) # s_start 为空 []
            full_sequence_ids.extend(chunk_ids)
        
        # 记录缓存点（前缀 + 所有文档）的长度
        cache_cutoff_point = len(full_sequence_ids)
        
        # 添加问题部分
        full_sequence_ids.extend(s_start) # s_start 为空 []
        full_sequence_ids.extend(q_ids)
        full_sequence_ids.extend(s_end)   # s_end 为空 []

        # --- 3. 准备Pytorch张量 ---
        input_tensor = torch.tensor([full_sequence_ids])
        prefix_ids = torch.tensor([full_sequence_ids[:cache_cutoff_point]])
        suffix_ids = torch.tensor([full_sequence_ids[cache_cutoff_point:]])

        # 安全检查：确保后缀不是空的
        if suffix_ids.shape[1] == 0:
            print(f"Warning: Skipping example {ex_index} due to empty suffix after tokenization.")
            continue

        print(f"\n--- Example {ex_index+1}/{len(eval_dataset)} ---")
        print(f"Input length: {input_tensor.shape[1]}, Prefix length: {prefix_ids.shape[1]}, Suffix length: {suffix_ids.shape[1]}")

        # --- 4. 实验组 (带缓存生成) ---
        try:
            # 步骤 4a: 计算并缓存前缀 (prefill)
            prefix_cache = blend_model.collect_kv_cache(prefix_ids)
            
            # 步骤 4b: 使用缓存生成后缀
            res_blend, ttft_cached = blend_model.generate_with_cache(
                suffix_ids, cached_kvs=prefix_cache, max_tokens=32
            )
            
            f1 = max([compute_f1(res_blend, answer) for answer in answers])
            results["ttft_blend"].append(ttft_cached)
            results["f1_blend"].append(f1)
            print(f"Cached generation result: '{res_blend}' (F1: {f1:.4f}, TTFT: {ttft_cached:.4f}s)")
            del prefix_cache # 释放内存

        except Exception as e:
            print(f"Error during cached generation for example {ex_index}: {e}")
            torch.cuda.empty_cache()

        # --- 5. 对照组 (无缓存生成) ---
        try:
            res_full, ttft_no_cache = blend_model.generate_with_cache(
                input_tensor, cached_kvs=None, max_tokens=32
            )
            
            f1 = max([compute_f1(res_full, answer) for answer in answers])
            results["ttft_full"].append(ttft_no_cache)
            results["f1_full"].append(f1)
            print(f"Full prefill result:    '{res_full}' (F1: {f1:.4f}, TTFT: {ttft_no_cache:.4f}s)")

        except Exception as e:
            print(f"Error during full prefill generation for example {ex_index}: {e}")

        torch.cuda.empty_cache()

    # --- 6. 打印总结 ---
    print("\n--------------- Result Summary ---------------")
    if results["ttft_blend"]:
        print(f"Avg TTFT with cache:      {np.mean(results['ttft_blend']):.4f} seconds")
        print(f"Avg F1 with cache:        {np.mean(results['f1_blend']):.4f}")
    if results["ttft_full"]:
        print(f"Avg TTFT with full prefill: {np.mean(results['ttft_full']):.4f} seconds")
        print(f"Avg F1 with full prefill: {np.mean(results['f1_full']):.4f}")

if __name__ == "__main__":
    main()