import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import json

# --- 配置参数 ---
MODEL_ID = "/root/playground/Qwen2-1.5B-Instruct"  # 替换成你的模型ID
DATASET_NAME = "c4"
DATASET_CONFIG = "en"
NUM_SAMPLES = 10000  # 用于计算的数据样本量，可根据算力调整
POOL_SIZE = 512      # 你想要的低重要性池的大小
OUTPUT_FILE = "qwen2-1.5b-low-attention-pool.json" # 输出文件名

def precompute_low_attention_pool():
    """
    一个完整的函数，用于预计算并保存全局低重要性Token池。
    """
    # --- 第1步：准备工作 ---
    # ... (这部分代码不变) ...
    print(f"--- 步骤 1: 加载模型 {MODEL_ID} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager" 
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print(f"模型已加载到 {model.device}")

    # --- 第2步：选择数据集 ---
    print(f"--- 步骤 2: 以流式模式加载数据集 {DATASET_NAME} ---")
    
    # ### START OF MODIFICATION ###
    # 1. 设置 streaming=True，并且移除 split 中的切片
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)
    # ### END OF MODIFICATION ###
    
    # --- 第3步：提取与聚合注意力分数 ---
    print("--- 步骤 3: 开始提取和聚合注意力分数 ---")
    
    token_attention_scores = defaultdict(lambda: [0.0, 0])

    with torch.no_grad():
        # ### START OF MODIFICATION ###
        # 2. 使用 .take(NUM_SAMPLES) 来限制处理的样本数量
        for sample in tqdm(dataset.take(NUM_SAMPLES), desc="处理样本中", total=NUM_SAMPLES):
        # ### END OF MODIFICATION ###
            text = sample['text']
            # ... (后续的循环内代码不变) ...
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
            input_ids = inputs["input_ids"][0]

            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions
            attentions = torch.stack(attentions).squeeze(1)

            received_attention = attentions.sum(dim=-2)
            max_head_attention, _ = torch.max(received_attention, dim=1)
            max_layer_attention, _ = torch.max(max_head_attention, dim=0)

            for i, token_id in enumerate(input_ids):
                token_id_item = token_id.item()
                if token_id_item in tokenizer.all_special_ids:
                    continue
                score = max_layer_attention[i].item()
                token_attention_scores[token_id_item][0] += score
                token_attention_scores[token_id_item][1] += 1
    
    # --- 第4步和第5步不变 ---
    # ... (计算、排序和保存的代码不变) ...
    print("--- 步骤 4: 计算平均分并排序 ---")
    average_scores = {}
    for token_id, (total_score, count) in token_attention_scores.items():
        if count > 0:
            average_scores[token_id] = total_score / count
    sorted_token_ids = sorted(average_scores.keys(), key=lambda token_id: average_scores[token_id])

    print(f"--- 步骤 5: 生成并保存Top-{POOL_SIZE}低重要性Token池到 {OUTPUT_FILE} ---")
    low_attention_pool = sorted_token_ids[:POOL_SIZE]
    
    print("\n--- 最不重要的10个Token示例 ---")
    for token_id in low_attention_pool[:10]:
        print(f"Token: '{tokenizer.decode([token_id])}' (ID: {token_id}), Score: {average_scores[token_id]:.4f}")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(low_attention_pool, f)
        
    print(f"\n成功！低重要性Token池已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    precompute_low_attention_pool()