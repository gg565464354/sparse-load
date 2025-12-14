# evaluate_sparsity.py
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
import time
import json
import os
from tqdm import tqdm

# --- 配置 ---
# 模型ID
MODEL_ID = "/share/models/opt-6.7b"
# 数据集ID
DATASET_ID = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"
# 用于评测的样本数量（设置为 None 将评测整个测试集）
NUM_SAMPLES = 10
# 生成摘要的最大 token 数
MAX_NEW_TOKENS = 150
# 输出结果文件名
OUTPUT_FILENAME = "evaluation_results.json"


def check_local_model_file():
    """检查当前目录下是否存在修改过的模型文件"""
    if not os.path.exists("modeling_opt.py"):
        raise FileNotFoundError(
            "错误：未在当前目录下找到 'modeling_opt.py' 文件。\n"
            "请将您修改后的模型代码保存为 'modeling_opt.py' 并与本脚本放在同一目录。"
        )
    print("✅ 成功找到 'modeling_opt.py'，将使用本地修改版的模型代码。")


def create_prompt(sample):
    """为 CNN/DailyMail 数据集构建标准的 zero-shot prompt"""
    return f"Article: {sample['article']}\n\nSummarize the above article in a few sentences.\n\nSummary:"


def evaluate_model():
    """
    加载模型和数据集，执行生成任务，并评测性能、缓存效率和推理速度。
    """
    # 1. 加载 Tokenizer 和模型
    print(f"🚀 正在加载模型: {MODEL_ID}...")
    # Transformers 会优先加载同目录下的 modeling_*.py 文件，从而使你的修改生效
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto", # 自动将模型分发到可用设备 (GPU/CPU)
    )
    # 确保模型处于评估模式
    model.eval()
    
    # 你的代码在 OPTAttention 中添加了 heavy_hitter_masker
    # 我们可以检查它是否已成功加载
    # try:
    #     # 访问模型深层结构来确认
    #     _ = model.model.decoder.layers[3].self_attn.heavy_hitter_masker
    #     print("✅ 自定义模块 'heavy_hitter_masker' 已成功加载。")
    # except AttributeError:
    #     print("⚠️ 警告：未找到自定义模块 'heavy_hitter_masker'。请确保 'modeling_opt.py' 的修改已生效。")


    # 2. 加载和准备数据集
    print(f"📚 正在加载数据集: {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split='test')
    
    if NUM_SAMPLES is not None:
        dataset = dataset.select(range(NUM_SAMPLES))
        print(f"选择了 {NUM_SAMPLES} 个样本进行评测。")


    # 3. 执行生成和评测
    predictions = []
    references = []
    generation_times = []

    print("\n🔍 开始生成摘要并收集统计数据...")
    
    # 在评测开始前，重置你添加的统计数据
    # 根据你的代码结构，该方法在 OPTModel -> OPTDecoder 中
    if hasattr(model, 'model') and hasattr(model.model, 'reset_cache_hit_stats'):
        model.model.reset_cache_hit_stats()
        print("🔄️ 缓存命中率统计已重置。")
    else:
        print("⚠️ 警告：未找到 `reset_cache_hit_stats` 方法。无法重置统计信息。")
    
    for sample in tqdm(dataset, desc="Generating Summaries"):
        prompt = create_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
        
        start_time = time.time()
        # 使用 model.generate() 来触发解码循环
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, # 使用贪心解码以获得确定性结果
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        
        # 记录生成时间和结果
        generation_times.append(end_time - start_time)
        
        # 解码生成的 token，跳过 prompt 部分
        output_ids = generated_ids[0, inputs.input_ids.shape[1]:]
        prediction = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        predictions.append(prediction)
        references.append(sample['highlights'])
        
    # 4. 收集和分析结果
    print("\n📊 评测完成，正在计算最终结果...")
    
    # 4.1 获取缓存和稀疏化统计
    # hit_report = {}
    # if hasattr(model, 'model') and hasattr(model.model, 'get_cache_hit_report'):
    #     hit_report = model.model.get_cache_hit_report()
    #     print("\n--- 稀疏注意力与缓存统计 ---")
    #     print(json.dumps(hit_report, indent=2))
    # else:
    #     print("⚠️ 警告：未找到 `get_cache_hit_report` 方法。")


    # 4.2 计算 ROUGE 分数
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=predictions, references=references)
    print("\n--- 模型性能 (ROUGE) ---")
    for key, value in rouge_results.items():
        print(f"{key}: {value:.4f}")

    # 4.3 计算平均推理延迟
    avg_latency = sum(generation_times) / len(generation_times)
    total_time = sum(generation_times)
    print("\n--- 推理速度 ---")
    print(f"处理样本总数: {len(generation_times)}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每个样本生成耗时: {avg_latency:.3f} 秒")

    # 5. 保存所有结果到文件
    final_results = {
        "model_id": MODEL_ID,
        "num_samples": NUM_SAMPLES or len(dataset),
        "sparsity_cache_stats": hit_report,
        "rouge_scores": rouge_results,
        "performance": {
            "average_latency_sec": avg_latency,
            "total_time_sec": total_time,
        }
    }
    
    with open(OUTPUT_FILENAME, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"\n✅ 所有评测结果已保存至: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    # check_local_model_file()
    evaluate_model()