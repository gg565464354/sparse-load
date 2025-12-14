import torch
import re
import time
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForCausalLM
import os 
import json
# --- 配置 ---
# ⚠️ 替换成你的模型路径！
model_path = "/root/playground/Qwen2-1.5B-Instruct" 
# ----------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载模型和分词器 (只需加载一次)
print(f"正在加载模型: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

# 2. 自动获取C-Eval所有可用的科目名称
try:
    all_subjects = get_dataset_config_names("ceval/ceval-exam")
    print(f"成功获取C-Eval所有科目，共 {len(all_subjects)} 个。")
except Exception as e:
    print(f"自动获取科目失败: {e}")
    # 如果自动获取失败，使用上次报错信息中的硬编码列表作为备用
    all_subjects = ['accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine', 'business_administration', 'chinese_language_and_literature', 'civil_servant', 'clinical_medicine', 'college_chemistry', 'college_economics', 'college_physics', 'college_programming', 'computer_architecture', 'computer_network', 'discrete_mathematics', 'education_science', 'electrical_engineer', 'environmental_impact_assessment_engineer', 'fire_engineer', 'high_school_biology', 'high_school_chemistry', 'high_school_chinese', 'high_school_geography', 'high_school_history', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'ideological_and_moral_cultivation', 'law', 'legal_professional', 'logic', 'mao_zedong_thought', 'marxism', 'metrology_engineer', 'middle_school_biology', 'middle_school_chemistry', 'middle_school_geography', 'middle_school_history', 'middle_school_mathematics', 'middle_school_physics', 'middle_school_politics', 'modern_chinese_history', 'operating_system', 'physician', 'plant_protection', 'probability_and_statistics', 'professional_tour_guide', 'sports_science', 'tax_accountant', 'teacher_qualification', 'urban_and_rural_planner', 'veterinary_medicine']

# 用于存储所有科目结果的字典
results = {}

# 3. 循环测试每一个科目
total_start_time = time.time()
for subject_name in all_subjects:
    print(f"\n--- 开始评测科目: {subject_name} ---")
    subject_start_time = time.time()
    
    dataset = load_dataset("ceval/ceval-exam", name=subject_name, split="dev")
    
    correct_count = 0
    total_count = len(dataset)
    
    for i, sample in enumerate(dataset):
        question = sample['question']
        choices = f"A. {sample['A']}\nB. {sample['B']}\nC. {sample['C']}\nD. {sample['D']}"
        prompt = f"以下是中国关于“{subject_name}”的单项选择题，请直接给出正确选项的字母。\n\n题目：{question}\n{choices}\n答案："
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=5)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("题目描述：", prompt)
        print("模型回答：",response_text)
        match = re.search(r'([A-D])', response_text)
        model_answer = match.group(1) if match else None
        
        if model_answer == sample['answer']:
            correct_count += 1
            
    accuracy = correct_count / total_count if total_count > 0 else 0
    results[subject_name] = accuracy
    
    subject_end_time = time.time()
    print(f"✅ 科目 '{subject_name}' 评测完成。准确率: {accuracy:.2%}, 耗时: {subject_end_time - subject_start_time:.2f} 秒")

# 4. 汇总并打印最终结果
print("\n\n--- 所有科目评测完成 ---")
print(f"模型: {model_path}")
print("-" * 50)
print(f"{'科目':<40} | {'准确率':<10}")
print("-" * 50)
for subject, acc in results.items():
    print(f"{subject:<40} | {acc:<10.2%}")
print("-" * 50)

average_accuracy = sum(results.values()) / len(results) if results else 0
total_end_time = time.time()

print(f"📊 **平均准确率 (Average Accuracy): {average_accuracy:.2%}**")
print(f"⏱️ **总耗时: {(total_end_time - total_start_time) / 60:.2f} 分钟**")

# 创建结果目录（如果不存在）
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# 准备保存的结果数据
result_data = {
    "model_path": model_path,
    "average_accuracy": average_accuracy,
    "total_time_minutes": (total_end_time - total_start_time) / 60,
    "subject_results": results,
    "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
}

# 生成文件名（使用模型名称和时间戳）
model_name = os.path.basename(model_path.rstrip("/"))
filename = f"{results_dir}/{model_name}_ceval_results_{time.strftime('%Y%m%d_%H%M%S')}.json"

# 保存结果到JSON文件
with open(filename, "w", encoding="utf-8") as f:
    json.dump(result_data, f, ensure_ascii=False, indent=4)

print(f"\n✅ 评测结果已保存到: {filename}")