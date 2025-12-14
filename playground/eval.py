import torch
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 配置 ---
# ⚠️ 替换成你的模型路径！分别测试原始模型和修改后的模型
model_path = "/root/playground/Qwen2-1.5B-Instruct" 
# 选择一个C-Eval的学科进行测试，dev是验证集
ceval_subject = "college_programming" 
# ----------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在加载模型: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

print(f"正在加载数据集: C-Eval ({ceval_subject})")
# 加载C-Eval指定学科的验证集
dataset = load_dataset("ceval/ceval-exam", name=ceval_subject, split="dev")

# 用于从模型输出中提取答案的辅助函数
def extract_answer(text):
    """从模型生成的文本中提取第一个大写字母A, B, C, D"""
    match = re.search(r'([A-D])', text)
    if match:
        return match.group(1)
    return None # 如果没有找到任何选项，返回None

correct_count = 0
total_count = len(dataset)

print("--- 开始评测 ---")
for i, sample in enumerate(dataset):
    # 1. 构建Prompt
    question = sample['question']
    choices = f"A. {sample['A']}\nB. {sample['B']}\nC. {sample['C']}\nD. {sample['D']}"
    # 我们构建一个 "few-shot" prompt 来引导模型输出格式
    prompt = f"以下是中国关于“{ceval_subject}”的单项选择题，请直接给出正确选项的字母。\n\n题目：{question}\n{choices}\n答案："

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 2. 模型生成答案
    # max_new_tokens=5 足够只生成一个字母和少量文本
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=5)
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 3. 提取并比对答案
    model_answer = extract_answer(response_text)
    correct_answer = sample['answer']
    
    if model_answer == correct_answer:
        correct_count += 1
        print(f"样本 {i+1}/{total_count}: ✅ 正确！ (模型输出: '{response_text.strip()}', 正确答案: {correct_answer})")
    else:
        print(f"样本 {i+1}/{total_count}: ❌ 错误。 (模型输出: '{response_text.strip()}', 正确答案: {correct_answer}, 模型提取: {model_answer})")


# 4. 计算并打印最终结果
accuracy = correct_count / total_count if total_count > 0 else 0

print("\n--- 评测完成 ---")
print(f"模型: {model_path}")
print(f"评测集: C-Eval ({ceval_subject})")
print(f"正确数量: {correct_count}/{total_count}")
print(f"准确率 (Accuracy): {accuracy:.2%}")