import sys
import os
import torch
import time
local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, local_lib_path)
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
# POOL_PATH = "qwen2-1.5b-low-attention-pool.json" 
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "/root/model/Qwen2-1.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
        # padding_strategy="fixed"
        # padding_strategy="least_important"
        # low_attention_pool_path=POOL_PATH 
    )
    message = []
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    while True:
        user_input = input("you: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        elif user_input.lower() == 'clear':
            os.system('clear' if os.name == 'posix' else 'cls')
            continue
        elif not user_input:
            continue
        message.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # --- 开始计时 ---
        if device == "cuda":
            torch.cuda.synchronize()  # 等待所有先前的CUDA核心完成
        start_time = time.perf_counter()

        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05
        )

        # --- 结束计时 ---
        if device == "cuda":
            torch.cuda.synchronize()  # 确保 generate 的所有CUDA核心都已完成
        end_time = time.perf_counter()
        
        # --- 计算并打印时间 ---
        duration = end_time - start_time
        print(f"模型生成耗时: {duration:.2f} 秒")

        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        message.append({"role": "assistant", "content": response})
        # model.model.print_hit_rate_summary()
        print("assistant: ", response)

if __name__ == "__main__":
    main()