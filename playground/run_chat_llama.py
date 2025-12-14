import sys
import os
import torch
import time
import pickle  # 用于加载 partial_weight
local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, local_lib_path)
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
# POOL_PATH = "qwen2-1.5b-low-attention-pool.json" 
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "/root/autodl-tmp/Llama-2-7B-32K-Instruct/"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
        # padding_strategy="fixed"
        # padding_strategy="least_important"
        # low_attention_pool_path=POOL_PATH 
    )
    
    # ===== 添加InfiniGen配置逻辑 =====
    print("启用InfiniGen优化...")
    
    # 路径配置
    skewing_matrix_path = "/root/sparse-load/SparseCache/accuracy/setup/skewing_matrix/Llama-2-7B-32K-Instruct.pt"
    partial_weight_dir = "/root/sparse-load/SparseCache/accuracy/setup/weights/Llama-2-7B-32K-Instruct_0.2"
    
    # 加载skewing matrix
    print(f"加载skewing matrix: {skewing_matrix_path}")
    try:
        A = torch.load(skewing_matrix_path, map_location=device)
        print(f"skewing_matrix 加载成功，形状: {A.shape}")
    except Exception as e:
        print(f"加载 skewing_matrix 失败: {e}")
        A = None
    
    # 获取模型的数据类型
    model_dtype = next(model.parameters()).dtype
    print(f"模型数据类型: {model_dtype}")
    
    # 为每一层初始化参数
    if A is not None:
        for layer in range(len(model.model.layers)):
            # 设置参数
            model.model.layers[layer].self_attn.partial_weight_ratio = 0.2
            
            # 加载partial_weight_q并确保数据类型匹配
            partial_weight_file = f"{partial_weight_dir}/partial_weight_q_{layer}.pt"
            try:
                partial_weight_q = torch.load(partial_weight_file, map_location=device)
                model.model.layers[layer].self_attn.partial_weight_q = partial_weight_q.to(dtype=model_dtype)
                print(f"  Layer {layer}: 加载 partial_weight_q 成功，形状: {partial_weight_q.shape}")
            except Exception as e:
                print(f"  Layer {layer}: 加载 partial_weight_q 失败: {e}")
                continue
            
            # 设置其他参数
            model.model.layers[layer].self_attn.alpha = 5
            model.model.layers[layer].self_attn.capacity = 1.0
            model.model.layers[layer].self_attn.budget = 1.0
            
            # 确保skewing_matrix的数据类型与模型一致
            skewing_matrix = A[layer].to(dtype=model_dtype)
            model.model.layers[layer].self_attn.skewing_matrix = skewing_matrix
        
        print(f"InfiniGen配置完成: partial_weight_ratio=0.2, alpha=5, capacity=1.0, budget=0.2")
    else:
        print("跳过InfiniGen配置，因为skewing_matrix加载失败")
    # ===== InfiniGen配置结束 =====
    
    message = []
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def format_llama2_prompt(messages):
        """手动格式化 Llama-2 聊天提示"""
        if not messages:
            return ""
        
        # 构建对话历史
        formatted = "<s>"
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if i == 0:
                    formatted += f"[INST] {msg['content']} [/INST]"
                else:
                    formatted += f"</s><s>[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                formatted += f" {msg['content']}"
        
        # 如果最后一条消息是用户消息，准备生成助手回复
        if messages[-1]["role"] == "user":
            formatted += " "
        
        return formatted

    while True:
        user_input = input("you: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        elif user_input.lower() == 'clear':
            os.system('clear' if os.name == 'posix' else 'cls')
            message = []  # 清空对话历史
            continue
        elif not user_input:
            continue
        
        message.append({"role": "user", "content": user_input})
        prompt = format_llama2_prompt(message)
        
        # 调试：打印生成的提示
        print(f"Debug - 生成的提示: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # --- 开始计时 ---
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # 减少生成长度避免乱码
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,  # 明确设置pad_token_id
            eos_token_id=tokenizer.eos_token_id   # 明确设置eos_token_id
        )

        # --- 结束计时 ---
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        print(f"模型生成耗时: {duration:.2f} 秒")

        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # 清理响应中的多余空格和换行
        response = response.strip()
        
        message.append({"role": "assistant", "content": response})
        print("assistant:", response)

if __name__ == "__main__":
    main()