import json
from transformers import AutoTokenizer

# --- 配置参数 ---
# 确保这里的模型ID与你计算池子时使用的模型ID完全一致
MODEL_ID = "/root/playground/Qwen2-1.5B-Instruct" 
# 你的低重要性Token池文件路径
POOL_FILE = "qwen2-1.5b-low-attention-pool.json"

def view_token_pool():
    """
    加载低重要性Token池并打印出每个ID对应的具体字符。
    """
    print(f"--- 正在加载分词器: {MODEL_ID} ---")
    try:
        # 1. 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"加载分词器失败，请检查模型ID是否正确: {e}")
        return

    print(f"--- 正在读取Token池文件: {POOL_FILE} ---")
    try:
        # 2. 读取JSON文件中的Token ID列表
        with open(POOL_FILE, 'r') as f:
            low_attention_ids = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{POOL_FILE}'。请确保文件路径正确。")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 '{POOL_FILE}' 不是一个有效的JSON文件。")
        return

    print("\n--- 低重要性Token池内容 ---")
    print("-" * 30)
    
    # 3. 遍历ID列表并解码
    for token_id in low_attention_ids:
        # 使用tokenizer.decode()将单个ID转换为字符串
        # 注意：将单个id放入列表中 `[token_id]` 是标准做法
        token_string = tokenizer.decode([token_id])
        
        # 为了更清晰地显示空格和换行符等特殊字符，我们用repr()
        print(f"ID: {token_id:<6}  |  Token: {repr(token_string)}")

    print("-" * 30)
    print(f"\n共显示 {len(low_attention_ids)} 个Token。")


if __name__ == "__main__":
    view_token_pool()