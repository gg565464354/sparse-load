import json
import os

def analyze_prompt_length(file_path):
    """
    分析 JSONL 文件中 'prompt' 字段的平均长度（字符和单词）。

    参数:
    file_path (str): .jsonl 文件的路径。
    """
    total_characters = 0
    total_words = 0
    valid_line_count = 0

    print(f"--- 正在处理文件: {os.path.basename(file_path)} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                try:
                    data = json.loads(line)
                    
                    # 检查 'prompt' 键是否存在且为字符串
                    if 'prompt' in data and isinstance(data['prompt'], str):
                        prompt_text = data['prompt']
                        
                        # 累加字符数
                        total_characters += len(prompt_text)
                        
                        # 累加单词数 (通过空格分割)
                        total_words += len(prompt_text.split())
                        
                        valid_line_count += 1
                    else:
                        print(f"  [警告] 第 {line_num} 行没有找到有效的 'prompt' 字段，已跳过。")

                except json.JSONDecodeError:
                    print(f"  [警告] 第 {line_num} 行 JSON 格式错误，已跳过。")
        
        if valid_line_count > 0:
            avg_chars = total_characters / valid_line_count
            avg_words = total_words / valid_line_count
            
            print(f"  分析完成！共处理了 {valid_line_count} 条有效数据。")
            print(f"  平均输入长度 (字符数): {avg_chars:.2f}")
            print(f"  平均输入长度 (单词数): {avg_words:.2f}\n")
        else:
            print("  文件中没有找到任何包含 'prompt' 字段的有效数据。\n")

    except FileNotFoundError:
        print(f"  [错误] 文件 '{file_path}' 未找到。\n")
    except Exception as e:
        print(f"  [错误] 处理文件时发生未知错误: {e}\n")


# --- 主程序 ---
if __name__ == "__main__":
    # 将 'your_file.jsonl' 替换成你的实际文件名
    # 如果想处理当前文件夹下所有的 .jsonl 文件，可以取消下面的注释
    
    # --- 方案一：处理单个指定文件 ---
    filename = 'rte-5.jsonl'  # <--- 在这里修改你的文件名
    analyze_prompt_length(filename)

    # --- 方案二：自动处理当前文件夹下所有 .jsonl 文件 ---
    # print("正在扫描当前文件夹下的所有 .jsonl 文件...\n")
    # current_directory = os.path.dirname(os.path.abspath(__file__))
    # for filename in os.listdir(current_directory):
    #     if filename.endswith(".jsonl"):
    #         full_path = os.path.join(current_directory, filename)
    #         analyze_prompt_length(full_path)