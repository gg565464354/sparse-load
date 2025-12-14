import os
import json

def calculate_average_for_single_file(file_path):
    """
    计算单个 JSONL 文件中 'length' 字段的平均值。

    参数:
    file_path (str): JSONL 文件的路径。

    返回:
    tuple: (平均值, 有效条目数)。如果无法计算则返回 (None, 0)。
    """
    total_length = 0
    item_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                
                try:
                    data = json.loads(line)
                    if 'length' in data and isinstance(data['length'], (int, float)):
                        total_length += data['length']
                        item_count += 1
                except json.JSONDecodeError:
                    print(f"  [警告] 文件 '{os.path.basename(file_path)}' 的第 {line_num} 行 JSON 格式错误，已跳过。")

        if item_count == 0:
            return None, 0

        return total_length / item_count, item_count

    except FileNotFoundError:
        print(f"  [错误] 找不到文件: '{file_path}'")
        return None, 0
    except Exception as e:
        print(f"  [错误] 处理文件 '{file_path}' 时发生未知错误: {e}")
        return None, 0

# --- 主程序入口 ---
if __name__ == "__main__":
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"正在扫描目录: {script_dir}\n")

    # 找到所有以 .jsonl 结尾的文件
    jsonl_files = [f for f in os.listdir(script_dir) if f.endswith(".jsonl")]

    if not jsonl_files:
        print("在此目录下未找到任何 .jsonl 文件。")
    else:
        print(f"找到了 {len(jsonl_files)} 个 .jsonl 文件，将分别为您计算平均 length：\n")
        
        # 遍历找到的每个 .jsonl 文件
        for filename in jsonl_files:
            full_path = os.path.join(script_dir, filename)
            
            # 为每个文件单独输出结果
            print(f"--- 文件: {filename} ---")
            
            avg, count = calculate_average_for_single_file(full_path)
            
            if avg is not None:
                print(f"  有效数据条数: {count}")
                print(f"  平均 length: {avg:.2f}") # 结果保留两位小数
            else:
                print(f"  未能在该文件中找到或计算 'length'。")
            
            print("-" * (len(filename) + 12) + "\n") # 打印分隔线