import os
def set_symlink(model_type, fname):
    model_path = "../../../playground/libs/transformers/src/transformers/models/" + model_type
    # 使用绝对路径而不是相对路径
    linker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src", fname))
    
    print(f"Source file: {linker_path}")
    print(f"Target path: {model_path}")
    
    if not os.path.exists(linker_path):
        print(f"No file exists at {linker_path}")
        exit(0)
    if not os.path.exists(model_path):
        print(f"No file exists at {model_path}")
        exit(0)
    
    curr_dir = os.getcwd()
    os.chdir(model_path)
    
    # 备份原文件
    if os.path.exists(f'modeling_{model_type}.py'):
        cmd = f"cp modeling_{model_type}.py modeling_{model_type}_backup.py"
        os.system(cmd)
        cmd = f"rm modeling_{model_type}.py"
        os.system(cmd)
    
    cmd = f"ln -s {linker_path} modeling_{model_type}.py"
    print(f"Creating symlink: {cmd}")
    result = os.system(cmd)
    print(f"Symlink creation result: {result}")
    
    os.chdir(curr_dir)
    
from datasets import load_dataset
import random

def get_qasper_calibration_data(num_samples=20, max_length=2048):
    """
    加载 Qasper 数据并构造 Prompt
    """
    dataset = load_dataset("json", data_files="/root/.cache/huggingface/datasets/downloads/extracted/a80cf5a629e2a7ff996c8e9ed1ba128cf5b5569c7668efc9bcc559283a51309a/data/qasper.jsonl", split="train")
    
    prompts = []
    # 随机采样，或者取前 N 个
    # 为了复现性，建议固定 seed 或取前 N 个
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
            
        # 提取论文全文
        context = sample.get("context", "")
        # 或者构造更有挑战性的 Prompt: Context + Question
        # if sample.get("qas") and len(sample["qas"]) > 0:
        #     question = sample["qas"][0]["question"]
        #     text = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        # else:
        #     text = context

        # 简单的截断处理
        if len(context) > 100: # 过滤太短的
             prompts.append(context[:max_length * 4]) # 粗略字符长度
             
    return prompts
