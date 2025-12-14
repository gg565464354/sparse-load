# 创建一个测试脚本来检查token ID
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/root/model/LLaMA-2-7B-32K", use_fast=False)

# 测试各种token组合
print("检查 s_start_full tokens:")
print(f"733: {repr(tokenizer.decode([733]))}")
print(f"4138: {repr(tokenizer.decode([4138]))}")  # blend.py中注释的版本
print(f"16289: {repr(tokenizer.decode([16289]))}")  # blend_musique.py中使用的版本
print(f"28793: {repr(tokenizer.decode([28793]))}")

print("\n检查 s_end tokens:")
print(f"733: {repr(tokenizer.decode([733]))}")
print(f"28748: {repr(tokenizer.decode([28748]))}")
print(f"16289: {repr(tokenizer.decode([16289]))}")
print(f"28793: {repr(tokenizer.decode([28793]))}")

print("\n组合测试:")
print(f"[733, 4138, 28793]: {repr(tokenizer.decode([733, 4138, 28793]))}")
print(f"[733, 16289, 28793]: {repr(tokenizer.decode([733, 16289, 28793]))}")
print(f"[733, 28748, 16289, 28793]: {repr(tokenizer.decode([733, 28748, 16289, 28793]))}")