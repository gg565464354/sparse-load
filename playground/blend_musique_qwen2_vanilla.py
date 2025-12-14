from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path

eval_dataset = load_dataset("/workspace/CacheBlend/inputs/musique_s.json")

llm = LLM(model="/share/models/Qwen2-1.5B-Instruct", gpu_memory_utilization=0.3, enforce_eager=True,
          tokenizer_mode = "slow", dtype = "float16"
          #tokenizer=tokenizer,
          )
tokenizer = AutoTokenizer.from_pretrained("/share/models/Qwen2-1.5B-Instruct", use_fast=False)
llm.set_tokenizer(tokenizer)

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

ttft_blend = []
ttft_full = []
f1_blend = []
f1_full = []

for ex in eval_dataset:
    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]

    #import pdb
    #pdb.set_trace()
    
    #while len(list(chain.from_iterable(doc_chunk_ids))) > max_ctx_len:
    #    del_idx = len(doc_chunk_ids)-1
    #    del doc_chunk_ids[del_idx]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Create an tokenizer and LLM.
    #cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    #cache_fuse_metadata['collect'] = False
    #cache_fuse_metadata['check'] = False

    bos_id = tokenizer.bos_token_id

    # 在系统消息后打开 user 段，让后续 docs + question 都在同一个 user 消息里
    s_start_full_text = "<|im_start|>system\n" + prefix_prompt + "<|im_end|>\n<|im_start|>user\n"
    s_start_full = [bos_id] + tokenizer.encode(s_start_full_text)[1:]
    s_start_len = len(s_start_full) + 1  # +1 计入 BOS

    # 这里保持不在每个 chunk 前重复开启 user 段
    s_start = []
    s_start_1_len = len(s_start) + 1

    # 结束 user 段并开启 assistant 段用于生成回答（注意不要带 BOS）
    s_end_text = "<|im_end|>\n<|im_start|>assistant\n"
    s_end = tokenizer.encode(s_end_text)[1:]
    s_end_len = len(s_end)
    #old_kvs = []

    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids+s_end]

    last_len = len(q_ids + s_end)

    #cache_fuse_metadata['collect'] = True
    #cache_fuse_metadata["check"] = False
    #base_model = llm.llm_engine.model_executor.driver_worker.model_runner.model.model
    #num_layer = len(base_model.layers)
    #chunk_past_key_values = []
    
    # Concatenate old KVs
    #for i in range(len(doc_chunk_ids)):
    #    prompts = [tokenizer.decode(doc_chunk_ids[i])]
    #    llm.generate(prompts, sampling_params)
        
    #    llm_layers = base_model.layers
    #    for j in range(num_layer):
    #        past_key_values = llm_layers[j].self_attn.hack_kv
    #        if i == 0:
    #            temp_k = past_key_values[0][:s_start_len].clone() # do not chage with s_start_1
    #            temp_v = past_key_values[1][:s_start_len].clone()
    #        else:
    #            temp_k = past_key_values[0][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
    #            temp_v = past_key_values[1][s_start_1_len:len(doc_chunk_ids[i])+1].clone()    

    #        if i == 0:
    #            chunk_past_key_values.append([temp_k, temp_v])
    #        else:
    #            chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=0)
    #            chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=0)

    #base_model.old_kvs = chunk_past_key_values
        
    input_ids = []

    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        input_ids += temp_ids
        
    # 用字符串直接构造最终提示
    # 注意：这里沿用你已有的 chat 模板字符串
    input_prompt = (
        (tokenizer.bos_token or "")  # 有的 tokenizer 有 BOS，就加上
        + s_start_full_text          # "<|im_start|>system\n" + prefix_prompt + "<|im_end|>\n<|im_start|>user\n"
        + "".join(doc_prompts)       # 拼接所有 passages 的文本
        + q_prompt                   # 问题文本
        + s_end_text                 # "<|im_end|>\n<|im_start|>assistant\n"
    )

    # 仅进行正常（full prefill）生成
    sampling_params = SamplingParams(temperature=0, max_tokens=32)
    output = llm.generate([input_prompt], sampling_params)

    res = output[0].outputs[0].text
    print(f"Normal generation: {res}")
    # ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
    # print(f"TTFT with full prefill: {ttft}")

    # ttft_full.append(ttft)
    f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
    f1_full.append(f1)
    print("------------")

print("---------------Result Summary---------------------")
# print(f"TTFT with full prefill: {np.mean(ttft_full)}")
print(f"F1 with full prefill: {np.mean(f1_full)}")
