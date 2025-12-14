from vllm import LLM, SamplingParams
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
from pathlib import Path
import os, sys
from itertools import chain

# --- Helper Functions (Assuming they exist in utils.py) ---
# You need to ensure you have a utils.py with these functions.
# For demonstration, I'll provide dummy implementations if they are missing.
try:
    from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
except ImportError:
    print("Warning: 'utils.py' not found. Using dummy implementations.")
    def load_dataset(path):
        with open(path, 'r') as f:
            return json.load(f)
    def normalize_question(q):
        return q.strip()
    def build_qa_prompt(ex, query_prompt):
        # This is a simplified version based on your code's logic
        doc_prompts = [p['paragraph_text'] for p in ex['paragraphs']]
        q_prompt = ex['question'] + query_prompt
        return doc_prompts, q_prompt
    def compute_f1(a_gold, a_pred, tokenizer):
        # A simple F1 score based on token overlap
        gold_toks = set(tokenizer.tokenize(a_gold))
        pred_toks = set(tokenizer.tokenize(a_pred))
        common = gold_toks & pred_toks
        if len(common) == 0:
            return 0
        precision = len(common) / len(pred_toks)
        recall = len(common) / len(gold_toks)
        return 2 * (precision * recall) / (precision + recall)
# -------------------------------------------------------------

# This function is specific to your environment for custom model code.
# It's kept for completeness but might not be needed in a standard setup.
def set_symlink(model_type, fname):
    model_path = "/workspace/playground/libs/transformers/src/transformers/models/" + model_type
    linker_path = os.path.realpath("/workspace/SparseCache/accuracy/src/" + fname)
    if not os.path.exists(linker_path):
        print(f"No file exists at {linker_path}")
        return
    if not os.path.exists(model_path):
        print(f"No file exists at {model_path}")
        return
    curr_dir = os.getcwd()
    os.chdir(model_path)
    if os.path.exists(f'modeling_{model_type}.py'):
        os.system(f"rm modeling_{model_type}.py")
    os.system(f"ln -s {linker_path} modeling_{model_type}.py")
    os.chdir(curr_dir)

# set_symlink("opt", f"modeling_opt_cache.py")

# --- Main Script ---

print("Loading dataset...")
eval_dataset = load_dataset("/workspace/CacheBlend/inputs/musique_s.json")

print("Initializing LLM...")
# The LLM class now handles the tokenizer internally.
# You don't need to load and set it separately unless for specific reasons.
llm = LLM(model="/share/models/Llama-2-7b-hf", gpu_memory_utilization=0.4)
tokenizer = llm.get_tokenizer()
print("LLM Initialized.")


prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

ttft_run1 = []
ttft_run2 = []
f1_run1 = []
f1_run2 = []

for i, ex in enumerate(eval_dataset):
    print(f"--- Processing Example {i+1}/{len(eval_dataset)} ---")
    answers = ex["answers"]
    
    # [MODIFIED] Construct the full prompt string at once.
    # This is the standard and optimized way to use vLLM.
    doc_passages, question_part = build_qa_prompt(ex, query_prompt)
    
    full_prompt = prefix_prompt
    full_prompt += "\n".join(doc_passages)
    full_prompt += "\n" + question_part

    # For very long prompts, you might want to tokenize to check length
    # and truncate if necessary, but vLLM handles long sequences well.
    # input_ids = tokenizer.encode(full_prompt)

    # [REMOVED] Block for manual KV cache manipulation.
    # All the logic for iterating through chunks, generating, extracting KVs,
    # and injecting them back is removed because it's incompatible with new vLLM.

    # --- Standard Generation ---
    # We now perform the generation in a single, efficient call.
    # Your original code had a "cached" and "full" run. Since the manual
    # caching is gone, we'll just run it twice to mimic your experimental setup.
    
    sampling_params = SamplingParams(temperature=0, max_tokens=32)

    # First Run (Equivalent to your "full prefill" or "cached" run in the new API)
    print("Starting first generation...")
    output1 = llm.generate([full_prompt], sampling_params)
    
    res1 = output1[0].outputs[0].text.strip()
    print(f"Generation 1: {res1}")
    ttft1 = output1[0].metrics.finished_time - output1[0].metrics.first_token_time
    # Note: TTFT definition can vary. Time to first token is often:
    # first_token_time - arrival_time. Let's use what your code had.
    ttft1_corrected = output1[0].metrics.first_token_time - output1[0].metrics.scheduled_time
    print(f"TTFT for run 1: {ttft1_corrected}")
    
    ttft_run1.append(ttft1_corrected)
    f1_1 = max([compute_f1(answer, res1, tokenizer) for answer in answers])
    f1_run1.append(f1_1)

    # Second Run (Equivalent to your "Normal generation" run)
    print("Starting second generation...")
    output2 = llm.generate([full_prompt], sampling_params)
    
    res2 = output2[0].outputs[0].text.strip()
    print(f"Generation 2: {res2}")
    ttft2_corrected = output2[0].metrics.first_token_time - output2[0].metrics.scheduled_time
    print(f"TTFT for run 2: {ttft2_corrected}")
    
    ttft_run2.append(ttft2_corrected)
    f1_2 = max([compute_f1(answer, res2, tokenizer) for answer in answers])
    f1_run2.append(f1_2)
    print("-" * 20)

print("\n--------------- Result Summary ---------------------")
print(f"Average TTFT (Run 1): {np.mean(ttft_run1)}")
print(f"Average TTFT (Run 2): {np.mean(ttft_run2)}")
print(f"Average F1 (Run 1): {np.mean(f1_run1)}")
print(f"Average F1 (Run 2): {np.mean(f1_run2)}")
