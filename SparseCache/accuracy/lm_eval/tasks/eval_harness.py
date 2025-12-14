from functools import partial

import os
import transformers
from lm_eval.api.model import TemplateLM, LM
from tqdm import tqdm
import numpy as np

from tasks.util import sample_batch, shrink_seq
import multiprocessing
import ftfy

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

tokenizer = None

def process_init():
    global tokenizer
    model_name = os.environ.get('MODEL_NAME', 'facebook/opt-1.3b')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = int(1e30)
    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False

    # pad_token 兜底：若无 pad_token，则回退到 eos_token，否则退回到 id 0

def process_request(x, seq):
    global tokenizer

    # 处理新版本lm_eval的Instance对象格式
    if hasattr(x, 'args') and len(x.args) >= 2:
        ctx, cont = x.args[0], x.args[1]
    elif isinstance(x, (tuple, list)) and len(x) >= 2:
        ctx, cont = x[0], x[1]
    else:
        # 尝试其他可能的属性访问方式
        try:
            ctx = getattr(x, 'context', getattr(x, 'ctx', ''))
            cont = getattr(x, 'continuation', getattr(x, 'cont', ''))
        except:
            raise ValueError(f"无法从请求对象中提取context和continuation: {type(x)}, {x}")

    ctx_text = ftfy.fix_text(ctx, normalization="NFKC")
    cont_text = ftfy.fix_text(cont, normalization="NFKC")
    all_text = ctx_text + cont_text

    ctx_tokens = tokenizer(ctx_text, add_special_tokens=False)['input_ids']
    cont_tokens = tokenizer(cont_text, add_special_tokens=False)['input_ids']

    all_tokens = ctx_tokens + cont_tokens
    all_tokens = np.array(all_tokens)[-seq:]  # truncate sequence at seq length

    provided_ctx = len(all_tokens) - 1
    pad_amount = seq - provided_ctx

    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0

    return {
        "obs": np.pad(all_tokens[:-1], ((0, pad_amount),), constant_values=pad_id),
        "target": np.pad(all_tokens[1:], ((0, pad_amount),), constant_values=pad_id),
        "ctx_length": seq,
        "eval_mask": np.logical_and(
            np.arange(0, seq) > len(all_tokens) - len(cont_tokens) - 2,
            np.arange(0, seq) < len(all_tokens) - 1
        ),
        "prompt": ctx_text,
        "target_text": cont_text,
        "text": all_text,
    }


class EvalHarnessAdaptor(LM):
    def generate_until(self, requests):
        """Generate greedily until a stopping sequence"""
        raise Exception("unimplemented")
    
    def greedy_until(self, requests):
        raise Exception("unimplemented")

    def loglikelihood_rolling(self, requests):
        raise Exception("unimplemented")

    def __init__(self, tpu_cluster, seq, batch, shrink, min_seq=None):
        super().__init__()
        self.tpu = tpu_cluster
        self.seq = seq
        self.batch = batch
        self.shrink = shrink
        self.min_seq = min_seq

        self.pool = multiprocessing.Pool(processes=1, initializer=process_init)
        # self.pool = multiprocessing.Pool(initializer=process_init)
        process_init()

    def convert_requests(self, requests):
        return self.pool.imap(partial(process_request, seq=self.seq), requests)

    def loglikelihood(self, requests):
        output = []

        r = self.convert_requests(requests)
        zero_example = process_request(requests[0], self.seq)

        for b in tqdm(sample_batch(r, self.batch, zero_example),
                      desc="LM eval harness",
                      total=len(requests) // self.batch):

            if self.shrink:
                b = shrink_seq(b, min_seq=self.min_seq)

            out = self.tpu.eval(b)

            for loss, correct in zip(out["mask_loss"], out["each_correct"]):
                output.append((float(-loss), bool(correct)))

        return output


