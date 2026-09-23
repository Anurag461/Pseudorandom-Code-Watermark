
import os

import time
from transformers import GenerationConfig
from tqdm import trange
import json
import matplotlib.pyplot as plt
from llm_prc_api import KeyGen, Encode, Detect, GF
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import LogitsProcessorList
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper
)
import argparse
# add arg: t, start, end
parser = argparse.ArgumentParser()
parser.add_argument("--prc_t", type=int, default=3)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=128)
parser.add_argument("--temperature", type=float, default=1.8)
parser.add_argument("--model_name", type=str, default="Deepseek", choices=["Deepseek", "Qwen"])
args = parser.parse_args()
model_name = args.model_name
if model_name == "Deepseek":
    model_path = ".models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/"
elif model_name == "Qwen":
    model_path = ".models/Qwen/Qwen3-8B/"
else:
    raise ValueError("Invalid model_id")
t = int(args.prc_t)
temperature = float(args.temperature)

model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
model.eval()
output_dir = f"llm/data/{model_name}_t_{t}_temp_{temperature}"
os.makedirs(output_dir, exist_ok=True)



def sample_token(x, probs):
    token_bits = math.ceil(math.log2(probs.shape[-1]))
    assert token_bits == x.shape[-1]

    sample_token = 0
    l, r = 0, 2**token_bits
    for extract_bit in range(token_bits):
        m = (l+r)//2
        p1 = probs[0][m:r].sum().item()
        p_all = probs[0][l:r].sum().item()
        p1 = p1/p_all
        p1 = np.clip(p1, 0, 1)
        t = np.random.binomial(
            1, 2*p1*x[0, extract_bit]) if p1 <= 0.5 else np.random.binomial(1, 1-(2*(1-p1)*(1-x[0, extract_bit])))
        l, r = (l, m) if t == 0 else (m, r)
        sample_token += (t << (token_bits-1-extract_bit))

    return sample_token


def int_to_bits(x, length):
    bits = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        bits[i] = (x >> (length - 1 - i)) & 1
    return bits


def get_entropy(x, softmax=True):
    if softmax:
        x = torch.nn.functional.softmax(x, dim=-1)
    x_nz = x[x != 0]
    e = -(torch.log2(x_nz)*x_nz).sum()
    return e.item()


def get_logits_processor(generation_config: GenerationConfig):
    processors = LogitsProcessorList()
    min_tokens_to_keep = 1
    if generation_config.temperature is not None and generation_config.temperature != 1.0:
        processors.append(TemperatureLogitsWarper(
            generation_config.temperature))
    if generation_config.top_k is not None and generation_config.top_k != 0:
        processors.append(
            TopKLogitsWarper(top_k=generation_config.top_k,
                             min_tokens_to_keep=min_tokens_to_keep)
        )
    if generation_config.top_p is not None and generation_config.top_p < 1.0:
        processors.append(
            TopPLogitsWarper(top_p=generation_config.top_p,
                             min_tokens_to_keep=min_tokens_to_keep)
        )
    return processors



head_token_length = 0
max_new_tokens = 1024
generation_config = GenerationConfig(
    temperature=temperature,
    top_k=0,
    top_p=1.0,
    do_sample=True,
    max_new_tokens=head_token_length+max_new_tokens,
    num_return_sequences=1
)

logits_processor = get_logits_processor(generation_config)

chats = [
    [
        {"role": "user", "content": "Write a long creative story of 10000 words about the sea."},
    ],
    [
        {"role": "user", "content": "Write a long poem about the future of AI."},
    ],
    [
        {"role": "user", "content": "Write a Python script to solve the Hanoi Tower problem."},
    ],
    [
        {"role": "user", "content": "How to teach my child to cook a perfect steak?"},
    ],
    [
        {"role": "user", "content": "What is the meaning of life, the universe and everything?"},
    ],
    [
        {"role": "user", "content": "Rewrite the following sentence: Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty and dedicated to the proposition that all men are created equal."},
    ],
    [
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "user", "content": "Can you tell me a joke?"},
    ],
    [
        {"role": "user", "content": "Is there any food recommendation around San Francisco?"}
    ],
    [
        {"role": "user", "content": "What is the best way to learn history?"},
    ],
    [
        {"role": "user", "content": "How to make a perfect cup of coffee?"},
    ],
    [
        {"role": "user", "content": "Can you sing a song for me?"},
    ],
    [
        {"role": "user", "content": "How to replace a tire on a car?"},
    ],
    [
        {"role": "user", "content": "Why does a fried chicken cross the street?"},
    ],
    [
        {"role": "user", "content": "Is there any life on Mars?"},
    ],
    [
        {"role": "user", "content": "My name of John Doe, I am a software engineer, please write a self-introduction for me."},
    ],
]


token_count_log2 = math.ceil(math.log2(model.config.vocab_size))
n = token_count_log2*max_new_tokens

start = int(args.start)
end = int(args.end)
for exp_i in range(start, end):
    enc, dec = KeyGen(n, t)
    filename = f"{output_dir}/{exp_i}.json"
    (generator_matrix, parity_check_matrix, one_time_pad, noise_rate, g, t) = dec

    _msgs = []
    _msg_decect = []
    _inv_msgs = []
    _inv_msg_detect = []
    _prompts = []
    _prompts_tokens = []
    _orginal_gen = []
    _orginal_gen_tokens = []
    _watermarked_gen = []
    _watermarked_gen_tokens = []


    def save_to_json(public_key: np.ndarray, secret_key, otp: np.ndarray,
                    msgs: list, msgs_decect: list,
                    inv_msgs: list, inv_msgs_detect: list,
                    prompts: list, original_gen: list,
                    watermarked_gen: list, prompt_tokens: list,
                    original_gen_tokens: list, watermarked_gen_tokens: list):

        public_key_list = public_key.tolist()
        secret_key_list = []
        for i in range(secret_key.shape[0]):
            row_start = secret_key.indptr[i]
            row_end = secret_key.indptr[i + 1]
            indices = secret_key.indices[row_start:row_end]
            sorted_indices = sorted(indices.tolist())
            secret_key_list.append(sorted_indices)

        otp_list = otp.tolist()

        data = {
            "secret_key": secret_key_list,
            "public_key": public_key_list,
            "one_time_pad": otp_list,
            "msg": {
                "num": str(len(msgs)),
                "succ": str(msgs_decect.count(True)),
                "val": np.array(msgs).tolist(),
            },
            "inv_msg": {
                "num": str(len(inv_msgs)),
                "succ": str(inv_msgs_detect.count(True)),
                "val": np.array(inv_msgs).tolist(),
            },
            "prompt": prompts,
            "origin_sentence": original_gen,
            "watermark_sentence": watermarked_gen,
            "prompt_tokens": prompt_tokens,
            "origin_sentence_tokens": original_gen_tokens,
            "watermark_sentence_tokens": watermarked_gen_tokens,
        }

        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False)


    for i, chat in enumerate(chats):
        print(f"=== Sample {i+1}/{len(chats)} ===")
        res = Encode(encoding_key=enc)
        origin_det = Detect(decoding_key=dec, posteriors=res)

        tok = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

        prompt_length = tok.shape[1]

        tok = model.generate(
            input_ids=tok,
            **generation_config.to_dict(),
        )

        prompt_content = tokenizer.decode(
            tok[0][:prompt_length], skip_special_tokens=True)
        head_gen = tokenizer.decode(
            tok[0][prompt_length:head_token_length+prompt_length], skip_special_tokens=True)
        benign_gen = tokenizer.decode(
            tok[0][prompt_length+head_token_length:], skip_special_tokens=True)
        prompt_tokens = tok[0][:prompt_length].tolist()
        benign_gen_tokens = tok[0][prompt_length+head_token_length:].tolist()

        tok = tok[:, :prompt_length+head_token_length]
        prompt_length = tok.shape[1]
        a_probs = []

        with torch.no_grad():
            for i in trange(max_new_tokens):
                x = np.array(res[i*token_count_log2:(i+1) *
                                token_count_log2].reshape(1, -1).tolist())
                output = model(input_ids=tok, return_dict=True)
                last_logits = output.logits[:, -1, :]
                last_score = logits_processor(tok, last_logits)
                probs = torch.nn.functional.softmax(last_score, dim=-1)
                en_score = get_entropy(probs, softmax=False)
                en_logit = get_entropy(last_logits, softmax=True)
                a_probs.append(en_score)
                sampled_token = sample_token(x, probs)
                tok = torch.cat(
                    [tok, torch.tensor([[sampled_token]]).to(tok.device)], dim=1)

        generated = tokenizer.decode(
            tok[0][prompt_length:], skip_special_tokens=True)
        res_inv = GF(np.concatenate(
            [int_to_bits(i.item(), token_count_log2) for i in tok[0][prompt_length:]]))
        det = Detect(decoding_key=dec, posteriors=res_inv)

        print("correct rate,", np.mean(res_inv == res),
            ", avg entropy,", np.array(a_probs).mean())
        error_rate = np.mean(
            (res_inv != res).reshape(-1, token_count_log2), axis=1)
        a_probs = np.array(a_probs)

        _msgs.append(res)
        _msg_decect.append(origin_det)
        _inv_msgs.append(res_inv)
        _inv_msg_detect.append(det)
        _prompts.append(prompt_content)
        _orginal_gen.append(benign_gen)
        _watermarked_gen.append(generated)
        _prompts_tokens.append(prompt_tokens)
        _orginal_gen_tokens.append(benign_gen_tokens)
        _watermarked_gen_tokens.append(tok[0][prompt_length:].tolist())
        save_to_json(generator_matrix, parity_check_matrix, one_time_pad,
                    _msgs, _msg_decect, _inv_msgs, _inv_msg_detect, _prompts, _orginal_gen, _watermarked_gen,
                    _prompts_tokens, _orginal_gen_tokens, _watermarked_gen_tokens)
