MODEL_ID = "Qwen/Qwen3-8B-Base"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 1.0
TOP_P = 1.0
REASONING = False
CONTEXT_LENGTH = 3
PRIMARY_SEED = 12345
SECONDARY_SEED = 67890
TEXTSEAL_ALPHA = 0.1
TEXTSEAL_KEY_A = 42
TEXTSEAL_KEY_B = TEXTSEAL_KEY_A + 12345
SYNTHID_KEYS = (654, 400, 836, 123, 340, 443, 597, 160, 57, 29)
SYNTHID_CONTEXT_HISTORY_SIZE = 1024
GUMBEL_KEY = 42
GENERATION_SETTINGS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "forced_length": True,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "reasoning": REASONING,
    "prompt_construction": "canonical prompts.jsonl prompt_tokens; no chat template",
    "sampling_distribution": "softmax(base logits / temperature), top-p=1.0",
}

NOMINAL_FPR = 0.001
TEXTSEAL_COMMIT = "c60d0d1da2e59f09a698438e218a07ee779b4616"
