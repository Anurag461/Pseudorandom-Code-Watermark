"""Frozen configuration and immutable upstream provenance."""

from __future__ import annotations

import hashlib
import json


SCHEMA_VERSION = 1
MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_DISPLAY = "Qwen3-8B-Base"
TOKENIZER_ID = MODEL_ID
PREFIX_LENGTHS = (128, 256, 400, 512, 768, 1024)
SMOKE_PROMPT_INDICES = (0, 1, 2, 3, 4)
MAX_NEW_TOKENS = 1024
TEMPERATURE = 1.0
TOP_P = 1.0
REASONING = False
CONTEXT_LENGTH = 3
NOMINAL_FPR = 1e-3
PRIMARY_SEED = 12345
SECONDARY_SEED = 67890

TEXTSEAL_REPOSITORY = "https://github.com/facebookresearch/textseal"
TEXTSEAL_COMMIT = "c60d0d1da2e59f09a698438e218a07ee779b4616"
TEXTSEAL_LICENSE = "Apache-2.0"
SYNTHID_REPOSITORY = "https://github.com/google-deepmind/synthid-text"
SYNTHID_COMMIT = "addb4a158143c7c6851a1308f78b89fceed59683"
SYNTHID_LICENSE = "Apache-2.0"
PRC_REPOSITORY = "https://github.com/Anurag461/Pseudorandom-Code-Watermark"
PRC_BASE_COMMIT = "13843c593209de3fc48acc2cda7f7869b2cf82b1"

TEXTSEAL_ALPHA = 0.1
TEXTSEAL_KEY_A = 42
TEXTSEAL_KEY_B = TEXTSEAL_KEY_A + 12345
SYNTHID_DEPTH = 10
SYNTHID_KEYS = (654, 400, 836, 123, 340, 443, 597, 160, 57, 29)
SYNTHID_CONTEXT_HISTORY_SIZE = 1024
GUMBEL_KEY = 42

ONLINE_PRC_SOURCE_TAG = (
    "online_causal_prc_v1/qwen3_8b_base/"
    "n1280_T1280_t3_eta0.05_rr99of100_sampler-poscdf-v1_"
    "kvcache-static-v1"
)
ONLINE_PRC_SOURCE_T = 1280
SHARED_NULL_SOURCE_T = 13088
ONLINE_PRC_T = 3
ONLINE_PRC_ETA = 0.05

# Direct requirements are fixed here. The Modal CPU reference check records the
# complete resolved environment (including transitives) in the smoke manifest.
PINNED_DEPENDENCIES = (
    "torch==2.4.0",
    "transformers==4.43.3",
    "tokenizers==0.19.1",
    "huggingface-hub==0.24.7",
    "safetensors==0.4.5",
    "numpy==1.26.0",
    "scipy==1.14.1",
    "immutabledict==4.2.0",
    "omegaconf==2.3.0",
    "msgspec==0.18.6",
    "rouge-score==0.1.2",
    "sacrebleu==2.4.3",
    "sentence-transformers==3.0.1",
    "sentencepiece==0.2.0",
    "tiktoken==0.7.0",
    "fsspec==2024.9.0",
    "blobfile==3.0.0",
    "orjson==3.10.7",
    "accelerate==0.34.2",
    "galois==0.4.2",
    "numba==0.59.1",
    "pytest==8.3.3",
)

IMAGE_DEFINITION = {
    "python": "3.11",
    "base": "modal.Image.debian_slim",
    "apt": ["git"],
    "dependencies": list(PINNED_DEPENDENCIES),
    "official_sources": {
        "textseal": f"git+{TEXTSEAL_REPOSITORY}.git@{TEXTSEAL_COMMIT}",
        "synthid_text": f"git+{SYNTHID_REPOSITORY}.git@{SYNTHID_COMMIT}",
    },
    "official_source_install": "--no-deps; dependencies pinned separately",
}
IMAGE_DEFINITION_SHA256 = hashlib.sha256(
    json.dumps(IMAGE_DEFINITION, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

GENERATION_SETTINGS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "forced_length": True,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "reasoning": REASONING,
    "prompt_construction": "canonical prompts.jsonl prompt_tokens; no chat template",
    "sampling_distribution": "softmax(base logits / temperature), top-p=1.0",
}
