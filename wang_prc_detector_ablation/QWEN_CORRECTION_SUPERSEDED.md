## Important correction after inspecting Wang et al.'s artifact

The Wang et al. repository already contains an official **Qwen3-8B implementation and ablation path**. Do not independently port their PRC sampler into our Qwen code unless necessary.

Their official generation script supports:

```bash
--model_name Qwen
```

which loads their pinned `Qwen/Qwen3-8B` checkpoint and applies the same `KeyGen`, `Encode`, hierarchical `sample_token`, OTP handling, and hard `Detect` used for their DeepSeek experiments.

Therefore revise the previous plan as follows.

### 1. Use Wang's official Qwen path as the source of truth

Start from the authors' `llm/generation/main.py` Qwen branch and associated `llm_prc_api.py`.

Treat these as authoritative for:

* Qwen prompt formatting;
* tokenizer behavior;
* vocabulary handling;
* `token_bits`;
* `KeyGen`;
* `Encode`;
* permutation / OTP semantics;
* hierarchical PRC-to-token sampling;
* hard detection.

Do not rewrite these semantics from memory.

Our changes should be limited to:

1. making transformer inference efficient with KV caching / batching;
2. saving all required intermediate probabilities;
3. adding our posterior detector;
4. adding matched-FPR calibration and analysis.

Before production, verify the optimized sampler against the authors' original implementation using fixed logits and fixed random variates.

---

### 2. Change the primary model to their exact Qwen ablation model

The previous prompt said to use our existing `Qwen3-8B-Base` checkpoint.

**Supersede that instruction.**

For this experiment use the exact `Qwen/Qwen3-8B` checkpoint/revision downloaded by Wang et al.'s artifact setup.

Reason: Wang et al. actually report an LLM ablation on this Qwen3-8B model. Using their exact checkpoint makes our experiment directly comparable to their reported Qwen ablation and removes unnecessary disagreement over model choice.

Record the exact Hugging Face model ID and git revision in the results.

Do **not** silently substitute our `Qwen3-8B-Base` checkpoint.

Our existing Qwen3-8B-Base results remain separate experiments in our paper.

---

### 3. First reproduce their Qwen T=1.8 setup

Before running the temperature sweep, run a small validation at:

```text
model = Wang Qwen/Qwen3-8B
t = 3
eta = 0.1
temperature = 1.8
length = 1024
prompts = Wang's exact 16 prompts
```

This corresponds to the Qwen3-8B ablation configuration reported in their appendix.

Use enough groups to verify that:

* generation is valid;
* their original hard detector runs correctly;
* the PRC/key/OTP conventions are aligned;
* qualitative behavior is consistent with their Qwen ablation.

Do not require exact reproduction of their cryptanalytic Attack-I/II percentages in the initial smoke test, because our main experiment is detection rather than attack reproduction.

If straightforward, also run their supplied attack script on a small subset as an additional implementation check.

---

### 4. Then extend their Qwen experiment across temperature

Run the detector experiment at:

```text
T = 1.0, 1.2, 1.4, 1.6, 1.8
```

with:

```text
10 independent PRC key groups
× 16 Wang prompts
= 160 watermarked responses per temperature
```

plus matched unwatermarked/null generations.

Use exactly the same Wang Qwen model, watermark construction, prompts, length, and sampler for all temperatures.

This extends their Qwen3-8B ablation into the low-entropy regime rather than introducing a different model.

---

### 5. Preserve the main detector comparison from the previous prompt

For every exact same generated watermarked response and secret key, compute:

**A. Wang hard detector**

* recovered hard token-ID bits;
* their exact OTP/parity-check statistic;
* their original published decision threshold;
* continuous hard score for recalibration.

**B. Our posterior detector**

* reconstruct the conditional probability at every hierarchical binary decision;
* convert each observed branch into the corresponding posterior mean for the latent PRC bit;
* propagate those soft values through Wang's exact parity checks and OTP;
* compute our normalized posterior score.

The posterior detector must model **Wang's hierarchical channel**, not our one-bit vocabulary partition.

---

### 6. Keep matched-FPR comparison unchanged

The main comparison remains:

```text
Wang hard statistic
vs.
our posterior statistic
```

on the **same samples**, calibrated to the **same FPR = 1e-3**.

Continue to report Wang's original theoretical threshold separately.

Do not interpret an improvement over only their published threshold as evidence that posterior scoring is better. The important result is improvement over the **recalibrated hard statistic at matched FPR**.

---

### 7. Keep prompt-free posterior detection as the primary result

Primary posterior re-detection remains completion-only:

* no original prompt;
* no chat-template prefix;
* no BOS/EOT context;
* first generated token's hierarchical coordinates abstain / contribute zero;
* subsequent token probabilities are reconstructed from previous completion tokens only.

Keep generation-time probabilities as an **oracle-context diagnostic** exactly as specified previously.

---

### 8. Efficiency change only — not a watermark change

Wang's original generation loop recomputes the entire transformer prefix at every generated token.

Optimize this with KV caching and batching.

However, verify that for the same model state, logits, random variates, PRC bits, and key, the optimized hierarchical sampler produces the same sampling decisions/distribution as their implementation.

The scientific experiment must change **inference efficiency**, not the Wang PRC channel.

---

### 9. Update the interpretation

The experiment can now be described as:

> We extend Wang et al.'s own Qwen3-8B PRC experiment across generation temperatures and compare their hard parity statistic against probability-aware posterior parity scoring on exactly the same watermarked outputs.

This is stronger and cleaner than saying we implemented their watermark on our own unrelated model.

The central question remains:

> Does posterior detection recover useful PRC signal in the low-entropy regimes where hard-bit decoding loses information?

---

Everything else in the previous experiment prompt — saved metadata, null calibration, ROC analysis, hierarchical bootstrap, result tables, figures, 10-key stopping point, and no automatic scaling — remains unchanged.
