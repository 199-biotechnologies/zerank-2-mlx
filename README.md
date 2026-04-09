# zerank-2-mlx

**The fastest way to run [ZeroEntropy's zerank-2](https://huggingface.co/zeroentropy/zerank-2) cross-encoder reranker on Apple Silicon.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-0.31%2B-orange.svg)](https://github.com/ml-explore/mlx)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20%7C%20M2%20%7C%20M3%20%7C%20M4-black.svg)](https://www.apple.com/mac/)
[![GitHub stars](https://img.shields.io/github/stars/199-biotechnologies/zerank-2-mlx?style=social)](https://github.com/199-biotechnologies/zerank-2-mlx)

zerank-2 is a 4B Qwen3-based cross-encoder reranker that tops the [BEIR benchmark](https://www.zeroentropy.dev/articles/zerank-2-advanced-instruction-following-multilingual-reranker) for biomedical, legal, and STEM retrieval. This repo ports it to **[Apple's MLX framework](https://github.com/ml-explore/mlx)** so you can serve it from M-series Macs at high throughput without the PyTorch + MPS memory tax or the OOM crashes that plague the upstream Python sidecar.

No model surgery, no custom Metal kernels — we reuse `mlx-lm`'s existing Qwen3 implementation and extract the calibrated yes-token logit directly. The result is a single-file HTTP sidecar and a small Python API that drop into any retrieval stack.

---

## 🏁 Benchmarks

Throughput and latency on a reference Mac Studio M-series with 64 GB unified memory. Workload: 20 distinct queries × 30 candidate documents = 600 pair scores per run. All numbers are measured, not estimated. See [`bench-results/`](bench-results/) for raw JSON.

| Variant | pairs/sec | mean latency (30 docs) | peak RAM | speedup vs PyTorch MPS |
|---|---:|---:|---:|---:|
| **v0 baseline** (bf16, full vocab proj, no compile) | **25.0** | **1200 ms** | **8.55 GB** | **~1.3×** |
| v1 yes-token shortcut *(planned)* | tbd | tbd | tbd | tbd |
| v2 + `mx.compile(shapeless=True)` *(planned)* | tbd | tbd | tbd | tbd |
| v3 + query prefix KV cache reuse *(planned)* | tbd | tbd | tbd | tbd |
| PyTorch + MPS reference (`zeroentropy/zerank-2`, bf16) | ~19 | ~1580 ms | ~8.5 GB | 1.0× |

v0 is **already 25–30 % faster than the PyTorch + MPS sidecar with zero optimization applied** — just switching the runtime. The optimized variants target **5–15× total speedup** over the PyTorch baseline once prefix-cache reuse lands.

---

## ✨ Why this exists

If you run retrieval-augmented systems on Apple Silicon, you probably already want a strong local reranker. Your options today:

- **Cohere Rerank 3.5 / Voyage Rerank 2.5** — great quality but a network hop, latency you can't control, and per-token pricing.
- **PyTorch + MPS upstream sidecar** — works, but `sentence-transformers` + custom `modeling_zeranker.py` is brittle: we hit two OOM events and a sidecar crash loop during a 1500-question benchmark that ate ~100 minutes of GPU time with zero partial results.
- **llama.cpp + community GGUF** — the GGUF conversion of Qwen3 classifier heads has a [known bug](https://github.com/ggml-org/llama.cpp/issues/16407) that produces garbage scores (~4e-23). Not trustworthy yet.
- **Ollama MLX backend (March 2026+)** — 57 % faster prefill, but Ollama [has no `/api/rerank`](https://github.com/ollama/ollama/issues/3368) and falls back to llama.cpp on Macs below 32 GB.

**zerank-2-mlx** gives you a single purpose-built file that loads the official zerank-2 weights straight into MLX, serves the exact same HTTP contract as the PyTorch sidecar, and runs natively on Apple Silicon unified memory. No dependency sprawl, no custom kernels, no tokenizer trust-remote-code footgun.

---

## 🚀 Quick start

### Install

You need a Mac with Apple Silicon (M1 / M2 / M3 / M4 / M5) and [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/199-biotechnologies/zerank-2-mlx.git
cd zerank-2-mlx
```

Everything runs under `uv`, so there is nothing to install globally.

### HTTP server (drop-in replacement for the PyTorch sidecar)

```bash
uv run --with mlx --with 'mlx-lm>=0.21' --with 'transformers<5.0,>=4.45' \
       --with safetensors --with huggingface_hub --with numpy \
       zerank_server_mlx.py
```

The server exposes the same contract as ZeroEntropy's reference Python sidecar:

```bash
# health
curl http://127.0.0.1:8766/health

# rerank
curl -X POST http://127.0.0.1:8766/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital and most populous city of France.",
      "Berlin is the capital of Germany."
    ],
    "top_k": 2
  }'
```

Port defaults to `8766` so it does not collide with any legacy `8765` sidecar. Override with `ENGRAM_ZERANK_MLX_PORT=9000`.

### Python API

```python
from zerank_server_mlx import load_model, score_pairs

bundle = load_model()           # downloads and patches the HF snapshot once
scores = score_pairs(bundle, [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of France?", "Berlin is the capital of Germany."),
])
# → [3.25, -1.48]   (calibrated yes-token logit / 5, matches PyTorch reference within bf16 grain)
```

The returned scores match the official ZeroEntropy calibration: `yes_token_logit / 5`.

### Validate correctness against the PyTorch reference

```bash
uv run --with mlx --with 'mlx-lm>=0.21' --with 'transformers<5.0,>=4.45' \
       --with torch --with safetensors --with huggingface_hub --with numpy \
       validate_mlx_reranker.py
```

The validator loads both the PyTorch reference (`AutoModelForSequenceClassification` from ZeroEntropy's custom modeling code) and the MLX port, scores 20 diverse pairs on both, and asserts that the top-10 ranking is identical and the per-pair score diffs stay within the bf16 cross-runtime tolerance (5e-2 absolute, ≈ 2× bf16 LSB).

### Benchmark a new variant

```bash
uv run --with mlx --with 'mlx-lm>=0.21' --with 'transformers<5.0,>=4.45' \
       --with safetensors --with huggingface_hub --with numpy \
       benchmark_mlx_reranker.py --variant v0-baseline
```

Writes `bench-results/<variant>.json` with throughput, per-call latency percentiles, and peak RAM. See [`bench-results/`](bench-results/) for the committed baselines.

---

## 🧠 How it works

### zerank-2 is Qwen3 with a yes-token trick

ZeroEntropy's [`modeling_zeranker.py`](https://huggingface.co/zeroentropy/zerank-2/blob/main/modeling_zeranker.py) inherits directly from `Qwen3PreTrainedModel` and `Qwen3Model`. There is no new classifier head — the `lm_head` is tied to `embed_tokens` (standard Qwen3), the model does a full forward pass, and scoring is just:

```python
last_pos = attention_mask.sum(dim=1) - 1
yes_logits = logits[batch_idx, last_pos, yes_token_id]
score = yes_logits / 5.0
```

where `yes_token_id = 9454`. The `/5` divisor is the calibration constant ZeroEntropy chose so scores live in a human-readable range.

### The MLX port in one paragraph

We symlink the HuggingFace snapshot into a tempdir, patch `config.json`'s `model_type` from `zeroentropy` to `qwen3` (so `mlx-lm` recognises the architecture), swap `tokenizer_config.json` from the custom `ZeroEntropyTokenizer` to the stock `Qwen2TokenizerFast` (the custom class only overrides `__call__` to accept pair tuples — we inline the chat template ourselves), and call `mlx_lm.load()`. The resulting MLX `Model` already does the full vocab projection via `embed_tokens.as_linear(hidden_state)` because `tie_word_embeddings=True`. We then advanced-index the `[batch, last_pos, yes_token_id]` logit and divide by 5. **That is the entire port.**

### Why no explicit padding mask

Qwen3 is decoder-only with causal attention. With right padding (`padding_side="right"`), the hidden state at `last_non_pad_position` only attends to real tokens — the causal mask enforces it. So we can batch variable-length `(query, doc)` pairs, right-pad them, and extract per-row logits without ever needing a padding mask. Verified against `mlx_lm/models/qwen3.py` and the reference PyTorch output.

### The chat template, inlined

```
<|im_start|>system
{query}
<|im_end|>
<|im_start|>user
{document}
<|im_end|>
<|im_start|>assistant
```

Matches exactly what `PreTrainedTokenizerFast.apply_chat_template(..., add_generation_prompt=True)` produces from zerank-2's `chat_template.jinja` in the no-tools case.

---

## 🔬 Optimization roadmap

All variants are gated by the validation harness — anything that breaks ranking gets reverted.

- [x] **v0 baseline** — mlx-lm Qwen3 forward + yes-token extraction. 25 pairs/sec, top-10 rank-identical to PyTorch.
- [ ] **v1 yes-token shortcut** — skip the full `[batch, seq, 151936]` vocab projection. Compute `score = hidden_at_last @ embed.weight[yes_token_id]` directly. Expected +15–25 %.
- [ ] **v2 `mx.compile(shapeless=True)`** — kernel fusion on the forward pass. Expected 1.5–2× over v1.
- [ ] **v3 Q4 quantization (optional)** — `mlx_lm.convert --quantize --q-bits 4 --q-group-size 64`. On M4 Max with high memory bandwidth this may be neutral or regress vs bf16; we benchmark it honestly and pick whichever wins.
- [ ] **v4 query prefix KV cache reuse** — the query system prompt is identical across all N docs in one rerank call. Cache its KV once, reuse across docs. Potentially 5–10× on `top_k ≈ 20–50`. Caveat: `mlx-lm`'s `make_prompt_cache` may silently fall back on hybrid attention layers; if so, we ship a Qwen3-specific manual implementation.
- [ ] **v5 concurrent rerank requests** — multiple inbound HTTP requests share the same MLX context. Metal is not thread-safe for parallel models, so we queue and batch internally.

Each variant is committed in its own change with updated benchmark JSON in `bench-results/`.

---

## 📂 Repository layout

```
zerank-2-mlx/
├── zerank_server_mlx.py         # MLX HTTP sidecar, drop-in replacement for zerank_server.py
├── validate_mlx_reranker.py     # Correctness gate vs the PyTorch reference
├── benchmark_mlx_reranker.py    # Throughput / latency / memory benchmark harness
├── bench-results/               # Committed benchmark JSON per variant (v0-baseline.json, ...)
├── README.md                    # You are here
└── LICENSE                      # MIT
```

---

## ⚠️ Model licence

[ZeroEntropy's zerank-2 weights](https://huggingface.co/zeroentropy/zerank-2) are released under **CC-BY-NC 4.0** (non-commercial). This repository is MIT-licensed, but the model weights it loads are not, so **check ZeroEntropy's terms before using this in a commercial product**. For commercial use, contact ZeroEntropy directly or use their hosted API.

The code in this repository (the MLX port, the validator, the benchmark harness) is MIT-licensed and can be adapted to other models (Qwen3-Reranker, Jina Reranker v3, etc.) without restriction.

---

## 🙏 Credits

- **ZeroEntropy** for releasing [zerank-2](https://huggingface.co/zeroentropy/zerank-2) and the reference PyTorch modeling code.
- **Apple MLX team** and the mlx-lm maintainers for the `Qwen3Model` implementation this port is built on.
- **199 Biotechnologies** for sponsoring the engineering work inside [engram](https://github.com/199-biotechnologies/engram) (upcoming).

Built because our long-running LoCoMo benchmark kept OOM-ing the PyTorch sidecar at question 1000-ish and we lost ~100 minutes of GPU time per run. Now it doesn't.

---

## 🤝 Contributing

Optimization ideas, bug reports, and weird edge cases welcome. Before sending a PR:

1. Run `validate_mlx_reranker.py` — it must pass (top-10 rank-identical to the PyTorch reference on the 20 test pairs).
2. Run `benchmark_mlx_reranker.py --variant <your-change-name>` and commit the `bench-results/<variant>.json` alongside the code.
3. Explain the change in the commit message: what it does, why, expected speedup, any risk.

If you add a new optimization variant, please keep the baseline path reproducible so we can always revert.

---

## 📖 Related projects

- [zeroentropy/zerank-2](https://huggingface.co/zeroentropy/zerank-2) — the model card
- [ml-explore/mlx](https://github.com/ml-explore/mlx) and [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
- [jina-ai/mlx-retrieval](https://github.com/jina-ai/mlx-retrieval) — reference Qwen2 cross-encoder implementation on MLX
- [willccbb/mlx_parallm](https://github.com/willccbb/mlx_parallm) — `BatchedKVCache` pattern we plan to borrow in v5
- [engram](https://github.com/199-biotechnologies/engram) — the 199 Biotechnologies memory engine this sidecar was built for

---

**Star the repo** ⭐ if this saved you from writing your own PyTorch → MLX port, and **open an issue** if your use case isn't covered yet.
