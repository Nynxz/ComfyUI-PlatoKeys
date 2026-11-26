# ComfyUI-PlatoKeys

Closed-form cross-encoder conditioning alignment for ComfyUI, based on Stanford Hazy Research's
[**Plato's Key**](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key).

Encode prompts with a **smaller, faster text encoder** (starting with **Qwen3.5-2B**, ~4.5 GB) and
project its hidden states into a **larger target encoder's space** (**Qwen3.5-9B**, ~18 GB — the text
encoder expected by **MOSS-Image**) with a single closed-form matrix multiply (`16 MB` `.safetensors`
key, zero gradient descent, zero retraining of the diffusion model or text encoder).

## Pre-calibrated keys (Hugging Face)

Download calibrated `.safetensors` keys from [**nynxz/PlatoKeys**](https://huggingface.co/nynxz/PlatoKeys)
and place them in `ComfyUI/models/plato_keys/`:

### Qwen3.5-2B → Qwen3.5-9B (`2048 → 4096`, MOSS-Image)

Calibrated on 2,000 diverse prompts (75,116 train tokens / 5,710 held-out validation tokens):

| Key (`16 MB` fp16) | Method | TokCos | CtrCos | CntCos | RelL2 | R² | NormR | ChStdR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen35_2b_to_qwen35_9b_moss_image_procrustes.safetensors` | Plato's Key (Procrustes + LS scale) | 0.8560 | 0.7850 | 0.7598 | 0.5163 | 0.6360 | 0.8640 | 0.9854 |
| `qwen35_2b_to_qwen35_9b_moss_image_procrustes_norm.safetensors` | Plato's Key (Procrustes + target variance scale) | 0.8514 | 0.7850 | 0.7598 | 0.5463 | 0.5925 | 1.0072 | 0.9854 |
| **`qwen35_2b_to_qwen35_9b_moss_image_ridge.safetensors`** | Closed-form Ridge (`alpha=0.01`) | **0.8978** | **0.8402** | **0.8088** | **0.4291** | **0.7486** | 0.9035 | **0.9981** |
| **`qwen35_2b_to_qwen35_9b_moss_image_ridge_norm.safetensors`** | Closed-form Ridge + target variance scale | **0.8951** | **0.8402** | **0.8088** | 0.4497 | 0.7238 | **1.0092** | **0.9981** |
| `qwen35_2b_to_qwen35_9b_moss_image_procrustes_pca256.safetensors` | Rank-256 PCA-Procrustes | 0.8375 | 0.7492 | 0.7109 | 0.5419 | 0.5991 | 0.8393 | 0.9855 |

Reverse keys (`Qwen3.5-9B → Qwen3.5-2B`, reaching `0.9433` token cosine / `0.8384` R² with Ridge) are also available in the same Hugging Face repo.

## How it works

Because Qwen3.5-2B (`D = 2048`) and Qwen3.5-9B (`D = 4096`) share the exact same tokenizer
(`qwen35_tokenizer`), every prompt produces identical token sequences of the same length $L$. Instead
of aligning only a single pooled vector, we align **every token position 1-to-1** across thousands of
shared prompts:

1. **Streaming `float64` sufficient statistics** (`SufficientStats`): accumulates $(W, \sum x, \sum y,
   X^\top X, X^\top Y, Y^\top Y)$ in constant ~224 MB RAM while loading the 2B and 9B models
   *sequentially* (runs on Colab L4/A100 in `bf16`, or T4 16 GB with `--target-8bit`).
2. **Exact semi-orthogonal Procrustes (no zero-padding waste)**: thin SVD directly on
   $M_{xy} = X_c^\top Y_c \in \mathbb{R}^{2048 \times 4096}$ produces $R = U V^\top$ (`16 MB` in fp16
   instead of `32 MB`) with $R R^\top = I_{2048}$.
3. **Closed-form Ridge comparison**: Hazy Research showed unconstrained Ridge overfit on < 5,000
   single-token samples; with 75,000+ paired token vectors ($N \approx 37 D$), regularized closed-form
   Ridge matches Qwen3.5's anisotropic outlier channels (`ChStdR = 0.9981`) while staying a single
   `16 MB` linear map at inference.
4. **Automatic layer & chat-template parity for MOSS-Image**: stock ComfyUI `CLIPLoader` loads
   `qwen3.5_2b` with `layer="hidden", layer_idx=-2` (unnormed) and `<think>\n</think>\n`, whereas
   MOSS-Image expects `last_hidden_state` (`layer="last"`, final RMSNorm) and open `<think>\n`. Every
   `.safetensors` PlatoKey embeds its calibrated layer and template in its header, and **Plato Key Encode**
   temporarily overrides `options_default` on the underlying `SDClipModel` (restoring it in `finally`)
   without modifying ComfyUI core.

## Install

Clone into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Nynxz/ComfyUI-PlatoKeys
```

Place `.safetensors` keys from [nynxz/PlatoKeys](https://huggingface.co/nynxz/PlatoKeys) into `ComfyUI/models/plato_keys/` (auto-registered via `folder_paths`).

## Nodes (`conditioning/plato keys`, ComfyUI V3)

| Node | Inputs → Outputs | Purpose |
|---|---|---|
| **Load Plato Key** | `key_name` → `PLATO_KEY`, `info` | Loads a `.safetensors` key from `models/plato_keys/` or `keys/` and outputs its metadata & held-out metrics. |
| **Plato Key Encode** | `clip`, `plato_key`, `text` → `CONDITIONING` | Drop-in replacement for `CLIPTextEncode`. Honors the key's calibrated layer (`last`) and chat template (MOSS-Image) automatically, then projects conditioning into target space. |
| **Apply Plato Key** | `conditioning`, `plato_key` → `CONDITIONING` | Projects any existing `CONDITIONING` wire directly (for custom pipelines or encoder pairs whose stock `CLIPTextEncode` already matches). |
| **Compare Conditioning (Plato Key)** | `cond_mapped`, `cond_target` → `report` | Diagnostic node comparing mapped vs native target conditioning (shape, RMS ratio, per-token cosine, head/tail tokens, relative L2). |

### MOSS-Image workflow with Qwen3.5-2B

```text
UNETLoader (moss_image / kromoss) ────────────────────────────────────────┐
                                                                          ▼
CLIPLoader (qwen3.5_2b_bf16.safetensors) ──┬──► Plato Key Encode (+) ──► KSampler
Load Plato Key (qwen35_2b_to_9b_moss_*) ───┼──► Plato Key Encode (-) ────┘
                                                                          ▲
VAELoader (flux2-vae.safetensors) ────────────────────────────────────────┘
```

*(For distilled `cfg = 1` models like 8-step DMD2, `ConditioningZeroOut` can feed the negative slot.)*

## Calibrating your own keys

Open [`colab/calibrate_qwen35_moss.ipynb`](colab/calibrate_qwen35_moss.ipynb) in Google Colab, or run locally:

```bash
python scripts/calibrate.py \
    --preset moss_image \
    --source Qwen/Qwen3.5-2B \
    --target Qwen/Qwen3.5-9B \
    --num-prompts 2000 \
    --out-dir out/qwen35_moss
```

Useful flags:
- `--target-8bit` / `--target-4bit`: fits Qwen3.5-9B on 16 GB GPUs (e.g. Colab T4).
- `--also-reverse`: also fits and exports `target → source` keys from the same sufficient stats.
- `--save-stats` / `--from-stats`: save sufficient stats (~224 MB) and refit on CPU in seconds.
- `--prompts-file my_prompts.txt` (or `.jsonl`): calibrate on your own prompt distribution.
- `--pca-rank 256`: also export a rank-256 PCA-Procrustes key.
- `--preset qwen35_default`: calibrate for stock ComfyUI `qwen35` `layer_idx=-2` extraction.

## Tests

```bash
pytest
ruff check .
```

## Credits

- **Plato's Key**: Vivien Cheng, Jerry Liu, Morris Yau, and Chris Ré — [Stanford Hazy Research (2026)](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key).
- **Qwen3.5** by Alibaba Qwen Team; **MOSS-Image** by OpenMOSS.
