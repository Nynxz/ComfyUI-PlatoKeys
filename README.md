# ComfyUI-PlatoKeys

Closed-form cross-encoder conditioning alignment for ComfyUI, based on Stanford Hazy Research's
[**Plato's Key**](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key).

Encode prompts (and images via **Patch CLIP**) with a **different text encoder** — smaller/faster
(e.g. **Qwen3.5-2B** for MOSS-Image or Krea 2) or stronger/larger (e.g. **Qwen3.5-9B** for Krea 2) —
and project its hidden states into the target encoder's space with a single closed-form linear map
(zero gradient descent, zero retraining of the diffusion model or text encoder).

Supports both **single-layer** targets (MOSS-Image `Qwen3.5-9B`) and **multi-layer tap** targets
(Krea 2 `Qwen3-VL-4B` 12-layer stack, including cross-tokenizer calibration via UTF-8 byte-span
alignment).

## Pre-calibrated keys (Hugging Face)

Download calibrated `.safetensors` keys from [**nynxz/PlatoKeys**](https://huggingface.co/nynxz/PlatoKeys)
and place them in `ComfyUI/models/plato_keys/`:

### 1. MOSS-Image: Qwen3.5-2B → Qwen3.5-9B (`2048 → 4096`)

Calibrated on 2,000 diverse prompts (75,116 train tokens / 5,710 held-out validation tokens):

| Key (`16 MB` fp16) | Method | TokCos | CtrCos | CntCos | RelL2 | R² | NormR | ChStdR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen35_2b_to_qwen35_9b_moss_image_procrustes.safetensors` | Plato's Key (Procrustes + LS scale) | 0.8560 | 0.7850 | 0.7598 | 0.5163 | 0.6360 | 0.8640 | 0.9854 |
| `qwen35_2b_to_qwen35_9b_moss_image_procrustes_norm.safetensors` | Plato's Key (Procrustes + target variance scale) | 0.8514 | 0.7850 | 0.7598 | 0.5463 | 0.5925 | 1.0072 | 0.9854 |
| **`qwen35_2b_to_qwen35_9b_moss_image_ridge.safetensors`** | Closed-form Ridge (`alpha=0.01`) | **0.8978** | **0.8402** | **0.8088** | **0.4291** | **0.7486** | 0.9035 | **0.9981** |
| **`qwen35_2b_to_qwen35_9b_moss_image_ridge_norm.safetensors`** | Closed-form Ridge + target variance scale | **0.8951** | **0.8402** | **0.8088** | 0.4497 | 0.7238 | **1.0092** | **0.9981** |
| `qwen35_2b_to_qwen35_9b_moss_image_procrustes_pca256.safetensors` | Rank-256 PCA-Procrustes | 0.8375 | 0.7492 | 0.7109 | 0.5419 | 0.5991 | 0.8393 | 0.9855 |

*(Reverse `Qwen3.5-9B → Qwen3.5-2B` keys reaching `0.9433` TokCos / `0.8384` R² are also hosted in the same repo.)*

### 2. Krea 2 (12-layer tap): Qwen3.5-2B → Qwen3-VL-4B (`12×2048 → 12×2560`, smaller/faster)

Cross-tokenizer aligned via UTF-8 byte spans (99.99% token alignment across Qwen3.5 248k vs Qwen3-VL 152k vocab), tapping 12 proportional layers `[1,3,5,7,9,11,13,15,17,19,21,23]` → Krea 2 `[2,5,8,11,14,17,20,23,26,29,32,35]`:

| Key (`120 MB` fp16) | Method | TokCos | CtrCos | CntCos | RelL2 | R² | NormR | ChStdR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen35_2b_to_qwen3vl_4b_krea2_procrustes.safetensors` | Plato's Key (Procrustes + LS scale) | 0.9008 | 0.7954 | 0.7920 | 0.4443 | 0.5965 | 0.9168 | 0.9711 |
| `qwen35_2b_to_qwen3vl_4b_krea2_procrustes_norm.safetensors` | Plato's Key (Procrustes + variance scale) | 0.8914 | 0.7954 | 0.7920 | 0.4817 | 0.5258 | 1.0360 | 0.9711 |
| **`qwen35_2b_to_qwen3vl_4b_krea2_ridge.safetensors`** | Closed-form Ridge | **0.9352** | **0.8638** | **0.8573** | **0.3548** | **0.7412** | 0.9377 | **0.9977** |
| **`qwen35_2b_to_qwen3vl_4b_krea2_ridge_norm.safetensors`** | Closed-form Ridge + variance scale | **0.9309** | **0.8638** | **0.8573** | 0.3756 | 0.7097 | **1.0181** | **0.9977** |

### 3. Krea 2 (12-layer tap): Qwen3.5-9B → Qwen3-VL-4B (`12×4096 → 12×2560`, bigger/stronger)

Tapping 12 proportional layers `[2,4,7,10,12,15,18,20,23,26,28,31]` → Krea 2 `[2,5,8,11,14,17,20,23,26,29,32,35]`:

| Key (`240 MB` fp16) | Method | TokCos | CtrCos | CntCos | RelL2 | R² | NormR | ChStdR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen35_9b_to_qwen3vl_4b_krea2_procrustes.safetensors` | Plato's Key (Procrustes + LS scale) | 0.9020 | 0.7932 | 0.7932 | 0.4417 | 0.6000 | 0.9155 | 0.9697 |
| `qwen35_9b_to_qwen3vl_4b_krea2_procrustes_norm.safetensors` | Plato's Key (Procrustes + variance scale) | 0.8937 | 0.7932 | 0.7932 | 0.4769 | 0.5334 | 1.0285 | 0.9697 |
| **`qwen35_9b_to_qwen3vl_4b_krea2_ridge.safetensors`** | Closed-form Ridge | **0.9432** | **0.8798** | **0.8759** | **0.3324** | **0.7723** | 0.9454 | **0.9978** |
| **`qwen35_9b_to_qwen3vl_4b_krea2_ridge_norm.safetensors`** | Closed-form Ridge + variance scale | **0.9400** | **0.8798** | **0.8759** | 0.3500 | 0.7472 | **1.0164** | **0.9978** |

## How it works

1. **Streaming `float64` sufficient statistics** (`SufficientStats`): accumulates $(W, \sum x, \sum y,
   X^\top X, X^\top Y, Y^\top Y)$ in constant memory while loading source and target encoders
   *sequentially*.
2. **Exact semi-orthogonal Procrustes (no zero-padding waste)** & **closed-form Ridge**: thin SVD on
   $M_{xy} = X_c^\top Y_c$ gives $R = U V^\top$ with $R R^\top = I$; with 70,000+ paired token vectors,
   scale-invariant closed-form Ridge matches anisotropic outlier channels (`ChStdR ≈ 0.998`) in a single
   linear projection at inference.
3. **Cross-tokenizer & multi-layer support**: for encoder families with different vocabularies
   (Qwen3.5 248k vs Qwen3-VL 152k), `scripts/calibrate_comfy.py` aligns shared UTF-8 byte spans across
   every prompt (~99.99% match rate) and stacks per-layer projections into a single `[N, d_target, d_source]`
   `.safetensors` key that flattens `[B, 12, seq, 2560]` → `[B, seq, 30720]` identically to `krea2.py`.
4. **Automatic layer, template, and prefix-strip parity**: every `.safetensors` PlatoKey embeds its
   calibrated layer(s), `layer_norm_hidden_state`, chat template, and prefix-strip mode in its header.
   **Plato Key Encode** and **Patch CLIP (Plato Key)** temporarily override `options_default` on the
   underlying `SDClipModel` (restoring it in `finally`) without modifying ComfyUI core.

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
| **Compose Plato Keys (Hub)** | `key_a_to_hub`, `key_hub_to_b` → `PLATO_KEY`, `info` | Analytically composes two keys (`A → Hub` and `Hub → B`) into a single zero-overhead `A → B` key via $W = W_2 W_1$ (supports 2D, 3D multi-layer, and 3D→2D). |
| **Patch CLIP (Plato Key)** | `clip`, `plato_key` → `CLIP` | Clones a `CLIP` and wraps `tokenize` / `encode_from_tokens*` / `clone` so **any downstream node** (`CLIPTextEncode`, Visual Fusion, Layout/Regions, `TextEncodeQwenImageEditPlus`, etc.) transparently taps the calibrated layer(s) and outputs target-space conditioning — including multi-layer Krea 2, `K → 1` multi-layer concat, and visual tokens. |
| **Plato Key Encode** | `clip`, `plato_key`, `text` → `CONDITIONING` | Drop-in replacement for `CLIPTextEncode`. Honors the key's calibrated layer(s), chat template, and prefix strip automatically, then projects conditioning into target space. |
| **Apply Plato Key** | `conditioning`, `plato_key` → `CONDITIONING` | Projects an existing single-layer `CONDITIONING` wire directly (for custom pipelines whose upstream encoder already matches the key's layer & template; for multi-layer keys like Krea 2, use **Patch CLIP (Plato Key)** or **Plato Key Encode** instead). |
| **Compare Conditioning (Plato Key)** | `cond_mapped`, `cond_target` → `report` | Diagnostic node comparing mapped vs native target conditioning (shape, RMS ratio, per-token cosine, head/tail tokens, relative L2). |

### Example 1: MOSS-Image with Qwen3.5-2B

```text
UNETLoader (moss_image / kromoss) ────────────────────────────────────────┐
                                                                          ▼
CLIPLoader (qwen3.5_2b_bf16, type=qwen_image) ─┬─► Plato Key Encode (+) ► KSampler
Load Plato Key (qwen35_2b_to_9b_moss_ridge) ───┼─► Plato Key Encode (-) ──┘
                                                                          ▲
VAELoader (flux2-vae.safetensors) ────────────────────────────────────────┘
```

### Example 2: Krea 2 + Visual Fusion (or any downstream CLIP node) with Qwen3.5-2B or Qwen3.5-9B

```text
CLIPLoader (qwen3.5_9b / 2b, type=qwen_image) ─┐
                                               ▼
Load Plato Key (qwen35_*_to_qwen3vl_4b_krea2) ► Patch CLIP (Plato Key) ► Text Encode (Fusion) / CLIPTextEncode ► KSampler (Krea 2)
```

*(For distilled `cfg = 1` models like Krea 2 Turbo or MOSS DMD2, `ConditioningZeroOut` can feed the negative slot.)*

## Calibrating your own keys

### Option A: Local ComfyUI checkpoints (supports multi-layer Krea 2, quantized `int8`/`fp8`, cross-tokenizer)

```bash
python scripts/calibrate_comfy.py \
    --comfy-dir /path/to/ComfyUI \
    --preset krea2 \
    --source-ckpt /path/to/qwen3.5_9b_int8_convrot.safetensors \
    --source-clip-type qwen_image \
    --target-ckpt /path/to/qwen3vl_4b_bf16.safetensors \
    --target-clip-type krea2 \
    --num-prompts 2000 \
    --out-dir keys \
    --file-prefix qwen35_9b_to_qwen3vl_4b_krea2
```

### Option B: Hugging Face models (or Google Colab)

Open [`colab/calibrate_qwen35_moss.ipynb`](colab/calibrate_qwen35_moss.ipynb) in Google Colab, or run locally:

```bash
python scripts/calibrate.py \
    --preset moss_image \
    --source Qwen/Qwen3.5-2B \
    --target Qwen/Qwen3.5-9B \
    --num-prompts 2000 \
    --out-dir out/qwen35_moss
```

Useful flags for `scripts/calibrate.py`:
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
- **Qwen3.5** & **Qwen3-VL** by Alibaba Qwen Team; **MOSS-Image** by OpenMOSS; **Krea 2** by Krea.
