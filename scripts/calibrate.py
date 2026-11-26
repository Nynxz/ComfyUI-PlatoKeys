#!/usr/bin/env python3
"""Calibrate closed-form PlatoKeys between two Hugging Face text encoders.

Default target: Qwen/Qwen3.5-2B -> Qwen/Qwen3.5-9B for MOSS-Image (`last_hidden_state` +
MOSS open-<think> chat template).

Memory-efficient sequential design:
  - Pass 1 runs the source model (e.g. Qwen3.5-2B, ~4 GB VRAM) and streams fp16 token chunks
    to a temporary cache directory (~700 MB for 3,000 prompts).
  - Unloads the source model completely.
  - Pass 2 runs the target model (e.g. Qwen3.5-9B, ~18 GB bf16 on L4/A100, or ~10 GB with
    `--target-8bit` on T4 16GB) and accumulates exact float64 sufficient statistics `(N, Sx, Sy,
    Cxx, Cxy, Cyy)` on the fly while keeping a held-out validation split for scoring.
  - Fits & compares 4 closed-form maps in ~5 seconds on CPU:
      1. `procrustes`      — exact Plato's Key (semi-orthogonal R + least-squares global scale)
      2. `procrustes_norm` — semi-orthogonal R + target-variance-matching global scale
      3. `ridge`           — closed-form Ridge regression (auto-tuned lambda on val split)
      4. `ridge_norm`      — closed-form Ridge + target-variance-matching scale
  - Exports ready-to-drop `.safetensors` files for `ComfyUI/models/plato_keys/`.

Example:
  python scripts/calibrate.py \\
      --preset moss_image \\
      --source Qwen/Qwen3.5-2B \\
      --target Qwen/Qwen3.5-9B \\
      --num-prompts 3000 \\
      --out-dir out/qwen35_moss
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Allow running directly as `python scripts/calibrate.py` from repo root or scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plato_keys.format import (  # noqa: E402
    MOSS_IMAGE_TEMPLATE,
    QWEN35_DEFAULT_TEMPLATE,
    PlatoKeyMeta,
    save_key,
)
from plato_keys.math import (  # noqa: E402
    EvalMetrics,
    FitResult,
    SufficientStats,
    evaluate_key,
    fit_all_methods,
)
from scripts.prompt_corpus import gather_calibration_prompts  # noqa: E402


@dataclass(frozen=True)
class Preset:
    name: str
    source_layer: str
    target_layer: str
    source_layer_norm: bool
    target_layer_norm: bool
    template: str
    strip_prompt: bool


PRESETS: dict[str, Preset] = {
    "moss_image": Preset(
        name="moss_image",
        source_layer="last",
        target_layer="last",
        source_layer_norm=True,
        target_layer_norm=True,
        template=MOSS_IMAGE_TEMPLATE,
        strip_prompt=True,
    ),
    "qwen35_default": Preset(
        name="qwen35_default",
        source_layer="-2",
        target_layer="-2",
        source_layer_norm=False,
        target_layer_norm=False,
        template=QWEN35_DEFAULT_TEMPLATE,
        strip_prompt=False,
    ),
    "raw_last": Preset(
        name="raw_last",
        source_layer="last",
        target_layer="last",
        source_layer_norm=True,
        target_layer_norm=True,
        template="",
        strip_prompt=False,
    ),
}


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_hf_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # CRITICAL for causal models & Qwen3.5 GatedDeltaNet:
    # Right-padding ensures every real token at positions 0..L-1 sees identical causal context
    # to batch_size=1 inference in ComfyUI!
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|endoftext|>"})
    return tok


def load_hf_backbone(
    model_id: str,
    *,
    dtype: torch.dtype,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
    """Load a Hugging Face model and return `(top_model, text_backbone)`.

    Prefers `AutoModel` (skips the ~2 GB `lm_head` projection matrix) and unwraps multimodal
    containers (e.g. Qwen3.5's `.language_model`) so text-only batches bypass the vision tower.
    """
    from transformers import AutoModel

    kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if load_in_8bit or load_in_4bit:
        from transformers import BitsAndBytesConfig

        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        # Dispatch directly to GPU 0 so accelerate doesn't conservatively spill layers to CPU
        # on 15-16 GB cards (like T4) where Qwen3.5-9B 8-bit fits in ~11 GB VRAM.
        kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else "auto"
    else:
        kwargs["dtype"] = dtype
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}

    def _load(cls):
        try:
            return cls.from_pretrained(model_id, **kwargs)
        except TypeError:
            # Older transformers (<4.56) expect `torch_dtype` instead of `dtype`.
            if "dtype" in kwargs:
                alt = dict(kwargs)
                alt["torch_dtype"] = alt.pop("dtype")
                return cls.from_pretrained(model_id, **alt)
            raise

    try:
        model = _load(AutoModel)
    except Exception:
        from transformers import AutoModelForCausalLM

        model = _load(AutoModelForCausalLM)

    model.eval()
    backbone = _unwrap_text_backbone(model)
    # Free unused vision tower VRAM on multimodal models during text-only calibration.
    for container in (model, getattr(model, "model", None)):
        if container is not None and hasattr(container, "visual"):
            try:
                container.visual = None
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, backbone


def _unwrap_text_backbone(model: Any) -> Any:
    """Locate the text transformer module inside multimodal or causal-LM wrappers."""
    # Qwen3.5 / Qwen3-VL AutoModel -> Qwen3_5Model with `.language_model`
    if hasattr(model, "language_model"):
        return model.language_model
    # *ForConditionalGeneration -> `.model.language_model`
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    # *ForCausalLM -> `.model`
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    return model


@torch.no_grad()
def forward_hidden_states(
    backbone: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_spec: str,
    apply_layer_norm: bool,
) -> torch.Tensor:
    """Run one forward pass and return `[B, L, D]` hidden states matching `layer_spec`."""
    want_last = layer_spec.strip().lower() == "last"
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=not want_last,
        use_cache=False,
        return_dict=True,
    )
    if want_last:
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        return outputs.hidden_states[-1]

    idx = int(layer_spec)
    hs = outputs.hidden_states[idx]
    if apply_layer_norm and hasattr(backbone, "norm") and backbone.norm is not None:
        hs = backbone.norm(hs)
    return hs


def format_prompts(prompts: list[str], preset: Preset) -> list[str]:
    out = []
    for p in prompts:
        s = p.strip() if preset.strip_prompt else p
        if preset.template and not s.startswith("<|im_start|>"):
            s = preset.template.format(prompt=s)
        out.append(s)
    return out


def detect_common_prefix_suffix(token_id_lists: list[list[int]]) -> tuple[int, int]:
    """Find how many leading/trailing tokens are shared identically across all prompts."""
    if len(token_id_lists) < 2:
        return 0, 0
    min_len = min(len(t) for t in token_id_lists)
    prefix = 0
    while prefix < min_len and all(t[prefix] == token_id_lists[0][prefix] for t in token_id_lists):
        prefix += 1

    suffix = 0
    while (
        suffix < (min_len - prefix)
        and all(t[-1 - suffix] == token_id_lists[0][-1 - suffix] for t in token_id_lists)
    ):
        suffix += 1
    return prefix, suffix


def extract_batch_tokens(
    hidden: torch.Tensor,
    lengths: list[int],
    prefix_len: int,
    suffix_len: int,
    prefix_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten unpadded tokens `[N_tok, D]`, their calibration weights `[N_tok]`, and content mask `[N_tok]`."""
    vecs = []
    weights = []
    content_masks = []
    h_cpu = hidden.detach().to(device="cpu", dtype=torch.float16)
    for b_idx, seq_len in enumerate(lengths):
        vecs.append(h_cpu[b_idx, :seq_len])
        w = torch.ones(seq_len, dtype=torch.float32)
        cm = torch.ones(seq_len, dtype=torch.bool)
        if prefix_len > 0:
            p_end = min(prefix_len, seq_len)
            w[:p_end] = float(prefix_weight)
            cm[:p_end] = False
        if suffix_len > 0 and seq_len > prefix_len:
            s_start = max(prefix_len, seq_len - suffix_len)
            cm[s_start:] = False
        weights.append(w)
        content_masks.append(cm)
    return torch.cat(vecs, dim=0), torch.cat(weights, dim=0), torch.cat(content_masks, dim=0)


def unload_model(model: Any, backbone: Any) -> None:
    del backbone
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_comparison_table(
    title: str,
    results: list[FitResult],
    content_metrics: dict[str, EvalMetrics] | None = None,
) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'Method':<18} {'TokCos':>8} {'CtrCos':>8} {'CntCos':>8} "
        f"{'RelL2':>8} {'R2':>8} {'NormR':>8} {'ChStdR':>8} {'Scale':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        m = r.metrics
        cm = content_metrics.get(r.name) if content_metrics else None
        if m is None:
            print(f"{r.name:<18} (no validation split)")
            continue
        cnt_cos = cm.centered_cosine if cm is not None else float("nan")
        print(
            f"{r.name:<18} {m.token_cosine:>8.4f} {m.centered_cosine:>8.4f} {cnt_cos:>8.4f} "
            f"{m.rel_l2:>8.4f} {m.r2:>8.4f} {m.norm_ratio:>8.4f} {m.channel_std_corr:>8.4f} "
            f"{r.key.scale:>8.4f}"
        )
    print(
        "\nColumns: TokCos=raw token cosine, CtrCos=mean-centered cosine (semantic direction), "
        "CntCos=content-only centered cosine (excluding template tokens), RelL2=relative L2 error, "
        "R2=explained variance, NormR=predicted/target L2 norm ratio, ChStdR=per-channel std correlation.\n"
    )


def fit_and_export(
    stats: SufficientStats,
    *,
    val_x: torch.Tensor | None,
    val_y: torch.Tensor | None,
    val_content_mask: torch.Tensor | None,
    base_meta: PlatoKeyMeta,
    out_dir: Path,
    file_prefix: str,
    pca_rank: int | None,
    save_dtype: torch.dtype,
) -> list[FitResult]:
    results = fit_all_methods(
        stats,
        val_x=val_x,
        val_y=val_y,
        meta=base_meta,
        pca_rank=pca_rank,
    )

    content_metrics: dict[str, EvalMetrics] = {}
    if val_x is not None and val_y is not None and val_content_mask is not None and val_content_mask.any():
        vx_c = val_x[val_content_mask]
        vy_c = val_y[val_content_mask]
        for r in results:
            cm = evaluate_key(r.key, vx_c, vy_c)
            content_metrics[r.name] = cm
            r.key.meta.metrics.update(cm.to_dict(prefix="content"))

    print_comparison_table(
        f"{base_meta.source_model} -> {base_meta.target_model} ({stats.d_x} -> {stats.d_y})",
        results,
        content_metrics,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        out_path = out_dir / f"{file_prefix}_{r.name}.safetensors"
        save_key(r.key, out_path, dtype=save_dtype)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  saved {out_path} ({size_mb:.1f} MB)")

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate closed-form PlatoKeys between two text encoders.")
    p.add_argument("--source", default="Qwen/Qwen3.5-2B", help="Source Hugging Face model id or local path.")
    p.add_argument("--target", default="Qwen/Qwen3.5-9B", help="Target Hugging Face model id or local path.")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="moss_image")
    p.add_argument("--num-prompts", type=int, default=3000, help="Total calibration + validation prompts.")
    p.add_argument("--val-prompts", type=int, default=150, help="Held-out prompts for validation & Ridge tuning.")
    p.add_argument("--prompts-file", type=str, default=None, help="Optional local .txt or .jsonl prompt file.")
    p.add_argument("--max-length", type=int, default=256, help="Max tokens per prompt.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for GPU encoding.")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--source-8bit", action="store_true", help="Load source model in 8-bit (bitsandbytes).")
    p.add_argument("--target-8bit", action="store_true", help="Load target model in 8-bit (useful for 16GB T4).")
    p.add_argument("--target-4bit", action="store_true", help="Load target model in 4-bit (fallback for small VRAM).")
    p.add_argument(
        "--prefix-weight",
        type=float,
        default=0.05,
        help="Calibration weight for repeated constant chat-template prefix tokens (default 0.05).",
    )
    p.add_argument("--pca-rank", type=int, default=None, help="Optional rank for PCA-Procrustes ablation (e.g. 256).")
    p.add_argument("--also-reverse", action="store_true", help="Also fit & export reverse target -> source keys.")
    p.add_argument("--save-stats", action="store_true", help="Save sufficient stats (.pt) for instant CPU refitting.")
    p.add_argument("--from-stats", type=str, default=None, help="Skip encoding and refit from a saved stats.pt bundle.")
    p.add_argument("--save-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--out-dir", type=str, default="out/plato_keys", help="Output directory for .safetensors keys.")
    p.add_argument("--prefix", type=str, default=None, help="Output filename prefix (auto-derived if omitted).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _default_prefix(source: str, target: str, preset_name: str) -> str:
    def _short(name: str) -> str:
        leaf = name.rstrip("/").split("/")[-1].lower()
        return leaf.replace("-", "_").replace(".", "")

    return f"{_short(source)}_to_{_short(target)}_{preset_name}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = resolve_dtype(args.save_dtype)

    if args.from_stats:
        bundle = torch.load(args.from_stats, map_location="cpu", weights_only=False)
        stats = SufficientStats.load(bundle["stats_path"]) if "stats_path" in bundle else bundle["stats"]
        meta = PlatoKeyMeta(**bundle["meta"])
        val_x = bundle.get("val_x")
        val_y = bundle.get("val_y")
        val_cm = bundle.get("val_content_mask")
        prefix = args.prefix or _default_prefix(meta.source_model, meta.target_model, meta.target_preset or "custom")
        fit_and_export(
            stats,
            val_x=val_x,
            val_y=val_y,
            val_content_mask=val_cm,
            base_meta=meta,
            out_dir=out_dir,
            file_prefix=prefix,
            pca_rank=args.pca_rank,
            save_dtype=save_dtype,
        )
        return 0

    preset = PRESETS[args.preset]
    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[calibrate] device={device}, dtype={dtype}, preset={preset.name}")
    print(f"[calibrate] source={args.source} -> target={args.target}")

    # 1. Gather prompts
    t0 = time.time()
    prompts, dataset_label = gather_calibration_prompts(
        num_prompts=args.num_prompts,
        prompts_file=args.prompts_file,
        include_domain_mix=(args.preset == "moss_image"),
        seed=args.seed,
    )
    formatted = format_prompts(prompts, preset)
    print(f"[calibrate] collected {len(formatted)} prompts ({dataset_label}) in {time.time() - t0:.1f}s")

    # 2. Tokenize once and verify source/target tokenizers agree 1-to-1
    tok_src = load_hf_tokenizer(args.source)
    tok_tgt = load_hf_tokenizer(args.target)
    sample_check = formatted[:8]
    src_sample = tok_src(sample_check, add_special_tokens=False)["input_ids"]
    tgt_sample = tok_tgt(sample_check, add_special_tokens=False)["input_ids"]
    if src_sample != tgt_sample:
        raise RuntimeError(
            "Source and target tokenizers produced different token IDs on the sample prompts! "
            "Token-level alignment requires a shared tokenizer (e.g. Qwen3.5 0.8B/2B/4B/9B/27B)."
        )
    del tok_tgt

    encoded_all = tok_src(
        formatted,
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_length,
    )["input_ids"]

    # Sort by length inside train/val splits? Keeping split deterministic first, then batching by length
    # saves massive compute on padded tokens!
    n_val = min(args.val_prompts, max(1, len(encoded_all) // 10))
    val_ids = encoded_all[:n_val]
    train_ids = encoded_all[n_val:]

    # Sort train prompts by length for minimal padding waste (exact same set of tokens).
    train_ids.sort(key=len)
    val_ids.sort(key=len)

    prefix_len, suffix_len = detect_common_prefix_suffix(encoded_all)
    print(
        f"[calibrate] split: {len(train_ids)} train / {len(val_ids)} val prompts; "
        f"detected common template prefix={prefix_len} tok, suffix={suffix_len} tok "
        f"(prefix_weight={args.prefix_weight})"
    )

    pad_id = int(tok_src.pad_token_id)

    def make_batches(id_lists: list[list[int]]) -> list[tuple[torch.Tensor, torch.Tensor, list[int]]]:
        batches = []
        for start in range(0, len(id_lists), args.batch_size):
            chunk = id_lists[start : start + args.batch_size]
            lengths = [len(x) for x in chunk]
            max_l = max(lengths)
            ids_t = torch.full((len(chunk), max_l), pad_id, dtype=torch.long)
            mask_t = torch.zeros((len(chunk), max_l), dtype=torch.long)
            for row_i, seq in enumerate(chunk):
                ids_t[row_i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                mask_t[row_i, : len(seq)] = 1
            batches.append((ids_t, mask_t, lengths))
        return batches

    train_batches = make_batches(train_ids)
    val_batches = make_batches(val_ids)

    cache_dir = Path(tempfile.mkdtemp(prefix="plato_calib_"))
    try:
        # --- PASS 1: Source model ---
        t1 = time.time()
        print(f"\n[pass 1/2] Loading source model: {args.source}")
        src_model, src_backbone = load_hf_backbone(
            args.source, dtype=dtype, load_in_8bit=args.source_8bit
        )
        d_source = None

        def run_source_split(batches, split_name: str) -> None:
            nonlocal d_source
            for b_idx, (ids_t, mask_t, lengths) in enumerate(batches):
                ids_dev = ids_t.to(device)
                mask_dev = mask_t.to(device)
                hs = forward_hidden_states(
                    src_backbone, ids_dev, mask_dev, preset.source_layer, preset.source_layer_norm
                )
                vecs, weights, cmask = extract_batch_tokens(
                    hs, lengths, prefix_len, suffix_len, args.prefix_weight
                )
                if d_source is None:
                    d_source = int(vecs.shape[-1])
                torch.save(
                    {"x": vecs, "w": weights, "cm": cmask, "n_prompts": len(lengths)},
                    cache_dir / f"{split_name}_{b_idx:05d}.pt",
                )
                if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(batches):
                    print(f"  [{split_name}] batch {b_idx + 1}/{len(batches)}")

        run_source_split(train_batches, "train")
        run_source_split(val_batches, "val")
        assert d_source is not None
        print(f"[pass 1/2] Done in {time.time() - t1:.1f}s (d_source={d_source}). Unloading source model...")
        unload_model(src_model, src_backbone)

        # --- PASS 2: Target model + streaming sufficient statistics ---
        t2 = time.time()
        print(f"\n[pass 2/2] Loading target model: {args.target}")
        tgt_model, tgt_backbone = load_hf_backbone(
            args.target,
            dtype=dtype,
            load_in_8bit=args.target_8bit,
            load_in_4bit=args.target_4bit,
        )
        stats: SufficientStats | None = None
        d_target: int | None = None

        for b_idx, (ids_t, mask_t, lengths) in enumerate(train_batches):
            ids_dev = ids_t.to(device)
            mask_dev = mask_t.to(device)
            hs = forward_hidden_states(
                tgt_backbone, ids_dev, mask_dev, preset.target_layer, preset.target_layer_norm
            )
            y_vecs, _, _ = extract_batch_tokens(
                hs, lengths, prefix_len, suffix_len, args.prefix_weight
            )
            src_chunk = torch.load(cache_dir / f"train_{b_idx:05d}.pt", map_location="cpu", weights_only=True)
            if stats is None:
                d_target = int(y_vecs.shape[-1])
                stats = SufficientStats(d_source, d_target)
            stats.update(
                src_chunk["x"],
                y_vecs,
                weights=src_chunk["w"],
                num_prompts=int(src_chunk["n_prompts"]),
            )
            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(train_batches):
                print(f"  [train] batch {b_idx + 1}/{len(train_batches)} ({stats.token_count:,} tokens)")

        val_x_list, val_y_list, val_cm_list = [], [], []
        for b_idx, (ids_t, mask_t, lengths) in enumerate(val_batches):
            ids_dev = ids_t.to(device)
            mask_dev = mask_t.to(device)
            hs = forward_hidden_states(
                tgt_backbone, ids_dev, mask_dev, preset.target_layer, preset.target_layer_norm
            )
            y_vecs, _, _ = extract_batch_tokens(
                hs, lengths, prefix_len, suffix_len, args.prefix_weight
            )
            src_chunk = torch.load(cache_dir / f"val_{b_idx:05d}.pt", map_location="cpu", weights_only=True)
            val_x_list.append(src_chunk["x"])
            val_y_list.append(y_vecs)
            val_cm_list.append(src_chunk["cm"])

        val_x = torch.cat(val_x_list, dim=0)
        val_y = torch.cat(val_y_list, dim=0)
        val_cm = torch.cat(val_cm_list, dim=0)
        print(f"[pass 2/2] Done in {time.time() - t2:.1f}s (d_target={d_target}, val_tokens={val_x.shape[0]:,}).")
        unload_model(tgt_model, tgt_backbone)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    assert stats is not None and d_target is not None

    # 3. Fit and export forward keys (source -> target)
    base_meta = PlatoKeyMeta(
        source_model=args.source,
        target_model=args.target,
        target_preset=preset.name,
        source_layer=preset.source_layer,
        target_layer=preset.target_layer,
        source_layer_norm=preset.source_layer_norm,
        target_layer_norm=preset.target_layer_norm,
        source_template=preset.template,
        strip_prompt=preset.strip_prompt,
        dataset=dataset_label,
        num_prompts=stats.prompt_count,
        num_tokens=stats.token_count,
    )

    if args.save_stats:
        stats_path = out_dir / "sufficient_stats.pt"
        stats.save(stats_path)
        torch.save(
            {
                "stats_path": str(stats_path),
                "meta": vars(base_meta),
                "val_x": val_x,
                "val_y": val_y,
                "val_content_mask": val_cm,
            },
            out_dir / "calibration_bundle.pt",
        )
        print(f"[calibrate] saved sufficient stats bundle to {out_dir / 'calibration_bundle.pt'}")

    t3 = time.time()
    prefix = args.prefix or _default_prefix(args.source, args.target, preset.name)
    fwd_results = fit_and_export(
        stats,
        val_x=val_x,
        val_y=val_y,
        val_content_mask=val_cm,
        base_meta=base_meta,
        out_dir=out_dir,
        file_prefix=prefix,
        pca_rank=args.pca_rank,
        save_dtype=save_dtype,
    )

    summary_payload: dict[str, Any] = {
        "forward": {r.name: r.key.meta.metrics for r in fwd_results},
    }

    if args.also_reverse:
        rev_meta = PlatoKeyMeta(
            source_model=args.target,
            target_model=args.source,
            target_preset=preset.name,
            source_layer=preset.target_layer,
            target_layer=preset.source_layer,
            source_layer_norm=preset.target_layer_norm,
            target_layer_norm=preset.source_layer_norm,
            source_template=preset.template,
            strip_prompt=preset.strip_prompt,
            dataset=dataset_label,
            num_prompts=stats.prompt_count,
            num_tokens=stats.token_count,
        )
        rev_prefix = _default_prefix(args.target, args.source, preset.name)
        rev_results = fit_and_export(
            stats.swap(),
            val_x=val_y,
            val_y=val_x,
            val_content_mask=val_cm,
            base_meta=rev_meta,
            out_dir=out_dir,
            file_prefix=rev_prefix,
            pca_rank=args.pca_rank,
            save_dtype=save_dtype,
        )
        summary_payload["reverse"] = {r.name: r.key.meta.metrics for r in rev_results}

    (out_dir / "metrics.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"[calibrate] fitting & export finished in {time.time() - t3:.1f}s. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
