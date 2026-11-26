#!/usr/bin/env python3
"""ComfyUI-native PlatoKey calibrator (single-layer & multi-layer, same & cross-tokenizer).

Loads local ComfyUI text-encoder `.safetensors` checkpoints directly through ComfyUI's own
`comfy.sd.load_clip(...)` so bf16 / fp8 / int8_convrot checkpoints and multi-layer taps (such as
Krea 2's 12-layer stack `[2, 5, 8, ..., 35]`) match ComfyUI inference bit-for-bit.

Supports both:
  - Same-tokenizer pairs (1-to-1 exact token pairing, e.g. Qwen3-VL 2B/8B -> Krea 2 4B,
    Qwen3.5 0.8B/2B/4B -> MOSS-Image 9B).
  - Cross-tokenizer pairs (UTF-8 byte-end boundary pairing inside the prompt body + 1-to-1 suffix
    token pairing, e.g. Qwen3.5-2B / Qwen3.5-9B -> Krea 2 Qwen3-VL-4B).

Run using ComfyUI's Python venv (which has `comfy` and GPU PyTorch):
  <ComfyUI>/.venv/bin/python scripts/calibrate_comfy.py \\
      --comfy /path/to/ComfyUI \\
      --preset krea2 \\
      --source /path/to/qwen3.5_2b_bf16.safetensors \\
      --target /path/to/qwen3vl_4b_bf16.safetensors \\
      --num-prompts 2000 \\
      --out-dir keys/
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plato_keys.encode import _inner_clip_models, find_krea2_prefix_end  # noqa: E402
from plato_keys.format import (  # noqa: E402
    KREA2_TAP_LAYERS,
    KREA2_TEMPLATE,
    MOSS_IMAGE_TEMPLATE,
    PlatoKey,
    PlatoKeyMeta,
    save_key,
)
from plato_keys.math import (  # noqa: E402
    SufficientStats,
    evaluate_key,
    fit_procrustes,
    fit_ridge,
)
from scripts.prompt_corpus import gather_calibration_prompts  # noqa: E402


@dataclass(frozen=True)
class ComfyPreset:
    name: str
    template: str
    strip_prompt: bool
    strip_prefix_mode: str
    target_layers: str | list[int]  # "last", int, or list[int]
    target_layer_norm: bool
    flatten_layers: bool
    suffix_tokens: int  # number of trailing assistant-turn tokens to align 1-to-1
    clip_type_target: str


COMFY_PRESETS: dict[str, ComfyPreset] = {
    "krea2": ComfyPreset(
        name="krea2",
        template=KREA2_TEMPLATE,
        strip_prompt=False,
        strip_prefix_mode="krea2_user",
        target_layers=list(KREA2_TAP_LAYERS),
        target_layer_norm=False,
        flatten_layers=True,
        suffix_tokens=5,  # <|im_end|>, \n, <|im_start|>, assistant, \n
        clip_type_target="krea2",
    ),
    "moss_image": ComfyPreset(
        name="moss_image",
        template=MOSS_IMAGE_TEMPLATE,
        strip_prompt=True,
        strip_prefix_mode="",
        target_layers="last",
        target_layer_norm=True,
        flatten_layers=False,
        suffix_tokens=7,  # <|im_end|>, \n, <|im_start|>, assistant, \n, <think>, \n
        clip_type_target="qwen_image",
    ),
}


def proportional_source_layers(target_layers: list[int], target_num_layers: int, source_num_layers: int) -> list[int]:
    """Map target tap indices proportionally onto `source_num_layers` (strictly preserving order)."""
    if source_num_layers == target_num_layers:
        return list(target_layers)
    mapped: list[int] = []
    for k in target_layers:
        # Scale k in [0, target_num_layers] to [0, source_num_layers], keeping < source_num_layers
        # whenever k < target_num_layers (so unnormed hidden taps stay unnormed).
        scaled = int(round(k * source_num_layers / target_num_layers))
        if k < target_num_layers:
            scaled = max(1, min(source_num_layers - 1, scaled))
        else:
            scaled = source_num_layers
        mapped.append(scaled)
    return mapped


def token_byte_ends(hf_tokenizer: Any, token_ids: list[int]) -> list[int]:
    """Cumulative UTF-8 byte end offsets for byte-level BPE token IDs (Qwen2.5/Qwen3/Qwen3.5)."""
    decoder = getattr(hf_tokenizer, "byte_decoder", None)
    ends: list[int] = []
    pos = 0
    if decoder is not None:
        for piece in hf_tokenizer.convert_ids_to_tokens(token_ids):
            if piece is None:
                ends.append(pos)
                continue
            b = bytes(decoder[c] for c in piece) if all(c in decoder for c in piece) else piece.encode("utf-8")
            pos += len(b)
            ends.append(pos)
        return ends

    # Fallback: incremental decode
    for k in range(1, len(token_ids) + 1):
        txt = hf_tokenizer.decode(token_ids[:k], skip_special_tokens=False).rstrip("\ufffd")
        pos = max(pos, len(txt.encode("utf-8")))
        ends.append(pos)
    return ends


def align_token_sequences(
    src_ids: list[int],
    tgt_ids: list[int],
    src_hf_tok: Any,
    tgt_hf_tok: Any,
    *,
    suffix_tokens: int = 5,
    same_tokenizer: bool = False,
) -> tuple[list[int], list[int], list[bool]]:
    """Return `(src_indices, tgt_indices, is_content_flags)` for one prompt after prefix stripping.

    For same-tokenizer pairs, every position aligns 1-to-1.
    For cross-tokenizer pairs, aligns every shared UTF-8 byte-end boundary in the prompt body (where
    both causal transformers have read the exact same prefix text) plus 1-to-1 on the trailing
    template suffix tokens.
    """
    if same_tokenizer and len(src_ids) == len(tgt_ids):
        n = len(src_ids)
        body_end = max(0, n - suffix_tokens)
        idx = list(range(n))
        return idx, idx, [i < body_end for i in range(n)]

    suf = min(suffix_tokens, len(src_ids), len(tgt_ids))
    src_body = src_ids[:-suf] if suf > 0 else src_ids
    tgt_body = tgt_ids[:-suf] if suf > 0 else tgt_ids

    src_ends = token_byte_ends(src_hf_tok, src_body)
    tgt_ends = token_byte_ends(tgt_hf_tok, tgt_body)

    tgt_by_end: dict[int, int] = {}
    for j, b_end in enumerate(tgt_ends):
        if b_end > 0 and b_end not in tgt_by_end:
            tgt_by_end[b_end] = j

    src_idx: list[int] = []
    tgt_idx: list[int] = []
    is_content: list[bool] = []
    seen_end: set[int] = set()
    for i, b_end in enumerate(src_ends):
        if b_end > 0 and b_end not in seen_end and b_end in tgt_by_end:
            seen_end.add(b_end)
            src_idx.append(i)
            tgt_idx.append(tgt_by_end[b_end])
            is_content.append(True)

    # Align trailing assistant-turn suffix tokens 1-to-1.
    for k in range(suf):
        src_idx.append(len(src_body) + k)
        tgt_idx.append(len(tgt_body) + k)
        is_content.append(False)

    return src_idx, tgt_idx, is_content


def _get_hf_tokenizer(clip: Any) -> Any:
    tok = clip.tokenizer
    inner = getattr(tok, getattr(tok, "clip", ""), None)
    return getattr(inner, "tokenizer", None)


def _load_comfy_clip(ckpt_path: str, clip_type_name: str):
    import comfy.sd

    clip_type = getattr(comfy.sd.CLIPType, clip_type_name.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
    return comfy.sd.load_clip(ckpt_paths=[ckpt_path], clip_type=clip_type)


def _tokenize_prompt_ids(clip: Any, formatted: str, strip_prefix_mode: str, max_length: int) -> tuple[list[int], int]:
    """Tokenize `formatted` with ComfyUI's tokenizer and return `(full_ids, prefix_cut)`."""
    tok_dict = clip.tokenize(formatted)
    pairs = None
    for v in tok_dict.values():
        if isinstance(v, list) and v and isinstance(v[0], list):
            pairs = v[0]
            break
    assert pairs is not None
    ids = [int(t[0]) for t in pairs[:max_length]]
    cut = find_krea2_prefix_end(pairs[:max_length], len(ids)) if strip_prefix_mode == "krea2_user" else 0
    return ids, cut


@torch.no_grad()
def _encode_comfy_batches(
    clip: Any,
    full_id_lists: list[list[int]],
    cuts: list[int],
    selected_post_cut_indices: list[list[int]],
    *,
    layer_spec: str | list[int],
    layer_norm: bool,
    batch_size: int,
    cache_dir: Path,
    tag: str,
) -> tuple[int, int]:
    """Run ComfyUI CLIP forward in length-sorted batches and save selected token states per prompt.

    Returns `(num_tap_layers, hidden_dim)`.
    """
    import comfy.model_management

    clip.load_model()
    device = clip.patcher.load_device
    inner_models = _inner_clip_models(clip.cond_stage_model)
    assert len(inner_models) == 1, f"Expected single inner SDClipModel, found {len(inner_models)}"
    inner = inner_models[0]

    inner.reset_clip_options()
    inner.execution_device = device
    inner.layer_norm_hidden_state = layer_norm
    if layer_spec == "last":
        inner.layer = "last"
        inner.layer_idx = None
    else:
        inner.layer = list(layer_spec)
        inner.layer_idx = None

    pad_id = int(inner.special_tokens.get("pad", 0))
    n_items = len(full_id_lists)
    order = sorted(range(n_items), key=lambda idx: len(full_id_lists[idx]))

    n_taps = 1 if layer_spec == "last" else len(layer_spec)
    hidden_dim: int | None = None

    with comfy.model_management.cuda_device_context(device):
        for b_start in range(0, n_items, batch_size):
            batch_orig_idx = order[b_start : b_start + batch_size]
            seqs = [full_id_lists[i] for i in batch_orig_idx]
            max_l = max(len(s) for s in seqs)
            padded = [s + [pad_id] * (max_l - len(s)) for s in seqs]

            out = inner.forward(padded)
            z = out[0]  # [B, S, D] (single-layer) or [B, N_taps, S, D] (multi-layer)
            if z.ndim == 3:
                z = z.unsqueeze(1)  # normalize to [B, 1, S, D]
            if hidden_dim is None:
                hidden_dim = int(z.shape[-1])
            z_cpu = z.detach().to(device="cpu", dtype=torch.float16)

            chunk_dict: dict[int, torch.Tensor] = {}
            for row_i, orig_i in enumerate(batch_orig_idx):
                cut = cuts[orig_i]
                sel = selected_post_cut_indices[orig_i]
                if not sel:
                    continue
                abs_idx = [cut + s for s in sel]
                # [N_taps, len(sel), D]
                chunk_dict[orig_i] = z_cpu[row_i, :, abs_idx, :].contiguous()

            torch.save(chunk_dict, cache_dir / f"{tag}_{b_start:06d}.pt")
            done = min(b_start + batch_size, n_items)
            if done == n_items or (b_start // batch_size + 1) % 25 == 0:
                print(f"  [{tag}] {done}/{n_items} prompts", flush=True)

    assert hidden_dim is not None
    return n_taps, hidden_dim


def _unload_comfy_clip(clip: Any) -> None:
    import comfy.model_management

    del clip
    gc.collect()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fit_multilayer_and_export(
    stats_per_layer: list[SufficientStats],
    val_x_per_layer: list[torch.Tensor],
    val_y_per_layer: list[torch.Tensor],
    val_content_mask: torch.Tensor,
    *,
    base_meta: PlatoKeyMeta,
    out_dir: Path,
    file_prefix: str,
    save_dtype: torch.dtype,
    methods: tuple[str, ...] = ("procrustes", "procrustes_norm", "ridge", "ridge_norm"),
) -> dict[str, dict[str, float]]:
    n_layers = len(stats_per_layer)
    d_x = stats_per_layer[0].d_x
    d_y = stats_per_layer[0].d_y

    print(f"\n=== {base_meta.source_model} -> {base_meta.target_model} ({n_layers}x{d_x} -> {n_layers}x{d_y}) ===")
    header = (
        f"{'Method':<18} {'TokCos':>8} {'CtrCos':>8} {'CntCos':>8} "
        f"{'RelL2':>8} {'R2':>8} {'NormR':>8} {'ChStdR':>8} {'MeanS':>8}"
    )
    print(header)
    print("-" * len(header))

    summary: dict[str, dict[str, float]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        layer_keys: list[PlatoKey] = []
        layer_metrics = []
        layer_cmetrics = []
        for l_idx in range(n_layers):
            st = stats_per_layer[l_idx]
            vx = val_x_per_layer[l_idx]
            vy = val_y_per_layer[l_idx]
            if method == "procrustes":
                lk = fit_procrustes(st, scale_mode="ls")
            elif method == "procrustes_norm":
                lk = fit_procrustes(st, scale_mode="variance")
            elif method == "ridge":
                lk = fit_ridge(st, val_x=vx, val_y=vy, match_variance=False)
            elif method == "ridge_norm":
                lk = fit_ridge(st, val_x=vx, val_y=vy, match_variance=True)
            else:
                raise ValueError(f"Unknown method: {method}")
            layer_keys.append(lk)
            layer_metrics.append(evaluate_key(lk, vx, vy))
            if val_content_mask.any():
                layer_cmetrics.append(evaluate_key(lk, vx[val_content_mask], vy[val_content_mask]))

        def _avg(vals: list[float]) -> float:
            return sum(vals) / max(len(vals), 1)

        agg = {
            "token_cosine": _avg([m.token_cosine for m in layer_metrics]),
            "centered_cosine": _avg([m.centered_cosine for m in layer_metrics]),
            "rel_l2": _avg([m.rel_l2 for m in layer_metrics]),
            "r2": _avg([m.r2 for m in layer_metrics]),
            "norm_ratio": _avg([m.norm_ratio for m in layer_metrics]),
            "channel_std_corr": _avg([m.channel_std_corr for m in layer_metrics]),
        }
        if layer_cmetrics:
            agg["content_centered_cosine"] = _avg([m.centered_cosine for m in layer_cmetrics])
            agg["content_r2"] = _avg([m.r2 for m in layer_cmetrics])

        scales = [lk.scale for lk in layer_keys]
        mean_scale = _avg(scales)
        src_rms = math.sqrt(_avg([st.source_token_rms ** 2 for st in stats_per_layer]))
        tgt_rms = math.sqrt(_avg([st.target_token_rms ** 2 for st in stats_per_layer]))

        print(
            f"{method:<18} {agg['token_cosine']:>8.4f} {agg['centered_cosine']:>8.4f} "
            f"{agg.get('content_centered_cosine', float('nan')):>8.4f} "
            f"{agg['rel_l2']:>8.4f} {agg['r2']:>8.4f} {agg['norm_ratio']:>8.4f} "
            f"{agg['channel_std_corr']:>8.4f} {mean_scale:>8.4f}"
        )

        # Print brief min/max per-layer centered cosine to spot any weak layers
        l_ctrs = [m.centered_cosine for m in layer_metrics]
        if n_layers > 1:
            print(
                f"  per-layer CtrCos: min={min(l_ctrs):.4f} (L{l_ctrs.index(min(l_ctrs))}), "
                f"max={max(l_ctrs):.4f} (L{l_ctrs.index(max(l_ctrs))})"
            )

        meta = PlatoKeyMeta(**vars(base_meta))
        meta.method = method
        meta.num_layers = n_layers
        meta.d_source = d_x
        meta.d_target = d_y
        meta.scale = mean_scale
        meta.num_prompts = stats_per_layer[0].prompt_count
        meta.num_tokens = stats_per_layer[0].token_count
        meta.metrics = dict(agg)
        alphas = [lk.meta.ridge_alpha for lk in layer_keys if lk.meta.ridge_alpha is not None]
        if alphas:
            meta.ridge_alpha = _avg(alphas)

        if n_layers == 1:
            key = PlatoKey(
                weight=layer_keys[0].weight,
                mu_x=layer_keys[0].mu_x,
                mu_y=layer_keys[0].mu_y,
                meta=meta,
                scale=mean_scale,
                source_token_rms=src_rms,
                target_token_rms=tgt_rms,
            )
        else:
            key = PlatoKey(
                weight=torch.stack([lk.weight for lk in layer_keys], dim=0),
                mu_x=torch.stack([lk.mu_x for lk in layer_keys], dim=0),
                mu_y=torch.stack([lk.mu_y for lk in layer_keys], dim=0),
                meta=meta,
                scale=mean_scale,
                layer_scales=scales,
                source_token_rms=src_rms,
                target_token_rms=tgt_rms,
            )

        out_path = out_dir / f"{file_prefix}_{method}.safetensors"
        save_key(key, out_path, dtype=save_dtype)
        print(f"  saved {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MB)")
        summary[method] = agg

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate PlatoKeys directly on local ComfyUI checkpoints.")
    p.add_argument("--comfy", required=True, help="Path to ComfyUI directory (containing `comfy/`).")
    p.add_argument("--source", required=True, help="Path to source text encoder .safetensors.")
    p.add_argument("--target", required=True, help="Path to target text encoder .safetensors.")
    p.add_argument("--preset", choices=sorted(COMFY_PRESETS.keys()), default="krea2")
    p.add_argument("--num-prompts", type=int, default=2000)
    p.add_argument("--val-prompts", type=int, default=150)
    p.add_argument("--prompts-file", type=str, default=None)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-dir", type=str, default="keys")
    p.add_argument("--prefix", type=str, default=None)
    p.add_argument("--save-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _short_ckpt_name(path_str: str) -> str:
    stem = Path(path_str).stem.lower()
    for suf in ("_bf16", "_fp16", "_fp8_scaled", "_int8_convrot", "_fp8mixed"):
        stem = stem.replace(suf, "")
    return stem.replace("-", "_").replace(".", "")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    comfy_dir = str(Path(args.comfy).resolve())
    if comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)

    preset = COMFY_PRESETS[args.preset]
    save_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.save_dtype]
    t0 = time.time()

    # 1. Gather calibration prompts
    prompts, dataset_label = gather_calibration_prompts(
        num_prompts=args.num_prompts,
        prompts_file=args.prompts_file,
        include_domain_mix=True,
        seed=args.seed,
    )
    formatted = [
        preset.template.format(prompt=(p.strip() if preset.strip_prompt else p)) if preset.template else p
        for p in prompts
    ]
    print(f"[calibrate_comfy] collected {len(formatted)} prompts ({dataset_label}) in {time.time() - t0:.1f}s")

    # 2. Load target briefly or tokenize with both models sequentially (save VRAM by loading 1 at a time!)
    cache_dir = Path(tempfile.mkdtemp(prefix="plato_comfy_"))
    try:
        # --- PASS 1: Load source model, tokenize, inspect layer count ---
        t1 = time.time()
        print(f"\n[pass 1/2] Loading source CLIP: {args.source}", flush=True)
        src_clip = _load_comfy_clip(args.source, "qwen_image")
        src_inner = _inner_clip_models(src_clip.cond_stage_model)[0]
        src_num_layers = int(src_inner.num_layers)
        src_hf_tok = _get_hf_tokenizer(src_clip)

        src_full_ids: list[list[int]] = []
        src_cuts: list[int] = []
        src_post_ids: list[list[int]] = []
        for text in formatted:
            ids, cut = _tokenize_prompt_ids(src_clip, text, preset.strip_prefix_mode, args.max_length)
            src_full_ids.append(ids)
            src_cuts.append(cut)
            src_post_ids.append(ids[cut:])

        # Determine source tap layers (if target is a multi-layer list, match proportionally!)
        if isinstance(preset.target_layers, list):
            # Target Qwen3-VL-4B / 8B has 36 layers for Krea 2
            assumed_tgt_layers = 36 if preset.name == "krea2" else max(preset.target_layers) + 1
            src_layers: str | list[int] = proportional_source_layers(
                preset.target_layers, assumed_tgt_layers, src_num_layers
            )
        else:
            src_layers = preset.target_layers

        print(
            f"[pass 1/2] source num_layers={src_num_layers}, source_layers={src_layers}, "
            f"sample post-cut len={len(src_post_ids[0])}",
            flush=True,
        )

        # Save ALL post-cut tokens for Pass 1 (we'll intersect with target byte-spans in Pass 2!)
        all_post_indices = [list(range(len(p))) for p in src_post_ids]
        n_taps_src, d_source = _encode_comfy_batches(
            src_clip,
            src_full_ids,
            src_cuts,
            all_post_indices,
            layer_spec=src_layers,
            layer_norm=preset.target_layer_norm,
            batch_size=args.batch_size,
            cache_dir=cache_dir,
            tag="src",
        )
        print(f"[pass 1/2] Done in {time.time() - t1:.1f}s (taps={n_taps_src}, d_source={d_source})", flush=True)
        _unload_comfy_clip(src_clip)

        # --- PASS 2: Load target model, align token spans, encode & accumulate stats ---
        t2 = time.time()
        print(f"\n[pass 2/2] Loading target CLIP: {args.target}", flush=True)
        tgt_clip = _load_comfy_clip(args.target, preset.clip_type_target)
        tgt_inner = _inner_clip_models(tgt_clip.cond_stage_model)[0]
        tgt_num_layers = int(tgt_inner.num_layers)
        tgt_hf_tok = _get_hf_tokenizer(tgt_clip)

        tgt_full_ids: list[list[int]] = []
        tgt_cuts: list[int] = []
        tgt_post_ids: list[list[int]] = []
        for text in formatted:
            ids, cut = _tokenize_prompt_ids(tgt_clip, text, preset.strip_prefix_mode, args.max_length)
            tgt_full_ids.append(ids)
            tgt_cuts.append(cut)
            tgt_post_ids.append(ids[cut:])

        same_tok = all(src_post_ids[i] == tgt_post_ids[i] for i in range(min(8, len(formatted))))
        aligned_src_idx: list[list[int]] = []
        aligned_tgt_idx: list[list[int]] = []
        aligned_content: list[list[bool]] = []
        total_src_tok = 0
        total_matched = 0
        for i in range(len(formatted)):
            si, ti, cm = align_token_sequences(
                src_post_ids[i],
                tgt_post_ids[i],
                src_hf_tok,
                tgt_hf_tok,
                suffix_tokens=preset.suffix_tokens,
                same_tokenizer=same_tok,
            )
            aligned_src_idx.append(si)
            aligned_tgt_idx.append(ti)
            aligned_content.append(cm)
            total_src_tok += len(src_post_ids[i])
            total_matched += len(si)

        match_pct = 100.0 * total_matched / max(total_src_tok, 1)
        print(
            f"[pass 2/2] target num_layers={tgt_num_layers}, same_tokenizer={same_tok}, "
            f"matched {total_matched:,}/{total_src_tok:,} source tokens ({match_pct:.1f}%)",
            flush=True,
        )

        # Load Pass 1 cached chunks into memory lookup (only ~2000 prompts * ~50 tok * 12 * 2048 * 2B = ~4 GB on RAM,
        # or stream chunk-by-chunk — actually let's index which file holds each prompt so peak RAM is tiny!)
        src_prompt_to_file: dict[int, Path] = {}
        src_chunks: dict[Path, dict[int, torch.Tensor]] = {}
        for fpath in sorted(cache_dir.glob("src_*.pt")):
            ch = torch.load(fpath, map_location="cpu", weights_only=True)
            for pid in ch:
                src_prompt_to_file[pid] = fpath
            del ch

        # Encode target directly in batches and accumulate SufficientStats on the fly!
        n_val = min(args.val_prompts, max(1, len(formatted) // 10))
        val_set = set(range(n_val))

        n_taps_tgt, d_target = _encode_comfy_batches(
            tgt_clip,
            tgt_full_ids,
            tgt_cuts,
            aligned_tgt_idx,
            layer_spec=preset.target_layers,
            layer_norm=preset.target_layer_norm,
            batch_size=args.batch_size,
            cache_dir=cache_dir,
            tag="tgt",
        )
        assert n_taps_src == n_taps_tgt, f"Layer tap count mismatch: {n_taps_src} vs {n_taps_tgt}"
        print(f"[pass 2/2] Target encoding done in {time.time() - t2:.1f}s (taps={n_taps_tgt}, d_target={d_target})", flush=True)
        _unload_comfy_clip(tgt_clip)

        # Stream-join src & tgt chunks into per-layer SufficientStats + val split
        stats_list = [SufficientStats(d_source, d_target) for _ in range(n_taps_src)]
        val_x_lists: list[list[torch.Tensor]] = [[] for _ in range(n_taps_src)]
        val_y_lists: list[list[torch.Tensor]] = [[] for _ in range(n_taps_src)]
        val_cm_list: list[torch.Tensor] = []

        # Cache a few src chunk files at a time (LRU-like simple dict capped at 16 files)
        for tgt_file in sorted(cache_dir.glob("tgt_*.pt")):
            tgt_chunk = torch.load(tgt_file, map_location="cpu", weights_only=True)
            batch_x = [[] for _ in range(n_taps_src)]
            batch_y = [[] for _ in range(n_taps_src)]
            train_prompts_in_batch = 0

            for pid, y_tensor in tgt_chunk.items():
                src_file = src_prompt_to_file.get(pid)
                if src_file is None:
                    continue
                if src_file not in src_chunks:
                    if len(src_chunks) >= 24:
                        src_chunks.clear()
                    src_chunks[src_file] = torch.load(src_file, map_location="cpu", weights_only=True)
                x_full = src_chunks[src_file][pid]  # [N_taps, len_src_post, d_source]
                x_sel = x_full[:, aligned_src_idx[pid], :]  # [N_taps, M, d_source]
                assert x_sel.shape[1] == y_tensor.shape[1]

                if pid in val_set:
                    for l_i in range(n_taps_src):
                        val_x_lists[l_i].append(x_sel[l_i])
                        val_y_lists[l_i].append(y_tensor[l_i])
                    val_cm_list.append(torch.tensor(aligned_content[pid], dtype=torch.bool))
                else:
                    for l_i in range(n_taps_src):
                        batch_x[l_i].append(x_sel[l_i])
                        batch_y[l_i].append(y_tensor[l_i])
                    train_prompts_in_batch += 1

            if train_prompts_in_batch > 0:
                for l_i in range(n_taps_src):
                    stats_list[l_i].update(
                        torch.cat(batch_x[l_i], dim=0),
                        torch.cat(batch_y[l_i], dim=0),
                        num_prompts=train_prompts_in_batch,
                    )

        val_x_per_layer = [torch.cat(v, dim=0) for v in val_x_lists]
        val_y_per_layer = [torch.cat(v, dim=0) for v in val_y_lists]
        val_cm = torch.cat(val_cm_list, dim=0)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    # 3. Fit & export all 4 closed-form methods
    src_label = Path(args.source).stem
    tgt_label = Path(args.target).stem
    src_layer_str = json.dumps(src_layers) if isinstance(src_layers, list) else str(src_layers)
    tgt_layer_str = (
        json.dumps(preset.target_layers) if isinstance(preset.target_layers, list) else str(preset.target_layers)
    )

    base_meta = PlatoKeyMeta(
        source_model=src_label,
        target_model=tgt_label,
        target_preset=preset.name,
        source_layer=src_layer_str,
        target_layer=tgt_layer_str,
        source_layer_norm=preset.target_layer_norm,
        target_layer_norm=preset.target_layer_norm,
        source_template=preset.template,
        strip_prompt=preset.strip_prompt,
        strip_prefix_mode=preset.strip_prefix_mode,
        flatten_layers=preset.flatten_layers,
        num_layers=n_taps_src,
        dataset=dataset_label,
        notes=f"comfy_native; same_tokenizer={same_tok}; aligned={match_pct:.1f}%",
    )

    out_dir = Path(args.out_dir)
    prefix = args.prefix or f"{_short_ckpt_name(args.source)}_to_{_short_ckpt_name(args.target)}_{preset.name}"
    t3 = time.time()
    summary = _fit_multilayer_and_export(
        stats_list,
        val_x_per_layer,
        val_y_per_layer,
        val_cm,
        base_meta=base_meta,
        out_dir=out_dir,
        file_prefix=prefix,
        save_dtype=save_dtype,
    )
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[calibrate_comfy] Completed in {time.time() - t0:.1f}s (fit={time.time() - t3:.1f}s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
