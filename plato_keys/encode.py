"""CLIP encoding bridge that honors a PlatoKey's calibrated layer and chat template.

Why this exists:
  In ComfyUI, loading `qwen3.5_2b_bf16.safetensors` with stock `CLIPLoader` constructs a generic
  `Qwen35TEModel` which defaults to `layer="hidden", layer_idx=-2, layer_norm_hidden_state=False`
  and wraps prompts in `<think>\\n</think>\\n`. MOSS-Image's Qwen3.5-9B uses `last_hidden_state`
  (`layer="last"`, with final RMSNorm) and the open-think template
  `<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n`.

  When `PlatoKeyEncode` runs, it reads the key's `.safetensors` metadata, clones `clip`,
  temporarily overrides the underlying `SDClipModel`'s `options_default` / `layer` / `layer_idx` /
  `layer_norm_hidden_state` (restoring them in `finally`), formats the prompt with the calibrated
  chat template, encodes, and projects the conditioning into target space.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from .apply import NORM_MATCH_OFF, apply_conditioning
from .format import PlatoKey


def format_prompt_for_key(text: str, key: PlatoKey, *, honor_template: bool = True) -> str:
    """Apply the key's calibrated chat template if configured and not already templated."""
    if text.lstrip().startswith("<|im_start|>"):
        return text
    prompt = text.strip() if key.meta.strip_prompt else text
    template = key.meta.source_template if honor_template else ""
    if not template:
        return prompt
    return template.format(prompt=prompt)


def parse_source_layer(layer_spec: str) -> str | int | None:
    """Parse `meta.source_layer` (`"last"`, integer string like `"-2"`, or empty)."""
    s = (layer_spec or "").strip().lower()
    if not s or s == "default":
        return None
    if s == "last":
        return "last"
    try:
        return int(s)
    except ValueError:
        return None


def _inner_clip_models(cond_stage_model: Any) -> list[Any]:
    """Find underlying SDClipModel instances inside a ComfyUI SD1ClipModel / wrapper."""
    if cond_stage_model is None:
        return []
    if hasattr(cond_stage_model, "layer_norm_hidden_state") or hasattr(cond_stage_model, "options_default"):
        return [cond_stage_model]
    found = []
    clip_attr = getattr(cond_stage_model, "clip", None)
    if isinstance(clip_attr, str) and hasattr(cond_stage_model, clip_attr):
        sub = getattr(cond_stage_model, clip_attr)
        if hasattr(sub, "layer_norm_hidden_state") or hasattr(sub, "options_default"):
            found.append(sub)
    if not found and hasattr(cond_stage_model, "children"):
        for child in cond_stage_model.children():
            if hasattr(child, "layer_norm_hidden_state") or hasattr(child, "options_default"):
                found.append(child)
    return found


@contextmanager
def _override_clip_extraction(
    cond_stage_model: Any,
    layer_spec: str | int | None,
    layer_norm: bool | None,
):
    """Temporarily override `options_default`, `layer`, `layer_idx`, and `layer_norm_hidden_state`.

    Overriding `options_default` ensures ComfyUI's `encode_from_tokens` -> `reset_clip_options()`
    lands directly on `layer="last"` (or the requested intermediate layer) across all ComfyUI
    versions, without relying on sentinel `clip_layer(...)` heuristics. Everything is restored in
    `finally` so shared `cond_stage_model` instances are never left mutated.
    """
    if layer_spec is None and layer_norm is None:
        yield
        return

    targets = _inner_clip_models(cond_stage_model)
    saved = []
    for m in targets:
        saved.append(
            (
                m,
                getattr(m, "options_default", None),
                getattr(m, "layer", None),
                getattr(m, "layer_idx", None),
                getattr(m, "layer_norm_hidden_state", None),
            )
        )
    try:
        for m, orig_default, _, _, _ in saved:
            if layer_spec == "last":
                m.layer = "last"
                m.layer_idx = None
                if isinstance(orig_default, tuple) and len(orig_default) >= 2:
                    m.options_default = ("last", None, *orig_default[2:])
            elif isinstance(layer_spec, int):
                m.layer = "hidden"
                m.layer_idx = layer_spec
                if isinstance(orig_default, tuple) and len(orig_default) >= 2:
                    m.options_default = ("hidden", layer_spec, *orig_default[2:])
            if layer_norm is not None and hasattr(m, "layer_norm_hidden_state"):
                m.layer_norm_hidden_state = layer_norm
        yield
    finally:
        for m, orig_default, orig_layer, orig_idx, orig_ln in saved:
            if orig_default is not None:
                m.options_default = orig_default
            if orig_layer is not None:
                m.layer = orig_layer
            if hasattr(m, "layer_idx"):
                m.layer_idx = orig_idx
            if orig_ln is not None and hasattr(m, "layer_norm_hidden_state"):
                m.layer_norm_hidden_state = orig_ln


def encode_with_key(
    clip: Any,
    text: str,
    key: PlatoKey,
    *,
    contrast: float = 1.0,
    scale_multiplier: float = 1.0,
    norm_match: str = NORM_MATCH_OFF,
    honor_key_template: bool = True,
    honor_key_layer: bool = True,
    transform_pooled: bool = True,
) -> list:
    """Encode `text` through `clip` matching `key`'s calibrated settings, then project into target space."""
    if clip is None:
        raise ValueError("PlatoKeyEncode requires a valid CLIP text encoder input.")

    work_clip = clip.clone() if hasattr(clip, "clone") else clip
    layer_spec = parse_source_layer(key.meta.source_layer) if honor_key_layer else None
    override_ln: bool | None = None

    if layer_spec == "last":
        # Clear any upstream clip_layer override so reset_clip_options() uses our overridden
        # options_default = ("last", None, ...) cleanly.
        if hasattr(work_clip, "clip_layer"):
            work_clip.clip_layer(None)
    elif isinstance(layer_spec, int):
        if hasattr(work_clip, "clip_layer"):
            work_clip.clip_layer(layer_spec)
        override_ln = key.meta.source_layer_norm

    formatted = format_prompt_for_key(text, key, honor_template=honor_key_template)
    tokens = work_clip.tokenize(formatted)

    cond_stage = getattr(work_clip, "cond_stage_model", None)
    with _override_clip_extraction(cond_stage, layer_spec, override_ln):
        if hasattr(work_clip, "encode_from_tokens_scheduled"):
            conditioning = work_clip.encode_from_tokens_scheduled(tokens)
        else:
            cond, pooled = work_clip.encode_from_tokens(tokens, return_pooled=True)
            conditioning = [[cond, {"pooled_output": pooled}]]

    return apply_conditioning(
        conditioning,
        key,
        contrast=contrast,
        scale_multiplier=scale_multiplier,
        norm_match=norm_match,
        transform_pooled=transform_pooled,
    )
