"""CLIP encoding bridge that honors a PlatoKey's calibrated layer(s), chat template, and prefix stripping.

Supports:
  - Single-layer keys (`source_layer="last"` or integer like `"-2"`), e.g. MOSS-Image (`Qwen3.5`).
  - Multi-layer tap keys (`source_layer="[1,3,5,...,23]"`), e.g. Krea 2 (12-layer stack -> flattened
    `[B, seq, 30720]`) with `strip_prefix_mode="krea2_user"` to strip the system + user-opening
    prefix identically to `comfy/text_encoders/krea2.py` across both Qwen3-VL and Qwen3.5 tokenizers.
"""

from __future__ import annotations

import json
import logging
import numbers
import types
from contextlib import contextmanager
from typing import Any

import torch

from .apply import NORM_MATCH_OFF, apply_conditioning
from .format import PlatoKey

log = logging.getLogger("plato_keys")

# `<|im_start|>` token IDs across Qwen2.5/Qwen3/Qwen3-VL (151644) and Qwen3.5 (248045).
_IM_START_IDS = {151644, 248045}
# `"user"` token IDs across Qwen2.5/Qwen3 (872) and Qwen3.5 (846).
_USER_IDS = {872, 846}
_NEWLINE_ID = 198


def format_prompt_for_key(text: str, key: PlatoKey, *, honor_template: bool = True) -> str:
    """Apply the key's calibrated chat template if configured and not already templated."""
    if text.lstrip().startswith(("<|im_start|>", "<|turn>")):
        return text
    prompt = text.strip() if key.meta.strip_prompt else text
    template = key.meta.source_template if honor_template else ""
    if not template:
        return prompt
    return template.format(prompt=prompt)


def parse_source_layer(layer_spec: str) -> str | int | list[int] | None:
    """Parse `meta.source_layer` (`"last"`, int string `"-2"`, or JSON int list `"[1,3,5]"`)."""
    s = (layer_spec or "").strip()
    if not s or s.lower() == "default":
        return None
    if s.lower() == "last":
        return "last"
    if s.startswith("["):
        parsed = json.loads(s)
        if not isinstance(parsed, list) or not all(isinstance(v, int) for v in parsed):
            raise ValueError(f"Invalid multi-layer source_layer spec: {layer_spec!r}")
        return [int(v) for v in parsed]
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
    found: list[Any] = []
    seen: set[int] = set()

    def _add(mod: Any) -> None:
        if mod is not None and id(mod) not in seen:
            if hasattr(mod, "layer_norm_hidden_state") or hasattr(mod, "options_default"):
                seen.add(id(mod))
                found.append(mod)

    clip_attr = getattr(cond_stage_model, "clip", None)
    if isinstance(clip_attr, str) and hasattr(cond_stage_model, clip_attr):
        _add(getattr(cond_stage_model, clip_attr))
    if not found and hasattr(cond_stage_model, "modules"):
        for mod in cond_stage_model.modules():
            if mod is not cond_stage_model:
                _add(mod)
    if not found and hasattr(cond_stage_model, "children"):
        for child in cond_stage_model.children():
            _add(child)
    return found


@contextmanager
def _override_clip_extraction(
    cond_stage_model: Any,
    layer_spec: str | int | list[int] | None,
    layer_norm: bool | None,
):
    """Temporarily override `options_default`, `layer`, `layer_idx`, and `layer_norm_hidden_state`.

    Restores every modified attribute in `finally` so shared `cond_stage_model` instances are never
    left mutated.
    """
    if layer_spec is None and layer_norm is None:
        yield
        return

    targets = _inner_clip_models(cond_stage_model)
    if not targets:
        log.warning(
            "[PlatoKeys] Could not find an inner SDClipModel inside %s to apply layer_spec=%r; "
            "output may use the encoder's default layer.",
            type(cond_stage_model).__name__,
            layer_spec,
        )
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
            elif isinstance(layer_spec, list):
                m.layer = list(layer_spec)
                m.layer_idx = None
                if isinstance(orig_default, tuple) and len(orig_default) >= 2:
                    m.options_default = (list(layer_spec), None, *orig_default[2:])
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


def _extract_first_token_pairs(tokens: Any) -> list[tuple[Any, ...]] | None:
    if not isinstance(tokens, dict):
        return None
    for value in tokens.values():
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            return list(value[0])
    return None


def find_krea2_prefix_end(token_pairs: list[tuple[Any, ...]], seq_len: int) -> int:
    """Locate the token index right after the second `<|im_start|>user\\n` (matching `krea2.py`).

    Only strips when the second `<|im_start|>` actually begins a `user\\n` turn (i.e. a system
    prefix precedes it). Templates without a system turn (like MOSS Image, where the second
    `<|im_start|>` is `assistant\\n`) return `0` so the user/vision tokens are never stripped.
    """
    template_end = 0
    count_im_start = 0
    for i, pair in enumerate(token_pairs):
        elem = pair[0]
        if not torch.is_tensor(elem) and isinstance(elem, numbers.Integral):
            if int(elem) in _IM_START_IDS and count_im_start < 2:
                template_end = i
                count_im_start += 1
    if count_im_start == 2 and len(token_pairs) > (template_end + 2) and seq_len > (template_end + 3):
        t1 = token_pairs[template_end + 1][0]
        t2 = token_pairs[template_end + 2][0]
        if (
            isinstance(t1, numbers.Integral)
            and isinstance(t2, numbers.Integral)
            and int(t1) in _USER_IDS
            and int(t2) == _NEWLINE_ID
        ):
            return template_end + 3
    # Gemma 4 `<bos><|turn>user\n` (IDs [2, 105, 2364, 107]) prefix detection for Krea 2
    for i in range(min(4, len(token_pairs) - 2)):
        t0, t1, t2 = token_pairs[i][0], token_pairs[i + 1][0], token_pairs[i + 2][0]
        if (
            isinstance(t0, numbers.Integral)
            and isinstance(t1, numbers.Integral)
            and isinstance(t2, numbers.Integral)
            and int(t0) == 105
            and int(t1) == 2364
            and int(t2) == 107
            and seq_len > (i + 3)
        ):
            return i + 3
    return 0


def _strip_prefix_conditioning(conditioning: list, tokens: Any, mode: str) -> list:
    if not mode:
        return conditioning
    if mode != "krea2_user":
        raise ValueError(f"Unknown strip_prefix_mode: {mode!r}")

    tok_pairs = _extract_first_token_pairs(tokens)
    if not tok_pairs:
        return conditioning

    out = []
    for cond_tensor, meta in conditioning:
        seq_dim = 2 if cond_tensor.ndim == 4 else 1
        seq_len = int(cond_tensor.shape[seq_dim])
        cut = find_krea2_prefix_end(tok_pairs, seq_len)
        new_meta = dict(meta) if meta else {}
        if cut > 0:
            if cond_tensor.ndim == 4:
                cond_tensor = cond_tensor[:, :, cut:]
            else:
                cond_tensor = cond_tensor[:, cut:]
            mask = new_meta.get("attention_mask")
            if mask is not None and mask.shape[-1] > cut:
                mask = mask[:, cut:]
                if mask.sum() == torch.numel(mask):
                    new_meta.pop("attention_mask", None)
                else:
                    new_meta["attention_mask"] = mask
        out.append([cond_tensor, new_meta])
    return out


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

    if layer_spec == "last" or isinstance(layer_spec, list):
        if hasattr(work_clip, "clip_layer"):
            work_clip.clip_layer(None)
        if isinstance(layer_spec, list):
            override_ln = key.meta.source_layer_norm
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

    if honor_key_template and key.meta.strip_prefix_mode:
        conditioning = _strip_prefix_conditioning(conditioning, tokens, key.meta.strip_prefix_mode)

    return apply_conditioning(
        conditioning,
        key,
        contrast=contrast,
        scale_multiplier=scale_multiplier,
        norm_match=norm_match,
        transform_pooled=transform_pooled,
    )


def patch_clip_with_key(
    clip: Any,
    key: PlatoKey,
    *,
    contrast: float = 1.0,
    scale_multiplier: float = 1.0,
    norm_match: str = NORM_MATCH_OFF,
    honor_key_template: bool = True,
    honor_key_layer: bool = True,
    transform_pooled: bool = True,
) -> Any:
    """Return a cloned `CLIP` whose tokenize/encode methods transparently apply `key`.

    Preserves the underlying `CLIP` type, `tokenizer`, `cond_stage_model`, and `patcher` so any
    downstream ComfyUI node (`CLIPTextEncode`, Visual Fusion, Layout Apply, Regions,
    `TextEncodeQwenImageEditPlus`, etc.) works out of the box without modifying the original CLIP.
    """
    if clip is None:
        raise ValueError("Patch CLIP (Plato Key) requires a valid CLIP input.")

    patched = clip.clone() if hasattr(clip, "clone") else clip
    layer_spec = parse_source_layer(key.meta.source_layer) if honor_key_layer else None
    override_ln: bool | None = (
        key.meta.source_layer_norm if isinstance(layer_spec, (int, list)) else None
    )

    orig_tokenize = patched.tokenize
    orig_encode_scheduled = getattr(patched, "encode_from_tokens_scheduled", None)
    orig_encode = getattr(patched, "encode_from_tokens", None)
    orig_clone = getattr(patched, "clone", None)

    state = {"active": False}

    def _tokenize(self, text, *args, **kwargs):
        if honor_key_template and isinstance(text, str):
            text = format_prompt_for_key(text, key, honor_template=True)
        return orig_tokenize(text, *args, **kwargs)

    def _prepare_layer(self) -> int | None:
        prev = getattr(self, "layer_idx", None)
        if layer_spec == "last" or isinstance(layer_spec, list):
            if hasattr(self, "clip_layer"):
                self.clip_layer(None)
        elif isinstance(layer_spec, int):
            if hasattr(self, "clip_layer"):
                self.clip_layer(layer_spec)
        return prev

    def _encode_scheduled(self, tokens, *args, **kwargs):
        if state["active"]:
            return orig_encode_scheduled(tokens, *args, **kwargs)
        state["active"] = True
        prev_layer_idx = _prepare_layer(self)
        cond_stage = getattr(self, "cond_stage_model", None)
        try:
            with _override_clip_extraction(cond_stage, layer_spec, override_ln):
                conditioning = orig_encode_scheduled(tokens, *args, **kwargs)
        finally:
            if hasattr(self, "clip_layer"):
                self.clip_layer(prev_layer_idx)
            state["active"] = False

        if honor_key_template and key.meta.strip_prefix_mode:
            conditioning = _strip_prefix_conditioning(conditioning, tokens, key.meta.strip_prefix_mode)
        return apply_conditioning(
            conditioning,
            key,
            contrast=contrast,
            scale_multiplier=scale_multiplier,
            norm_match=norm_match,
            transform_pooled=transform_pooled,
        )

    def _encode_from_tokens(self, tokens, return_pooled=False, return_dict=False, **kwargs):
        if state["active"]:
            return orig_encode(tokens, return_pooled=return_pooled, return_dict=return_dict, **kwargs)
        state["active"] = True
        prev_layer_idx = _prepare_layer(self)
        cond_stage = getattr(self, "cond_stage_model", None)
        try:
            with _override_clip_extraction(cond_stage, layer_spec, override_ln):
                result = orig_encode(tokens, return_pooled=return_pooled, return_dict=return_dict, **kwargs)
        finally:
            if hasattr(self, "clip_layer"):
                self.clip_layer(prev_layer_idx)
            state["active"] = False

        if return_dict and isinstance(result, dict):
            cond = result.pop("cond")
            wrapped = [[cond, result]]
            if honor_key_template and key.meta.strip_prefix_mode:
                wrapped = _strip_prefix_conditioning(wrapped, tokens, key.meta.strip_prefix_mode)
            mapped = apply_conditioning(
                wrapped,
                key,
                contrast=contrast,
                scale_multiplier=scale_multiplier,
                norm_match=norm_match,
                transform_pooled=transform_pooled,
            )
            out_dict = dict(mapped[0][1])
            out_dict["cond"] = mapped[0][0]
            return out_dict

        if return_pooled and isinstance(result, tuple):
            cond, pooled = result[0], result[1]
            wrapped = [[cond, {"pooled_output": pooled}]]
            if honor_key_template and key.meta.strip_prefix_mode:
                wrapped = _strip_prefix_conditioning(wrapped, tokens, key.meta.strip_prefix_mode)
            mapped = apply_conditioning(
                wrapped,
                key,
                contrast=contrast,
                scale_multiplier=scale_multiplier,
                norm_match=norm_match,
                transform_pooled=transform_pooled,
            )
            return mapped[0][0], mapped[0][1].get("pooled_output")

        wrapped = [[result, {}]]
        if honor_key_template and key.meta.strip_prefix_mode:
            wrapped = _strip_prefix_conditioning(wrapped, tokens, key.meta.strip_prefix_mode)
        mapped = apply_conditioning(
            wrapped,
            key,
            contrast=contrast,
            scale_multiplier=scale_multiplier,
            norm_match=norm_match,
            transform_pooled=transform_pooled,
        )
        return mapped[0][0]

    def _clone(self, *args, **kwargs):
        cloned_base = orig_clone(*args, **kwargs)
        return patch_clip_with_key(
            cloned_base,
            key,
            contrast=contrast,
            scale_multiplier=scale_multiplier,
            norm_match=norm_match,
            honor_key_template=honor_key_template,
            honor_key_layer=honor_key_layer,
            transform_pooled=transform_pooled,
        )

    patched.tokenize = types.MethodType(_tokenize, patched)
    if orig_encode_scheduled is not None:
        patched.encode_from_tokens_scheduled = types.MethodType(_encode_scheduled, patched)
    if orig_encode is not None:
        patched.encode_from_tokens = types.MethodType(_encode_from_tokens, patched)
    if orig_clone is not None:
        patched.clone = types.MethodType(_clone, patched)
    return patched
