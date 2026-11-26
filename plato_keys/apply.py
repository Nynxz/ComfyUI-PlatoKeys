"""Pure conditioning & tensor transformation for PlatoKeys (no ComfyUI imports).

At inference time, every closed-form PlatoKey (Procrustes, variance-matched Procrustes, Ridge,
PCA-Procrustes) applies the exact same fast affine map on the last dimension:

    y_hat = mu_y + contrast * ((x - mu_x) @ weight.T)

where `weight` has shape `[d_target, d_source]` and already carries the calibrated global scale.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .format import PlatoKey

log = logging.getLogger("plato_keys")
_EPS = 1e-8

NORM_MATCH_OFF = "off"
NORM_MATCH_TARGET_RMS = "target_rms"
NORM_MATCH_SOURCE_NORM = "preserve_source_norm"
NORM_MATCH_MODES = (NORM_MATCH_OFF, NORM_MATCH_TARGET_RMS, NORM_MATCH_SOURCE_NORM)


@torch.no_grad()
def apply_tensor(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    mu_x: torch.Tensor,
    mu_y: torch.Tensor,
    *,
    contrast: float = 1.0,
    scale_multiplier: float = 1.0,
    norm_match: str = NORM_MATCH_OFF,
    target_token_rms: float | None = None,
) -> torch.Tensor:
    """Project `tensor` (`[..., d_source]`) into target space (`[..., d_target]`).

    Always runs the projection in float32 for numerical accuracy (LLM outlier channels can exceed
    fp16 dynamic range when subtracting `mu_x`), then casts back to `tensor.dtype`.

    Controls:
      - `contrast`: scales the deviation from the target mean `mu_y` (`1.0` = exact calibrated map;
        `>1.0` sharpens deviation from average prompt; `<1.0` pulls toward average target prompt).
      - `scale_multiplier`: global output multiplier (`1.0` = exact calibrated magnitude).
      - `norm_match`:
          * `"off"`: use exact calibrated output (recommended default).
          * `"target_rms"`: rescale the whole tensor's RMS to the calibrated `target_token_rms`
            (useful if a least-squares/Ridge map slightly under-scales energy).
          * `"preserve_source_norm"`: rescale each token vector to match its input L2 norm.
    """
    d_target, d_source = weight.shape
    if tensor.shape[-1] != d_source:
        raise ValueError(
            f"PlatoKey expects source embedding dimension {d_source}, but input conditioning has "
            f"dimension {tensor.shape[-1]} (shape {tuple(tensor.shape)}). "
            "Check that the wired text encoder matches the key's source model."
        )

    orig_dtype = tensor.dtype
    device = tensor.device

    x32 = tensor.to(dtype=torch.float32)
    w32 = weight.to(device=device, dtype=torch.float32)
    mx32 = mu_x.to(device=device, dtype=torch.float32)
    my32 = mu_y.to(device=device, dtype=torch.float32)

    projected = F.linear(x32 - mx32, w32)
    if contrast != 1.0:
        projected = projected * float(contrast)
    out = projected + my32

    if norm_match == NORM_MATCH_TARGET_RMS:
        if target_token_rms is not None and target_token_rms > 0.0:
            cur_rms = out.pow(2).mean().sqrt().clamp_min(_EPS)
            out = out * (float(target_token_rms) / cur_rms)
    elif norm_match == NORM_MATCH_SOURCE_NORM:
        src_norm = x32.norm(dim=-1, keepdim=True)
        out_norm = out.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        out = out * (src_norm / out_norm)
    elif norm_match != NORM_MATCH_OFF:
        raise ValueError(f"Unknown norm_match mode {norm_match!r}; expected one of {NORM_MATCH_MODES}")

    if scale_multiplier != 1.0:
        out = out * float(scale_multiplier)

    return out.to(dtype=orig_dtype)


def apply_conditioning(
    conditioning: list,
    key: PlatoKey,
    *,
    contrast: float = 1.0,
    scale_multiplier: float = 1.0,
    norm_match: str = NORM_MATCH_OFF,
    transform_pooled: bool = True,
) -> list:
    """Apply a PlatoKey to every entry of a ComfyUI `CONDITIONING` list.

    Each entry is `[cond_tensor, meta_dict]`. Returns a new list with copied metadata dicts so
    upstream conditioning is never mutated in place.
    """
    out = []
    for cond_tensor, meta in conditioning:
        x32 = cond_tensor.detach().to(device="cpu", dtype=torch.float32)
        in_rms = float(x32.pow(2).mean().sqrt().item())
        mx32 = key.mu_x.detach().to(device="cpu", dtype=torch.float32)
        if x32.shape[-1] == mx32.shape[0]:
            x_mean = x32.reshape(-1, x32.shape[-1]).mean(dim=0)
            cos_mux = float(
                ((x_mean * mx32).sum() / (x_mean.norm().clamp_min(_EPS) * mx32.norm().clamp_min(_EPS))).item()
            )
            log.debug(
                "[PlatoKeys] Apply (%s, %d->%d): in_shape=%s, in_rms=%.4f (calib=%.4f), "
                "cos(x_mean,mu_x)=%.4f, contrast=%s, scale_mult=%s, norm_match=%s",
                key.meta.method,
                key.d_source,
                key.d_target,
                tuple(cond_tensor.shape),
                in_rms,
                key.source_token_rms or 0.0,
                cos_mux,
                contrast,
                scale_multiplier,
                norm_match,
            )
        if key.source_token_rms and key.source_token_rms > 0.0:
            ratio = in_rms / key.source_token_rms
            if ratio > 2.5 or ratio < 0.4:
                log.warning(
                    "[PlatoKeys] Input conditioning RMS %.4f differs strongly from calibrated "
                    "source_token_rms %.4f (%.2fx) for key %s (expected layer=%s, norm=%s). "
                    "If you wired CLIPTextEncode -> Apply Plato Key on Qwen3.5-2B, switch to "
                    "Plato Key Encode so the calibrated layer and chat template are applied.",
                    in_rms,
                    key.source_token_rms,
                    ratio,
                    key.meta.method,
                    key.meta.source_layer,
                    key.meta.source_layer_norm,
                )
        mapped = apply_tensor(
            cond_tensor,
            key.weight,
            key.mu_x,
            key.mu_y,
            contrast=contrast,
            scale_multiplier=scale_multiplier,
            norm_match=norm_match,
            target_token_rms=key.target_token_rms,
        )
        new_meta = dict(meta) if meta else {}
        if transform_pooled and key.pooled_weight is not None:
            pooled = new_meta.get("pooled_output")
            if pooled is not None and pooled.shape[-1] == key.pooled_weight.shape[1]:
                new_meta["pooled_output"] = apply_tensor(
                    pooled,
                    key.pooled_weight,
                    key.pooled_mu_x,
                    key.pooled_mu_y,
                    contrast=contrast,
                    scale_multiplier=scale_multiplier,
                    norm_match=NORM_MATCH_OFF,
                )
        out.append([mapped, new_meta])
    return out


@torch.no_grad()
def compare_conditioning(cond_a: list, cond_b: list) -> str:
    """Diagnostic comparison between two ComfyUI CONDITIONING wires (e.g. mapped 2B vs native 9B)."""
    if not cond_a or not cond_b:
        return "Empty conditioning input."

    ta, ma = cond_a[0]
    tb, mb = cond_b[0]
    a = ta.detach().to(device="cpu", dtype=torch.float32)
    b = tb.detach().to(device="cpu", dtype=torch.float32)

    rms_a = float(a.pow(2).mean().sqrt().item())
    rms_b = float(b.pow(2).mean().sqrt().item())
    lines = [
        f"A: shape={tuple(ta.shape)}, dtype={ta.dtype}, finite={bool(torch.isfinite(a).all())}, "
        f"rms={rms_a:.4f}, max_abs={float(a.abs().max()):.4f}",
        f"B: shape={tuple(tb.shape)}, dtype={tb.dtype}, finite={bool(torch.isfinite(b).all())}, "
        f"rms={rms_b:.4f}, max_abs={float(b.abs().max()):.4f}",
        f"RMS ratio (A/B): {rms_a / max(rms_b, _EPS):.4f}",
    ]

    mask_a = ma.get("attention_mask") if isinstance(ma, dict) else None
    mask_b = mb.get("attention_mask") if isinstance(mb, dict) else None
    lines.append(
        f"Masks: A={tuple(mask_a.shape) if mask_a is not None else None}, "
        f"B={tuple(mask_b.shape) if mask_b is not None else None}"
    )

    if a.shape == b.shape and a.ndim == 3:
        va = a.reshape(-1, a.shape[-1])
        vb = b.reshape(-1, b.shape[-1])
        na = va.norm(dim=-1).clamp_min(_EPS)
        nb = vb.norm(dim=-1).clamp_min(_EPS)
        cos = (va * vb).sum(dim=-1) / (na * nb)
        rel_l2 = float(((va - vb).pow(2).sum() / vb.pow(2).sum().clamp_min(_EPS)).sqrt().item())
        lines.append(
            f"All tokens ({va.shape[0]}): cos_mean={float(cos.mean()):.4f}, "
            f"cos_min={float(cos.min()):.4f}, rel_l2={rel_l2:.4f}, "
            f"norm_ratio={float((na / nb).mean()):.4f}"
        )
        # First 4 and last 4 tokens (attention sink + chat template boundaries)
        per_tok = [f"t{i}:cos={float(cos[i]):.3f}(normA={float(na[i]):.1f}/B={float(nb[i]):.1f})" for i in range(min(4, va.shape[0]))]
        lines.append("Head: " + ", ".join(per_tok))
        if va.shape[0] > 4:
            tail = [
                f"t{i}:cos={float(cos[i]):.3f}(normA={float(na[i]):.1f}/B={float(nb[i]):.1f})"
                for i in range(max(4, va.shape[0] - 4), va.shape[0])
            ]
            lines.append("Tail: " + ", ".join(tail))
    else:
        lines.append("Shapes differ — cannot compute token-by-token cosine (check template/tokenizer/layer).")

    return "\n".join(lines)
