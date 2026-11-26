"""Pure conditioning & tensor transformation for PlatoKeys (no ComfyUI imports).

Supports both:
  - Single-layer 2D keys (`weight: [d_target, d_source]`):
        y_hat = mu_y + contrast * ((x - mu_x) @ weight.T)
  - Multi-layer 3D stacked keys (`weight: [N, d_target, d_source]`, e.g. Krea 2's 12-layer tap):
        accepts either 4D `[B, N, seq, d_source]` (direct from ComfyUI's multi-layer CLIP tap)
        or 3D `[B, seq, N * d_source]` (pre-flattened), projects each layer independently via one
        batched einsum, and optionally flattens to `[B, seq, N * d_target]`.
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
    flatten_layers: bool = False,
) -> torch.Tensor:
    """Project `tensor` from source space into target space (single-layer 2D or multi-layer 3D)."""
    orig_dtype = tensor.dtype
    device = tensor.device

    if weight.ndim == 2:
        _, d_source = weight.shape
        if tensor.shape[-1] != d_source:
            raise ValueError(
                f"PlatoKey expects source embedding dimension {d_source}, but input conditioning has "
                f"dimension {tensor.shape[-1]} (shape {tuple(tensor.shape)}). "
                "Check that the wired text encoder matches the key's source model."
            )
        x32 = tensor.to(dtype=torch.float32)
        w32 = weight.to(device=device, dtype=torch.float32)
        mx32 = mu_x.to(device=device, dtype=torch.float32)
        my32 = mu_y.to(device=device, dtype=torch.float32)

        projected = F.linear(x32 - mx32, w32)
        if contrast != 1.0:
            projected = projected * float(contrast)
        out = projected + my32
        src_for_norm = x32
    elif weight.ndim == 3:
        n_layers, d_target, d_source = weight.shape
        if tensor.ndim == 4 and tensor.shape[1] == n_layers and tensor.shape[-1] == d_source:
            x32 = tensor.to(dtype=torch.float32)  # [B, N, S, d_source]
        elif tensor.ndim == 3 and tensor.shape[-1] == n_layers * d_source:
            b, s, _ = tensor.shape
            x32 = tensor.to(dtype=torch.float32).reshape(b, s, n_layers, d_source).permute(0, 2, 1, 3)
        else:
            hint = ""
            if tensor.ndim == 3 and tensor.shape[-1] == d_source:
                hint = (
                    " This conditioning only has 1 layer instead of "
                    f"{n_layers}. 'Apply Plato Key' cannot extract multiple hidden layers from an "
                    "already-encoded CONDITIONING wire — replace 'Apply Plato Key' with "
                    "'Patch CLIP (Plato Key)' BEFORE Visual Fusion / CLIPTextEncode "
                    "(or use 'Plato Key Encode' for plain text)."
                )
            raise ValueError(
                f"Multi-layer PlatoKey expects [B, {n_layers}, seq, {d_source}] or "
                f"[B, seq, {n_layers * d_source}], got shape {tuple(tensor.shape)}.{hint}"
            )

        w32 = weight.to(device=device, dtype=torch.float32)  # [N, d_target, d_source]
        mx32 = mu_x.to(device=device, dtype=torch.float32).view(1, n_layers, 1, d_source)
        my32 = mu_y.to(device=device, dtype=torch.float32).view(1, n_layers, 1, d_target)

        projected = torch.einsum("bnsd,ntd->bnst", x32 - mx32, w32)
        if contrast != 1.0:
            projected = projected * float(contrast)
        out = projected + my32  # [B, N, S, d_target]
        src_for_norm = x32
    else:
        raise ValueError(f"Unsupported PlatoKey weight ndim={weight.ndim}")

    if norm_match == NORM_MATCH_TARGET_RMS:
        if target_token_rms is not None and target_token_rms > 0.0:
            cur_rms = out.pow(2).mean().sqrt().clamp_min(_EPS)
            out = out * (float(target_token_rms) / cur_rms)
    elif norm_match == NORM_MATCH_SOURCE_NORM:
        src_norm = src_for_norm.norm(dim=-1, keepdim=True)
        out_norm = out.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        out = out * (src_norm / out_norm)
    elif norm_match != NORM_MATCH_OFF:
        raise ValueError(f"Unknown norm_match mode {norm_match!r}; expected one of {NORM_MATCH_MODES}")

    if scale_multiplier != 1.0:
        out = out * float(scale_multiplier)

    if weight.ndim == 3 and flatten_layers:
        b, n, s, dt = out.shape
        out = out.permute(0, 2, 1, 3).reshape(b, s, n * dt).contiguous()

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
    """Apply a PlatoKey to every entry of a ComfyUI `CONDITIONING` list."""
    out = []
    for cond_tensor, meta in conditioning:
        x32 = cond_tensor.detach().to(device="cpu", dtype=torch.float32)
        in_rms = float(x32.pow(2).mean().sqrt().item())
        if not key.is_multilayer and x32.shape[-1] == key.d_source:
            mx32 = key.mu_x.detach().to(device="cpu", dtype=torch.float32)
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
                    "If you wired CLIPTextEncode -> Apply Plato Key, switch to Plato Key Encode "
                    "so the calibrated layer(s) and chat template are applied.",
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
            flatten_layers=key.meta.flatten_layers,
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
        per_tok = [
            f"t{i}:cos={float(cos[i]):.3f}(normA={float(na[i]):.1f}/B={float(nb[i]):.1f})"
            for i in range(min(4, va.shape[0]))
        ]
        lines.append("Head: " + ", ".join(per_tok))
        if va.shape[0] > 4:
            tail = [
                f"t{i}:cos={float(cos[i]):.3f}(normA={float(na[i]):.1f}/B={float(nb[i]):.1f})"
                for i in range(max(4, va.shape[0] - 4), va.shape[0])
            ]
            lines.append("Tail: " + ", ".join(tail))
    else:
        lines.append("Shapes differ (expected for cross-tokenizer pairs) — check RMS ratio and feature dims.")

    return "\n".join(lines)
