"""Unit tests for ComfyUI-PlatoKeys (math, format, apply, encode bridge, calibration helpers)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from plato_keys.apply import (
    NORM_MATCH_OFF,
    NORM_MATCH_SOURCE_NORM,
    NORM_MATCH_TARGET_RMS,
    apply_conditioning,
    apply_tensor,
)
from plato_keys.encode import (
    encode_with_key,
    find_krea2_prefix_end,
    format_prompt_for_key,
    parse_source_layer,
    patch_clip_with_key,
)
from plato_keys.format import (
    KREA2_TAP_LAYERS,
    KREA2_TEMPLATE,
    MOSS_IMAGE_TEMPLATE,
    PlatoKey,
    PlatoKeyMeta,
    load_key,
    resolve_key_path,
    save_key,
)
from plato_keys.math import (
    SufficientStats,
    evaluate_key,
    fit_all_methods,
    fit_procrustes,
    fit_procrustes_pca,
    fit_ridge,
)
from scripts.calibrate import (
    PRESETS,
    detect_common_prefix_suffix,
    extract_batch_tokens,
    fit_and_export,
    format_prompts,
)
from scripts.calibrate_comfy import align_token_sequences, proportional_source_layers
from scripts.prompt_corpus import build_domain_prompts, load_prompts_from_file


def _random_orthonormal(rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    """Return a semi-orthogonal matrix of shape [rows, cols] with orthonormal rows if rows <= cols."""
    k = max(rows, cols)
    a = torch.randn((k, k), generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    return q[:rows, :cols].contiguous()


def test_sufficient_stats_matches_batch_and_swap(tmp_path: Path) -> None:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn((500, 16), generator=gen, dtype=torch.float32) * 2.0 + 3.0
    y = torch.randn((500, 32), generator=gen, dtype=torch.float32) * 0.5 - 1.5
    w = torch.rand(500, generator=gen, dtype=torch.float32) + 0.1

    stats = SufficientStats(16, 32)
    for start in range(0, 500, 73):
        stats.update(x[start : start + 73], y[start : start + 73], weights=w[start : start + 73], num_prompts=1)

    mu_x, mu_y, m_xx, m_xy, m_yy = stats.centered()

    w64 = w.double()
    w_sum = w64.sum()
    x64, y64 = x.double(), y.double()
    ref_mux = (w64[:, None] * x64).sum(0) / w_sum
    ref_muy = (w64[:, None] * y64).sum(0) / w_sum
    xc, yc = x64 - ref_mux, y64 - ref_muy
    ref_mxx = (xc * w64[:, None]).T @ xc
    ref_mxy = (xc * w64[:, None]).T @ yc
    ref_myy = (yc * w64[:, None]).T @ yc

    assert torch.allclose(mu_x, ref_mux, atol=1e-6)
    assert torch.allclose(mu_y, ref_muy, atol=1e-6)
    assert torch.allclose(m_xx, ref_mxx, atol=1e-5)
    assert torch.allclose(m_xy, ref_mxy, atol=1e-5)
    assert torch.allclose(m_yy, ref_myy, atol=1e-5)

    # Swap produces exact transpose statistics
    rev = stats.swap()
    r_mux, r_muy, r_mxx, r_mxy, r_myy = rev.centered()
    assert torch.allclose(r_mux, mu_y)
    assert torch.allclose(r_muy, mu_x)
    assert torch.allclose(r_mxx, m_yy)
    assert torch.allclose(r_mxy, m_xy.T)
    assert torch.allclose(r_myy, m_xx)

    # Save / load roundtrip
    p = stats.save(tmp_path / "stats.pt")
    loaded = SufficientStats.load(p)
    assert loaded.token_count == 500
    assert torch.allclose(loaded.sum_xy, stats.sum_xy)


@pytest.mark.parametrize("d_x,d_y", [(32, 32), (32, 64), (64, 32)])
def test_procrustes_exact_recovery(d_x: int, d_y: int) -> None:
    gen = torch.Generator().manual_seed(123 + d_x + d_y)
    true_r = _random_orthonormal(d_x, d_y, gen)
    true_scale = 1.75
    true_mux = torch.randn(d_x, generator=gen, dtype=torch.float64) * 2.0
    true_muy = torch.randn(d_y, generator=gen, dtype=torch.float64) * 1.5

    x = torch.randn((1500, d_x), generator=gen, dtype=torch.float64) + true_mux
    y = true_scale * ((x - true_mux) @ true_r) + true_muy

    stats = SufficientStats(d_x, d_y)
    stats.update(x[:1200], y[:1200], num_prompts=40)

    key = fit_procrustes(stats, scale_mode="ls")
    metrics = evaluate_key(key, x[1200:], y[1200:])

    if d_x <= d_y:
        # When d_x <= d_y, R R^T = I makes tr(R^T M_xx R) constant, so thin-SVD Procrustes is exact
        # even at finite N.
        assert math.isclose(key.scale, true_scale, rel_tol=1e-3)
        assert metrics.token_cosine > 0.9999
        assert metrics.centered_cosine > 0.9999
        assert metrics.rel_l2 < 1e-3
        assert metrics.r2 > 0.9999
    else:
        # When d_x > d_y, finite-sample M_xx fluctuations slightly perturb thin-SVD Procrustes,
        # while Ridge recovers the exact linear projection.
        assert math.isclose(key.scale, true_scale, rel_tol=0.08)
        assert metrics.centered_cosine > 0.98
        ridge = fit_ridge(stats, alpha=1e-8)
        rm = evaluate_key(ridge, x[1200:], y[1200:])
        assert rm.centered_cosine > 0.9999
        assert rm.r2 > 0.9999


def test_ridge_beats_procrustes_on_anisotropic_outlier_channels() -> None:
    """Mimic Qwen3.5 outlier channels: target has non-isotropic per-channel scaling."""
    gen = torch.Generator().manual_seed(7)
    d_x, d_y = 32, 64
    r_true = _random_orthonormal(d_x, d_y, gen)
    channel_gains = torch.ones(d_y, dtype=torch.float64)
    channel_gains[:4] = 15.0  # outlier channels
    channel_gains[4:12] = 0.2

    x = torch.randn((2500, d_x), generator=gen, dtype=torch.float64)
    noise = torch.randn((2500, d_y), generator=gen, dtype=torch.float64) * 0.05
    y = ((x @ r_true) * channel_gains) + noise

    stats = SufficientStats(d_x, d_y)
    stats.update(x[:2000], y[:2000], num_prompts=100)

    proc = fit_procrustes(stats, scale_mode="ls")
    proc_norm = fit_procrustes(stats, scale_mode="variance")
    ridge = fit_ridge(stats, val_x=x[2000:], val_y=y[2000:], match_variance=False)
    ridge_norm = fit_ridge(stats, val_x=x[2000:], val_y=y[2000:], match_variance=True)

    m_proc = evaluate_key(proc, x[2000:], y[2000:])
    m_ridge = evaluate_key(ridge, x[2000:], y[2000:])
    m_rnorm = evaluate_key(ridge_norm, x[2000:], y[2000:])

    assert m_ridge.centered_cosine > m_proc.centered_cosine + 0.05
    assert m_ridge.r2 > 0.98
    assert m_ridge.channel_std_corr > 0.99
    assert abs(m_rnorm.norm_ratio - 1.0) < 0.05
    # Variance-matched Procrustes matches total RMS energy even under extreme outlier channels.
    y_hat_pnorm = apply_tensor(x[2000:].float(), proc_norm.weight, proc_norm.mu_x, proc_norm.mu_y)
    rms_ratio = float((y_hat_pnorm.pow(2).mean().sqrt() / y[2000:].float().pow(2).mean().sqrt()).item())
    assert abs(rms_ratio - 1.0) < 0.08


def test_procrustes_pca_low_rank_recovery() -> None:
    gen = torch.Generator().manual_seed(11)
    d_x, d_y, rank = 48, 64, 8
    z = torch.randn((1000, rank), generator=gen, dtype=torch.float64)
    bx = _random_orthonormal(rank, d_x, gen)
    by = _random_orthonormal(rank, d_y, gen)
    x = z @ bx
    y = 2.0 * (z @ by) + 1.0

    stats = SufficientStats(d_x, d_y)
    stats.update(x[:800], y[:800])
    key = fit_procrustes_pca(stats, rank=rank, scale_mode="ls")
    metrics = evaluate_key(key, x[800:], y[800:])
    assert key.meta.pca_rank == rank
    assert metrics.centered_cosine > 0.9999
    assert metrics.r2 > 0.9999


def test_safetensors_save_load_roundtrip(tmp_path: Path) -> None:
    gen = torch.Generator().manual_seed(99)
    meta = PlatoKeyMeta(
        source_model="Qwen/Qwen3.5-2B",
        target_model="Qwen/Qwen3.5-9B",
        target_preset="moss_image",
        method="ridge",
        source_layer="last",
        target_layer="last",
        source_layer_norm=True,
        target_layer_norm=True,
        source_template=MOSS_IMAGE_TEMPLATE,
        strip_prompt=True,
        ridge_alpha=1e-3,
        num_prompts=3000,
        num_tokens=180000,
        dataset="moss-domain+sd-prompts",
        metrics={"token_cosine": 0.9312, "centered_cosine": 0.8921},
    )
    key = PlatoKey(
        weight=torch.randn((64, 32), generator=gen),
        mu_x=torch.randn(32, generator=gen),
        mu_y=torch.randn(64, generator=gen),
        meta=meta,
        scale=1.25,
        target_token_rms=0.87,
        source_token_rms=1.12,
        pooled_weight=torch.randn((16, 8), generator=gen),
        pooled_mu_x=torch.randn(8, generator=gen),
        pooled_mu_y=torch.randn(16, generator=gen),
    )

    out = save_key(key, tmp_path / "test_key.safetensors", dtype=torch.float16)
    loaded = load_key(out)

    assert loaded.d_source == 32
    assert loaded.d_target == 64
    assert loaded.weight.dtype == torch.float16
    assert loaded.mu_x.dtype == torch.float32
    assert loaded.mu_y.dtype == torch.float32
    assert torch.allclose(loaded.weight.float(), key.weight, atol=2e-3)
    assert torch.allclose(loaded.mu_x, key.mu_x)
    assert torch.allclose(loaded.mu_y, key.mu_y)
    assert math.isclose(loaded.scale, 1.25, rel_tol=1e-5)
    assert loaded.target_token_rms is not None and math.isclose(loaded.target_token_rms, 0.87, rel_tol=1e-5)
    assert loaded.pooled_weight is not None and loaded.pooled_weight.shape == (16, 8)
    assert loaded.meta.source_template == MOSS_IMAGE_TEMPLATE
    assert loaded.meta.ridge_alpha == pytest.approx(1e-3)
    assert loaded.meta.metrics["centered_cosine"] == pytest.approx(0.8921)
    assert "Qwen/Qwen3.5-2B -> Qwen/Qwen3.5-9B" in loaded.summary()


def test_resolve_key_path_rejects_traversal() -> None:
    with pytest.raises(ValueError):
        resolve_key_path("../secret.safetensors")
    with pytest.raises(ValueError):
        resolve_key_path("/etc/passwd")


def test_apply_conditioning_and_controls() -> None:
    gen = torch.Generator().manual_seed(5)
    key = PlatoKey(
        weight=torch.randn((64, 32), generator=gen),
        mu_x=torch.randn(32, generator=gen),
        mu_y=torch.randn(64, generator=gen),
        target_token_rms=2.0,
        pooled_weight=torch.randn((16, 8), generator=gen),
        pooled_mu_x=torch.randn(8, generator=gen),
        pooled_mu_y=torch.randn(16, generator=gen),
    )

    cond_in = torch.randn((1, 12, 32), generator=gen, dtype=torch.bfloat16)
    pooled_in = torch.randn((1, 8), generator=gen, dtype=torch.float32)
    mask_in = torch.ones((1, 12), dtype=torch.long)
    orig_meta = {"pooled_output": pooled_in, "attention_mask": mask_in, "custom": 42}
    conditioning = [[cond_in, orig_meta]]

    out = apply_conditioning(conditioning, key)
    assert len(out) == 1
    out_tensor, out_meta = out[0]
    assert out_tensor.shape == (1, 12, 64)
    assert out_tensor.dtype == torch.bfloat16
    assert out_meta["pooled_output"].shape == (1, 16)
    assert out_meta["custom"] == 42
    assert torch.equal(out_meta["attention_mask"], mask_in)
    # Original conditioning untouched
    assert orig_meta["pooled_output"].shape == (1, 8)

    # Contrast = 0 collapses every token to mu_y
    zero_c = apply_tensor(cond_in.float(), key.weight, key.mu_x, key.mu_y, contrast=0.0)
    assert torch.allclose(zero_c, key.mu_y.view(1, 1, -1).expand_as(zero_c), atol=1e-5)

    # target_rms mode matches target RMS
    rms_out = apply_tensor(
        cond_in.float(),
        key.weight,
        key.mu_x,
        key.mu_y,
        norm_match=NORM_MATCH_TARGET_RMS,
        target_token_rms=2.0,
    )
    assert math.isclose(float(rms_out.pow(2).mean().sqrt().item()), 2.0, rel_tol=1e-4)

    # preserve_source_norm matches per-token input L2 norm
    pnorm_out = apply_tensor(
        cond_in.float(),
        key.weight,
        key.mu_x,
        key.mu_y,
        norm_match=NORM_MATCH_SOURCE_NORM,
    )
    assert torch.allclose(pnorm_out.norm(dim=-1), cond_in.float().norm(dim=-1), rtol=1e-4)

    # Dimension mismatch raises informative error
    with pytest.raises(ValueError, match="expects source embedding dimension 32"):
        apply_tensor(torch.randn(1, 5, 99), key.weight, key.mu_x, key.mu_y, norm_match=NORM_MATCH_OFF)


class _DummyInnerClip:
    def __init__(self) -> None:
        self.layer = "hidden"
        self.layer_idx = -2
        self.layer_norm_hidden_state = False
        self.options_default = ("hidden", -2, False)

    def reset_clip_options(self) -> None:
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]


class _DummyCondStage:
    def __init__(self) -> None:
        self.clip = "qwen35_2b"
        self.qwen35_2b = _DummyInnerClip()

    def reset_clip_options(self) -> None:
        self.qwen35_2b.reset_clip_options()


class _DummyCLIP:
    """Minimal mock of ComfyUI's comfy.sd.CLIP for testing PlatoKeyEncode."""

    def __init__(self, hidden_dim: int = 32) -> None:
        self.hidden_dim = hidden_dim
        self.layer_idx = None
        self.cond_stage_model = _DummyCondStage()
        self.seen_text: str | None = None
        self.seen_layer = None
        self.seen_layer_idx = None
        self.seen_ln: bool | None = None

    def clone(self) -> _DummyCLIP:
        c = _DummyCLIP(self.hidden_dim)
        c.layer_idx = self.layer_idx
        c.cond_stage_model = self.cond_stage_model
        return c

    def clip_layer(self, layer_idx: int | None) -> None:
        self.layer_idx = layer_idx

    def tokenize(self, text: str):
        self.seen_text = text
        return {"tokens": text}

    def encode_from_tokens_scheduled(self, tokens):
        # Mirror ComfyUI's CLIP.encode_from_tokens: always calls reset_clip_options() first!
        self.cond_stage_model.reset_clip_options()
        inner = self.cond_stage_model.qwen35_2b
        if self.layer_idx is not None:
            inner.layer = "hidden"
            inner.layer_idx = self.layer_idx
        self.seen_layer = inner.layer
        self.seen_layer_idx = inner.layer_idx
        self.seen_ln = inner.layer_norm_hidden_state
        n_tok = max(1, len(tokens["tokens"].split()))
        return [[torch.ones((1, n_tok, self.hidden_dim), dtype=torch.float32), {"pooled_output": None}]]


def test_encode_with_key_moss_template_and_last_layer() -> None:
    key = PlatoKey(
        weight=torch.eye(64, 32),
        mu_x=torch.zeros(32),
        mu_y=torch.zeros(64),
        meta=PlatoKeyMeta(
            source_layer="last",
            source_layer_norm=True,
            source_template=MOSS_IMAGE_TEMPLATE,
            strip_prompt=True,
        ),
    )

    # Template formatting strips prompt and avoids double-wrapping
    formatted = format_prompt_for_key("  a red fox  ", key)
    assert formatted == MOSS_IMAGE_TEMPLATE.format(prompt="a red fox")
    assert format_prompt_for_key(formatted, key) == formatted

    orig_clip = _DummyCLIP(hidden_dim=32)
    orig_clip.layer_idx = -2  # even if upstream set clip_layer(-2), key's layer="last" must win!
    captured: list[_DummyCLIP] = []
    real_clone = orig_clip.clone

    def _spy_clone() -> _DummyCLIP:
        c = real_clone()
        captured.append(c)
        return c

    orig_clip.clone = _spy_clone  # type: ignore[method-assign]

    cond = encode_with_key(orig_clip, "  a red fox  ", key)
    assert orig_clip.layer_idx == -2  # original CLIP untouched
    assert orig_clip.cond_stage_model.qwen35_2b.options_default == ("hidden", -2, False)  # restored
    assert orig_clip.cond_stage_model.qwen35_2b.layer == "hidden"  # restored
    assert len(captured) == 1
    assert captured[0].seen_layer == "last"
    assert captured[0].seen_layer_idx is None
    assert captured[0].seen_text == MOSS_IMAGE_TEMPLATE.format(prompt="a red fox")
    assert cond[0][0].shape[-1] == 64


def test_encode_with_key_intermediate_layer_restores_layer_norm() -> None:
    key = PlatoKey(
        weight=torch.eye(64, 32),
        mu_x=torch.zeros(32),
        mu_y=torch.zeros(64),
        meta=PlatoKeyMeta(
            source_layer="-2",
            source_layer_norm=True,
            source_template="",
        ),
    )
    orig_clip = _DummyCLIP(hidden_dim=32)
    orig_clip.cond_stage_model.qwen35_2b.layer_norm_hidden_state = False

    captured: list[_DummyCLIP] = []
    real_clone = orig_clip.clone

    def _spy_clone() -> _DummyCLIP:
        c = real_clone()
        captured.append(c)
        return c

    orig_clip.clone = _spy_clone  # type: ignore[method-assign]

    encode_with_key(orig_clip, "hello", key)
    assert captured[0].seen_layer_idx == -2
    assert captured[0].seen_ln is True
    # Restored to original False after encode!
    assert orig_clip.cond_stage_model.qwen35_2b.layer_norm_hidden_state is False


def test_calibration_helpers_and_end_to_end_export(tmp_path: Path) -> None:
    prompts = build_domain_prompts(seed=42)
    assert "" in prompts
    assert len(prompts) >= 40

    formatted = format_prompts(prompts[:4], PRESETS["moss_image"])
    assert all(s.startswith("<|im_start|>user\n") and s.endswith("<think>\n") for s in formatted)

    # Common prefix/suffix detection
    seqs = [
        [1, 2, 3, 10, 11, 90, 91],
        [1, 2, 3, 20, 21, 22, 90, 91],
        [1, 2, 3, 30, 90, 91],
    ]
    assert detect_common_prefix_suffix(seqs) == (3, 2)

    hidden = torch.randn(3, 8, 16)
    vecs, w, cm = extract_batch_tokens(hidden, [7, 8, 6], prefix_len=3, suffix_len=2, prefix_weight=0.05)
    assert vecs.shape == (21, 16)
    assert cm.sum().item() == (2 + 3 + 1)  # only content tokens
    assert math.isclose(float(w[0].item()), 0.05, rel_tol=1e-5)
    assert math.isclose(float(w[3].item()), 1.0, rel_tol=1e-5)

    # Prompt file loader (.txt and .jsonl)
    txt = tmp_path / "p.txt"
    txt.write_text("cat\ndog\n\n", encoding="utf-8")
    assert load_prompts_from_file(txt) == ["cat", "dog"]

    jsonl = tmp_path / "p.jsonl"
    jsonl.write_text(json.dumps({"prompt": "fox"}) + "\n" + json.dumps({"caption": "owl"}) + "\n", encoding="utf-8")
    assert load_prompts_from_file(jsonl) == ["fox", "owl"]

    # End-to-end fit_all_methods & fit_and_export
    gen = torch.Generator().manual_seed(17)
    r_mat = _random_orthonormal(16, 32, gen)
    x = torch.randn((400, 16), generator=gen)
    y = 1.4 * (x @ r_mat.float()) + 0.3
    stats = SufficientStats(16, 32)
    stats.update(x[:320], y[:320], num_prompts=20)
    val_cm = torch.ones(80, dtype=torch.bool)
    val_cm[:10] = False

    results = fit_and_export(
        stats,
        val_x=x[320:],
        val_y=y[320:],
        val_content_mask=val_cm,
        base_meta=PlatoKeyMeta(source_model="src", target_model="tgt", target_preset="moss_image"),
        out_dir=tmp_path / "keys",
        file_prefix="src_to_tgt",
        pca_rank=8,
        save_dtype=torch.float16,
    )
    assert len(results) == 5
    for r in results:
        assert (tmp_path / "keys" / f"src_to_tgt_{r.name}.safetensors").is_file()
        assert "content_centered_cosine" in r.key.meta.metrics
    # Also check fit_all_methods directly without PCA
    assert len(fit_all_methods(stats)) == 4


def test_multilayer_key_roundtrip_apply_and_krea2_encode(tmp_path: Path) -> None:
    gen = torch.Generator().manual_seed(2026)
    n_layers, d_src, d_tgt = 4, 16, 24
    w = torch.randn((n_layers, d_tgt, d_src), generator=gen)
    mx = torch.randn((n_layers, d_src), generator=gen)
    my = torch.randn((n_layers, d_tgt), generator=gen)
    meta = PlatoKeyMeta(
        source_model="qwen35_2b",
        target_model="qwen3vl_4b",
        target_preset="krea2",
        method="ridge",
        source_layer="[1,7,15,23]",
        target_layer="[2,11,23,35]",
        source_layer_norm=False,
        target_layer_norm=False,
        source_template=KREA2_TEMPLATE,
        strip_prompt=False,
        strip_prefix_mode="krea2_user",
        flatten_layers=True,
    )
    key = PlatoKey(weight=w, mu_x=mx, mu_y=my, meta=meta, scale=1.1, layer_scales=[0.9, 1.0, 1.1, 1.4])
    assert key.is_multilayer and key.num_layers == 4 and key.d_source == 16 and key.d_target == 24
    assert parse_source_layer(key.meta.source_layer) == [1, 7, 15, 23]

    # Save / load roundtrip
    out_p = save_key(key, tmp_path / "ml_key.safetensors", dtype=torch.float16)
    loaded = load_key(out_p)
    assert loaded.is_multilayer and loaded.num_layers == 4
    assert loaded.layer_scales == pytest.approx([0.9, 1.0, 1.1, 1.4], rel=1e-5)
    assert loaded.meta.strip_prefix_mode == "krea2_user"
    assert loaded.meta.flatten_layers is True
    assert "4x16 -> 4x24" in loaded.summary()

    # Apply on 4D [B, N, S, d_src] -> flattened [B, S, N*d_tgt]
    x4d = torch.randn((1, n_layers, 7, d_src), generator=gen)
    y_flat = apply_tensor(x4d, w, mx, my, flatten_layers=True)
    assert y_flat.shape == (1, 7, n_layers * d_tgt)

    # Apply on pre-flattened 3D [B, S, N*d_src] gives identical output!
    x3d = x4d.permute(0, 2, 1, 3).reshape(1, 7, n_layers * d_src)
    y_from_3d = apply_tensor(x3d, w, mx, my, flatten_layers=True)
    assert torch.allclose(y_flat, y_from_3d, atol=1e-5)

    # Prefix stripping detects second <|im_start|>user\n for BOTH Qwen3-VL (151644/872) and Qwen3.5 (248045/846)
    q35_pairs = [(248045, 1.0), (9, 1.0), (248046, 1.0), (198, 1.0), (248045, 1.0), (846, 1.0), (198, 1.0), (42, 1.0)]
    assert find_krea2_prefix_end(q35_pairs, len(q35_pairs)) == 7
    q3vl_pairs = [(151644, 1.0), (9, 1.0), (151645, 1.0), (198, 1.0), (151644, 1.0), (872, 1.0), (198, 1.0), (42, 1.0)]
    assert find_krea2_prefix_end(q3vl_pairs, len(q3vl_pairs)) == 7
    # Templates without a system turn (2nd <|im_start|> is assistant\n, not user\n) must NOT be stripped
    moss_pairs = [(248045, 1.0), (846, 1.0), (198, 1.0), (42, 1.0), (248046, 1.0), (198, 1.0), (248045, 1.0), (77091, 1.0), (198, 1.0)]
    assert find_krea2_prefix_end(moss_pairs, len(moss_pairs)) == 0

    # Single-layer [B, seq, d_src] fed to a multi-layer key gives actionable Patch CLIP hint
    with pytest.raises(ValueError, match="Patch CLIP"):
        apply_tensor(torch.randn(1, 7, d_src), w, mx, my, flatten_layers=True)


def test_proportional_layers_and_cross_tokenizer_alignment() -> None:
    assert proportional_source_layers(KREA2_TAP_LAYERS, 36, 36) == KREA2_TAP_LAYERS
    l24 = proportional_source_layers(KREA2_TAP_LAYERS, 36, 24)
    assert len(l24) == 12 and all(1 <= v <= 23 for v in l24)
    l32 = proportional_source_layers(KREA2_TAP_LAYERS, 36, 32)
    assert len(l32) == 12 and all(1 <= v <= 31 for v in l32)

    # Synthetic cross-tokenizer byte-span alignment:
    # Source splits "hello world" into ["hel", "lo", " world"] (byte ends: 3, 5, 11) + 2 suffix tokens
    # Target splits "hello world" into ["hello", " wo", "rld"] (byte ends: 5, 8, 11) + 2 suffix tokens
    class _FakeTok:
        def __init__(self, vocab: dict[int, str]):
            self.vocab = vocab

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            return "".join(self.vocab[i] for i in ids)

    src_tok = _FakeTok({1: "hel", 2: "lo", 3: " world", 90: "<end>", 91: "\n"})
    tgt_tok = _FakeTok({10: "hello", 11: " wo", 12: "rld", 99: "<end>", 98: "\n"})

    si, ti, cm = align_token_sequences(
        [1, 2, 3, 90, 91],
        [10, 11, 12, 99, 98],
        src_tok,
        tgt_tok,
        suffix_tokens=2,
        same_tokenizer=False,
    )
    # Byte ends 5 ("hello") and 11 ("hello world") match, plus the 2 suffix tokens!
    assert si == [1, 2, 3, 4]
    assert ti == [0, 2, 3, 4]
    assert cm == [True, True, False, False]


def test_patch_clip_with_key_works_with_multilayer_and_visual_tokens() -> None:
    """Verify Patch CLIP handles Visual Fusion's pre-templated image tokens, clone(), and no double-apply."""

    class _MultiLayerDummyCLIP:
        def __init__(self, d_in: int = 16) -> None:
            self.d_in = d_in
            self.layer_idx = None
            self.cond_stage_model = _DummyCondStage()
            self.last_text: str | None = None
            self.last_layer = None

        def clone(self) -> _MultiLayerDummyCLIP:
            c = _MultiLayerDummyCLIP(self.d_in)
            c.layer_idx = self.layer_idx
            c.cond_stage_model = self.cond_stage_model
            return c

        def clip_layer(self, layer_idx: int | None) -> None:
            self.layer_idx = layer_idx

        def tokenize(self, text: str, images=None):
            self.last_text = text
            # Simulate Qwen3.5 system + <|im_start|>user\n (7 prefix tokens) + optional image + prompt
            pairs: list[tuple] = [
                (248045, 1.0), (10, 1.0), (248046, 1.0), (198, 1.0),
                (248045, 1.0), (846, 1.0), (198, 1.0),
            ]
            if images:
                pairs.append(({"type": "image", "data": images[0]}, 1.0))
            pairs.extend([(42, 1.0), (43, 1.0), (248046, 1.0)])
            return {"qwen35_2b": [pairs]}

        def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
            self.cond_stage_model.reset_clip_options()
            inner = self.cond_stage_model.qwen35_2b
            self.last_layer = inner.layer
            # Simulate 1 image expanding to 4 visual tokens -> total seq = 7 + 4 + 3 = 14 (or 10 without image)
            pairs = tokens["qwen35_2b"][0]
            has_img = any(isinstance(t[0], dict) for t in pairs)
            seq = 14 if has_img else 10
            n_taps = len(inner.layer) if isinstance(inner.layer, list) else 1
            tensor = torch.ones((1, n_taps, seq, self.d_in), dtype=torch.float32) if n_taps > 1 else torch.ones((1, seq, self.d_in))
            if return_dict:
                return {"cond": tensor, "pooled_output": None, "attention_mask": torch.ones((1, seq), dtype=torch.long)}
            return (tensor, None) if return_pooled else tensor

        def encode_from_tokens_scheduled(self, tokens):
            # Calls self.encode_from_tokens internally — tests that patching NEVER double-applies!
            d = self.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = d.pop("cond")
            return [[cond, d]]

    n_layers, d_src, d_tgt = 3, 16, 24
    key = PlatoKey(
        weight=torch.eye(d_tgt, d_src).unsqueeze(0).repeat(n_layers, 1, 1),
        mu_x=torch.zeros((n_layers, d_src)),
        mu_y=torch.zeros((n_layers, d_tgt)),
        meta=PlatoKeyMeta(
            source_layer="[1,11,23]",
            source_layer_norm=False,
            source_template=KREA2_TEMPLATE,
            strip_prefix_mode="krea2_user",
            flatten_layers=True,
        ),
    )

    base = _MultiLayerDummyCLIP(d_in=d_src)
    patched = patch_clip_with_key(base, key)
    # Cloning a patched CLIP keeps the patch intact!
    patched2 = patched.clone()

    # 1. Pre-templated Visual Fusion call with an image: template passes through untouched,
    #    7 prefix tokens stripped -> 14 - 7 = 7 tokens (4 visual + 3 text), projected & flattened to 3*24=72
    fusion_prompt = KREA2_TEMPLATE.format(prompt="<|vision_start|><|image_pad|><|vision_end|>cat")
    toks = patched2.tokenize(fusion_prompt, images=["fake_img"])
    assert patched2.last_text == fusion_prompt
    cond = patched2.encode_from_tokens_scheduled(toks)
    assert patched2.last_layer == [1, 11, 23]
    assert cond[0][0].shape == (1, 7, n_layers * d_tgt)
    assert "attention_mask" not in cond[0][1]  # all-1 mask dropped like krea2.py
    # Original CLIP's inner model restored cleanly!
    assert base.cond_stage_model.qwen35_2b.layer == "hidden"

    # 2. Raw text call (CLIPTextEncode): gets KREA2_TEMPLATE automatically
    toks_raw = patched.tokenize("a red fox")
    assert patched.last_text == KREA2_TEMPLATE.format(prompt="a red fox")
    cond_raw = patched.encode_from_tokens_scheduled(toks_raw)
    assert cond_raw[0][0].shape == (1, 3, n_layers * d_tgt)
