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
from plato_keys.encode import encode_with_key, format_prompt_for_key
from plato_keys.format import (
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
