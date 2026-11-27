"""PlatoKey `.safetensors` file format, metadata schema, and ComfyUI folder discovery.

Pure Python + torch + safetensors (no ComfyUI imports required at import time).

Tensor layout inside a `.safetensors` PlatoKey file:
  - `weight`        : `[d_target, d_source]` (PyTorch `F.linear` layout; already includes `scale`
                      for Procrustes keys so inference is one GEMM: `F.linear(x - mu_x, weight) + mu_y`)
  - `mu_x`          : `[d_source]` source mean vector
  - `mu_y`          : `[d_target]` target mean vector
  - `scale`         : `[1]` scalar global scale (stored for inspection / diagnostics; baked into `weight`
                      when `weight_includes_scale == True`, which is always true for v1 keys)
  - Optional pooled-embedding map (for dual-output encoders like CLIP/SDXL/FLUX):
      `pooled_weight` : `[d_pooled_target, d_pooled_source]`
      `pooled_mu_x`   : `[d_pooled_source]`
      `pooled_mu_y`   : `[d_pooled_target]`
  - Optional target-norm stats (for `norm_match="target_rms"` / `"per_token"` diagnostics):
      `target_token_rms` : `[1]` mean per-token L2 norm / sqrt(d_target) in target space
      `source_token_rms` : `[1]` mean per-token L2 norm / sqrt(d_source) in source space

All string metadata lives in the safetensors header under the `plato.*` prefix so any tool can
inspect a key without loading weights.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

FORMAT_VERSION = "1"
KEY_EXTENSION = ".safetensors"
FOLDER_NAME = "plato_keys"

# Known ComfyUI prompt/extraction presets so PlatoKeyEncode can reproduce exact calibration conditions.
MOSS_IMAGE_TEMPLATE = (
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)
QWEN35_DEFAULT_TEMPLATE = (
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n</think>\n"
)
KREA2_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
KREA2_TAP_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]


@dataclass
class PlatoKeyMeta:
    """Metadata stored in the `.safetensors` header describing how a key was calibrated."""

    source_model: str = ""
    target_model: str = ""
    target_preset: str = ""  # e.g. "moss_image", "qwen35_default", "generic"
    method: str = "procrustes"  # "procrustes", "procrustes_norm", "ridge", "ridge_norm", "procrustes_pca"
    source_layer: str = "last"  # "last" or integer string like "-2"
    target_layer: str = "last"
    source_layer_norm: bool = True
    target_layer_norm: bool = True
    source_template: str = ""  # Python format string with `{prompt}`, or "" to use CLIP's default
    strip_prompt: bool = True  # MOSS-Image strips leading/trailing whitespace before templating
    strip_prefix_mode: str = ""  # "" (none) or "krea2_user" (strip system + <|im_start|>user\n prefix)
    source_prefix_len: int = 0   # Optional source prefix token count stripped before wrapping `target_prefix`
    source_suffix_len: int = 0   # Optional source suffix token count stripped before wrapping `target_suffix`
    flatten_layers: bool = False  # True for multi-layer DiTs like Krea 2: [B, N, S, D] -> [B, S, N*D]
    num_layers: int = 1  # 1 for single-layer keys, >1 for stacked per-layer keys (e.g. 12 for Krea 2)
    d_source: int = 0
    d_target: int = 0
    scale: float = 1.0
    ridge_alpha: float | None = None
    pca_rank: int | None = None
    num_prompts: int = 0
    num_tokens: int = 0
    dataset: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    format_version: str = FORMAT_VERSION

    def to_header(self) -> dict[str, str]:
        """Serialize metadata to string-only safetensors header dict."""
        raw = asdict(self)
        out: dict[str, str] = {}
        for k, v in raw.items():
            if v is None:
                continue
            key = f"plato.{k}"
            if isinstance(v, bool):
                out[key] = "true" if v else "false"
            elif isinstance(v, (dict, list)):
                out[key] = json.dumps(v, sort_keys=True)
            else:
                out[key] = str(v)
        return out

    @classmethod
    def from_header(cls, header: dict[str, str] | None) -> PlatoKeyMeta:
        if not header:
            return cls()

        def _get(name: str, default: Any = None) -> Any:
            return header.get(f"plato.{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _get(name)
            if val is None:
                return default
            return str(val).strip().lower() in ("1", "true", "yes")

        def _int(name: str, default: int) -> int:
            val = _get(name)
            if val is None or val == "":
                return default
            try:
                return int(val)
            except ValueError:
                return default

        def _opt_int(name: str) -> int | None:
            val = _get(name)
            if val is None or val == "":
                return None
            try:
                return int(val)
            except ValueError:
                return None

        def _float(name: str, default: float) -> float:
            val = _get(name)
            if val is None or val == "":
                return default
            try:
                return float(val)
            except ValueError:
                return default

        def _opt_float(name: str) -> float | None:
            val = _get(name)
            if val is None or val == "":
                return None
            try:
                return float(val)
            except ValueError:
                return None

        metrics_raw = _get("metrics")
        metrics: dict[str, float] = {}
        if metrics_raw:
            try:
                parsed = json.loads(metrics_raw)
                if isinstance(parsed, dict):
                    metrics = {str(k): float(v) for k, v in parsed.items()}
            except (json.JSONDecodeError, TypeError, ValueError):
                metrics = {}

        return cls(
            source_model=str(_get("source_model", "")),
            target_model=str(_get("target_model", "")),
            target_preset=str(_get("target_preset", "")),
            method=str(_get("method", "procrustes")),
            source_layer=str(_get("source_layer", "last")),
            target_layer=str(_get("target_layer", "last")),
            source_layer_norm=_bool("source_layer_norm", True),
            target_layer_norm=_bool("target_layer_norm", True),
            source_template=str(_get("source_template", "")),
            strip_prompt=_bool("strip_prompt", True),
            strip_prefix_mode=str(_get("strip_prefix_mode", "")),
            source_prefix_len=_int("source_prefix_len", 0),
            source_suffix_len=_int("source_suffix_len", 0),
            flatten_layers=_bool("flatten_layers", False),
            num_layers=_int("num_layers", 1),
            d_source=_int("d_source", 0),
            d_target=_int("d_target", 0),
            scale=_float("scale", 1.0),
            ridge_alpha=_opt_float("ridge_alpha"),
            pca_rank=_opt_int("pca_rank"),
            num_prompts=_int("num_prompts", 0),
            num_tokens=_int("num_tokens", 0),
            dataset=str(_get("dataset", "")),
            metrics=metrics,
            notes=str(_get("notes", "")),
            format_version=str(_get("format_version", FORMAT_VERSION)),
        )


@dataclass
class PlatoKey:
    """In-memory cross-encoder alignment map (single-layer 2D or multi-layer 3D)."""

    weight: torch.Tensor  # [d_target, d_source] or [num_layers, d_target, d_source]
    mu_x: torch.Tensor  # [d_source] or [num_layers, d_source]
    mu_y: torch.Tensor  # [d_target] or [num_layers, d_target]
    meta: PlatoKeyMeta = field(default_factory=PlatoKeyMeta)
    scale: float = 1.0
    layer_scales: list[float] | None = None
    target_token_rms: float | None = None
    source_token_rms: float | None = None
    pooled_weight: torch.Tensor | None = None  # [d_pooled_target, d_pooled_source]
    pooled_mu_x: torch.Tensor | None = None  # [d_pooled_source]
    pooled_mu_y: torch.Tensor | None = None  # [d_pooled_target]
    target_prefix: torch.Tensor | None = None  # [P, d_target] exact target attention-sink prefix (e.g. MOSS 3-tok)
    target_suffix: torch.Tensor | None = None  # [T, d_target] exact target turn-closing suffix (e.g. MOSS 7-tok)

    def __post_init__(self) -> None:
        if self.weight.ndim == 2:
            d_target, d_source = self.weight.shape
            n_layers = 1
            if self.mu_x.shape != (d_source,):
                raise ValueError(f"mu_x shape {tuple(self.mu_x.shape)} does not match d_source={d_source}")
            if self.mu_y.shape != (d_target,):
                raise ValueError(f"mu_y shape {tuple(self.mu_y.shape)} does not match d_target={d_target}")
        elif self.weight.ndim == 3:
            n_layers, d_target, d_source = self.weight.shape
            if self.mu_x.shape != (n_layers, d_source):
                raise ValueError(
                    f"mu_x shape {tuple(self.mu_x.shape)} does not match (num_layers={n_layers}, d_source={d_source})"
                )
            if self.mu_y.shape != (n_layers, d_target):
                raise ValueError(
                    f"mu_y shape {tuple(self.mu_y.shape)} does not match (num_layers={n_layers}, d_target={d_target})"
                )
        else:
            raise ValueError(
                f"PlatoKey weight must be 2D [d_target, d_source] or 3D [num_layers, d_target, d_source], "
                f"got {tuple(self.weight.shape)}"
            )
        if self.pooled_weight is not None:
            if self.pooled_mu_x is None or self.pooled_mu_y is None:
                raise ValueError("pooled_weight requires pooled_mu_x and pooled_mu_y")
            pt, ps = self.pooled_weight.shape
            if self.pooled_mu_x.shape != (ps,) or self.pooled_mu_y.shape != (pt,):
                raise ValueError("pooled_mu_x / pooled_mu_y shapes do not match pooled_weight")
        self.meta.num_layers = n_layers
        self.meta.d_source = d_source
        self.meta.d_target = d_target
        self.meta.scale = float(self.scale)

    @property
    def is_multilayer(self) -> bool:
        return self.weight.ndim == 3

    @property
    def num_layers(self) -> int:
        return int(self.weight.shape[0]) if self.weight.ndim == 3 else 1

    @property
    def d_source(self) -> int:
        return int(self.weight.shape[-1])

    @property
    def d_target(self) -> int:
        return int(self.weight.shape[-2])

    def summary(self) -> str:
        """Human-readable multi-line summary for inspection nodes and CLI output."""
        m = self.meta
        dim_str = (
            f"{self.num_layers}x{self.d_source} -> {self.num_layers}x{self.d_target}"
            if self.is_multilayer
            else f"{self.d_source} -> {self.d_target}"
        )
        lines = [f"PlatoKey ({m.method}) — {dim_str}"]
        if m.source_model or m.target_model:
            lines.append(f"Models:   {m.source_model or '?'} -> {m.target_model or '?'}")
        if m.target_preset:
            lines.append(f"Preset:   {m.target_preset}")
        lines.append(
            f"Layers:   source={m.source_layer} (norm={m.source_layer_norm}), "
            f"target={m.target_layer} (norm={m.target_layer_norm})"
        )
        if m.strip_prefix_mode or m.flatten_layers:
            lines.append(f"Post:     strip_prefix={m.strip_prefix_mode or 'none'}, flatten_layers={m.flatten_layers}")
        if m.source_template:
            escaped = m.source_template.replace("\n", "\\n")
            lines.append(f"Template: {escaped}")
        params = [f"scale={self.scale:.4f}"]
        if m.ridge_alpha is not None:
            params.append(f"ridge_alpha={m.ridge_alpha:.2e}")
        if m.pca_rank is not None:
            params.append(f"pca_rank={m.pca_rank}")
        if self.target_token_rms is not None:
            params.append(f"target_rms={self.target_token_rms:.4f}")
        lines.append("Params:   " + ", ".join(params))
        if m.num_prompts or m.num_tokens:
            lines.append(f"Calib:    {m.num_prompts} prompts, {m.num_tokens:,} tokens ({m.dataset or 'custom'})")
        if m.metrics:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(m.metrics.items()))
            lines.append(f"Metrics:  {metric_str}")
        if self.pooled_weight is not None:
            lines.append(f"Pooled:   {self.pooled_weight.shape[1]} -> {self.pooled_weight.shape[0]}")
        if m.notes:
            lines.append(f"Notes:    {m.notes}")
        return "\n".join(lines)


def save_key(
    key: PlatoKey,
    path: str | os.PathLike[str],
    *,
    dtype: torch.dtype = torch.float16,
) -> Path:
    """Save a PlatoKey to `.safetensors`.

    Weights are cast to `dtype` (fp16 by default: 2048x4096 is 16 MB); `mu_x` and `mu_y` are
    always saved in float32 because mean vectors are tiny (8–16 KB) and subtracting a quantized
    mean introduces avoidable bias in outlier channels.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scale_tensor = (
        torch.tensor([float(s) for s in key.layer_scales], dtype=torch.float32)
        if key.layer_scales is not None
        else torch.tensor([float(key.scale)], dtype=torch.float32)
    )
    tensors: dict[str, torch.Tensor] = {
        "weight": key.weight.detach().to(device="cpu", dtype=dtype).contiguous(),
        "mu_x": key.mu_x.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        "mu_y": key.mu_y.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        "scale": scale_tensor,
    }
    if key.target_token_rms is not None:
        tensors["target_token_rms"] = torch.tensor([float(key.target_token_rms)], dtype=torch.float32)
    if key.source_token_rms is not None:
        tensors["source_token_rms"] = torch.tensor([float(key.source_token_rms)], dtype=torch.float32)
    if key.pooled_weight is not None and key.pooled_mu_x is not None and key.pooled_mu_y is not None:
        tensors["pooled_weight"] = key.pooled_weight.detach().to(device="cpu", dtype=dtype).contiguous()
        tensors["pooled_mu_x"] = key.pooled_mu_x.detach().to(device="cpu", dtype=torch.float32).contiguous()
        tensors["pooled_mu_y"] = key.pooled_mu_y.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if key.target_prefix is not None:
        tensors["target_prefix"] = key.target_prefix.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if key.target_suffix is not None:
        tensors["target_suffix"] = key.target_suffix.detach().to(device="cpu", dtype=torch.float32).contiguous()

    save_file(tensors, str(out_path), metadata=key.meta.to_header())
    return out_path


def load_key(path: str | os.PathLike[str]) -> PlatoKey:
    """Load a `.safetensors` PlatoKey file from disk."""
    from safetensors import safe_open

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"PlatoKey file not found: {p}")

    with safe_open(str(p), framework="pt", device="cpu") as f:
        header = f.metadata()
    tensors = load_file(str(p), device="cpu")

    if "weight" not in tensors or "mu_x" not in tensors or "mu_y" not in tensors:
        raise ValueError(
            f"{p.name} is not a valid PlatoKey file (expected 'weight', 'mu_x', 'mu_y'; "
            f"found {sorted(tensors.keys())})"
        )

    meta = PlatoKeyMeta.from_header(header)
    layer_scales: list[float] | None = None
    if "scale" in tensors:
        s_t = tensors["scale"].reshape(-1)
        if s_t.numel() > 1:
            layer_scales = [float(v) for v in s_t.tolist()]
            scale = float(s_t.mean().item())
        else:
            scale = float(s_t.item())
    else:
        scale = meta.scale
    target_rms = float(tensors["target_token_rms"].item()) if "target_token_rms" in tensors else None
    source_rms = float(tensors["source_token_rms"].item()) if "source_token_rms" in tensors else None

    return PlatoKey(
        weight=tensors["weight"],
        mu_x=tensors["mu_x"],
        mu_y=tensors["mu_y"],
        meta=meta,
        scale=scale,
        layer_scales=layer_scales,
        target_token_rms=target_rms,
        source_token_rms=source_rms,
        pooled_weight=tensors.get("pooled_weight"),
        pooled_mu_x=tensors.get("pooled_mu_x"),
        pooled_mu_y=tensors.get("pooled_mu_y"),
        target_prefix=tensors.get("target_prefix"),
        target_suffix=tensors.get("target_suffix"),
    )


def pack_keys_dir() -> Path:
    """Bundled `keys/` folder inside ComfyUI-PlatoKeys."""
    return Path(__file__).resolve().parent.parent / "keys"


def search_dirs() -> list[Path]:
    """Ordered list of directories scanned for `.safetensors` PlatoKey files.

    Checks `ComfyUI/models/plato_keys/` (registered via `folder_paths` when running inside ComfyUI)
    first, followed by the pack's local `keys/` folder.
    """
    dirs: list[Path] = []
    try:
        import folder_paths

        if FOLDER_NAME not in folder_paths.folder_names_and_paths:
            models_sub = os.path.join(folder_paths.models_dir, FOLDER_NAME)
            os.makedirs(models_sub, exist_ok=True)
            folder_paths.add_model_folder_path(FOLDER_NAME, models_sub)
        for p in folder_paths.get_folder_paths(FOLDER_NAME):
            path_obj = Path(p)
            if path_obj not in dirs:
                dirs.append(path_obj)
    except ImportError:
        pass

    local_dir = pack_keys_dir()
    if local_dir not in dirs:
        dirs.append(local_dir)
    return dirs


def discover_keys() -> dict[str, Path]:
    """Map display names (relative `.safetensors` paths) to resolved absolute paths."""
    found: dict[str, Path] = {}
    for root in search_dirs():
        if not root.is_dir():
            continue
        for file_path in sorted(root.rglob(f"*{KEY_EXTENSION}")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(root).as_posix()
            if rel not in found:
                found[rel] = file_path
    return found


def resolve_key_path(name: str) -> Path:
    """Resolve a dropdown key name to a file inside allowed PlatoKey directories.

    Rejects path traversal (`..`, absolute paths outside allowed dirs) so shared workflows cannot
    read arbitrary files off disk.
    """
    if not name or name == "(no keys found)":
        raise FileNotFoundError(
            "No PlatoKey `.safetensors` files found. Place calibrated keys in "
            "`ComfyUI/models/plato_keys/` or `ComfyUI-PlatoKeys/keys/` and refresh."
        )
    cleaned = str(name).strip().replace("\\", "/")
    if os.path.isabs(cleaned) or ".." in cleaned.split("/"):
        raise ValueError(f"Invalid PlatoKey name: {name!r}")

    keys = discover_keys()
    if cleaned in keys:
        return keys[cleaned]

    # Also allow matching by bare filename if unambiguous.
    matches = [p for rel, p in keys.items() if Path(rel).name == cleaned]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"PlatoKey {name!r} not found in {[str(d) for d in search_dirs()]}. "
        f"Available: {sorted(keys.keys())}"
    )
