"""ComfyUI-PlatoKeys core package: closed-form cross-encoder conditioning alignment."""

from .apply import apply_conditioning, apply_tensor
from .format import (
    FORMAT_VERSION,
    KEY_EXTENSION,
    PlatoKey,
    PlatoKeyMeta,
    load_key,
    save_key,
)
from .math import (
    EvalMetrics,
    FitResult,
    SufficientStats,
    evaluate_key,
    fit_all_methods,
    fit_procrustes,
    fit_procrustes_pca,
    fit_ridge,
)

__all__ = [
    "EvalMetrics",
    "FORMAT_VERSION",
    "FitResult",
    "KEY_EXTENSION",
    "PlatoKey",
    "PlatoKeyMeta",
    "SufficientStats",
    "apply_conditioning",
    "apply_tensor",
    "evaluate_key",
    "fit_all_methods",
    "fit_procrustes",
    "fit_procrustes_pca",
    "fit_ridge",
    "load_key",
    "save_key",
]
