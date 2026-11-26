"""ComfyUI-PlatoKeys entry point (V3).

Closed-form cross-encoder conditioning alignment inspired by Stanford Hazy Research's
"Plato's Key" — rotate/project one text encoder's hidden space into another's without
gradient descent or retraining the diffusion model.
"""

if __package__:
    from .plato_keys.nodes import PlatoKeysExtension, comfy_entrypoint  # noqa: F401

    __all__ = ["PlatoKeysExtension", "comfy_entrypoint"]

