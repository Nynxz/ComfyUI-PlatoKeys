"""ComfyUI V3 nodes for ComfyUI-PlatoKeys.

Three composable nodes:
  - `Load Plato Key`: loads a `.safetensors` key from `ComfyUI/models/plato_keys/` or the pack's
    `keys/` folder, caching it by path + mtime and exposing a readable metadata/metrics summary.
  - `Plato Key Encode`: drop-in replacement for `CLIPTextEncode` that honors the key's calibrated
    layer and chat template (so stock `CLIPLoader` + `qwen3.5_2b` works out of the box for
    MOSS-Image without modifying ComfyUI core), then projects conditioning into target space.
  - `Apply Plato Key`: transforms any existing `CONDITIONING` wire directly (for custom encode
    pipelines, region/layout nodes, or encoders whose stock `CLIPTextEncode` already matches).
"""

from __future__ import annotations

import os
from functools import lru_cache

from comfy_api.latest import ComfyExtension, io

from .apply import NORM_MATCH_MODES, NORM_MATCH_OFF, apply_conditioning, compare_conditioning
from .encode import encode_with_key
from .format import PlatoKey, discover_keys, load_key, resolve_key_path

CATEGORY = "conditioning/plato keys"
NAMESPACE = "plato_keys"
NO_KEYS_LABEL = "(no keys found)"

PlatoKeyType = io.Custom("PLATO_KEY")


@lru_cache(maxsize=8)
def _load_cached(path_str: str, mtime_ns: int) -> PlatoKey:
    del mtime_ns  # part of cache key for automatic reload on overwrite
    return load_key(path_str)


def _load_by_name(key_name: str) -> PlatoKey:
    path = resolve_key_path(key_name)
    stat = os.stat(path)
    return _load_cached(str(path), stat.st_mtime_ns)


def _available_key_names() -> list[str]:
    names = sorted(discover_keys().keys())
    return names if names else [NO_KEYS_LABEL]


class PlatoKeyLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = _available_key_names()
        return io.Schema(
            node_id=f"{NAMESPACE}.Load",
            display_name="Load Plato Key",
            category=CATEGORY,
            description=(
                "Loads a closed-form PlatoKey (.safetensors) cross-encoder alignment map from "
                "ComfyUI/models/plato_keys/ or ComfyUI-PlatoKeys/keys/."
            ),
            inputs=[
                io.Combo.Input(
                    "key_name",
                    options=options,
                    default=options[0],
                    tooltip="Calibrated .safetensors PlatoKey file.",
                ),
            ],
            outputs=[
                PlatoKeyType.Output(display_name="plato_key"),
                io.String.Output(
                    display_name="info",
                    tooltip="Source/target models, method, dimensions, and held-out calibration metrics.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, key_name: str) -> str:
        try:
            path = resolve_key_path(key_name)
            st = os.stat(path)
            return f"{path}:{st.st_mtime_ns}:{st.st_size}"
        except Exception:
            return str(key_name)

    @classmethod
    def execute(cls, key_name: str) -> io.NodeOutput:
        key = _load_by_name(key_name)
        return io.NodeOutput(key, key.summary())


class PlatoKeyEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"{NAMESPACE}.Encode",
            display_name="Plato Key Encode",
            category=CATEGORY,
            description=(
                "Encodes a prompt with the source CLIP/text encoder using the key's calibrated "
                "hidden layer and chat template, then projects the conditioning into the target "
                "encoder's space (e.g. Qwen3.5-2B -> Qwen3.5-9B for MOSS-Image)."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="Source text encoder (e.g. qwen3.5_2b_bf16 loaded with stock CLIPLoader).",
                ),
                PlatoKeyType.Input("plato_key"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
                io.Float.Input(
                    "contrast",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    advanced=True,
                    tooltip="Scales deviation from the target mean mu_y. 1.0 = exact calibrated map.",
                ),
                io.Float.Input(
                    "scale_multiplier",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    advanced=True,
                    tooltip="Global output magnitude multiplier. 1.0 = exact calibrated scale.",
                ),
                io.Combo.Input(
                    "norm_match",
                    options=list(NORM_MATCH_MODES),
                    default=NORM_MATCH_OFF,
                    advanced=True,
                    tooltip="Optional norm correction: 'off' uses the calibrated map verbatim; "
                    "'target_rms' matches overall RMS to the calibrated target RMS; "
                    "'preserve_source_norm' keeps each token's input L2 norm.",
                ),
                io.Boolean.Input(
                    "honor_key_template",
                    default=True,
                    advanced=True,
                    tooltip="Format the prompt with the chat template stored in the key metadata "
                    "(e.g. MOSS-Image's open-<think> template). Leave enabled for MOSS-Image.",
                ),
                io.Boolean.Input(
                    "honor_key_layer",
                    default=True,
                    advanced=True,
                    tooltip="Extract the exact hidden layer stored in the key metadata (e.g. "
                    "'last' for MOSS-Image instead of Qwen3.5's default layer -2).",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        plato_key: PlatoKey,
        text: str,
        contrast: float = 1.0,
        scale_multiplier: float = 1.0,
        norm_match: str = NORM_MATCH_OFF,
        honor_key_template: bool = True,
        honor_key_layer: bool = True,
    ) -> io.NodeOutput:
        conditioning = encode_with_key(
            clip,
            text,
            plato_key,
            contrast=contrast,
            scale_multiplier=scale_multiplier,
            norm_match=norm_match,
            honor_key_template=honor_key_template,
            honor_key_layer=honor_key_layer,
        )
        return io.NodeOutput(conditioning)


class PlatoKeyApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"{NAMESPACE}.Apply",
            display_name="Apply Plato Key",
            category=CATEGORY,
            description=(
                "Projects an existing CONDITIONING wire from a source text encoder's space into "
                "a target text encoder's space using a loaded PlatoKey."
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                PlatoKeyType.Input("plato_key"),
                io.Float.Input(
                    "contrast",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    advanced=True,
                    tooltip="Scales deviation from the target mean mu_y. 1.0 = exact calibrated map.",
                ),
                io.Float.Input(
                    "scale_multiplier",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    advanced=True,
                    tooltip="Global output magnitude multiplier. 1.0 = exact calibrated scale.",
                ),
                io.Combo.Input(
                    "norm_match",
                    options=list(NORM_MATCH_MODES),
                    default=NORM_MATCH_OFF,
                    advanced=True,
                    tooltip="Optional norm correction after projection.",
                ),
                io.Boolean.Input(
                    "transform_pooled",
                    default=True,
                    advanced=True,
                    tooltip="Also transform pooled_output when the key includes a pooled map.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning,
        plato_key: PlatoKey,
        contrast: float = 1.0,
        scale_multiplier: float = 1.0,
        norm_match: str = NORM_MATCH_OFF,
        transform_pooled: bool = True,
    ) -> io.NodeOutput:
        mapped = apply_conditioning(
            conditioning,
            plato_key,
            contrast=contrast,
            scale_multiplier=scale_multiplier,
            norm_match=norm_match,
            transform_pooled=transform_pooled,
        )
        return io.NodeOutput(mapped)


class PlatoKeyCompare(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"{NAMESPACE}.Compare",
            display_name="Compare Conditioning (Plato Key)",
            category=CATEGORY,
            description=(
                "Diagnostic node: compares a PlatoKey-mapped CONDITIONING against a native target "
                "CONDITIONING on the same prompt (shape, RMS, attention-sink token 0, per-token cosine, rel L2)."
            ),
            inputs=[
                io.Conditioning.Input("cond_mapped"),
                io.Conditioning.Input("cond_target"),
            ],
            outputs=[
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, cond_mapped, cond_target) -> io.NodeOutput:
        report = compare_conditioning(cond_mapped, cond_target)
        print(f"[PlatoKeys Compare]\n{report}")
        return io.NodeOutput(report)


NODES = [PlatoKeyLoad, PlatoKeyEncode, PlatoKeyApply, PlatoKeyCompare]


class PlatoKeysExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODES


async def comfy_entrypoint() -> ComfyExtension:
    return PlatoKeysExtension()
