"""Calibration prompt collection for PlatoKeys.

Builds a diverse, representative prompt corpus for calibrating text-encoder alignments (e.g.
Qwen3.5-2B -> Qwen3.5-9B for MOSS-Image):
  1. Streams ungated public image-prompt & caption datasets from Hugging Face (`datasets`).
  2. Mixes in MOSS-Image domain coverage (empty/negative prompts, short/long natural captions,
     tag-style prompts, bilingual EN/ZH & typography prompts, and MOSS JSON layout prompts).
  3. Supports user-supplied `.txt` or `.jsonl` prompt files.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Public, ungated, TEXT-ONLY Hugging Face datasets (no image columns -> streams in seconds).
HF_PROMPT_SOURCES: list[dict[str, str]] = [
    {
        "repo": "Gustavosta/Stable-Diffusion-Prompts",
        "split": "train",
        "column": "Prompt",
        "label": "sd-prompts",
    },
    {
        "repo": "nateraw/parti-prompts",
        "split": "train",
        "column": "Prompt",
        "label": "parti-prompts",
    },
    {
        "repo": "ChristophSchuhmann/MS_COCO_2017_URL_TEXT",
        "split": "train",
        "column": "TEXT",
        "label": "coco-text",
    },
    {
        "repo": "FredZhang7/stable-diffusion-prompts-2.47M",
        "split": "train",
        "column": "text",
        "label": "sd-prompts-2m",
    },
]

# Essential short / negative / empty prompts so CFG unconditional passes stay in-distribution.
NEGATIVE_AND_EDGE_PROMPTS: list[str] = [
    "",
    " ",
    "blurry, low quality, distorted, ugly, watermark, text, signature",
    "bad anatomy, extra fingers, deformed hands, mutated, lowres, jpeg artifacts",
    "oversaturated, overexposed, underexposed, flat lighting, amateur photo",
    "nsfw, nude, gore, violence, low quality, worst quality",
    "watermark, logo, username, border, frame, cropped, out of frame",
    "3d render, cgi, plastic skin, uncanny valley, airbrushed",
    "monochrome, grayscale, washed out, noisy, grainy",
    "duplicate, clone, extra limbs, missing fingers, bad proportions",
    "a photo",
    "abstract background",
    "white background",
    "black background",
    "minimalist geometric shapes",
]

# Synthetic MOSS-Image coverage: typography, bilingual EN/ZH, booru tags, and layout JSON prompts.
_TYPOGRAPHY_TEMPLATES = [
    'A minimalist graphic design poster with bold white typography reading "{text}" centered on a {color} background, Swiss modernist grid layout.',
    'A glowing neon sign mounted on a wet brick wall at night spelling "{text}" in {color} light, cinematic rain reflections, 35mm photography.',
    'A vintage hardcover book cover titled "{text}" embossed in gold foil on textured {color} linen cloth, studio lighting.',
    'A wooden cafe chalkboard menu outside a sunny street shop with hand-lettered chalk writing "{text}", shallow depth of field.',
    'A street-level photograph of a storefront banner in {color} and white displaying "{text}", crisp afternoon sunlight.',
]
_TYPOGRAPHY_TEXTS = [
    "PLATO'S KEY",
    "OPEN 24 HOURS",
    "COFFEE & BOOKS",
    "TOKYO DRIFT 1989",
    "HELLO WORLD",
    "AUTUMN FESTIVAL",
    "MIDNIGHT EXPRESS",
    "SILENT GARDEN",
    "QUANTUM MECHANICS",
    "FRESH BAKERY",
    "山海之间",
    "深夜食堂",
    "未来都市 2026",
    "春日茶馆",
    "清风明月",
    "MOSS IMAGE STUDIO",
    "LIMITED EDITION",
    "WARNING: HIGH VOLTAGE",
    "GRAND OPENING",
    "NO PARKING ANYTIME",
]
_COLORS = [
    "cobalt blue",
    "crimson red",
    "emerald green",
    "warm amber",
    "matte black",
    "ivory cream",
    "terracotta orange",
    "deep indigo",
]

_TAG_GROUPS = [
    "1girl, solo, silver hair, amber eyes, winter coat, snowing, city street at dusk, bokeh, cinematic lighting, masterpiece, detailed",
    "1boy, cyberpunk jacket, neon alley, rain droplets, wet pavement, reflective visor, volumetric fog, anime style, high contrast",
    "landscape, mountain lake, autumn foliage, wooden dock, morning mist, golden hour, ultra detailed, photorealistic, 8k",
    "still life, ceramic teapot, matcha cup, bamboo tray, tatami mat, soft window light, shallow depth of field, film grain",
    "mecha, weathered white armor, hangar bay, maintenance scaffolding, sparks, dramatic rim lighting, concept art, sharp focus",
    "cute cat, tortoiseshell fur, curled up on knitted blanket, sunlit windowsill, dust motes, cozy atmosphere, candid photo",
    "fantasy castle, floating island, waterfalls, twilight sky, glowing runes, painterly brushwork, oil on canvas, epic scale",
    "vintage sports car, metallic silver, coastal highway, sunset clouds, motion blur, automotive photography, wide angle lens",
    "botanical illustration, pressed fern leaves, watercolor wash, cream parchment paper, fine ink outlines, scientific plate",
    "cyberpunk food stall, steam rising, hanging paper lanterns, bilingual signage, night market, crowded alley, 50mm lens",
    "portrait, elderly craftsman, wrinkled hands, woodworking workshop, wood shavings, warm tungsten lamp, Leica monochrome",
    "isometric diorama, tiny cozy bookstore, glowing windows, cobblestone sidewalk, autumn leaves, miniature tilt-shift style",
]

_ZH_PROMPTS = [
    "一幅中国水墨画风格的江南古镇，细雨蒙蒙，白墙黛瓦倒映在平静的河面上，远处有一叶扁舟和石拱桥。",
    "赛博朋克风格的重庆夜景，错落有致的立体高楼，霓虹灯牌写着“未来已来”，雨后的街道泛着五彩倒影，电影感构图。",
    "清晨阳光透过落地窗洒在原木书桌上，一杯冒着热气的拿铁咖啡旁边放着翻开的笔记本和钢笔，日系清新摄影风格。",
    "一只橘猫慵懒地趴在古老红砖墙头晒太阳，背景是盛开的粉色樱花树，浅景深，温暖胶片色调。",
    "身穿汉服的少女手持油纸伞站在竹林小径中，微风吹动衣袂，薄雾缭绕，唯美古风插画，细节丰富。",
    "现代极简室内建筑摄影，大面积清水混凝土墙面搭配温润橡木家具，斜射的自然光影勾勒出几何层次感。",
    "宇航员独自站在火星赤红荒漠的山脊上，抬头仰望星云密布的深邃夜空，科幻概念艺术，史诗级宏大视角。",
    "传统中式茶室特写，紫砂壶正往青瓷茶杯中倾倒琥珀色茶汤，袅袅茶烟升腾，暗调静物布光。",
]

_LAYOUT_SCENES = [
    (
        "A cozy living room with a red velvet sofa, a round wooden coffee table, and a tall floor lamp beside a window.",
        [
            {"label": "red velvet sofa", "bbox": [360, 120, 740, 680]},
            {"label": "round wooden coffee table", "bbox": [650, 260, 880, 620]},
            {"label": "tall brass floor lamp", "bbox": [180, 720, 780, 880]},
        ],
    ),
    (
        "A flat-lay product photograph of a black mirrorless camera, leather strap, and two lenses arranged on light grey stone.",
        [
            {"label": "black mirrorless camera", "bbox": [200, 180, 620, 560]},
            {"label": "prime lens", "bbox": [240, 620, 520, 860]},
            {"label": "woven leather strap", "bbox": [640, 220, 880, 800]},
        ],
    ),
    (
        "A sunny botanical greenhouse interior with terracotta potted ferns on a wooden bench and hanging glass planters overhead.",
        [
            {"label": "hanging glass planters", "bbox": [40, 160, 320, 840]},
            {"label": "terracotta potted ferns", "bbox": [420, 120, 760, 540]},
            {"label": "weathered wooden bench", "bbox": [680, 80, 940, 920]},
        ],
    ),
    (
        "A street photography shot of a cyclist passing a bright yellow kiosk under rain-soaked trees.",
        [
            {"label": "bright yellow street kiosk", "bbox": [220, 80, 780, 420]},
            {"label": "cyclist in a dark raincoat", "bbox": [380, 480, 860, 820]},
            {"label": "wet overhanging tree branches", "bbox": [20, 200, 300, 960]},
        ],
    ),
]


def build_domain_prompts(seed: int = 42) -> list[str]:
    """Deterministic set of MOSS-relevant prompts (empty/negative, typography, tags, ZH, layout JSON)."""
    rng = random.Random(seed)
    prompts: list[str] = list(NEGATIVE_AND_EDGE_PROMPTS)

    for text in _TYPOGRAPHY_TEXTS:
        tpl = rng.choice(_TYPOGRAPHY_TEMPLATES)
        color = rng.choice(_COLORS)
        prompts.append(tpl.format(text=text, color=color))

    prompts.extend(_TAG_GROUPS)
    prompts.extend(_ZH_PROMPTS)

    for caption, objects in _LAYOUT_SCENES:
        prompts.append(json.dumps({"caption": caption, "objects": objects}, ensure_ascii=False))

    return prompts


def load_prompts_from_file(path: str | Path) -> list[str]:
    """Load prompts from a `.txt` (one per line) or `.jsonl` (`prompt`/`text`/`caption` field) file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    prompts: list[str] = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    prompts.append(obj)
                elif isinstance(obj, dict):
                    for key in ("prompt", "Prompt", "text", "caption"):
                        if key in obj and isinstance(obj[key], str):
                            prompts.append(obj[key])
                            break
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    prompts.append(s)
    return prompts


def stream_hf_prompts(
    target_count: int,
    *,
    sources: list[dict[str, str]] | None = None,
    min_chars: int = 10,
    max_chars: int = 1200,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Stream up to `target_count` unique prompts from ungated Hugging Face datasets.

    Returns `(prompts, used_source_labels)`. Gracefully falls back across sources if any single
    dataset is unreachable.
    """
    from datasets import load_dataset

    src_list = sources or HF_PROMPT_SOURCES
    per_source = max(1, (target_count + len(src_list) - 1) // len(src_list))
    seen: set[str] = set()
    collected: list[str] = []
    used_labels: list[str] = []

    for src in src_list:
        if len(collected) >= target_count:
            break
        repo = src["repo"]
        split = src.get("split", "train")
        col = src["column"]
        label = src.get("label", repo)
        want = min(per_source, target_count - len(collected))
        try:
            print(f"[datasets] streaming up to {want} prompts from {repo}...", flush=True)
            ds = load_dataset(repo, split=split, streaming=True)
            if hasattr(ds, "shuffle"):
                ds = ds.shuffle(seed=seed, buffer_size=512)
            got = 0
            for row in ds:
                val = row.get(col)
                if not isinstance(val, str):
                    continue
                text = " ".join(val.split())
                if len(text) < min_chars or len(text) > max_chars:
                    continue
                if text in seen:
                    continue
                seen.add(text)
                collected.append(text)
                got += 1
                if got >= want:
                    break
            if got > 0:
                used_labels.append(f"{label}({got})")
                print(f"[datasets]   got {got} from {label}", flush=True)
        except Exception as exc:
            print(f"[datasets] skipping {repo}: {exc}", flush=True)

    # Top up from the first working source if an earlier one had fewer rows than `per_source`.
    if len(collected) < target_count:
        for src in src_list:
            if len(collected) >= target_count:
                break
            try:
                ds = load_dataset(src["repo"], split=src.get("split", "train"), streaming=True)
                col = src["column"]
                for row in ds:
                    val = row.get(col)
                    if not isinstance(val, str):
                        continue
                    text = " ".join(val.split())
                    if len(text) < min_chars or len(text) > max_chars or text in seen:
                        continue
                    seen.add(text)
                    collected.append(text)
                    if len(collected) >= target_count:
                        break
            except Exception:
                continue

    return collected, used_labels


def gather_calibration_prompts(
    num_prompts: int = 3000,
    *,
    prompts_file: str | Path | None = None,
    include_domain_mix: bool = True,
    seed: int = 42,
) -> tuple[list[str], str]:
    """Gather `num_prompts` diverse calibration prompts and return `(prompts, dataset_summary)`."""
    rng = random.Random(seed)
    prompts: list[str] = []
    tags: list[str] = []

    if include_domain_mix:
        dom = build_domain_prompts(seed=seed)
        prompts.extend(dom)
        tags.append(f"moss-domain({len(dom)})")

    remaining = max(0, num_prompts - len(prompts))
    if prompts_file is not None:
        local = load_prompts_from_file(prompts_file)
        rng.shuffle(local)
        take = local[:remaining] if remaining > 0 else []
        prompts.extend(take)
        tags.append(f"file:{Path(prompts_file).name}({len(take)})")
    elif remaining > 0:
        hf_prompts, hf_labels = stream_hf_prompts(remaining, seed=seed)
        prompts.extend(hf_prompts)
        tags.extend(hf_labels)

    # Deduplicate while preserving empty prompt and order, then deterministically shuffle.
    seen: set[str] = set()
    unique: list[str] = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    rng.shuffle(unique)
    return unique[:num_prompts], "+".join(tags) if tags else "custom"
