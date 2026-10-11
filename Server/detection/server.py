"""
Object Detection Server - Multi-Transport Support

Supports three transport protocols:
1. HTTP (FastAPI) - Traditional REST API, reliable
2. UDP - Low-latency binary protocol for real-time detection
3. WebSocket - Reliable streaming with connection persistence

YOLO Optimizations:
- Lazy model loading with CUDA acceleration
- FP16 precision on GPU for faster inference
- Model warmup for optimized CUDA kernels
"""

import re
import os
import sys
import warnings

# Suppress warnings before imports
warnings.filterwarnings("ignore", message=".*config attributes.*")
warnings.filterwarnings("ignore", message=".*Keyword arguments.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import io
import base64
import json
import argparse
import asyncio
import socket
import struct
import threading
import time
import logging
from PIL import Image
from typing import Optional, List, Dict, Set
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ============================================================================
# MASK FORMAT OPTIONS
# ============================================================================
class MaskFormat(str, Enum):
    NONE = "none"           # No mask data
    RASTER = "raster"       # Raw grayscale bitmap (original, large)
    POLYGON = "polygon"     # Vector polygon points (small, efficient)
    RLE = "rle"             # Run-length encoded (medium, lossless)

# ============================================================================
# LAZY MODEL LOADING - Optimized for CUDA
# ============================================================================
_yolo_model = None
_yolo_model_name = None
_yoloe_model = None
_yoloe_model_name = None
_rfdetr_model = None
_device = None

# Model names as given on the command line, kept so a runtime detector switch reloads the
# SAME weights instead of load_model's built-in defaults. Set in the __main__ block; the
# None defaults keep the module importable (self-tests, tooling) without it.
_yolo_model_name_arg = None
_yoloe_model_name_arg = None

# Thread pool for async operations
_executor = ThreadPoolExecutor(max_workers=2)

# Global model reference (set at startup) - maintained for backwards compatibility
active_model = None
active_model_type = None

# ============================================================================
# OPEN-VOCABULARY STATE (YOLOE)
# ============================================================================
# The active per-shader vocabulary. For an open-vocabulary detector this list is
# pushed by the Unity client whenever a shader becomes active (set-once-per-shader,
# never per frame). A detection's label_id is the SLOT INDEX into this list, which
# the generated shader filters against - fully decoupled from any fixed dataset.
_active_vocabulary: List[str] = []
_active_vocabulary_confidence: float = 0.25
_applied_vocabulary_key: Optional[str] = None  # guards redundant text-embedding recompute
_vocabulary_status: str = "none"  # none | empty | applied | not_open_vocab | error
_vocabulary_lock = threading.Lock()

# {model class index -> shader slot} for the CLOSED-vocabulary detector, rebuilt whenever the
# vocabulary or the detector changes. Precomputed so the per-frame path is a dict lookup
# instead of string matching. Empty while yoloe is active (it needs no translation).
_coco_slot_map: dict = {}

# {YOLOE prompt index -> shader slot}. YOLOE gets each slot's phrase PLUS that slot's COCO
# name from @detect-coco as a second prompt for the SAME slot: short category names are what
# it was trained on, and a descriptive phrase can miss completely ("computer monitor" finds
# nothing on yoloe-26x, "tv" scores 0.9 on the same TV). Rebuilt with the vocabulary.
_yoloe_prompt_slots: List[int] = []

# The shader's own @detect-coco list, as last sent by the client. Kept so an internal
# re-apply (detector switch -> load_model -> apply_vocabulary) does not silently fall back to
# name matching: those calls pass coco_classes=None, which means "keep what you have".
_active_coco_classes: Optional[List[str]] = None


# Fraction by which segmentation mask crop-boxes are expanded before cropping.
# ultralytics crops instance masks tightly to the detection box (ops.crop_mask),
# which shaves off object parts that extend just past the (often slightly tight) box
# -> straight "cut" edges. Expanding the crop box recovers that silhouette from the
# prototype masks. Too large risks bleed from neighbouring instances.
_mask_crop_margin: float = 0.15
_mask_crop_patched: bool = False


def install_mask_crop_expansion(margin: float):
    """Monkeypatch ultralytics' crop_mask to expand the crop box by `margin`.

    process_mask() resolves crop_mask via the module global at call time, so replacing
    ops.crop_mask affects the live predict path (verified). Idempotent.
    """
    global _mask_crop_patched
    if _mask_crop_patched or margin <= 0:
        return

    import torch
    import ultralytics.utils.ops as ops

    def expanded_crop(masks, boxes):
        if boxes.device != masks.device:
            boxes = boxes.to(masks.device)
        _, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
        bw = x2 - x1
        bh = y2 - y1
        mx = bw * margin
        my = bh * margin
        x1 = (x1 - mx).clamp(min=0)
        y1 = (y1 - my).clamp(min=0)
        x2 = (x2 + mx).clamp(max=w)
        y2 = (y2 + my).clamp(max=h)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    ops.crop_mask = expanded_crop
    _mask_crop_patched = True
    logger.info(f"Installed expanded mask crop (margin={margin}) to reduce mask edge clipping.")


def get_yoloe_model(model_name: str = "yoloe-26x-seg.pt"):
    """
    Lazy-load a YOLOE open-vocabulary segmentation model.

    YOLOE accepts text prompts via set_classes(): the (expensive) text-embedding
    pass runs only when the vocabulary changes, after which inference runs at full
    YOLO speed. Predictions' class indices are positions into the prompt list, i.e.
    they ARE the slot index expected by the shader.
    """
    global _yoloe_model, _yoloe_model_name, _device

    if _yoloe_model is not None and _yoloe_model_name == model_name:
        return _yoloe_model

    import torch
    from ultralytics import YOLOE

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path, model_format = _find_optimized_model(model_name)
    logger.info(f"Loading YOLOE model: {model_path} ({model_format}) on {_device}")
    _yoloe_model = YOLOE(model_path)
    _yoloe_model_name = model_name

    if _device == "cuda" and model_format == "PyTorch":
        _yoloe_model.to("cuda")
        logger.info("YOLOE model moved to CUDA")

    logger.info(f"YOLOE model loaded successfully ({model_format})")
    return _yoloe_model


def apply_vocabulary(classes: List[str], confidence: Optional[float] = None,
                     coco_classes: Optional[List[str]] = None) -> List[str]:
    """
    Set the active open-vocabulary target classes (and optional confidence).

    Applies the text prompts to the YOLOE model only when the list actually changed
    (the text-embedding pass is the costly part). Returns the normalized class list.
    """
    global _active_vocabulary, _active_vocabulary_confidence, _applied_vocabulary_key, _vocabulary_status, _coco_slot_map, _active_coco_classes, _yoloe_prompt_slots

    cleaned = [c.strip() for c in (classes or []) if c and c.strip()]

    with _vocabulary_lock:
        # None = internal re-apply, keep the stored list. A real client always sends the field
        # (empty list when the shader has no @detect-coco), so an actual shader change always
        # replaces it and can never inherit the previous shader's mapping.
        if coco_classes is not None:
            _active_coco_classes = list(coco_classes)

        if confidence is not None and confidence >= 0:
            _active_vocabulary_confidence = float(confidence)

        _active_vocabulary = cleaned
        prompts, prompt_slots = build_yoloe_prompts(cleaned, _active_coco_classes)
        key = "".join(prompts)

        if active_model_type == "yoloe" and active_model is not None:
            if not cleaned:
                _vocabulary_status = "empty"
            elif key == _applied_vocabulary_key:
                _vocabulary_status = "applied"  # the model still holds exactly these prompts
            else:
                try:
                    active_model.set_classes(prompts, active_model.get_text_pe(prompts))
                    _yoloe_prompt_slots = prompt_slots
                    _vocabulary_status = "applied"
                    logger.info(f"YOLOE vocabulary applied ({len(cleaned)} slots): {prompts} -> slots {prompt_slots}")
                except Exception as e:
                    _vocabulary_status = "error"
                    logger.error(
                        f"Failed to apply YOLOE vocabulary (set_classes/get_text_pe): {e}. "
                        f"Common causes: dtype mismatch from half-precision inference (YOLOE "
                        f"must run FP32), or MobileCLIP text encoder unavailable "
                        f"('mobileclip2_b.ts' download blocked / no internet)."
                    )
                _applied_vocabulary_key = key
            _coco_slot_map = {}  # open vocab needs no translation
        else:
            # Closed vocabulary (yolo26-seg on COCO): the model cannot be told what to look
            # for, so the translation happens here — ONCE per vocabulary change, never per
            # frame. build_coco_slot_map logs what mapped and warns about what did not.
            _coco_slot_map = build_coco_slot_map(
                cleaned, _closed_vocabulary_class_names(), _active_coco_classes)
            _vocabulary_status = "coco_mapped" if _coco_slot_map else (
                "empty" if not cleaned else "no_coco_match"
            )

    return cleaned


def _closed_vocabulary_class_names() -> dict:
    """
    {class index -> name} of the closed detector. Prefer the LOADED model's own names — an
    ultralytics model is the authority on its own indices — and fall back to the static COCO
    table when no model is loaded yet (a vocabulary can arrive before the first switch).
    """
    names = getattr(active_model, "names", None) if active_model_type != "yoloe" else None
    if isinstance(names, dict) and names:
        return names
    if isinstance(names, (list, tuple)) and names:
        return {i: n for i, n in enumerate(names)}
    return COCO_ID_TO_NAME


def get_active_inference_model():
    """Return the active detection model, loading lazily if needed."""
    if active_model is not None:
        return active_model
    return get_yolo_model()


# Detectors the client is allowed to switch to at runtime. rfdetr is deliberately absent:
# it produces no masks, which the shader pipeline requires.
SWITCHABLE_DETECTORS = ("yoloe", "yolo")


def switch_detector(model_type: str) -> bool:
    """
    Switch the active detector at runtime (open-vocabulary YOLOE <-> closed-vocabulary
    yolo26-seg). Returns True when the active detector changed.

    Cheap after the first switch: get_yoloe_model/get_yolo_model cache their model in a
    module global, so flipping back and forth only re-points active_model. The FIRST
    switch to yoloe still pays the MobileCLIP text-encoder load (~242MB on a cold cache).

    load_model() re-applies the current vocabulary at the end, so the shader's classes
    survive the switch without the client having to re-send them.
    """
    global active_model_type

    model_type = (model_type or "").strip().lower()
    if model_type not in SWITCHABLE_DETECTORS:
        logger.warning(
            f"Ignoring detector switch to '{model_type}' (allowed: {', '.join(SWITCHABLE_DETECTORS)})"
        )
        return False

    if model_type == active_model_type:
        return False

    previous = active_model_type
    try:
        load_model(model_type, yolo_model=_yolo_model_name_arg, yoloe_model=_yoloe_model_name_arg)
    except Exception as e:
        # Keep serving with the previous detector rather than dropping to no model at all.
        logger.error(f"Detector switch to '{model_type}' FAILED, staying on '{previous}': {e}")
        active_model_type = previous
        return False

    logger.info(f"Detector switched: {previous} -> {active_model_type}")
    return True


# Synonyms for the closed COCO vocabulary. A generated shader asks in natural language
# ("television screen", "computer monitor"), COCO answers in 80 terse labels ("tv"), and the
# two rarely share a substring — matching them textually is what silently produced ZERO
# detections in COCO mode. This table is the bridge, and it is meant to be EDITED: adding a
# phrase here is what lets an already-generated shader work under yolo26-seg without being
# regenerated.
#
# Key = COCO class name, value = phrases that should resolve to it. Matching is word-based,
# so "monitor" already covers "the big monitor" and "computer monitor" — only genuinely
# different words need an entry.
COCO_SYNONYMS = {
    "tv": ["television", "tv screen", "television screen", "monitor", "computer monitor",
           "computer screen", "desktop monitor", "display", "screen", "smart tv", "flatscreen",
           "flat screen"],
    "laptop": ["notebook", "macbook", "laptop computer", "portable computer", "netbook"],
    "cell phone": ["phone", "smartphone", "mobile phone", "mobile", "iphone", "android phone"],
    "cup": ["mug", "coffee cup", "coffee mug", "teacup", "tea cup"],
    "couch": ["sofa", "settee"],
    "potted plant": ["plant", "houseplant", "pot plant"],
    "dining table": ["table", "desk", "worktable"],
    "keyboard": ["computer keyboard"],
    "mouse": ["computer mouse"],
    "remote": ["remote control", "tv remote"],
    "book": ["notebook paper", "textbook"],
    "bottle": ["water bottle", "plastic bottle"],
    "person": ["human", "people", "man", "woman", "someone"],
    "chair": ["seat", "office chair", "stool"],
    "backpack": ["rucksack", "bag"],
    "refrigerator": ["fridge"],
    "microwave": ["microwave oven"],
    "tvmonitor": [],  # some COCO exports use this spelling; keeps the lookup total
}


def _words(text: str) -> set:
    """Lowercase word set. Word-based beats substring: it matches 'cup' inside 'red coffee
    cup' while refusing 'car' inside 'cardboard'."""
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _phrase_matches(phrase: str, wanted_words: set) -> bool:
    """True when every word of `phrase` appears in the requested phrase."""
    pw = _words(phrase)
    return bool(pw) and pw <= wanted_words


def build_coco_slot_map(vocabulary, class_names, coco_classes=None) -> dict:
    """
    Precompute {model class index -> shader slot} for the closed-vocabulary detector.

    Done ONCE per vocabulary change rather than per detection per frame: the previous version
    ran string matching inside the hot loop, which is exactly what this project must not do.

    A COCO class can serve only ONE slot — a detection carries a single label_id on the wire —
    so when several requested phrases resolve to the same class, the EARLIEST slot wins. That
    is a real limitation and the log distinguishes it from a genuine miss, because the two need
    opposite fixes: a miss wants a new synonym, a collision wants the shader to merge the two
    phrases into one class.
    """
    class_names = class_names or {}
    vocabulary = list(vocabulary or [])

    # The LLM's own mapping, written into the shader header as @detect-coco when it was
    # generated. It beats any runtime guess: the model knew the intent of the shader, saw all
    # slots at once, and could avoid two slots needing the same COCO class. Only accepted when
    # it lines up slot-for-slot AND names a class this detector actually has — a hallucinated
    # or shifted list would point slots at the wrong objects, which is worse than guessing.
    by_name = {(n or "").strip().lower(): i for i, n in class_names.items()}
    explicit = {}
    if coco_classes and len(coco_classes) == len(vocabulary):
        for slot, coco_name in enumerate(coco_classes):
            key = (coco_name or "").strip().lower()
            if not key:
                continue  # the LLM said "no COCO equivalent" for this slot
            if key in by_name:
                explicit[slot] = by_name[key]
            else:
                logger.warning(
                    f"COCO mapping: shader asked for COCO class '{coco_name}' (slot {slot}), "
                    f"which this detector does not have — falling back to name matching."
                )
    elif coco_classes:
        logger.warning(
            f"COCO mapping: @detect-coco has {len(coco_classes)} entries but the shader has "
            f"{len(vocabulary)} classes — ignoring it, slots would not line up."
        )

    # Pass 1: everything each slot COULD take, before anyone claims it.
    candidates = {}
    for slot, wanted in enumerate(vocabulary):
        if slot in explicit:
            candidates[slot] = {explicit[slot]}
            continue

        wanted_words = _words(wanted)
        if not wanted_words:
            continue

        wanted_exact = (wanted or "").strip().lower()
        hits = set()
        for cls_id, coco_name in class_names.items():
            name = (coco_name or "").strip().lower()
            if not name:
                continue
            if (
                name == wanted_exact
                # "cup" <= {"red","coffee","cup"}: the COCO label names the requested thing.
                or _phrase_matches(name, wanted_words)
                # "television screen" -> tv, via the synonym table.
                or any(_phrase_matches(syn, wanted_words) for syn in COCO_SYNONYMS.get(name, []))
            ):
                hits.add(cls_id)
        if hits:
            candidates[slot] = hits

    # Pass 2: assign each class to the earliest slot that wants it.
    slot_map = {}
    claimed_by = {}
    for slot in sorted(candidates):
        for cls_id in candidates[slot]:
            if cls_id not in slot_map:
                slot_map[cls_id] = slot
                claimed_by.setdefault(slot, []).append(class_names[cls_id])

    # Pass 3: say precisely what happened, per slot.
    for slot, wanted in enumerate(vocabulary):
        if slot in claimed_by:
            source = "from shader @detect-coco" if slot in explicit else "matched by name"
            logger.info(
                f"COCO mapping: '{wanted}' (slot {slot}) <- {sorted(claimed_by[slot])} ({source})")
        elif slot in candidates:
            taken = sorted({(class_names[c], slot_map[c]) for c in candidates[slot]})
            detail = ", ".join(f"'{n}' already serves slot {s}" for n, s in taken)
            logger.warning(
                f"COCO mapping: '{wanted}' (slot {slot}) stays EMPTY — {detail}. A detection "
                f"carries one slot only, so two phrases cannot share a COCO class. Merge them "
                f"in the shader, or use the open-vocabulary detector to tell them apart."
            )
        elif _words(wanted):
            logger.warning(
                f"COCO mapping: '{wanted}' (slot {slot}) has NO COCO equivalent — this slot "
                f"stays empty while the closed detector is active. Add a synonym to "
                f"COCO_SYNONYMS in server.py, or switch back to the open-vocabulary detector."
            )

    return slot_map


def build_yoloe_prompts(vocabulary, coco_classes=None):
    """
    Prompt list for YOLOE's set_classes() and the slot each prompt reports.

    Every slot keeps its own phrase at its own index, so a shader without @detect-coco behaves
    exactly as before. A valid COCO name from @detect-coco is appended as an extra prompt for
    its slot unless it duplicates a phrase; a COCO name two slots share goes to the earliest,
    the same rule build_coco_slot_map applies. "-" / empty / unknown names add nothing, and a
    list whose length does not line up is ignored (a shifted list would aim at wrong slots).
    """
    prompts = list(vocabulary)
    slots = list(range(len(vocabulary)))
    if not coco_classes or len(coco_classes) != len(vocabulary):
        return prompts, slots

    seen = {p.strip().lower() for p in prompts}
    for slot, coco in enumerate(coco_classes):
        name = (coco or "").strip().lower()
        if name in COCO_NAME_TO_ID and name not in seen:
            seen.add(name)
            prompts.append(name)
            slots.append(slot)
    return prompts, slots


def _agnostic_nms() -> bool:
    """
    Cross-class NMS while a slot has two prompts: the phrase and its COCO alias usually both
    fire on the same object, and two boxes in one slot would apply the shader's effect twice.
    The cost is that two DIFFERENT slots overlapping above the IoU threshold also merge.
    """
    return active_model_type == "yoloe" and len(_yoloe_prompt_slots) > len(_active_vocabulary)


def resolve_label_id(cls: int, names) -> int:
    """
    Map a model class index onto the slot index the SHADER expects.

    Open-vocabulary (yoloe): set_classes() was called with the shader's phrases followed by
    their COCO aliases (build_yoloe_prompts), so the index is looked up in that prompt->slot
    list; for the phrases themselves it is the slot index unchanged.

    Closed-vocabulary (yolo26-seg on COCO): the model's index is a COCO id, unrelated to the
    shader's slot order — a shader asking for "chair, laptop" would otherwise light up on
    COCO 0/1 (person, bicycle). The translation table is built once per vocabulary change by
    build_coco_slot_map(); this is a plain dict lookup on the per-frame path.

    -1 means "this shader did not ask for that object" and the caller drops the detection.
    With no vocabulary set (no shader active) the raw index is returned, which keeps the
    plain COCO behaviour for debugging.
    """
    if active_model_type == "yoloe":
        return _yoloe_prompt_slots[cls] if cls < len(_yoloe_prompt_slots) else cls

    if not _active_vocabulary:
        return cls

    return _coco_slot_map.get(cls, -1)


# COCO class names for reference (yolo26 uses these IDs)
# Key classes: 62=tv, 63=laptop, 41=cup, 39=bottle, 56=chair, 57=couch, etc.

# Mapping from class name to COCO ID (for RF-DETR compatibility)
COCO_NAME_TO_ID = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
    "bus": 5, "train": 6, "truck": 7, "boat": 8, "traffic light": 9,
    "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13,
    "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
    "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23, "backpack": 24,
    "umbrella": 25, "handbag": 26, "tie": 27, "suitcase": 28, "frisbee": 29,
    "skis": 30, "snowboard": 31, "sports ball": 32, "kite": 33, "baseball bat": 34,
    "baseball glove": 35, "skateboard": 36, "surfboard": 37, "tennis racket": 38,
    "bottle": 39, "wine glass": 40, "cup": 41, "fork": 42, "knife": 43,
    "spoon": 44, "bowl": 45, "banana": 46, "apple": 47, "sandwich": 48,
    "orange": 49, "broccoli": 50, "carrot": 51, "hot dog": 52, "pizza": 53,
    "donut": 54, "cake": 55, "chair": 56, "couch": 57, "potted plant": 58,
    "bed": 59, "dining table": 60, "toilet": 61, "tv": 62, "laptop": 63,
    "mouse": 64, "remote": 65, "keyboard": 66, "cell phone": 67, "microwave": 68,
    "oven": 69, "toaster": 70, "sink": 71, "refrigerator": 72, "book": 73,
    "clock": 74, "vase": 75, "scissors": 76, "teddy bear": 77, "hair drier": 78,
    "toothbrush": 79
}

# Reverse mapping: ID to name
COCO_ID_TO_NAME = {v: k for k, v in COCO_NAME_TO_ID.items()}


def _find_optimized_model(base_name: str) -> tuple[str, str]:
    """
    Find the best available model format.
    Priority: TensorRT (.engine) > ONNX (.onnx) > PyTorch (.pt)
    
    Returns (model_path, format_type)
    """
    # Get base name without extension
    base = os.path.splitext(base_name)[0]
    
    # Search directories
    search_dirs = [
        os.getcwd(),
        os.path.dirname(__file__),
        os.path.join(os.path.dirname(__file__), '..'),
    ]
    
    # Priority order: TensorRT > ONNX > PyTorch
    formats = [
        ('.engine', 'TensorRT'),
        ('_fp16.engine', 'TensorRT-FP16'),
        ('.onnx', 'ONNX'),
        ('.pt', 'PyTorch'),
    ]
    
    for search_dir in search_dirs:
        for ext, fmt in formats:
            model_path = os.path.join(search_dir, base + ext)
            if os.path.exists(model_path):
                return model_path, fmt
    
    # Fallback to original name (ultralytics will download if needed)
    return base_name, 'PyTorch'


def get_yolo_model(model_name: str = "yolo26n-seg.pt", use_half: bool = True):
    """
    Lazy-load YOLO segmentation model with automatic optimization.
    
    This provides:
    - Automatic format selection (TensorRT > ONNX > PyTorch)
    - Automatic GPU detection and placement
    - FP16 half precision support
    - Model caching to avoid reloading
    - Warmup for CUDA kernel optimization
    
    Args:
        model_name: Base model name (e.g., "yolo26n-seg.pt")
        use_half: Use FP16 half precision on CUDA (default True)
    """
    global _yolo_model, _yolo_model_name, _device
    
    # Return cached model if same name requested
    if _yolo_model is not None and _yolo_model_name == model_name:
        return _yolo_model
    
    import torch
    from ultralytics import YOLO
    
    # Set device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Find best available model format
    model_path, model_format = _find_optimized_model(model_name)
    
    logger.info(f"Loading YOLO model: {model_path} ({model_format}) on {_device}")
    _yolo_model = YOLO(model_path)
    _yolo_model_name = model_name
    
    # Move to CUDA if available and using PyTorch format
    if _device == "cuda" and model_format == "PyTorch":
        _yolo_model.to('cuda')
        logger.info("YOLO model moved to CUDA")
        
        # Warmup for CUDA kernel optimization
        _warmup_yolo_model(_yolo_model, use_half=use_half)
    elif _device == "cuda":
        # TensorRT/ONNX handle device placement internally
        _warmup_yolo_model(_yolo_model, use_half=use_half)
    
    logger.info(f"YOLO model loaded successfully ({model_format})")
    return _yolo_model


def get_rfdetr_model():
    """Lazy-load RF-DETR model."""
    global _rfdetr_model
    
    if _rfdetr_model is None:
        from inference import get_model
        logger.info("Loading RF-DETR model...")
        _rfdetr_model = get_model("rfdetr-base")
        logger.info("RF-DETR model loaded successfully")
    
    return _rfdetr_model


def _warmup_yolo_model(model, use_half: bool = True):
    """Run warmup inference to optimize CUDA kernels and enable FP16."""
    import torch
    
    half_str = "FP16" if use_half else "FP32"
    logger.info(f"Running YOLO warmup ({half_str})...")
    try:
        # Create dummy image
        dummy_image = Image.new("RGB", (640, 640), color=(128, 128, 128))
        
        # Run warmup iterations with half precision if enabled
        for i in range(3):
            model(dummy_image, verbose=False, half=use_half)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"YOLO warmup complete ({half_str}, 3 iterations)")
    except Exception as e:
        logger.warning(f"YOLO warmup failed: {e}")

# ============================================================================
# UDP SERVER GLOBALS
# ============================================================================
udp_server_running = False
udp_server_socket = None
udp_confidence_threshold = 0.25
udp_mask_format = MaskFormat.POLYGON
udp_max_polygon_points = 64  # Simplify polygons for bandwidth

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def simplify_polygon(points: np.ndarray, max_points: int = 64) -> np.ndarray:
    """
    Simplify a polygon using Douglas-Peucker algorithm to reduce point count.
    This dramatically reduces data size while preserving shape.
    """
    if len(points) <= max_points:
        return points
    
    # Use cv2.approxPolyDP for Douglas-Peucker simplification
    # Start with a small epsilon and increase until we get target point count
    epsilon = 0.001
    for _ in range(20):
        simplified = cv2.approxPolyDP(points.astype(np.float32), epsilon, True)
        if len(simplified) <= max_points:
            return simplified.reshape(-1, 2)
        epsilon *= 1.5
    
    # Fallback: uniform sampling if simplification fails
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indices]


# Fraction of the mask's own size added as a margin around the polygon box, so the
# rasterized mask never touches the cell border (prevents the hard cut-off edges that
# happen when the mask is clamped to the tight YOLO detection box).
MASK_BOX_MARGIN = 0.06


def extract_polygon_from_mask(mask_data: np.ndarray, img_width: int, img_height: int,
                               bbox_norm: tuple, max_points: int = 64):
    """
    Extract a polygon contour from a segmentation mask.

    The polygon is normalized to a box that tightly wraps the MASK CONTOUR (plus a
    small margin) rather than the YOLO detection box. This (a) stops mask edges from
    being clipped to the detection box and (b) makes the mask fill its atlas cell, so
    the effective mask resolution is higher. The adjusted box is returned so the caller
    can update the detection's bounding box to match (shader gate + mask stay aligned).

    Args:
        mask_data: Binary mask array from YOLO
        img_width, img_height: Original image dimensions
        bbox_norm: Normalized detection bounding box (cx, cy, w, h) — fallback only
        max_points: Maximum polygon points after simplification

    Returns:
        (normalized_points, (cx, cy, w, h)) where points are 0-1 relative to the
        returned (mask-tight) box and the box is normalized to the image, or None.
    """
    try:
        # Resize mask to image dimensions
        mask_full = cv2.resize(
            mask_data.astype(np.float32),
            (img_width, img_height),
            interpolation=cv2.INTER_LINEAR
        )

        # Threshold to binary
        mask_binary = (mask_full > 0.5).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) < 3:
            return None

        # Reshape to (N, 2)
        points = largest_contour.reshape(-1, 2).astype(np.float32)

        # Simplify polygon
        points = simplify_polygon(points, max_points)

        # Tight box around the mask contour (pixels), expanded by a small margin and
        # clamped to the image. This replaces clamping to the YOLO detection box.
        min_x = float(points[:, 0].min())
        min_y = float(points[:, 1].min())
        max_x = float(points[:, 0].max())
        max_y = float(points[:, 1].max())

        cw = max(max_x - min_x, 1.0)
        ch = max(max_y - min_y, 1.0)
        pad_x = cw * MASK_BOX_MARGIN
        pad_y = ch * MASK_BOX_MARGIN

        bx1 = max(0.0, min_x - pad_x)
        by1 = max(0.0, min_y - pad_y)
        bx2 = min(float(img_width), max_x + pad_x)
        by2 = min(float(img_height), max_y + pad_y)

        box_w = max(bx2 - bx1, 1.0)
        box_h = max(by2 - by1, 1.0)

        # Normalize polygon points into the mask-tight box (no clipping needed: the box
        # contains the contour by construction).
        normalized_points = []
        for px, py in points:
            nx = min(max((px - bx1) / box_w, 0.0), 1.0)
            ny = min(max((py - by1) / box_h, 0.0), 1.0)
            normalized_points.append([float(nx), float(ny)])

        # Adjusted detection box (center form), normalized to the image.
        adj_box = (
            ((bx1 + bx2) * 0.5) / img_width,
            ((by1 + by2) * 0.5) / img_height,
            box_w / img_width,
            box_h / img_height,
        )

        return normalized_points, adj_box

    except Exception as e:
        print(f"Warning: Polygon extraction failed: {e}")
        return None


def encode_rle(mask_binary: np.ndarray) -> str:
    """
    Run-length encode a binary mask for efficient transmission.
    Format: comma-separated runs of (start_index, length)
    """
    flat = mask_binary.flatten()
    # Find where values change
    changes = np.diff(flat)
    change_indices = np.where(changes != 0)[0] + 1
    
    # Build runs
    runs = []
    prev_idx = 0
    for idx in change_indices:
        if flat[prev_idx] == 1:
            runs.append(f"{prev_idx}:{idx-prev_idx}")
        prev_idx = idx
    
    # Handle last run
    if flat[prev_idx] == 1:
        runs.append(f"{prev_idx}:{len(flat)-prev_idx}")
    
    return ",".join(runs)


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    include_masks: bool = Query(False, description="Include segmentation masks (YOLO only)"),
    mask_format: MaskFormat = Query(MaskFormat.POLYGON, description="Mask format: 'none', 'raster', 'polygon', or 'rle'"),
    mask_resolution: int = Query(512, description="Resolution for raster mask encoding (64-1024)"),
    max_polygon_points: int = Query(64, description="Max points for polygon simplification (16-256)"),
    confidence: float = Query(0.25, description="Confidence threshold")
):
    """
    Detect objects in an image using the model selected at server startup.
    
    Args:
        file: Image file to process
        include_masks: If True, includes base64-encoded segmentation masks (YOLO only)
        mask_resolution: Resolution to resize masks before encoding (default 160)
        confidence: Confidence threshold for detections
    
    Returns:
        JSON with detections including bounding boxes and optionally masks
    """
    global active_model, active_model_type
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_width, img_height = image.size
        
        detections = []
        
        if active_model_type == "rfdetr":
            # RF-DETR inference
            predictions = active_model.infer(image, confidence=confidence)[0]
            
            for pred in predictions.predictions:
                # Normalize and convert to center format
                center_x = pred.x / img_width
                center_y = pred.y / img_height
                width_norm = pred.width / img_width
                height_norm = pred.height / img_height
                
                # Map RF-DETR class names to COCO IDs
                label = pred.class_name
                label_id = COCO_NAME_TO_ID.get(label.lower(), -1)
                
                detection = {
                    "label": label,
                    "label_id": label_id,
                    "x": center_x,
                    "y": center_y,
                    "w": width_norm,
                    "h": height_norm,
                    "confidence": float(pred.confidence)
                }
                detections.append(detection)
            
            print(f"[RF-DETR] Processed image {img_width}x{img_height}, found {len(detections)} objects")
        
        else:
            # YOLO inference with FP16 on CUDA
            # YOLOE must run FP32: half-precision inference flips the model to Half,
            # but set_classes/get_text_pe produce Float text embeddings -> dtype mismatch.
            use_half = _device == "cuda" and active_model_type != "yoloe"
            results = active_model(image, conf=confidence, half=use_half, agnostic_nms=_agnostic_nms())
            
            for result in results:
                boxes = result.boxes
                masks = result.masks
                
                for i, box in enumerate(boxes):
                    xywhn = box.xywhn[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = result.names[cls]

                    # Closed-vocab detectors return COCO ids; map them onto the shader's slots.
                    label_id = resolve_label_id(cls, result.names)
                    if label_id < 0:
                        continue

                    detection = {
                        "label": label,
                        "label_id": label_id,
                        "x": xywhn[0],
                        "y": xywhn[1],
                        "w": xywhn[2],
                        "h": xywhn[3],
                        "confidence": conf
                    }
                    
                    # Include segmentation mask if requested and available
                    if include_masks and masks is not None and i < len(masks) and mask_format != MaskFormat.NONE:
                        try:
                            mask_data = masks.data[i].cpu().numpy()  # Shape: (mask_h, mask_w) - masks.data is tensor [N, H, W]
                            bbox_norm = (xywhn[0], xywhn[1], xywhn[2], xywhn[3])
                            
                            if mask_format == MaskFormat.POLYGON:
                                # ================================================
                                # POLYGON FORMAT - Smallest, most efficient
                                # ================================================
                                poly_result = extract_polygon_from_mask(
                                    mask_data, img_width, img_height,
                                    bbox_norm, max_polygon_points
                                )

                                if poly_result:
                                    polygon, adj_box = poly_result
                                    # Align the reported box with the mask-tight box so
                                    # the shader gate matches the (un-clipped) mask.
                                    detection["x"], detection["y"], detection["w"], detection["h"] = adj_box
                                    # Flatten polygon for Unity JsonUtility compatibility
                                    # Format: [x0, y0, x1, y1, x2, y2, ...]
                                    flat_polygon = []
                                    for px, py in polygon:
                                        flat_polygon.append(px)
                                        flat_polygon.append(py)
                                    detection["mask_polygon"] = flat_polygon
                                    detection["mask_polygon_count"] = len(polygon)
                                    detection["mask_format"] = "polygon"
                                    polygon_bytes = len(flat_polygon) * 4  # Approximate size
                                    print(f"    [POLYGON] {label}: {len(polygon)} points (~{polygon_bytes} bytes)")
                                    
                            elif mask_format == MaskFormat.RLE:
                                # ================================================
                                # RLE FORMAT - Lossless compression
                                # ================================================
                                mask_full = cv2.resize(
                                    mask_data.astype(np.float32),
                                    (img_width, img_height),
                                    interpolation=cv2.INTER_LINEAR
                                )
                                mask_binary = (mask_full > 0.5).astype(np.uint8)
                                
                                # Crop to bounding box
                                cx, cy, w, h = bbox_norm
                                x1 = max(0, int((cx - w/2) * img_width))
                                y1 = max(0, int((cy - h/2) * img_height))
                                x2 = min(img_width, int((cx + w/2) * img_width))
                                y2 = min(img_height, int((cy + h/2) * img_height))
                                
                                if x2 > x1 and y2 > y1:
                                    mask_cropped = mask_binary[y1:y2, x1:x2]
                                    # Resize to standard size for consistent decoding
                                    mask_resized = cv2.resize(mask_cropped, (mask_resolution, mask_resolution), 
                                                            interpolation=cv2.INTER_NEAREST)
                                    rle_data = encode_rle(mask_resized)
                                    
                                    detection["mask_rle"] = rle_data
                                    detection["mask_width"] = mask_resolution
                                    detection["mask_height"] = mask_resolution
                                    detection["mask_format"] = "rle"
                                    print(f"    [RLE] {label}: {len(rle_data)} chars")
                                    
                            else:  # MaskFormat.RASTER
                                # ================================================
                                # RASTER FORMAT - Original (large but exact)
                                # ================================================
                                mask_full = cv2.resize(
                                    mask_data.astype(np.float32),
                                    (img_width, img_height),
                                    interpolation=cv2.INTER_LINEAR
                                )
                                
                                cx, cy, w, h = bbox_norm
                                x1 = max(0, int((cx - w/2) * img_width))
                                y1 = max(0, int((cy - h/2) * img_height))
                                x2 = min(img_width, int((cx + w/2) * img_width))
                                y2 = min(img_height, int((cy + h/2) * img_height))
                                
                                if x2 > x1 and y2 > y1:
                                    mask_cropped = mask_full[y1:y2, x1:x2]
                                    target_size = (mask_resolution, mask_resolution)
                                    mask_resized = cv2.resize(mask_cropped, target_size,
                                                            interpolation=cv2.INTER_AREA)
                                    mask_uint8 = (np.clip(mask_resized, 0, 1) * 255).astype(np.uint8)
                                    mask_uint8 = np.flipud(mask_uint8)  # Flip for Unity
                                    
                                    mask_base64 = base64.b64encode(mask_uint8.tobytes()).decode('utf-8')
                                    detection["mask"] = mask_base64
                                    detection["mask_width"] = mask_resolution
                                    detection["mask_height"] = mask_resolution
                                    detection["mask_format"] = "raw"
                                    
                                    raster_bytes = mask_resolution * mask_resolution
                                    print(f"    [RASTER] {label}: {mask_resolution}x{mask_resolution} ({raster_bytes} bytes)")
                                    
                        except Exception as mask_error:
                            import traceback
                            traceback.print_exc()
                            print(f"Warning: Could not encode mask for {label}: {mask_error}")
                    
                    detections.append(detection)
            
            print(f"[YOLO] Processed image {img_width}x{img_height}, found {len(detections)} objects")
        
        for det in detections:
            print(f"  - {det['label']} (ID={det['label_id']}): center=({det['x']:.3f},{det['y']:.3f}) size=({det['w']:.3f}x{det['h']:.3f}) conf={det['confidence']:.3f}")
        
        return JSONResponse(content={
            "detections": detections, 
            "image_size": [img_width, img_height],
            "model": active_model_type
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/configure")
async def configure(request: Request):
    """
    Set the active open-vocabulary target classes for the current shader.

    Body: {"classes": ["person", "red coffee mug", ...], "confidence": 0.3}

    The class order is significant: a detection's label_id is the slot index into
    this list. Sent over HTTP (reliable TCP) on shader switch and decoupled from the
    low-latency frame transport (UDP).
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(content={"error": "Invalid JSON body"}, status_code=400)

    classes = data.get("classes", []) or []
    confidence = data.get("confidence", None)

    # Same optional detector switch the UDP control packet accepts, so HTTP clients are
    # not a second-class citizen. Before the vocabulary for the same ordering reason.
    if data.get("detector"):
        switch_detector(data["detector"])

    applied = apply_vocabulary(classes, confidence, data.get("coco_classes", None))

    return JSONResponse(content={
        "status": "ok",
        "active_model": active_model_type,
        "open_vocabulary": active_model_type == "yoloe",
        "classes": applied,
        "confidence": _active_vocabulary_confidence,
    })


@app.get("/labels")
async def get_labels():
    """
    Return the active label list.

    Open-vocabulary (yoloe): the current per-shader vocabulary with slot indices.
    Otherwise: the static COCO label set.
    """
    if active_model_type == "yoloe":
        labels = {i: name for i, name in enumerate(_active_vocabulary)}
    else:
        labels = {v: k for k, v in COCO_NAME_TO_ID.items()}
    return JSONResponse(content={"labels": labels})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "ok",
        "active_model": active_model_type,
        "supports_masks": active_model_type in ("yolo", "yoloe"),
        "open_vocabulary": active_model_type == "yoloe",
        "active_classes": _active_vocabulary,
    })


@app.get("/models")
async def list_models():
    """List available models and their capabilities."""
    return JSONResponse(content={
        "active_model": active_model_type,
        "available_models": [
            {
                "name": "yolo",
                "description": "yolo26-seg: Fast segmentation model with masks",
                "supports_masks": True,
                "speed": "fast"
            },
            {
                "name": "rfdetr",
                "description": "RF-DETR: Transformer-based detector, higher accuracy",
                "supports_masks": False,
                "speed": "medium"
            }
        ]
    })


def load_model(model_type: str, yolo_model: str = None, yoloe_model: str = None):
    """Load the specified model using optimized lazy loading."""
    global active_model, active_model_type

    if model_type == "rfdetr":
        logger.info("Loading RF-DETR model...")
        active_model = get_rfdetr_model()
        active_model_type = "rfdetr"
    elif model_type == "yoloe":
        # Open-vocabulary segmentation model
        model_name = yoloe_model if yoloe_model else "yoloe-26x-seg.pt"
        if not model_name.endswith(".pt"):
            model_name += ".pt"
        logger.info(f"Loading YOLOE (open-vocabulary) model: {model_name}...")
        active_model = get_yoloe_model(model_name)
        active_model_type = "yoloe"
        # Recover mask silhouette that ultralytics would otherwise crop to the tight box.
        install_mask_crop_expansion(_mask_crop_margin)
    else:
        # Use provided YOLO model name or default to yolo26n-seg.pt
        model_name = yolo_model if yolo_model else "yolo26n-seg.pt"
        # Ensure .pt extension
        if not model_name.endswith(".pt"):
            model_name += ".pt"
        logger.info(f"Loading YOLO model: {model_name}...")
        active_model = get_yolo_model(model_name)
        active_model_type = "yolo"

    logger.info(f"Model '{active_model_type}' loaded successfully!")

    # Warm up the YOLOE text encoder at startup. get_text_pe() lazily downloads the
    # MobileCLIP weights (~242MB 'mobileclip2_b.ts') on first use; forcing it here
    # surfaces download/network failures loudly at boot instead of silently failing
    # on the first vocabulary packet (which would leave YOLOE with no classes -> 0 detections).
    if active_model_type == "yoloe":
        try:
            logger.info("Warming up YOLOE text encoder (first run downloads MobileCLIP ~242MB)...")
            active_model.get_text_pe(["object"])
            logger.info("YOLOE text encoder ready.")
        except Exception as e:
            logger.error(
                f"YOLOE text encoder warmup FAILED — open-vocabulary detection will not work: {e}. "
                f"Ensure the server host has internet access to download 'mobileclip2_b.ts'."
            )

    # Apply any vocabulary that may already be configured (e.g. set before reload).
    if active_model_type == "yoloe" and _active_vocabulary:
        apply_vocabulary(_active_vocabulary, _active_vocabulary_confidence)


# ============================================================================
# UDP SERVER - Low latency transport
# ============================================================================

def process_udp_detection(image_bytes: bytes, frame_id: int, 
                          confidence: float = None, mask_format: int = None,
                          request_masks: bool = None, mask_resolution: int = None,
                          max_polygon_points: int = None) -> bytes:
    """
    Process detection for UDP protocol. Returns binary response.
    
    Binary Protocol Format (v2 - extended header):
    Request:  [frame_id:4][chunk_idx:2][total_chunks:2][confidence:4]
              [mask_format:1][request_masks:1][mask_resolution:2][max_polygon_points:2][jpeg_data:N]
    Response: [frame_id:4][num_detections:2][detections...]
    
    Detection format:
    [label_id:2][x:4][y:4][w:4][h:4][conf:4][num_points:2][polygon_points...]
    Polygon points: [x:2][y:2] as uint16 (0-65535 normalized)
    
    Parameters from client are used, but server command-line args override if set differently from defaults.
    """
    global active_model, udp_confidence_threshold, udp_max_polygon_points, udp_mask_format
    
    # Resolve parameters: use client values, but server CLI overrides defaults
    # Server default values (matching Unity defaults)
    SERVER_DEFAULT_CONFIDENCE = 0.25
    SERVER_DEFAULT_MAX_POINTS = 64
    SERVER_DEFAULT_MASK_FORMAT = MaskFormat.POLYGON
    
    # If server has non-default settings, they take priority (allows server-side control)
    if udp_confidence_threshold != SERVER_DEFAULT_CONFIDENCE:
        effective_confidence = udp_confidence_threshold
    else:
        effective_confidence = confidence if confidence is not None else udp_confidence_threshold
    
    if udp_max_polygon_points != SERVER_DEFAULT_MAX_POINTS:
        effective_max_points = udp_max_polygon_points
    else:
        effective_max_points = max_polygon_points if max_polygon_points is not None else udp_max_polygon_points
    
    if udp_mask_format != SERVER_DEFAULT_MASK_FORMAT:
        effective_mask_format = udp_mask_format
    else:
        # Convert int from client to MaskFormat enum
        if mask_format is not None:
            format_map = {0: MaskFormat.NONE, 1: MaskFormat.RASTER, 2: MaskFormat.POLYGON, 3: MaskFormat.RLE}
            effective_mask_format = format_map.get(mask_format, udp_mask_format)
        else:
            effective_mask_format = udp_mask_format
    
    effective_request_masks = request_masks if request_masks is not None else True
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = image.size
        
        # Build binary response
        response = bytearray()
        response.extend(struct.pack('<I', frame_id))  # Frame ID (4 bytes)
        
        detections_data = bytearray()
        num_detections = 0

        if active_model_type in ("yolo", "yoloe"):
            # YOLO / YOLOE inference with FP16 on CUDA. For YOLOE the class indices
            # are slot indices into the active vocabulary (set via /configure).
            # YOLOE must run FP32: half-precision inference flips the model to Half,
            # but set_classes/get_text_pe produce Float text embeddings -> dtype mismatch.
            use_half = _device == "cuda" and active_model_type != "yoloe"
            results = active_model(image, conf=effective_confidence, half=use_half, agnostic_nms=_agnostic_nms())
            
            for result in results:
                boxes = result.boxes
                masks = result.masks
                
                has_masks = masks is not None and len(masks) > 0
                if len(boxes) > 0 and not has_masks:
                    print(f"[UDP] Warning: No masks available! Make sure you're using a -seg model (e.g., yolo26n-seg.pt)")
                
                for i, box in enumerate(boxes):
                    xywhn = box.xywhn[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    # Closed-vocab detectors return COCO ids; map them onto the shader's slots.
                    # Done BEFORE the mask work below so a dropped detection costs no polygon
                    # extraction — this is the per-frame hot path.
                    label_id = resolve_label_id(cls, result.names)
                    if label_id < 0:
                        continue

                    # Box defaults to the YOLO detection box; overridden below by the
                    # mask-tight box when a polygon is extracted (keeps gate + mask aligned).
                    box = [xywhn[0], xywhn[1], xywhn[2], xywhn[3]]

                    # Extract polygon if masks requested and available
                    polygon_data = bytearray()
                    num_points = 0

                    # Check mask conditions using effective parameters
                    mask_available = masks is not None and i < len(masks.data)
                    format_is_polygon = effective_mask_format == MaskFormat.POLYGON

                    if mask_available and format_is_polygon and effective_request_masks:
                        try:
                            mask_data = masks.data[i].cpu().numpy()  # Shape: (mask_h, mask_w) - note: masks.data not masks[i].data
                            bbox_norm = (xywhn[0], xywhn[1], xywhn[2], xywhn[3])
                            poly_result = extract_polygon_from_mask(
                                mask_data, img_width, img_height,
                                bbox_norm, effective_max_points
                            )

                            if poly_result:
                                polygon, adj_box = poly_result
                                box = list(adj_box)
                                num_points = len(polygon)
                                for px, py in polygon:
                                    # Pack as uint16 (0-65535) for efficiency
                                    polygon_data.extend(struct.pack('<HH',
                                        int(px * 65535),
                                        int(py * 65535)
                                    ))
                            else:
                                print(f"[UDP] Warning: Polygon extraction returned None for detection {i}")
                        except Exception as e:
                            print(f"[UDP] Error extracting polygon: {e}")

                    # Pack detection data (after possible box adjustment from the mask).
                    # Format: H=uint16(label), f=float(x), f=float(y), f=float(w), f=float(h), H=uint16(conf)
                    det_data = struct.pack('<HffffH',
                        label_id,              # slot index the shader filters on (2 bytes)
                        box[0],                # x center normalized (4 bytes)
                        box[1],                # y center normalized (4 bytes)
                        box[2],                # width normalized (4 bytes)
                        box[3],                # height normalized (4 bytes)
                        int(conf * 65535)      # confidence as uint16 (2 bytes)
                    )

                    # Append point count and polygon
                    detections_data.extend(det_data)
                    detections_data.extend(struct.pack('<H', num_points))
                    detections_data.extend(polygon_data)
                    num_detections += 1
        
        response.extend(struct.pack('<H', num_detections))
        response.extend(detections_data)
        return bytes(response)
        
    except Exception as e:
        print(f"[UDP] Error processing frame {frame_id}: {e}")
        # Return error response (frame_id + 0 detections)
        return struct.pack('<IH', frame_id, 0)


def run_udp_server(host: str, port: int):
    """Run UDP detection server in a separate thread."""
    global udp_server_running, udp_server_socket
    
    udp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Increase buffer sizes for large images
    udp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB receive buffer
    udp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024)   # 256KB send buffer
    
    udp_server_socket.bind((host, port))
    udp_server_running = True
    
    print(f"[UDP] Server listening on {host}:{port}")
    print(f"[UDP] Extended header format (18 bytes): frame_id, chunk_idx, total_chunks, confidence, mask_format, request_masks, mask_resolution, max_polygon_points")
    
    # Buffer for reassembling fragmented images
    frame_buffers = {}  # frame_id -> {chunks: [], expected_chunks: int, params: dict}
    
    while udp_server_running:
        try:
            # Max UDP datagram size (theoretical limit ~65KB, practical ~65507 bytes)
            data, addr = udp_server_socket.recvfrom(65507)

            # Control packet: reserved frame-id sentinel 0xFFFFFFFF + JSON vocabulary
            # payload. Delivered over the same socket as frames so it is guaranteed to
            # reach the server (no dependency on a separate HTTP port).
            if len(data) >= 4 and data[:4] == b"\xff\xff\xff\xff":
                try:
                    payload = json.loads(data[4:].decode("utf-8"))

                    # Optional detector switch, handled BEFORE the vocabulary: load_model()
                    # re-applies the active vocabulary, and apply_vocabulary() below then
                    # runs against the detector the client actually asked for.
                    if payload.get("detector"):
                        switch_detector(payload["detector"])

                    applied = apply_vocabulary(
                        payload.get("classes", []) or [],
                        payload.get("confidence", None),
                        payload.get("coco_classes", None),
                    )
                    print(f"[UDP] Vocabulary control ({_vocabulary_status}, {len(applied)}): {applied}")
                    # Ack back so the client can confirm receipt + apply status in its own
                    # log (sentinel 0xFFFFFFFE so it is not mistaken for a detection response).
                    ack = b"\xfe\xff\xff\xff" + json.dumps({
                        "status": _vocabulary_status,
                        "model": active_model_type,
                        "count": len(applied),
                    }).encode("utf-8")
                    try:
                        udp_server_socket.sendto(ack, addr)
                    except Exception:
                        pass
                except Exception as ctrl_err:
                    print(f"[UDP] Failed to parse vocabulary control packet: {ctrl_err}")
                continue

            if len(data) < 18:  # Minimum header size (extended)
                continue
            
            # Parse packet header (extended format - 18 bytes)
            # Format: [frame_id:4][chunk_idx:2][total_chunks:2][confidence:4]
            #         [mask_format:1][request_masks:1][mask_resolution:2][max_polygon_points:2]
            header_format = '<I H H f B B H H'
            frame_id, chunk_idx, total_chunks, confidence, mask_format, request_masks, mask_resolution, max_polygon_points = struct.unpack(header_format, data[:18])
            chunk_data = data[18:]
            
            # Store parameters for multi-chunk frames
            params = {
                'confidence': confidence,
                'mask_format': mask_format,
                'request_masks': bool(request_masks),
                'mask_resolution': mask_resolution,
                'max_polygon_points': max_polygon_points
            }
            
            if total_chunks == 1:
                # Single packet - process immediately with parameters
                response = process_udp_detection(
                    chunk_data, frame_id,
                    confidence=params['confidence'],
                    mask_format=params['mask_format'],
                    request_masks=params['request_masks'],
                    mask_resolution=params['mask_resolution'],
                    max_polygon_points=params['max_polygon_points']
                )
                try:
                    udp_server_socket.sendto(response, addr)
                except Exception as send_err:
                    print(f"[UDP] ERROR sending to {addr}: {send_err}")
            else:
                # Multi-packet frame - reassemble
                if frame_id not in frame_buffers:
                    frame_buffers[frame_id] = {
                        'chunks': [None] * total_chunks,
                        'received': 0,
                        'params': params,
                        'addr': addr
                    }
                
                buf = frame_buffers[frame_id]
                if buf['chunks'][chunk_idx] is None:
                    buf['chunks'][chunk_idx] = chunk_data
                    buf['received'] += 1
                
                # Check if complete
                if buf['received'] == total_chunks:
                    # Reassemble and process with stored parameters
                    full_data = b''.join(buf['chunks'])
                    p = buf['params']
                    response = process_udp_detection(
                        full_data, frame_id,
                        confidence=p['confidence'],
                        mask_format=p['mask_format'],
                        request_masks=p['request_masks'],
                        mask_resolution=p['mask_resolution'],
                        max_polygon_points=p['max_polygon_points']
                    )
                    try:
                        udp_server_socket.sendto(response, buf['addr'])
                    except Exception as send_err:
                        print(f"[UDP] ERROR sending to {buf['addr']}: {send_err}")
                    del frame_buffers[frame_id]
                    
                    # Clean old incomplete frames
                    old_frames = [fid for fid in frame_buffers if fid < frame_id - 10]
                    for fid in old_frames:
                        del frame_buffers[fid]
                        
        except socket.timeout:
            continue
        except Exception as e:
            if udp_server_running:
                print(f"[UDP] Error: {e}")
    
    print("[UDP] Server stopped")


def stop_udp_server():
    """Stop the UDP server."""
    global udp_server_running, udp_server_socket
    udp_server_running = False
    if udp_server_socket:
        udp_server_socket.close()


# ============================================================================
# WEBSOCKET SERVER - Reliable streaming with connection persistence
# ============================================================================

@dataclass
class WebSocketClientState:
    """State for a connected WebSocket client."""
    is_streaming: bool = False
    last_frame_time: float = 0.0
    frames_processed: int = 0
    confidence_threshold: float = 0.25
    mask_format: MaskFormat = MaskFormat.POLYGON
    mask_resolution: int = 512
    max_polygon_points: int = 64
    request_masks: bool = True


class DetectionWebSocketServer:
    """WebSocket server for object detection streaming."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8002,
        yolo_model: str = "yolo26n-seg.pt"
    ):
        self.host = host
        self.port = port
        self.yolo_model = yolo_model
        self.client_states: Dict[str, WebSocketClientState] = {}
    
    def decode_image(self, base64_str: str) -> Image.Image:
        """Decode base64 image to PIL Image."""
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        image_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    def run_detection(
        self,
        image: Image.Image,
        confidence: float = 0.25,
        mask_format: MaskFormat = MaskFormat.POLYGON,
        max_polygon_points: int = 64,
        request_masks: bool = True
    ) -> List[dict]:
        """Run detection on an image using the active model (YOLO or YOLOE)."""
        model = get_active_inference_model()
        img_width, img_height = image.size
        
        # Use FP16 half precision on CUDA (but NOT for YOLOE: it would flip the model to
        # Half while set_classes/get_text_pe produce Float embeddings -> dtype mismatch).
        use_half = _device == "cuda" and active_model_type != "yoloe"
        results = model(image, conf=confidence, verbose=False, half=use_half, agnostic_nms=_agnostic_nms())
        detections = []
        
        for result in results:
            boxes = result.boxes
            masks = result.masks
            
            if boxes is None:
                continue
            
            for i, box in enumerate(boxes):
                xywhn = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls]

                # Closed-vocab detectors return COCO ids; map them onto the shader's slots.
                label_id = resolve_label_id(cls, result.names)
                if label_id < 0:
                    continue

                detection = {
                    "label": label,
                    "label_id": label_id,
                    "x": xywhn[0],
                    "y": xywhn[1],
                    "w": xywhn[2],
                    "h": xywhn[3],
                    "confidence": conf
                }
                
                # Extract mask if requested
                if request_masks and masks is not None and i < len(masks) and mask_format != MaskFormat.NONE:
                    try:
                        mask_data = masks.data[i].cpu().numpy()
                        bbox_norm = (xywhn[0], xywhn[1], xywhn[2], xywhn[3])
                        
                        if mask_format == MaskFormat.POLYGON:
                            poly_result = extract_polygon_from_mask(
                                mask_data, img_width, img_height,
                                bbox_norm, max_polygon_points
                            )
                            if poly_result:
                                polygon, adj_box = poly_result
                                detection["x"], detection["y"], detection["w"], detection["h"] = adj_box
                                flat_polygon = []
                                for px, py in polygon:
                                    flat_polygon.append(px)
                                    flat_polygon.append(py)
                                detection["mask_polygon"] = flat_polygon
                                detection["mask_polygon_count"] = len(polygon)
                                detection["mask_format"] = "polygon"
                    except Exception as e:
                        logger.warning(f"Mask extraction failed: {e}")
                
                detections.append(detection)
        
        return detections
    
    async def handle_client(self, websocket, path=None):
        """Handle WebSocket client connection."""
        import websockets
        
        client_id = str(id(websocket))
        self.client_states[client_id] = WebSocketClientState()
        state = self.client_states[client_id]
        
        logger.info(f"[WebSocket] Client {client_id} connected")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "detect")
                    
                    if msg_type == "config":
                        # Update client configuration
                        if "confidence" in data:
                            state.confidence_threshold = data["confidence"]
                        if "mask_format" in data:
                            state.mask_format = MaskFormat(data["mask_format"])
                        if "max_polygon_points" in data:
                            state.max_polygon_points = data["max_polygon_points"]
                        if "request_masks" in data:
                            state.request_masks = data["request_masks"]
                        
                        await websocket.send(json.dumps({
                            "type": "config_ack",
                            "confidence": state.confidence_threshold,
                            "mask_format": state.mask_format.value,
                            "max_polygon_points": state.max_polygon_points,
                            "request_masks": state.request_masks
                        }))

                    elif msg_type == "set_classes":
                        # Open-vocabulary target list for the active shader. Applied
                        # once here (text-embedding pass), then detection runs at full
                        # speed. label_id becomes the slot index into this list.
                        # Optional detector switch rides along, as on UDP and HTTP.
                        if data.get("detector"):
                            switch_detector(data["detector"])
                        applied = apply_vocabulary(
                            data.get("classes", []) or [],
                            data.get("confidence", None),
                            data.get("coco_classes", None),
                        )
                        if data.get("confidence", None) is not None and data["confidence"] >= 0:
                            state.confidence_threshold = float(data["confidence"])
                        await websocket.send(json.dumps({
                            "type": "classes_ack",
                            "open_vocabulary": active_model_type == "yoloe",
                            "classes": applied,
                        }))
                    
                    elif msg_type == "start_stream":
                        state.is_streaming = True
                        state.frames_processed = 0
                        
                        # Apply any settings from start message
                        if "confidence" in data:
                            state.confidence_threshold = data["confidence"]
                        if "mask_format" in data:
                            state.mask_format = MaskFormat(data["mask_format"])
                        if "request_masks" in data:
                            state.request_masks = data["request_masks"]
                        
                        await websocket.send(json.dumps({
                            "type": "stream_started"
                        }))
                        logger.info(f"[WebSocket] Client {client_id} started streaming")
                    
                    elif msg_type == "stop_stream":
                        state.is_streaming = False
                        await websocket.send(json.dumps({
                            "type": "stream_stopped",
                            "frames_processed": state.frames_processed
                        }))
                        logger.info(f"[WebSocket] Client {client_id} stopped streaming")
                    
                    elif msg_type == "frame" or msg_type == "detect":
                        # Process detection frame
                        start_time = time.time()
                        
                        image = self.decode_image(data["image"])
                        img_width, img_height = image.size
                        
                        # Run detection in executor to not block event loop
                        loop = asyncio.get_event_loop()
                        detections = await loop.run_in_executor(
                            _executor,
                            lambda: self.run_detection(
                                image=image,
                                confidence=state.confidence_threshold,
                                mask_format=state.mask_format,
                                max_polygon_points=state.max_polygon_points,
                                request_masks=state.request_masks
                            )
                        )
                        
                        processing_time = (time.time() - start_time) * 1000
                        state.frames_processed += 1
                        fps = 1000.0 / processing_time if processing_time > 0 else 0
                        
                        await websocket.send(json.dumps({
                            "type": "detection_result",
                            "success": True,
                            "detections": detections,
                            "num_detections": len(detections),
                            "image_size": [img_width, img_height],
                            "processing_time_ms": processing_time,
                            "frame_number": state.frames_processed,
                            "fps": round(fps, 1)
                        }))
                        
                        if state.frames_processed % 30 == 0:
                            logger.info(f"[WebSocket] Client {client_id}: {state.frames_processed} frames, {fps:.1f} FPS")
                    
                    elif msg_type == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
                except KeyError as e:
                    await websocket.send(json.dumps({"type": "error", "error": f"Missing field: {e}"}))
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"[WebSocket] Client {client_id} disconnected during processing")
                    break
                except Exception as e:
                    logger.exception(f"[WebSocket] Error handling message: {e}")
                    try:
                        await websocket.send(json.dumps({"type": "error", "error": str(e)}))
                    except:
                        break
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[WebSocket] Client {client_id} disconnected")
        finally:
            if client_id in self.client_states:
                del self.client_states[client_id]
    
    async def start(self):
        """Start the WebSocket server."""
        import websockets
        
        logger.info(f"[WebSocket] Starting detection server on ws://{self.host}:{self.port}")
        
        # Pre-load active model (YOLO or YOLOE)
        logger.info("[WebSocket] Pre-loading detection model...")
        get_active_inference_model()
        
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=50 * 1024 * 1024,  # 50MB max message size
            ping_interval=20,
            ping_timeout=60,
            close_timeout=30
        ):
            logger.info(f"[WebSocket] Detection server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever


def run_websocket_server(host: str, port: int, yolo_model: str):
    """Run WebSocket server in a separate thread."""
    server = DetectionWebSocketServer(host=host, port=port, yolo_model=yolo_model)
    asyncio.run(server.start())


@app.get("/udp/status")
async def udp_status():
    """Get UDP server status."""
    return JSONResponse(content={
        "running": udp_server_running,
        "confidence_threshold": udp_confidence_threshold,
        "mask_format": udp_mask_format.value,
        "max_polygon_points": udp_max_polygon_points
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection Server - Multi-Transport Support")
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["yolo", "yoloe", "rfdetr"],
        default="yoloe",
        help="Model to use: 'yoloe' (open-vocabulary text-prompted seg, set classes via "
             "/configure — default), 'yolo' (yolo26-seg, legacy fixed COCO labels + masks), "
             "or 'rfdetr' (RF-DETR, accurate)"
    )
    parser.add_argument(
        "--yolo-model", "-y",
        type=str,
        default="yolo26n-seg",
        help="YOLO model variant to use (e.g., 'yolo26n-seg', 'yolo26s-seg', 'yolo26m-seg', 'yolo26l-seg', 'yolo26x-seg')"
    )
    parser.add_argument(
        "--yoloe-model",
        type=str,
        default="yoloe-26x-seg",
        help="YOLOE open-vocabulary model variant: 'yoloe-26x-seg' (most accurate, default), 'yoloe-26l-seg', 'yoloe-26m-seg', 'yoloe-26s-seg', 'yoloe-26n-seg' "
             "(fastest) (also 'yoloe-11{s,m,l}-seg' / 'yoloe-v8{s,m,l}-seg')"
    )
    parser.add_argument(
        "--mask-crop-margin",
        type=float,
        default=0.15,
        help="Fraction to expand instance-mask crop boxes (YOLOE) so mask edges are not "
             "shaved off by the tight detection box. 0 disables. Higher recovers more "
             "silhouette but risks bleed from neighbours. Default: 0.15"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="HTTP port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=8001,
        help="UDP port for low-latency detection (default: 8001, 0 to disable)"
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8002,
        help="WebSocket port for reliable streaming (default: 8002, 0 to disable)"
    )
    parser.add_argument(
        "--udp-confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for UDP detections (default: 0.25)"
    )
    parser.add_argument(
        "--udp-max-points",
        type=int,
        default=64,
        help="Max polygon points for UDP masks (default: 64)"
    )
    parser.add_argument(
        "--udp-mask-format",
        type=str,
        choices=["none", "polygon", "raster", "rle"],
        default="polygon",
        help="Mask format for UDP: 'none', 'polygon' (default), 'raster', 'rle'"
    )
    
    args = parser.parse_args()

    # Mask crop expansion margin (applied when a YOLOE model is loaded)
    _mask_crop_margin = args.mask_crop_margin

    # Update UDP settings
    udp_confidence_threshold = args.udp_confidence
    udp_max_polygon_points = args.udp_max_points
    udp_mask_format = MaskFormat(args.udp_mask_format)
    
    # Ensure .pt extension for YOLO model
    yolo_model_name = args.yolo_model
    if not yolo_model_name.endswith(".pt"):
        yolo_model_name += ".pt"
    
    print("=" * 70)
    print("Object Detection Server - Multi-Transport Support")
    print("=" * 70)
    print("\nSupported transports:")
    print("  - HTTP:      Reliable REST API (for single requests)")
    print("  - UDP:       Low-latency binary protocol (may drop frames)")
    print("  - WebSocket: Reliable streaming with connection persistence")
    print("")
    
    # Remember the CLI model names so a runtime detector switch loads the SAME weights the
    # command line asked for, instead of silently falling back to load_model's defaults.
    _yolo_model_name_arg = args.yolo_model
    _yoloe_model_name_arg = args.yoloe_model

    # Load the selected model
    load_model(args.model, args.yolo_model, args.yoloe_model)
    
    # Start UDP server if enabled
    if args.udp_port > 0:
        udp_thread = threading.Thread(
            target=run_udp_server, 
            args=(args.host, args.udp_port),
            daemon=True
        )
        udp_thread.start()
        print(f"[UDP]       udp://{args.host}:{args.udp_port}")
    else:
        print("[UDP]       Disabled")
    
    # Start WebSocket server if enabled (YOLO and YOLOE support streaming + masks)
    if args.ws_port > 0 and args.model in ("yolo", "yoloe"):
        ws_thread = threading.Thread(
            target=run_websocket_server,
            args=(args.host, args.ws_port, yolo_model_name),
            daemon=True
        )
        ws_thread.start()
        print(f"[WebSocket] ws://{args.host}:{args.ws_port}")
    elif args.ws_port > 0 and args.model not in ("yolo", "yoloe"):
        print("[WebSocket] Disabled (only available with YOLO / YOLOE models)")
    else:
        print("[WebSocket] Disabled")
    
    print(f"[HTTP]      http://{args.host}:{args.port}")
    
    print("\n" + "-" * 70)
    print("HTTP Endpoints:")
    print("  POST /detect       - Detect objects in image")
    print("       ?mask_format=polygon  (default, efficient)")
    print("       ?mask_format=raster   (original, large)")
    print("       ?mask_format=rle      (lossless compression)")
    print("  GET  /models       - List available models")
    print("  GET  /labels       - Get COCO class labels")
    print("  GET  /health       - Health check")
    print("  GET  /udp/status   - UDP server status")
    print("-" * 70)
    print("\nWebSocket Protocol (JSON):")
    print("  -> {type: 'config', confidence: 0.25, mask_format: 'polygon', ...}")
    print("  <- {type: 'config_ack', ...}")
    print("  -> {type: 'start_stream'}")
    print("  <- {type: 'stream_started'}")
    print("  -> {type: 'frame', image: 'base64...'}")
    print("  <- {type: 'detection_result', detections: [...], fps: 30.0}")
    print("  -> {type: 'stop_stream'}")
    print("  <- {type: 'stream_stopped', frames_processed: 100}")
    print("-" * 70)
    print("\nUsage examples:")
    print(f"  python server.py                                   # All transports")
    print(f"  python server.py --ws-port 0 --udp-port 0          # HTTP only")
    print(f"  python server.py --ws-port 9000                    # Custom WebSocket port")
    print("=" * 70)
    
    uvicorn.run(app, host=args.host, port=args.port)

