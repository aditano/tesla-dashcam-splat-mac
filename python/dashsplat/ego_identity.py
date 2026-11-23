"""Identify the owner's Tesla (ego) paint color and make/model from repeater views.

Paint names/codes: Tesla Body Repair / Service Manual "Vehicle and Wheel Paint Codes"
  https://service.tesla.com/docs/ModelY/ServiceManual/2025/en-us/GUID-769C9625-76EE-467B-B756-C9032AC2B99A.html
  https://service.tesla.com/docs/Model3/ServiceManual/2024/en-us/GUID-769C9625-76EE-467B-B756-C9032AC2B99A.html

Approximate screen hex swatches (for Lab nearest-neighbor only, not OEM formulas):
  https://www.car-editor.com/tools/paint-code/tesla/

Model guesses are soft heuristics (camera set / HW gen / silhouette / optional vision-LLM).
Never emit a hard 100% claim.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from dashsplat.masks.ego import ego_mask_array

# ---------------------------------------------------------------------------
# Factory paint catalog
# ---------------------------------------------------------------------------

PAINT_LIST_SOURCES = (
    "https://service.tesla.com/docs/ModelY/ServiceManual/2025/en-us/GUID-769C9625-76EE-467B-B756-C9032AC2B99A.html",
    "https://service.tesla.com/docs/Model3/ServiceManual/2024/en-us/GUID-769C9625-76EE-467B-B756-C9032AC2B99A.html",
    "https://www.car-editor.com/tools/paint-code/tesla/",
)


@dataclass(frozen=True)
class FactoryPaint:
    code: str
    name: str
    hex: str  # approximate display swatch
    finish: str = "metallic"


# Codes/names from Tesla service manuals; hex approximations from car-editor.com.
TESLA_FACTORY_PAINTS: tuple[FactoryPaint, ...] = (
    FactoryPaint("PPSW", "Pearl White Multi-Coat", "#E0DFDE", "tricoat"),
    FactoryPaint("PBSB", "Solid Black", "#222224", "solid"),
    FactoryPaint("PMNG", "Midnight Silver Metallic", "#5A5C61", "metallic"),
    FactoryPaint("PN01", "Stealth Grey", "#33343D", "metallic"),
    FactoryPaint("PN00", "Quicksilver", "#9D9CA1", "tricoat"),
    FactoryPaint("PPSB", "Deep Blue Metallic", "#10396E", "metallic"),
    FactoryPaint("PR01", "Ultra Red", "#86001D", "tricoat"),
    FactoryPaint("PPMR", "Red Multi-Coat", "#8B1A1A", "tricoat"),
    FactoryPaint("PR00", "Midnight Cherry Red", "#4A0A14", "quadcoat"),
    FactoryPaint("PX02", "Diamond Black", "#1A1A1C", "metallic"),
    FactoryPaint("PB00", "Frost Blue Metallic", "#6B8CAE", "metallic"),
    FactoryPaint("PB01", "Glacier Blue", "#7BA3C4", "metallic"),
    FactoryPaint("PB02", "Marine Blue", "#1E4D7B", "metallic"),
    FactoryPaint("PB03", "Coastal Blue", "#3A6B8C", "metallic"),
    FactoryPaint("PN02", "Lunar Silver", "#C0C0C4", "metallic"),
    FactoryPaint("PN03", "Cosmic Silver", "#A8A9AE", "metallic"),
    FactoryPaint("PPSR", "Signature Red", "#A01020", "tricoat"),
)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Color space helpers (sRGB → Lab), no heavy deps
# ---------------------------------------------------------------------------

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb float 0..1 or uint8 → Lab (D65). Last axis = RGB."""
    was_uint = rgb.dtype == np.uint8
    x = rgb.astype(np.float64)
    if was_uint or x.max() > 1.5:
        x = x / 255.0
    lin = _srgb_to_linear(x)
    # sRGB D65
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ m.T
    # D65 white
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x = xyz[..., 0] / xn
    y = xyz[..., 1] / yn
    z = xyz[..., 2] / zn

    def f(t: np.ndarray) -> np.ndarray:
        delta = 6 / 29
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb_hex(lab: np.ndarray) -> str:
    """Rough Lab → display hex (for diagnostics)."""
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    eps = 6 / 29

    def finv(t: float) -> float:
        return t**3 if t > eps else 3 * eps**2 * (t - 4 / 29)

    xn, yn, zn = 0.95047, 1.0, 1.08883
    X, Y, Z = xn * finv(fx), yn * finv(fy), zn * finv(fz)
    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    rgb = np.array([X, Y, Z]) @ m_inv.T

    def encode(c: float) -> int:
        c = float(np.clip(c, 0, 1))
        if c <= 0.0031308:
            v = 12.92 * c
        else:
            v = 1.055 * (c ** (1 / 2.4)) - 0.055
        return int(round(np.clip(v, 0, 1) * 255))

    return f"#{encode(rgb[0]):02X}{encode(rgb[1]):02X}{encode(rgb[2]):02X}"


# ---------------------------------------------------------------------------
# Paint sampling / matching
# ---------------------------------------------------------------------------

@dataclass
class PaintGuess:
    code: str
    name: str
    hex: str
    confidence: float
    lab: list[float] = field(default_factory=list)
    method: str = "lab_cluster"


def sample_ego_paint_lab(
    image: np.ndarray,
    camera: str,
    *,
    max_samples: int = 4000,
) -> np.ndarray:
    """Sample body pixels inside the ego mask; skip overexposed/dark/specular."""
    h, w = image.shape[:2]
    mask = ego_mask_array(camera, w, h)  # 0 = ego
    ego = mask == 0
    rgb = image.astype(np.float64)
    lum = rgb.mean(axis=2)
    # Skip very dark / overexposed
    valid = ego & (lum > 25) & (lum < 245)
    # Skip high-specular: near-white relative to neighborhood or high max channel
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    chroma = mx - mn
    # Specular-ish: bright + low chroma
    valid &= ~((lum > 200) & (chroma < 18))
    # Also skip extreme single-channel spikes (chrome trim / lights)
    valid &= chroma < 140

    coords = np.argwhere(valid)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(coords) > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(coords), size=max_samples, replace=False)
        coords = coords[idx]
    samples = rgb[coords[:, 0], coords[:, 1]]
    return rgb_to_lab(samples / 255.0)


def _road_sky_wb_gain(image: np.ndarray) -> np.ndarray:
    """Estimate simple RGB gains from sky (top) and road (bottom) midtones."""
    h, w = image.shape[:2]
    img = image.astype(np.float64)
    sky = img[: h // 4, w // 4 : 3 * w // 4]
    road = img[3 * h // 4 :, w // 4 : 3 * w // 4]
    refs = []
    for region in (sky, road):
        lum = region.mean(axis=2)
        mid = (lum > 40) & (lum < 220)
        if mid.sum() < 32:
            continue
        refs.append(region[mid].mean(axis=0))
    if not refs:
        return np.ones(3)
    gray = np.mean(refs, axis=0)
    mean = gray.mean()
    if mean < 1e-3:
        return np.ones(3)
    gains = mean / np.maximum(gray, 1.0)
    return np.clip(gains, 0.7, 1.4)


def _kmeans_lab(samples: np.ndarray, k: int = 3, iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Tiny k-means; returns (centroids, counts)."""
    if len(samples) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=int)
    k = min(k, len(samples))
    rng = np.random.default_rng(1)
    cents = samples[rng.choice(len(samples), size=k, replace=False)].copy()
    labels = np.zeros(len(samples), dtype=int)
    for _ in range(iters):
        d = ((samples[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        for i in range(k):
            m = labels == i
            if m.any():
                cents[i] = samples[m].mean(axis=0)
    counts = np.bincount(labels, minlength=k)
    return cents, counts


def match_paint(lab: np.ndarray) -> tuple[FactoryPaint, float]:
    """Nearest factory paint in Lab; confidence from distance."""
    lab = np.asarray(lab, dtype=np.float64).reshape(3)
    best: FactoryPaint | None = None
    best_d = 1e9
    for p in TESLA_FACTORY_PAINTS:
        rgb = np.array(_hex_to_rgb(p.hex), dtype=np.float64) / 255.0
        plab = rgb_to_lab(rgb.reshape(1, 3))[0]
        d = float(np.linalg.norm(lab - plab))
        if d < best_d:
            best_d = d
            best = p
    assert best is not None
    # Lab ΔE-ish: ~10 is close, ~40+ is weak
    conf = float(np.clip(1.0 - best_d / 55.0, 0.05, 0.92))
    return best, conf


def guess_paint_from_frames(
    frames: Sequence[dict[str, Any]],
) -> PaintGuess | None:
    """Cluster Lab samples across left/right repeater ego regions."""
    all_lab: list[np.ndarray] = []
    for fr in frames:
        cam = fr.get("camera")
        if cam not in ("left_repeater", "right_repeater", "left_pillar", "right_pillar"):
            continue
        path = Path(fr["path"])
        if not path.is_file():
            continue
        img = np.asarray(Image.open(path).convert("RGB"))
        gains = _road_sky_wb_gain(img)
        balanced = np.clip(img.astype(np.float64) * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        lab = sample_ego_paint_lab(balanced, cam)
        if len(lab):
            all_lab.append(lab)
    if not all_lab:
        return None
    samples = np.concatenate(all_lab, axis=0)
    cents, counts = _kmeans_lab(samples, k=3)
    if len(cents) == 0:
        return None
    # Prefer mid-chroma body cluster (skip near-gray road bleed / sky)
    scores = []
    for i, c in enumerate(cents):
        chroma = float(np.hypot(c[1], c[2]))
        # Body paint often has moderate L* and some chroma; white/black special-cased
        L = float(c[0])
        bodyish = 1.0
        if L < 8 or L > 98:
            bodyish = 0.3
        scores.append(counts[i] * bodyish * (1.0 + 0.02 * min(chroma, 40)))
    idx = int(np.argmax(scores))
    paint, conf = match_paint(cents[idx])
    # Boost confidence with sample count
    n = int(counts.sum())
    conf = float(np.clip(conf * (0.55 + 0.45 * min(1.0, n / 800.0)), 0.05, 0.92))
    return PaintGuess(
        code=paint.code,
        name=paint.name,
        hex=paint.hex,
        confidence=conf,
        lab=[float(x) for x in cents[idx]],
        method="lab_cluster_wb",
    )


# ---------------------------------------------------------------------------
# Make / model heuristics
# ---------------------------------------------------------------------------

@dataclass
class ModelGuess:
    model: str
    confidence: float
    reasons: list[str] = field(default_factory=list)


def infer_model_guesses(
    *,
    cameras: Sequence[str],
    hw_generation: str | None = None,
    silhouette_features: dict[str, float] | None = None,
    sei_hints: dict[str, Any] | None = None,
    vision_llm: bool = False,
) -> list[ModelGuess]:
    """Soft ranking of Model 3/Y/S/X (+ Highland/Juniper when cues exist).

    No permissive on-device car-make classifier is bundled; heuristics only.
    Optional vision_llm is off by default and currently a no-op stub.
    """
    cams = set(cameras)
    hw = (hw_generation or "unknown").lower()
    feats = silhouette_features or {}
    guesses: list[ModelGuess] = []

    # Camera set: HW3 4-cam common on 3/Y; pillars → HW4 (Highland 3 / Juniper Y / recent S/X)
    if {"left_pillar", "right_pillar"} & cams or hw == "hw4":
        guesses.append(
            ModelGuess(
                "Model 3 Highland",
                0.38,
                ["HW4 / pillar cameras often on refreshed Model 3"],
            )
        )
        guesses.append(
            ModelGuess(
                "Model Y Juniper",
                0.34,
                ["HW4 pillars also appear on refreshed Model Y"],
            )
        )
        guesses.append(ModelGuess("Model S", 0.18, ["HW4 also on recent S/X"]))
        guesses.append(ModelGuess("Model X", 0.16, ["HW4 also on recent S/X"]))
    elif cams >= {"front", "back", "left_repeater", "right_repeater"} or hw == "hw3":
        guesses.append(ModelGuess("Model Y", 0.42, ["HW3 four-camera TeslaCam is common on Model Y"]))
        guesses.append(ModelGuess("Model 3", 0.40, ["HW3 four-camera TeslaCam is common on Model 3"]))
        guesses.append(ModelGuess("Model S", 0.14, ["Less common four-cam HW3 layout"]))
        guesses.append(ModelGuess("Model X", 0.12, ["Less common four-cam HW3 layout"]))
    else:
        guesses.append(ModelGuess("Model 3", 0.28, ["Partial camera set; weak prior"]))
        guesses.append(ModelGuess("Model Y", 0.28, ["Partial camera set; weak prior"]))

    # Silhouette / geometry cues from repeater view (optional measured features)
    # mirror_height_norm: 0..1 fraction from bottom of mirror housing in ego region
    # fender_curve: higher → more arched crossover-like (Y) vs sedan (3)
    # camera_height_proxy: relative to horizon (higher → taller vehicle / SUV)
    mh = feats.get("mirror_height_norm")
    fc = feats.get("fender_curve")
    ch = feats.get("camera_height_proxy")
    if mh is not None or fc is not None or ch is not None:
        adj: dict[str, float] = {}
        if fc is not None and fc > 0.55:
            adj["Model Y"] = adj.get("Model Y", 0) + 0.08
            adj["Model Y Juniper"] = adj.get("Model Y Juniper", 0) + 0.06
            adj["Model 3"] = adj.get("Model 3", 0) - 0.04
        if ch is not None and ch > 0.55:
            adj["Model Y"] = adj.get("Model Y", 0) + 0.06
            adj["Model X"] = adj.get("Model X", 0) + 0.05
        if mh is not None and mh < 0.35:
            adj["Model 3"] = adj.get("Model 3", 0) + 0.05
            adj["Model 3 Highland"] = adj.get("Model 3 Highland", 0) + 0.04
        for g in guesses:
            if g.model in adj:
                g.confidence = float(np.clip(g.confidence + adj[g.model], 0.05, 0.9))
                g.reasons.append("repeater silhouette / mounting-height cues")

    # SEI: Tesla SEI (2025.44.25+) does not currently expose make/model fields in
    # the public dashcam protobuf we parse — only gear/speed/GPS/heading.
    # Keep a hook if future firmware adds VIN/model.
    if sei_hints:
        if sei_hints.get("model"):
            m = str(sei_hints["model"])
            guesses.insert(0, ModelGuess(m, 0.7, ["SEI metadata model field"]))
        else:
            for g in guesses:
                g.reasons.append("SEI present but no model field")

    if vision_llm:
        # Optional off-by-default hook — no network call unless explicitly enabled
        # and DASHSPLAT_VISION_LLM=1. Stub keeps license surface clean.
        if os.environ.get("DASHSPLAT_VISION_LLM") == "1":
            for g in guesses:
                g.reasons.append("vision-LLM step requested (not run in this build)")

    guesses.sort(key=lambda g: g.confidence, reverse=True)
    # Cap confidence — never a hard claim
    for g in guesses:
        g.confidence = float(min(g.confidence, 0.9))
    return guesses


def measure_silhouette_features(frames: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Cheap geometry proxies from repeater ego silhouette."""
    feats: dict[str, float] = {}
    curves = []
    heights = []
    for fr in frames:
        cam = fr.get("camera")
        if cam not in ("left_repeater", "right_repeater"):
            continue
        path = Path(fr["path"])
        if not path.is_file():
            continue
        img = np.asarray(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        mask = ego_mask_array(cam, w, h) == 0
        # Top edge of ego blob as fender curve proxy
        ys = []
        xs = []
        for x in range(0, w, max(1, w // 40)):
            col = mask[:, x]
            if col.any():
                y0 = int(np.argmax(col))
                ys.append(y0 / h)
                xs.append(x / w)
        if len(ys) >= 4:
            # curvature: variance of top edge
            curves.append(float(np.std(ys)))
            heights.append(1.0 - float(np.mean(ys)))
    if curves:
        feats["fender_curve"] = float(np.clip(np.mean(curves) * 8.0, 0.0, 1.0))
    if heights:
        feats["camera_height_proxy"] = float(np.clip(np.mean(heights), 0.0, 1.0))
        feats["mirror_height_norm"] = float(np.clip(1.0 - np.mean(heights), 0.0, 1.0))
    return feats


# ---------------------------------------------------------------------------
# Aggregate identity
# ---------------------------------------------------------------------------

@dataclass
class EgoIdentity:
    make: str = "Tesla"
    paint: PaintGuess | None = None
    model_guesses: list[ModelGuess] = field(default_factory=list)
    override: bool = False
    method: str = "heuristics+lab"
    paint_sources: tuple[str, ...] = PAINT_LIST_SOURCES
    device: str = "cpu"

    def top_model(self) -> ModelGuess | None:
        return self.model_guesses[0] if self.model_guesses else None

    def summary_line(self) -> str:
        model = self.top_model().model if self.top_model() else "Unknown"
        paint = self.paint.name if self.paint else "Unknown color"
        conf = 0.0
        if self.paint and self.top_model():
            conf = 0.5 * self.paint.confidence + 0.5 * self.top_model().confidence
        elif self.paint:
            conf = self.paint.confidence
        elif self.top_model():
            conf = self.top_model().confidence
        pct = int(round(conf * 100))
        return f"Your car: {model}, {paint}, {pct}%"

    def with_override(
        self,
        *,
        model: str | None = None,
        paint_name: str | None = None,
        paint_code: str | None = None,
        paint_hex: str | None = None,
    ) -> EgoIdentity:
        paint = self.paint
        if paint_code or paint_name:
            match = None
            for p in TESLA_FACTORY_PAINTS:
                if paint_code and p.code.upper() == paint_code.upper():
                    match = p
                    break
                if paint_name and p.name.lower() == paint_name.lower():
                    match = p
                    break
            if match:
                paint = PaintGuess(
                    code=match.code,
                    name=match.name,
                    hex=paint_hex or match.hex,
                    confidence=1.0,
                    method="user_override",
                )
            else:
                paint = PaintGuess(
                    code=paint_code or "CUSTOM",
                    name=paint_name or "Custom",
                    hex=paint_hex or "#808080",
                    confidence=1.0,
                    method="user_override",
                )
        guesses = list(self.model_guesses)
        if model:
            guesses = [ModelGuess(model, 1.0, ["user override"])] + [
                g for g in guesses if g.model != model
            ]
        return replace(self, paint=paint, model_guesses=guesses, override=True, method="user_override")

    def to_dict(self) -> dict[str, Any]:
        return {
            "make": self.make,
            "paint": asdict(self.paint) if self.paint else None,
            "model_guesses": [asdict(g) for g in self.model_guesses],
            "override": self.override,
            "method": self.method,
            "summary": self.summary_line(),
            "paint_sources": list(self.paint_sources),
            "device": self.device,
        }


def _pick_accel_device() -> str:
    try:
        from dashsplat.accel.device import detect_host

        host = detect_host()
        if host.is_apple_silicon and host.coreml_compute_units in ("ALL", "all", "CPU_AND_NE"):
            return "coreml_ane"
        if host.is_apple_silicon and host.metal_available:
            return "metal_gpu"
        if host.mlx_device:
            return str(host.mlx_device)
    except Exception:
        pass
    return "cpu"


def identify_ego_vehicle(
    frames: Sequence[dict[str, Any]],
    *,
    cameras: Sequence[str],
    hw_generation: str | None = None,
    sei_hints: dict[str, Any] | None = None,
    vision_llm: bool = False,
    override: dict[str, Any] | None = None,
) -> EgoIdentity:
    device = _pick_accel_device()
    # Sampling is NumPy; ANE/GPU reserved for optional Core ML path later.
    paint = guess_paint_from_frames(frames)
    feats = measure_silhouette_features(frames)
    models = infer_model_guesses(
        cameras=cameras,
        hw_generation=hw_generation,
        silhouette_features=feats,
        sei_hints=sei_hints,
        vision_llm=vision_llm,
    )
    identity = EgoIdentity(
        paint=paint,
        model_guesses=models,
        method="heuristics+lab",
        device=device,
    )
    if override:
        identity = identity.with_override(
            model=override.get("model"),
            paint_name=override.get("paint_name") or override.get("paint"),
            paint_code=override.get("paint_code"),
            paint_hex=override.get("paint_hex"),
        )
    return identity


def default_override_path() -> Path:
    try:
        from dashsplat.bootstrap import default_support_dir

        return default_support_dir() / "ego_override.json"
    except Exception:
        return Path.home() / ".dashsplat" / "ego_override.json"


def load_override(path: Path | None = None) -> dict[str, Any] | None:
    p = path or default_override_path()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def save_override(data: dict[str, Any], path: Path | None = None) -> Path:
    p = path or default_override_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    return p


def enrich_event_dict_with_identity(
    event_dict: dict[str, Any],
    *,
    frame_paths: dict[str, Sequence[Path]] | None = None,
    frames: Sequence[dict[str, Any]] | None = None,
    hw_generation: str | None = None,
    sei_hints: dict[str, Any] | None = None,
    override: dict[str, Any] | None = None,
    vision_llm: bool = False,
) -> dict[str, Any]:
    """Attach ego_identity to an event.to_dict() payload."""
    fr: list[dict[str, Any]] = list(frames or [])
    if frame_paths:
        for cam, paths in frame_paths.items():
            for p in paths:
                fr.append({"path": Path(p), "camera": cam})
    cameras = event_dict.get("cameras") or []
    ov = override if override is not None else load_override()
    identity = identify_ego_vehicle(
        fr,
        cameras=cameras,
        hw_generation=hw_generation,
        sei_hints=sei_hints,
        vision_llm=vision_llm,
        override=ov,
    )
    out = dict(event_dict)
    out["ego_identity"] = identity.to_dict()
    out["ego_summary"] = identity.summary_line()
    return out


def extract_repeater_stills_for_identity(
    event: Any,
    work_dir: Path,
    *,
    max_frames: int = 4,
) -> list[dict[str, Any]]:
    """Pull a few stills from left/right repeater MP4s for paint sampling."""
    from dashsplat.frames import extract_frames_ffmpeg, extract_placeholder_frames

    work_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for seg in getattr(event, "segments", []) or []:
        for cam, path in seg.cameras.items():
            if cam not in ("left_repeater", "right_repeater", "left_pillar", "right_pillar"):
                continue
            out = work_dir / cam
            try:
                paths = extract_frames_ffmpeg(path, out, max_frames=max_frames, prefix=cam)
            except Exception:
                paths = []
            if not paths:
                paths = extract_placeholder_frames(
                    path, out, max_frames=max_frames, prefix=cam
                )
            for p in paths:
                frames.append({"path": p, "camera": cam})
    return frames
