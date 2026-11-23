"""Failing-first tests for ego paint + make/model identification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dashsplat.ego_identity import (
    TESLA_FACTORY_PAINTS,
    EgoIdentity,
    identify_ego_vehicle,
    infer_model_guesses,
    match_paint,
    sample_ego_paint_lab,
)
from dashsplat.events import scan_teslacam
from dashsplat.masks.ego import ego_mask_array


def _synthetic_repeater(
    path: Path,
    *,
    body_rgb: tuple[int, int, int],
    size: tuple[int, int] = (320, 240),
) -> Path:
    """Paint solid body color into the typical left_repeater ego region."""
    w, h = size
    img = Image.new("RGB", (w, h), (90, 140, 200))  # sky-ish
    # Road
    arr = np.asarray(img).copy()
    arr[h // 2 :, :] = (60, 60, 60)
    # Ego body on right side of left-repeater view (matches ego polygon)
    mask = ego_mask_array("left_repeater", w, h)
    # ego_mask: 0 = ego (ignore for splat). Fill those pixels with body paint.
    body = np.zeros_like(arr)
    body[:] = body_rgb
    ego = mask == 0
    arr[ego] = body[ego]
    # Add mild noise / specular candidates
    rng = np.random.default_rng(0)
    noise = rng.integers(-8, 8, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, quality=92)
    return path


def test_paint_catalog_has_cited_tesla_names():
    names = {p.name for p in TESLA_FACTORY_PAINTS}
    for required in (
        "Pearl White Multi-Coat",
        "Solid Black",
        "Stealth Grey",
        "Quicksilver",
        "Deep Blue Metallic",
        "Ultra Red",
        "Midnight Silver Metallic",
        "Red Multi-Coat",
        "Midnight Cherry Red",
    ):
        assert required in names
    assert all(p.code for p in TESLA_FACTORY_PAINTS)
    assert all(p.hex.startswith("#") for p in TESLA_FACTORY_PAINTS)


def test_match_paint_pearl_white():
    # Lab-ish white sample
    paint, conf = match_paint(np.array([95.0, 0.0, 0.0]))  # near D65 white in Lab L*
    assert "White" in paint.name or paint.code == "PPSW"
    assert conf > 0.2


def test_identify_ego_from_synthetic_white_frames(tmp_path: Path):
    frames = []
    for i in range(6):
        p = tmp_path / f"left_{i}.jpg"
        _synthetic_repeater(p, body_rgb=(235, 235, 230))
        frames.append({"path": p, "camera": "left_repeater"})
    for i in range(6):
        p = tmp_path / f"right_{i}.jpg"
        # Mirror: body on left for right_repeater
        w, h = 320, 240
        img = Image.new("RGB", (w, h), (90, 140, 200))
        arr = np.asarray(img).copy()
        arr[h // 2 :, :] = (60, 60, 60)
        mask = ego_mask_array("right_repeater", w, h)
        body = np.zeros_like(arr)
        body[:] = (235, 235, 230)
        arr[mask == 0] = body[mask == 0]
        Image.fromarray(arr).save(p)
        frames.append({"path": p, "camera": "right_repeater"})

    identity = identify_ego_vehicle(
        frames,
        cameras=["front", "back", "left_repeater", "right_repeater"],
        hw_generation="hw3",
    )
    assert isinstance(identity, EgoIdentity)
    assert identity.paint is not None
    assert "White" in identity.paint.name or identity.paint.code == "PPSW"
    assert identity.paint.confidence > 0.15
    assert identity.make == "Tesla"
    assert identity.model_guesses
    assert identity.model_guesses[0].confidence <= 1.0
    assert "Your car:" in identity.summary_line()


def test_identify_ego_red(tmp_path: Path):
    frames = []
    for i in range(5):
        p = tmp_path / f"L{i}.jpg"
        _synthetic_repeater(p, body_rgb=(160, 30, 40))
        frames.append({"path": p, "camera": "left_repeater"})
    identity = identify_ego_vehicle(frames, cameras=["left_repeater", "front"], hw_generation="hw3")
    assert identity.paint is not None
    assert identity.paint.code in {"PR01", "PPMR", "PR00", "PPSR"}


def test_model_guesses_hw3_four_cam():
    guesses = infer_model_guesses(cameras=["front", "back", "left_repeater", "right_repeater"], hw_generation="hw3")
    assert guesses
    names = [g.model for g in guesses]
    assert any(n.startswith("Model") for n in names)
    # Never a hard claim of 100%
    assert all(g.confidence < 0.95 for g in guesses)


def test_override_identity(tmp_path: Path):
    frames = [{"path": _synthetic_repeater(tmp_path / "a.jpg", body_rgb=(20, 20, 20)), "camera": "left_repeater"}]
    identity = identify_ego_vehicle(frames, cameras=["left_repeater"], hw_generation="hw3")
    overridden = identity.with_override(model="Model Y", paint_name="Solid Black", paint_code="PBSB")
    assert overridden.override is True
    assert overridden.model_guesses[0].model == "Model Y"
    assert overridden.paint and overridden.paint.code == "PBSB"
    assert overridden.paint.confidence == 1.0


def test_list_json_includes_ego_identity(tmp_path: Path):
    # Build a mini TeslaCam tree with colored repeater stubs + front/back
    root = tmp_path / "TeslaCam" / "SavedClips" / "2025-11-20_14-03-12"
    root.mkdir(parents=True)
    for cam in ("front", "back", "left_repeater", "right_repeater"):
        # Minimal mp4 stub so scan finds cameras; frames for identity come from jpgs via helper
        (root / f"2025-11-20_14-03-12-{cam}.mp4").write_bytes(
            b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64
        )
    (root / "event.json").write_text('{"city":"Palo Alto"}')
    events = scan_teslacam(tmp_path / "TeslaCam")
    assert events
    # Identity attachment is done at list time via enrich helper
    from dashsplat.ego_identity import enrich_event_dict_with_identity

    d = enrich_event_dict_with_identity(
        events[0].to_dict(),
        frame_paths={
            "left_repeater": [
                _synthetic_repeater(tmp_path / "lr.jpg", body_rgb=(240, 240, 235))
            ]
        },
        hw_generation="hw3",
    )
    assert "ego_identity" in d
    assert d["ego_identity"]["paint"]["name"]
