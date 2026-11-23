from dashsplat.timing_offsets import DEFAULT_OFFSETS_SEC, apply_offset, load_offsets, save_offsets


def test_default_offsets_zero():
    assert all(v == 0.0 for v in DEFAULT_OFFSETS_SEC.values())


def test_apply_offset():
    assert apply_offset(1.5, "back", {"back": 0.04}) == 1.54
    assert apply_offset(1.0, "front") == 1.0


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "timing_offsets.json"
    save_offsets(path, {"front": 0.0, "back": 0.03})
    loaded = load_offsets(path)
    assert loaded["back"] == 0.03
    assert "left_repeater" in loaded
