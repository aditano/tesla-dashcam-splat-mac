from __future__ import annotations

import numpy as np

from dashsplat.trainers.colmap_io import write_colmap_sparse


def test_one_camera_per_physical_shared_intrinsics(tmp_path):
    means = np.zeros((3, 3), np.float32)
    colors = np.ones((3, 3), np.float32)
    cameras = []
    for i in range(4):
        cameras.append(
            {
                "camera": "front" if i < 2 else "back",
                "file_name": f"{'front' if i < 2 else 'back'}_{i:04d}.jpg",
                "transform_matrix": np.eye(4).tolist(),
                "fl_x": 500.0 if i < 2 else 300.0,
                "fl_y": 500.0 if i < 2 else 300.0,
                "cx": 160.0,
                "cy": 120.0,
                "w": 320,
                "h": 240,
            }
        )
    out = write_colmap_sparse(tmp_path / "proj", means, colors, cameras)
    cam_txt = (out / "sparse" / "0" / "cameras.txt").read_text()
    data_lines = [ln for ln in cam_txt.splitlines() if ln and not ln.startswith("#")]
    assert len(data_lines) == 2  # front + back only
    # Distortion all zero → PINHOLE (do not invent k/p or pretend OPENCV)
    assert "PINHOLE" in data_lines[0]
    assert "OPENCV" not in data_lines[0]
    # Existing focals preserved, not reinvented
    assert "500.0" in data_lines[0] or "500" in data_lines[0]
    img_txt = (out / "sparse" / "0" / "images.txt").read_text()
    # Four images referencing camera ids 1 or 2
    img_data = [ln for ln in img_txt.splitlines() if ln and not ln.startswith("#")]
    # Every other line is empty POINTS2D
    named = [ln for ln in img_data if "front_" in ln or "back_" in ln]
    assert len(named) == 4
