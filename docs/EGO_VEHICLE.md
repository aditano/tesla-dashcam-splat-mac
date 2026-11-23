# Ego vehicle (path-following 3D proxy)

The ego body is **masked out of training**. In the viewer, a low-poly glTF fills that gap and follows the driven path from SEI telemetry (position + heading), scrubbed with the timeline.

## Default mesh

| | |
| --- | --- |
| File | `viewer/assets/ego/sedan_crossover.glb` |
| License | **MIT** (original DashSplat procedural low-poly sedan/crossover) |
| Cite | `viewer/assets/ego/LICENSE.txt` |

No automatic paint/make/model identification. A later version can swap in a Model 3/Y/S/X + paint mesh via `ego/ego_vehicle.manifest.json` (`swap_hint`) without changing the placement API.

## Pipeline output

Each job folder gets:

```
ego/
  sedan_crossover.glb
  LICENSE.txt
  ego_vehicle.manifest.json
  ego_path.json          # dense ENU samples for the scrubber
ego_path.json           # mirror at job root
```

## Viewer

- Loads PLY + `egoPath` + `egoGlb` query params
- Places the car on the path (wheels on ground, yaw from SEI)
- Timeline scrubber interpolates pose
- **Show ego car** toggle (Swift + HTML checkbox)
- Path drawn as a blue polyline

## Export

The `ego/` folder is a **separate object** alongside `splat.ply` / `cameras.json` — include it when packaging results.
