# HW3 (and HW4) TeslaCam support

Owner target: **HW3 first-class** (four cameras). HW4 six-camera events are auto-detected when pillar files are present.

## Camera layout

| Generation | Cameras written to USB |
| --- | --- |
| **HW3 / AI3** | `front`, `back`, `left_repeater`, `right_repeater` (no pillars) |
| **HW4 / AI4** | same four **plus** `left_pillar`, `right_pillar` |

Sources:
- https://www.clipsalvage.com/articles/how-tesla-stores-video.html (HW3 four vs HW4 six)
- https://github.com/geeksbaek/tesla-dashcam-viewer/ (HW3/HW4 filename examples)

Folder layout: `TeslaCam/{RecentClips,SavedClips,SentryClips}` — see same ClipSalvage article and https://github.com/conscientsystems/teslacam.

## Resolution and FPS (cited — prefer probing real clips)

Dashcam encodes on HW3 are commonly reported as **1280×960 at about 36 fps**:

- https://github.com/conscientsystems/teslacam — README: “Clips are 1280x960 at 36 fps”
- https://github.com/kandotrun/teslacam-viewer — “1280 × 960 at 36 fps, like HW3 cars”
- https://www.nasrtech.dev/tools/video-repair/tesla-dashcam-video-corrupted/ — “1280×960 H.264, about 36 fps”

These are **not invented factory specs**. DashSplat probes `ffprobe` when available and falls back to these cited nominals only when metadata is missing. Frame rate can vary; do not assume constant 30 fps (see Tesla Replay notes on `stts`).

## Intrinsics / extrinsics

Tables in `python/dashsplat/cameras.py` are **approximations** for MapAnything initialization (HFOV → pinhole K; rough body-frame positions). They are **not** factory calibration.

Refine with:
- Self-calibration approach documented in [R44VC0RP/dashsplat](https://github.com/R44VC0RP/dashsplat) (Apache-2.0; built on HW4 Model 3; upstream notes HW3 four-cam untested there)
- SEI poses from [teslamotors/dashcam](https://github.com/teslamotors/dashcam) when firmware provides them

## SEI telemetry

Only firmware **2025.44.25+ on HW3+** embeds SEI; parked clips may have none ([teslamotors/dashcam](https://github.com/teslamotors/dashcam)). Older clips get a clear UI/CLI message and MapAnything pose fallback (lower quality).

## Ego-car masks

Fixed per-camera polygons in `python/dashsplat/masks/ego.py` for HW3 four cams + HW4 pillars (approximations; refine from temporal variance on the owner’s car).

## Auto-detect

`dashsplat.hw_profile.detect_hw_profile` / `profile_for_cameras` choose HW3 vs HW4 from the camera set in the folder (pillars ⇒ HW4).
