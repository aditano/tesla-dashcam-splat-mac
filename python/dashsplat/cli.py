#!/usr/bin/env python3
"""dashsplat CLI — TeslaCam → MapAnything → Brush/msplat 3DGS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dashsplat",
        description="TeslaCam → MapAnything (MLX) → 3D Gaussian Splat (Brush/msplat Metal)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List events in a TeslaCam folder")
    p_list.add_argument("folder", type=Path)
    p_list.add_argument("--json", action="store_true")
    p_list.add_argument(
        "--no-ego-identity",
        action="store_true",
        help="Skip ego paint/model identification (faster)",
    )
    p_list.add_argument(
        "--ego-override",
        type=Path,
        default=None,
        help="JSON override {model,paint_name,paint_code} for ego identity",
    )

    p_ego = sub.add_parser("ego-override", help="Set/clear owner vehicle paint+model override")
    p_ego.add_argument("--model", default=None, help='e.g. "Model Y"')
    p_ego.add_argument("--paint", dest="paint_name", default=None, help='e.g. "Pearl White Multi-Coat"')
    p_ego.add_argument("--paint-code", default=None, help="e.g. PPSW")
    p_ego.add_argument("--clear", action="store_true", help="Remove override file")
    p_ego.add_argument("--path", type=Path, default=None, help="Override JSON path")

    def add_build_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("folder", type=Path)
        p.add_argument("--event", action="append", dest="events", help="Event ID (repeatable)")
        p.add_argument("--all", action="store_true", help="Build every event (batch queue)")
        p.add_argument("--out", type=Path, default=None)
        p.add_argument("--tier", choices=["preview", "full"], default="preview")
        p.add_argument("--trainer", choices=["brush", "msplat", "seeded"], default=None)
        p.add_argument("--no-mlx", action="store_true")
        p.add_argument("--nc-mapanything", action="store_true", help="Personal-use CC BY-NC MapAnything (off by default)")
        p.add_argument("--force-fallback-trainer", action="store_true")
        p.add_argument("--iterations", type=int, default=None)
        p.add_argument("--max-frames", type=int, default=None)

    p_build = sub.add_parser("build", help="Build Preview (default) or Full splat(s)")
    add_build_args(p_build)

    p_refine = sub.add_parser("refine", help="Refine an event to Full quality")
    add_build_args(p_refine)
    # refine forces full tier
    p_refine.set_defaults(tier="full")

    p_bench = sub.add_parser("bench", help="Held-out proxy PSNR + stage timings")
    p_bench.add_argument("folder", type=Path)
    p_bench.add_argument("--event", dest="event_id", default=None)
    p_bench.add_argument("--out", type=Path, default=None)
    p_bench.add_argument("--trainer", default="seeded")
    p_bench.add_argument("--no-mlx", action="store_true")

    p_boot = sub.add_parser("bootstrap", help="Install deps/models/trainers into Application Support")
    p_boot.add_argument("--support-dir", type=Path, default=None)
    p_boot.add_argument("--no-mlx", action="store_true")
    p_boot.add_argument("--no-model", action="store_true")
    p_boot.add_argument("--no-trainers", action="store_true")

    p_doc = sub.add_parser("doctor", help="Report chip, RAM, MLX, Core ML, Metal, VideoToolbox")
    p_doc.add_argument("--json", action="store_true", default=True)
    p_doc.add_argument("--text", action="store_true", help="Human-readable instead of JSON")

    p_quick = sub.add_parser("run", help="Alias: build")
    add_build_args(p_quick)

    args = parser.parse_args(argv)

    if args.cmd == "list":
        from dashsplat.events import scan_teslacam
        from dashsplat.hw_profile import profile_for_cameras

        events = scan_teslacam(args.folder)
        payloads: list[dict] = []
        if args.no_ego_identity:
            payloads = [e.to_dict() for e in events]
        else:
            import tempfile

            from dashsplat.ego_identity import (
                enrich_event_dict_with_identity,
                extract_repeater_stills_for_identity,
                load_override,
            )

            override = None
            if args.ego_override:
                override = json.loads(Path(args.ego_override).read_text())
            else:
                override = load_override()
            with tempfile.TemporaryDirectory(prefix="dashsplat-ego-") as td:
                for e in events:
                    hw = profile_for_cameras(e.cameras)
                    frames = extract_repeater_stills_for_identity(
                        e, Path(td) / e.id, max_frames=3
                    )
                    payloads.append(
                        enrich_event_dict_with_identity(
                            e.to_dict(),
                            frames=frames,
                            hw_generation=hw.generation.value,
                            override=override,
                        )
                    )
        if args.json:
            print(json.dumps(payloads, indent=2))
        else:
            if not payloads:
                print("No events found", file=sys.stderr)
                return 2
            for d in payloads:
                cams = ",".join(d.get("cameras") or [])
                loc = d.get("location_label") or "-"
                ego = d.get("ego_summary") or ""
                print(
                    f"{d['id']}\t{d['label']}\t{d.get('duration_sec', 0):.0f}s\t"
                    f"{cams}\t{loc}\t{d.get('source_folder') or ''}\t{ego}"
                )
        return 0

    if args.cmd == "ego-override":
        from dashsplat.ego_identity import default_override_path, load_override, save_override

        path = args.path or default_override_path()
        if args.clear:
            if path.is_file():
                path.unlink()
            print(json.dumps({"cleared": True, "path": str(path)}))
            return 0
        data = load_override(path) or {}
        if args.model:
            data["model"] = args.model
        if args.paint_name:
            data["paint_name"] = args.paint_name
        if args.paint_code:
            data["paint_code"] = args.paint_code
        if not data:
            print(json.dumps({"error": "provide --model/--paint/--paint-code or --clear"}, indent=2))
            return 2
        save_override(data, path)
        print(json.dumps({"saved": data, "path": str(path)}, indent=2))
        return 0

    if args.cmd == "bootstrap":
        from dashsplat.bootstrap import bootstrap

        result = bootstrap(
            support_dir=args.support_dir,
            with_mlx=not args.no_mlx,
            download_model=not args.no_model,
            with_trainers=not args.no_trainers,
        )
        print(json.dumps(result, indent=2))
        return int(result.get("exit_code", 1))

    if args.cmd == "doctor":
        from dashsplat.doctor import run_doctor

        report = run_doctor(as_json=not args.text)
        # Exit 0 always for diagnostics; warnings are in the payload.
        return 0 if report else 1

    if args.cmd == "bench":
        from dashsplat.bench import run_bench
        from dashsplat.bootstrap import default_support_dir

        out = args.out or (default_support_dir() / "bench" / args.folder.name)
        report = run_bench(
            args.folder,
            args.event_id,
            out,
            use_mlx=not args.no_mlx,
            trainer=args.trainer,
        )
        return int(report.get("exit_code", 1))

    if args.cmd in ("build", "run", "refine"):
        from dashsplat.bootstrap import default_support_dir
        from dashsplat.pipeline import build_folder

        out = args.out or (default_support_dir() / "jobs" / args.folder.name)
        summary = build_folder(
            args.folder,
            out,
            event_ids=args.events,
            all_events=bool(args.all),
            tier=args.tier,
            use_mlx=not args.no_mlx,
            use_nc_mapanything=bool(getattr(args, "nc_mapanything", False)),
            force_fallback_trainer=bool(getattr(args, "force_fallback_trainer", False)),
            trainer_name=getattr(args, "trainer", None),
            iterations=getattr(args, "iterations", None),
            max_frames_per_camera=getattr(args, "max_frames", None),
        )
        print(json.dumps({"type": "summary", "exit_code": summary["exit_code"], "ok": summary["ok"], "tier": summary.get("tier")}, indent=2))
        return int(summary.get("exit_code", 1))

    return 1


def main_folder_argv() -> int:
    argv = sys.argv[1:]
    known = {"list", "build", "bootstrap", "run", "refine", "bench", "doctor", "ego-override", "help"}
    if argv and not argv[0].startswith("-") and argv[0] not in known:
        return main(["run", *argv])
    if not argv:
        return main(["--help"])
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main_folder_argv())
