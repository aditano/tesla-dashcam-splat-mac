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

    p_quick = sub.add_parser("run", help="Alias: build")
    add_build_args(p_quick)

    args = parser.parse_args(argv)

    if args.cmd == "list":
        from dashsplat.events import scan_teslacam

        events = scan_teslacam(args.folder)
        if args.json:
            print(json.dumps([e.to_dict() for e in events], indent=2))
        else:
            if not events:
                print("No events found", file=sys.stderr)
                return 2
            for e in events:
                cams = ",".join(e.cameras)
                loc = e.location_label or "-"
                print(
                    f"{e.id}\t{e.label}\t{e.duration_sec:.0f}s\t{cams}\t{loc}\t{e.source_folder or ''}"
                )
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
    known = {"list", "build", "bootstrap", "run", "refine", "bench", "help"}
    if argv and not argv[0].startswith("-") and argv[0] not in known:
        return main(["run", *argv])
    if not argv:
        return main(["--help"])
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main_folder_argv())
