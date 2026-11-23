#!/usr/bin/env python3
"""dashsplat CLI — list / build TeslaCam events into 3D Gaussian Splats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dashsplat",
        description="TeslaCam → MapAnything (MLX) → 3D Gaussian Splat (OpenSplat/Metal)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List events in a TeslaCam folder")
    p_list.add_argument("folder", type=Path)
    p_list.add_argument("--json", action="store_true")

    p_build = sub.add_parser("build", help="Build splat(s) for selected events")
    p_build.add_argument("folder", type=Path)
    p_build.add_argument("--event", action="append", dest="events", help="Event ID (repeatable)")
    p_build.add_argument("--all", action="store_true", help="Build every event")
    p_build.add_argument("--out", type=Path, default=None)
    p_build.add_argument("--no-mlx", action="store_true")
    p_build.add_argument("--force-fallback-trainer", action="store_true")
    p_build.add_argument("--iterations", type=int, default=1500)
    p_build.add_argument("--max-frames", type=int, default=6)

    p_boot = sub.add_parser("bootstrap", help="Install deps/models into Application Support")
    p_boot.add_argument("--support-dir", type=Path, default=None)
    p_boot.add_argument("--no-mlx", action="store_true")
    p_boot.add_argument("--no-model", action="store_true")

    # Convenience: `dashsplat <folder> [--event ID] [--all]`
    p_quick = sub.add_parser("run", help="Alias: list then build")
    p_quick.add_argument("folder", type=Path)
    p_quick.add_argument("--event", action="append", dest="events")
    p_quick.add_argument("--all", action="store_true")
    p_quick.add_argument("--out", type=Path, default=None)
    p_quick.add_argument("--no-mlx", action="store_true")
    p_quick.add_argument("--force-fallback-trainer", action="store_true")

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
        )
        print(json.dumps(result, indent=2))
        return int(result.get("exit_code", 1))

    if args.cmd in ("build", "run"):
        from dashsplat.bootstrap import default_support_dir
        from dashsplat.pipeline import build_folder

        out = args.out or (default_support_dir() / "jobs" / args.folder.name)
        summary = build_folder(
            args.folder,
            out,
            event_ids=args.events,
            all_events=bool(args.all),
            use_mlx=not args.no_mlx,
            force_fallback_trainer=bool(getattr(args, "force_fallback_trainer", False)),
            iterations=getattr(args, "iterations", 1500),
            max_frames_per_camera=getattr(args, "max_frames", 6),
        )
        print(json.dumps({"type": "summary", "exit_code": summary["exit_code"], "ok": summary["ok"]}, indent=2))
        return int(summary.get("exit_code", 1))

    return 1


# Allow `python -m dashsplat.cli <folder>` style via a thin wrapper below
def main_folder_argv() -> int:
    """Entry used by console script when invoked as `dashsplat <folder> ...` without subcommand.

    If the first arg looks like a path, treat as `run`.
    """
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-") and argv[0] not in {
        "list",
        "build",
        "bootstrap",
        "run",
        "help",
    }:
        # dashsplat <folder> [--event ID] [--all]
        return main(["run", *argv])
    if not argv:
        return main(["--help"])
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main_folder_argv())
