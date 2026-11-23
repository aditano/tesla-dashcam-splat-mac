"""Pluggable 3DGS trainers — Brush (default), msplat, seeded fallback.

OpenSplat (AGPL) is intentionally not supported.
"""

from __future__ import annotations

from dashsplat.trainers.base import Trainer, TrainerConfig, TrainResult
from dashsplat.trainers.registry import available_trainers, get_trainer, resolve_trainer_name

__all__ = [
    "Trainer",
    "TrainerConfig",
    "TrainResult",
    "available_trainers",
    "get_trainer",
    "resolve_trainer_name",
]
