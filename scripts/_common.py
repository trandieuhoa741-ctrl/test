from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_project_root_on_path() -> Path:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


ensure_project_root_on_path()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def collect_trajectory_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.tar", "*.TAR", "*.tar.gz", "*.tgz", "*.csv"):
        files.extend(root.rglob(pattern))
    return sorted(set(files))


def extract_chengdu_yyyymmdd(path: Path) -> int | None:
    text = str(path)
    patterns = (
        r"(2016)[-_\u5e74]?11[-_\u6708]?([0-3]\d)",
        r"(201611[0-3]\d)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        if len(match.groups()) == 1:
            return int(match.group(1))
        return int(f"{match.group(1)}11{match.group(2)}")
    return None


def parse_bounds(value: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bounds must have four comma-separated floats")
    return (parts[0], parts[1], parts[2], parts[3])


def parse_date_range(value: str) -> tuple[int, int]:
    parts = [part.strip().replace("-", "") for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("date range must be START,END")
    start, end = (int(parts[0]), int(parts[1]))
    if start > end:
        raise argparse.ArgumentTypeError("date range START must be <= END")
    return start, end


def parse_duration_clip(value: str) -> tuple[float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 2 or parts[0] <= 0 or parts[0] > parts[1]:
        raise argparse.ArgumentTypeError("duration clip must be MIN,MAX with 0 < MIN <= MAX")
    return parts[0], parts[1]


def cache_bounds(pack: dict) -> tuple[float, float, float, float] | None:
    stats = pack.get("stats")
    value = stats.get("bounds") if isinstance(stats, dict) else getattr(stats, "bounds", None)
    if value is None:
        return None
    parts = [float(x) for x in value]
    if len(parts) != 4:
        return None
    return (parts[0], parts[1], parts[2], parts[3])


def assert_same_grid(train_grid: object, test_grid: object) -> None:
    if train_grid.num_cells != test_grid.num_cells:
        raise ValueError(f"train/test num_cells differ: {train_grid.num_cells} != {test_grid.num_cells}")
    if train_grid.coords != test_grid.coords:
        raise ValueError("train/test grid coords differ; rebuild test cache using the training selected cells")
    if train_grid.projection_origin != test_grid.projection_origin:
        raise ValueError("train/test projection_origin differ")
    if not np.allclose(train_grid.xy, test_grid.xy):
        raise ValueError("train/test grid xy coordinates differ")


def write_matrix_long_csv(path: Path, matrices: object, *, distance_field: str, time_field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    distance = np.asarray(matrices.distance_km)
    time = np.asarray(matrices.time_minutes)
    n = int(distance.shape[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["origin", "destination", distance_field, time_field],
        )
        writer.writeheader()
        for origin in range(n):
            for destination in range(n):
                writer.writerow(
                    {
                        "origin": origin,
                        "destination": destination,
                        distance_field: float(distance[origin, destination]),
                        time_field: float(time[origin, destination]),
                    }
                )


def road_matrix_stats(matrices: object) -> dict[str, object]:
    distance = np.asarray(matrices.distance_km, dtype=np.float32)
    time = np.asarray(matrices.time_minutes, dtype=np.float32)
    return {
        "num_cells": int(distance.shape[0]),
        "cell_nodes": list(getattr(matrices, "cell_nodes", ())),
        "distance_km_min": float(np.nanmin(distance)),
        "distance_km_max": float(np.nanmax(distance)),
        "time_min_min": float(np.nanmin(time)),
        "time_min_max": float(np.nanmax(time)),
        "distance_nonfinite": int((~np.isfinite(distance)).sum()),
        "time_nonfinite": int((~np.isfinite(time)).sum()),
        "fallback_distance_count": int(matrices.fallback_distance_count),
        "fallback_time_count": int(matrices.fallback_time_count),
        "fallback_distance_ratio": float(matrices.fallback_distance_count / max(distance.size, 1)),
        "fallback_time_ratio": float(matrices.fallback_time_count / max(time.size, 1)),
    }
