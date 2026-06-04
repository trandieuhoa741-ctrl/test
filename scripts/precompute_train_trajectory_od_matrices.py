from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts._common import (
        cache_bounds,
        collect_trajectory_files,
        extract_chengdu_yyyymmdd,
        load_pickle,
        parse_bounds,
        parse_date_range,
        parse_duration_clip,
        save_pickle,
        write_matrix_long_csv,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from _common import (
        cache_bounds,
        collect_trajectory_files,
        extract_chengdu_yyyymmdd,
        load_pickle,
        parse_bounds,
        parse_date_range,
        parse_duration_clip,
        save_pickle,
        write_matrix_long_csv,
    )

from taxi_dispatch.chengdu import build_chengdu_trajectory_od_matrices
from taxi_dispatch.grid import HexGrid
from taxi_dispatch.road_network import RoadCostMatrices


DEFAULT_RAW_DIR = Path("data/2016年成都滴滴轨迹数据")
DEFAULT_TRAIN_CACHE = Path("data/processed/chengdu_train_20161108_20161130_N142_T108.pkl")
DEFAULT_TEST_CACHE = Path("data/processed/chengdu_test_20161101_20161107_N142_T108.pkl")
DEFAULT_OUT = Path("data/processed/chengdu_train_trajectory_od_matrix_20161108_20161130_N142.npz")
DEFAULT_TRAIN_RANGE = (20161108, 20161130)


def build_matrices_from_train_cache(
    pack: dict,
    *,
    default_speed_kmph: float = 30.0,
    min_samples_per_od: int = 1,
) -> tuple[RoadCostMatrices, dict[str, object]]:
    grid = pack["grid"]
    demand = pack["demand"]
    num_cells = int(grid.num_cells)
    distance_sum = np.zeros((num_cells, num_cells), dtype=np.float64)
    time_sum = np.zeros((num_cells, num_cells), dtype=np.float64)
    sample_count = np.zeros((num_cells, num_cells), dtype=np.int64)

    rows_used = 0
    events_by_day = getattr(demand, "events_by_day", None)
    if events_by_day:
        for day_steps in events_by_day:
            for events in day_steps:
                for event in events:
                    origin = int(event.origin)
                    destination = int(event.destination)
                    if not (0 <= origin < num_cells and 0 <= destination < num_cells):
                        continue
                    minutes = float(event.trip_minutes)
                    if not np.isfinite(minutes) or minutes <= 0.0:
                        continue
                    if event.pickup_xy is not None and event.dropoff_xy is not None:
                        pickup = np.asarray(event.pickup_xy, dtype=np.float32)
                        dropoff = np.asarray(event.dropoff_xy, dtype=np.float32)
                        distance = float(np.linalg.norm(pickup - dropoff))
                    else:
                        distance = float(grid.distance(origin, destination))
                    if not np.isfinite(distance) or distance < 0.0:
                        continue
                    distance_sum[origin, destination] += distance
                    time_sum[origin, destination] += minutes
                    sample_count[origin, destination] += 1
                    rows_used += 1
    else:
        mean_trip_minutes = np.asarray(getattr(demand, "mean_trip_minutes", np.empty((0,))), dtype=np.float32)
        rates = np.asarray(getattr(demand, "rates", np.empty((0,))), dtype=np.float32)
        od_probs = np.asarray(getattr(demand, "od_probs", np.empty((0,))), dtype=np.float32)
        if mean_trip_minutes.ndim != 3 or od_probs.ndim != 3 or rates.ndim != 2:
            raise ValueError("train cache demand must provide events_by_day or mean_trip_minutes/rates/od_probs")
        od_counts = rates[:, :, None] * od_probs
        counts = od_counts.sum(axis=0)
        weighted_minutes = (mean_trip_minutes * od_counts).sum(axis=0)
        observed = counts > 0.0
        time_sum[observed] = weighted_minutes[observed]
        sample_count[observed] = np.rint(counts[observed]).astype(np.int64)
        rows_used = int(sample_count.sum())
        for origin in range(num_cells):
            for destination in range(num_cells):
                if sample_count[origin, destination] > 0:
                    distance_sum[origin, destination] = float(grid.distance(origin, destination)) * sample_count[origin, destination]

    fallback_distance = np.linalg.norm(
        np.asarray(grid.xy, dtype=np.float32)[:, None, :]
        - np.asarray(grid.xy, dtype=np.float32)[None, :, :],
        axis=2,
    ).astype(np.float32)
    observed_distance_total = float(distance_sum.sum())
    observed_time_total = float(time_sum.sum())
    observed_speed_kmph = (
        observed_distance_total / max(observed_time_total / 60.0, 1e-6)
        if observed_distance_total > 0.0 and observed_time_total > 0.0
        else float(default_speed_kmph)
    )
    if not np.isfinite(observed_speed_kmph) or observed_speed_kmph <= 0.0:
        observed_speed_kmph = float(default_speed_kmph)
    fallback_time = fallback_distance / max(observed_speed_kmph, 1e-6) * 60.0

    min_samples = max(int(min_samples_per_od), 1)
    observed_mask = sample_count >= min_samples
    distance = fallback_distance.astype(np.float64)
    time = fallback_time.astype(np.float64)
    np.divide(distance_sum, sample_count, out=distance, where=observed_mask)
    np.divide(time_sum, sample_count, out=time, where=observed_mask)
    np.fill_diagonal(distance, 0.0)
    np.fill_diagonal(time, 0.0)

    off_diagonal = ~np.eye(num_cells, dtype=bool)
    fallback_count = int((~observed_mask & off_diagonal).sum())
    matrices = RoadCostMatrices(
        distance_km=distance.astype(np.float32),
        time_minutes=time.astype(np.float32),
        fallback_distance_count=fallback_count,
        fallback_time_count=fallback_count,
    )
    stats = {
        "source": "train_cache_event_od_matrices",
        "rows_used": rows_used,
        "days": int(len(events_by_day)) if events_by_day else None,
        "num_cells": num_cells,
        "observed_od_pairs": int((observed_mask & off_diagonal).sum()),
        "fallback_distance_count": fallback_count,
        "fallback_time_count": fallback_count,
        "fallback_distance_ratio": float(fallback_count / max(num_cells * num_cells - num_cells, 1)),
        "fallback_time_ratio": float(fallback_count / max(num_cells * num_cells - num_cells, 1)),
        "mean_observed_speed_kmph": float(observed_speed_kmph),
        "min_samples_per_od": min_samples,
    }
    return matrices, stats


def inject_matrix(path: Path, matrices: RoadCostMatrices, matrix_path: Path) -> None:
    pack = load_pickle(path)
    pack["road_distance_matrix"] = matrices.distance_km
    pack["road_time_matrix"] = matrices.time_minutes
    pack["road_matrix_path"] = str(matrix_path)
    pack["road_matrix_source"] = "train_trajectory_od"
    pack["road_matrix_fallback_distance_count"] = int(matrices.fallback_distance_count)
    pack["road_matrix_fallback_time_count"] = int(matrices.fallback_time_count)
    save_pickle(path, pack)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute OD distance/time matrices from Chengdu training trajectories only. "
            "Defaults use 2016-11-08 through 2016-11-30 and output the same .npz format as --road-matrix."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--train-cache", type=Path, default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--test-cache", type=Path, default=DEFAULT_TEST_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stats-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument(
        "--source",
        choices=["cache", "raw"],
        default="cache",
        help="cache is fast and uses the train cache generated from 2016-11-08..30; raw rescans original GPS archives.",
    )
    parser.add_argument("--train-date-range", type=parse_date_range, default=DEFAULT_TRAIN_RANGE)
    parser.add_argument("--bounds", type=parse_bounds, default=None)
    parser.add_argument("--grid-padding-km", type=float, default=2.5)
    parser.add_argument("--default-speed-kmph", type=float, default=30.0)
    parser.add_argument("--duration-clip-minutes", type=parse_duration_clip, default=(1.0, 180.0))
    parser.add_argument("--min-samples-per-od", type=int, default=1)
    parser.add_argument(
        "--inject-caches",
        action="store_true",
        help="Store the train-only trajectory OD matrices inside train and test cache pickle files.",
    )
    parser.add_argument(
        "--no-test-cache-inject",
        action="store_true",
        help="With --inject-caches, inject only the train cache.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_pack = load_pickle(args.train_cache)
    grid = train_pack["grid"]
    bounds = args.bounds or cache_bounds(train_pack)
    if bounds is None:
        raise ValueError("bounds were not found in train cache stats; pass --bounds explicitly")

    start_date, end_date = args.train_date_range
    all_files: list[Path] = []
    train_paths: list[Path] = []
    skipped_test_or_other = 0
    undated: list[str] = []
    if args.source == "cache":
        matrices, stats = build_matrices_from_train_cache(
            train_pack,
            default_speed_kmph=args.default_speed_kmph,
            min_samples_per_od=args.min_samples_per_od,
        )
    else:
        all_files = collect_trajectory_files(args.raw_dir)
        if not all_files:
            raise FileNotFoundError(
                f"no trajectory CSV/tar.gz files found under {args.raw_dir}; "
                "raw source mode needs the original training trajectory archives"
            )
        for path in all_files:
            date = extract_chengdu_yyyymmdd(path)
            if date is None:
                undated.append(str(path))
                continue
            if start_date <= date <= end_date:
                train_paths.append(path)
            else:
                skipped_test_or_other += 1
        if not train_paths:
            raise RuntimeError(f"no trajectory files found in train date range {start_date}-{end_date}")

        assignment_grid = HexGrid.create_geographic_fixed(
            bounds,
            cell_width_km=float(grid.cell_width_km),
            padding_km=float(args.grid_padding_km),
        )
        matrices, stats = build_chengdu_trajectory_od_matrices(
            train_paths,
            grid=grid,
            assignment_grid=assignment_grid,
            bounds=bounds,
            filter_to_bounds=True,
            duration_clip_minutes=args.duration_clip_minutes,
            default_speed_kmph=args.default_speed_kmph,
            min_samples_per_od=args.min_samples_per_od,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    matrices.save_npz(args.out)
    stats_path = args.stats_out or args.out.with_suffix(".json")
    run_stats = {
        "train_cache": str(args.train_cache),
        "test_cache": str(args.test_cache) if args.test_cache else None,
        "road_matrix": str(args.out),
        "source": "train_trajectory_od",
        "matrix_source_mode": args.source,
        "train_date_range": [start_date, end_date],
        "all_files": len(all_files),
        "train_files": len(train_paths),
        "skipped_test_or_other_files": skipped_test_or_other,
        "undated_files": undated[:20],
        "bounds": bounds,
        "grid_padding_km": float(args.grid_padding_km),
        **stats,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(run_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.csv_out:
        write_matrix_long_csv(
            args.csv_out,
            matrices,
            distance_field="trajectory_distance_km",
            time_field="trajectory_time_min",
        )

    if args.inject_caches:
        inject_matrix(args.train_cache, matrices, args.out)
        if not args.no_test_cache_inject and args.test_cache:
            inject_matrix(args.test_cache, matrices, args.out)

    print("saved train-only trajectory OD matrix:", args.out)
    print("saved stats:", stats_path)
    print("train files:", len(train_paths))
    print("rows used:", stats["rows_used"])
    print("observed od pairs:", stats["observed_od_pairs"])
    print("fallback ratio:", f"{stats['fallback_time_ratio']:.4f}")
    if args.inject_caches:
        print("injected train cache:", args.train_cache)
        if not args.no_test_cache_inject and args.test_cache:
            print("injected test cache:", args.test_cache)


if __name__ == "__main__":
    main()
