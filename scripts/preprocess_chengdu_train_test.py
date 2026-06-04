from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

try:
    from scripts._common import collect_trajectory_files, extract_chengdu_yyyymmdd, parse_bounds, save_pickle
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from _common import collect_trajectory_files, extract_chengdu_yyyymmdd, parse_bounds, save_pickle

from taxi_dispatch.chengdu import (
    DEFAULT_PEAK_HOTSPOT_WINDOWS_TEXT,
    build_chengdu_trajectory_demand,
    select_high_demand_trajectory_cells,
)
from taxi_dispatch.chengdu import build_chengdu_trajectory_event_demand
from taxi_dispatch.env import EnvConfig
from taxi_dispatch.grid import DEFAULT_HEX_CENTER_SPACING_KM, HexGrid


DEFAULT_RAW_DIR = Path("data/2016年成都滴滴轨迹数据")
DEFAULT_OUT_DIR = Path("data/processed")

DEFAULT_BOUNDS = (
    103.95202562612084,
    30.563249036822317,
    104.168179909498,
    30.79947020070271,
)

TEST_RANGE = (20161101, 20161107)
TRAIN_RANGE = (20161108, 20161130)


def save_cache(
    path: Path,
    config: EnvConfig,
    demand: object,
    grid: HexGrid,
    initial_taxi_distribution: object,
    stats: object,
) -> None:
    save_pickle(
        path,
        {
            "config": config,
            "demand": demand,
            "grid": grid,
            "initial_taxi_distribution": initial_taxi_distribution,
            "stats": stats,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Chengdu trajectory files into leak-free train/test dispatch caches. "
            "The high-demand cells are selected from the training split only."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-cells", type=int, default=142)
    parser.add_argument("--fleet-size", type=int, default=6000)
    parser.add_argument("--horizon-steps", type=int, default=108)
    parser.add_argument("--step-minutes", type=int, default=10)
    parser.add_argument("--start-hour", type=float, default=6.0)
    parser.add_argument(
        "--cell-width-km",
        type=float,
        default=DEFAULT_HEX_CENTER_SPACING_KM,
        help="Adjacent hex-center spacing in km; default corresponds to average hex side length 1.22 km.",
    )
    parser.add_argument("--grid-padding-km", type=float, default=2.5)
    parser.add_argument("--min-orders-per-minute", type=float, default=1.0)
    parser.add_argument("--demand-scale", type=float, default=1.0)
    parser.add_argument("--peak-hotspot-multiplier", type=float, default=1.0)
    parser.add_argument("--peak-hotspot-windows", default=DEFAULT_PEAK_HOTSPOT_WINDOWS_TEXT)
    parser.add_argument("--peak-hotspot-cells", default="")
    parser.add_argument("--peak-hotspot-top-cells", type=int, default=10)
    parser.add_argument("--bounds", default=",".join(str(x) for x in DEFAULT_BOUNDS))
    parser.add_argument("--train-cache-name", default="")
    parser.add_argument("--test-cache-name", default="")
    parser.add_argument("--split-stats-name", default="chengdu_train_test_split_stats.json")
    demand_group = parser.add_mutually_exclusive_group()
    demand_group.add_argument(
        "--aggregate-od",
        dest="aggregate_od",
        action="store_true",
        default=True,
        help="Save aggregate OD demand. This is the default for hotspot shock evaluation.",
    )
    demand_group.add_argument(
        "--event-stream",
        dest="aggregate_od",
        action="store_false",
        help="Save one-minute event-stream demand for minute-level replay experiments.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = args.raw_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = parse_bounds(args.bounds)
    train_cache_name = args.train_cache_name or (
        f"chengdu_train_20161108_20161130_N{args.num_cells}_T{args.horizon_steps}.pkl"
    )
    test_cache_name = args.test_cache_name or (
        f"chengdu_test_20161101_20161107_N{args.num_cells}_T{args.horizon_steps}.pkl"
    )
    train_cache = out_dir / train_cache_name
    test_cache = out_dir / test_cache_name
    split_stats = out_dir / args.split_stats_name

    all_files = collect_trajectory_files(raw_dir)
    if not all_files:
        raise FileNotFoundError(f"No trajectory files found under {raw_dir}")

    dated: list[tuple[Path, int]] = []
    undated: list[str] = []
    for path in all_files:
        date = extract_chengdu_yyyymmdd(path)
        if date is None:
            undated.append(str(path))
        else:
            dated.append((path, date))

    train_paths = [path for path, date in dated if TRAIN_RANGE[0] <= date <= TRAIN_RANGE[1]]
    test_paths = [path for path, date in dated if TEST_RANGE[0] <= date <= TEST_RANGE[1]]

    print("all files:", len(all_files))
    print("dated files:", len(dated))
    print("train files:", len(train_paths))
    print("test files:", len(test_paths))
    print("undated files:", len(undated))

    if not train_paths:
        raise RuntimeError("train_paths is empty. Check file names/date parsing.")
    if not test_paths:
        raise RuntimeError("test_paths is empty. Check file names/date parsing.")

    config = EnvConfig(
        num_cells=args.num_cells,
        cell_width_km=args.cell_width_km,
        fleet_size=args.fleet_size,
        horizon_steps=args.horizon_steps,
        step_minutes=args.step_minutes,
        demand_scale=args.demand_scale,
        seed=0,
    )

    assignment_grid = HexGrid.create_geographic_fixed(
        bounds,
        cell_width_km=config.cell_width_km,
        padding_km=args.grid_padding_km,
    )

    selected_coords = select_high_demand_trajectory_cells(
        train_paths,
        grid=assignment_grid,
        num_cells=config.num_cells,
        horizon_steps=config.horizon_steps,
        step_minutes=config.step_minutes,
        start_hour=args.start_hour,
        bounds=bounds,
        average_by_day=True,
        filter_to_bounds=True,
        min_orders_per_minute=args.min_orders_per_minute,
    )

    grid = HexGrid.from_axial_coords(
        selected_coords,
        cell_width_km=assignment_grid.cell_width_km,
        projection_origin=assignment_grid.projection_origin,
    )
    config = replace(config, cell_width_km=grid.cell_width_km)

    demand_builder = build_chengdu_trajectory_demand if args.aggregate_od else build_chengdu_trajectory_event_demand
    demand_kind = "aggregate_od" if args.aggregate_od else "event_stream"

    train_demand, train_stats, train_initial = demand_builder(
        train_paths,
        raw_file_count=len(train_paths),
        grid=grid,
        assignment_grid=assignment_grid,
        horizon_steps=config.horizon_steps,
        step_minutes=config.step_minutes,
        start_hour=args.start_hour,
        bounds=bounds,
        bounds_quantiles=(0.01, 0.99),
        average_by_day=True,
        filter_to_bounds=True,
        stochastic=False,
        demand_scale=args.demand_scale,
        high_demand_min_orders_per_minute=args.min_orders_per_minute,
        peak_hotspot_multiplier=args.peak_hotspot_multiplier,
        peak_hotspot_windows=args.peak_hotspot_windows,
        peak_hotspot_cells=args.peak_hotspot_cells,
        peak_hotspot_top_cells=args.peak_hotspot_top_cells,
    )

    test_demand, test_stats, test_initial = demand_builder(
        test_paths,
        raw_file_count=len(test_paths),
        grid=grid,
        assignment_grid=assignment_grid,
        horizon_steps=config.horizon_steps,
        step_minutes=config.step_minutes,
        start_hour=args.start_hour,
        bounds=bounds,
        bounds_quantiles=(0.01, 0.99),
        average_by_day=True,
        filter_to_bounds=True,
        stochastic=False,
        demand_scale=args.demand_scale,
        high_demand_min_orders_per_minute=args.min_orders_per_minute,
        peak_hotspot_multiplier=args.peak_hotspot_multiplier,
        peak_hotspot_windows=args.peak_hotspot_windows,
        peak_hotspot_cells=args.peak_hotspot_cells,
        peak_hotspot_top_cells=args.peak_hotspot_top_cells,
    )

    save_cache(train_cache, config, train_demand, grid, train_initial, train_stats)
    save_cache(test_cache, config, test_demand, grid, test_initial, test_stats)

    split_info = {
        "train_date_range": ["2016-11-08", "2016-11-30"],
        "test_date_range": ["2016-11-01", "2016-11-07"],
        "all_files": len(all_files),
        "dated_files": len(dated),
        "train_files": len(train_paths),
        "test_files": len(test_paths),
        "undated_files": undated[:20],
        "bounds": bounds,
        "num_cells": args.num_cells,
        "cell_width_km": args.cell_width_km,
        "average_side_length_km": grid.average_side_length_km,
        "horizon_steps": args.horizon_steps,
        "step_minutes": args.step_minutes,
        "demand_scale": args.demand_scale,
        "peak_hotspot_multiplier": args.peak_hotspot_multiplier,
        "peak_hotspot_windows": args.peak_hotspot_windows,
        "peak_hotspot_cells": args.peak_hotspot_cells,
        "peak_hotspot_top_cells": args.peak_hotspot_top_cells,
        "demand_kind": demand_kind,
        "train_stats": asdict(train_stats),
        "test_stats": asdict(test_stats),
        "train_cache": str(train_cache),
        "test_cache": str(test_cache),
    }

    split_stats.write_text(json.dumps(split_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("saved train cache:", train_cache)
    print("saved test cache:", test_cache)
    print("saved split stats:", split_stats)
    print("train orders_per_episode:", train_stats.orders_per_episode)
    print("test orders_per_episode:", test_stats.orders_per_episode)
    print("train rows_used:", train_stats.rows_used)
    print("test rows_used:", test_stats.rows_used)

if __name__ == "__main__":
    main()
