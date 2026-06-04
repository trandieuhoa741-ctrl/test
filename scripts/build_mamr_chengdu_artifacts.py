from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    from scripts._common import collect_trajectory_files, extract_chengdu_yyyymmdd, parse_bounds, save_pickle
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from _common import collect_trajectory_files, extract_chengdu_yyyymmdd, parse_bounds, save_pickle

from taxi_dispatch.chengdu import iter_trajectory_trips, select_high_demand_trajectory_cells
from taxi_dispatch.grid import DEFAULT_HEX_CENTER_SPACING_KM, HexGrid


DEFAULT_BOUNDS = (
    103.95202562612084,
    30.563249036822317,
    104.168179909498,
    30.79947020070271,
)

TEST_RANGE = (20161101, 20161107)
TRAIN_RANGE = (20161108, 20161130)

NEIGHBOR_COLS = [
    "north_east_neighbor",
    "north_neighbor",
    "north_west_neighbor",
    "south_east_neighbor",
    "south_neighbor",
    "south_west_neighbor",
]


def valid_lonlat(lon: float, lat: float) -> bool:
    return np.isfinite(lon) and np.isfinite(lat) and 70 <= lon <= 140 and 10 <= lat <= 60


def inside_bounds(lon: float, lat: float, bounds: tuple[float, float, float, float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bounds
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def minute_of_horizon(
    dt: datetime,
    *,
    start_hour: float,
    horizon_steps: int,
    step_minutes: int,
) -> int | None:
    start_minutes = int(round(start_hour * 60))
    current_minutes = dt.hour * 60 + dt.minute
    minute = current_minutes - start_minutes
    if minute < 0 or minute >= horizon_steps * step_minutes:
        return None
    return int(minute)


def trip_fare(distance_km: float, duration_minutes: float, base: float, per_km: float, per_minute: float) -> float:
    distance = max(float(distance_km), 0.0) if np.isfinite(distance_km) else 0.0
    minutes = max(float(duration_minutes), 0.0) if np.isfinite(duration_minutes) else 0.0
    return float(max(float(base), 0.0) + max(float(per_km), 0.0) * distance + max(float(per_minute), 0.0) * minutes)


def lonlat_offset(lon: float, lat: float, dx_km: float, dy_km: float) -> list[float]:
    lat_scale = 111.0
    lon_scale = 111.0 * max(math.cos(math.radians(lat)), 1e-6)
    return [lon + dx_km / lon_scale, lat + dy_km / lat_scale]


def write_hex_attributes(path: Path, grid: HexGrid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if grid.lonlat is None:
        raise ValueError("grid.lonlat is required to export MAMR hex attributes")

    radius = float(grid.average_side_length_km)

    fieldnames = [
        "hex_id",
        *NEIGHBOR_COLS,
        "east",
        "north_east",
        "north_west",
        "south_east",
        "south_west",
        "west",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(grid.num_cells):
            lon, lat = map(float, grid.lonlat[i])
            row: dict[str, object] = {"hex_id": i}

            for action, col in enumerate(NEIGHBOR_COLS, start=1):
                nb = int(grid.neighbors[i, action])
                row[col] = "" if nb < 0 else nb

            # 这些顶点列主要用于 MAMR/DataProvider 和你的 mamr_data.py 解析中心点。
            row["east"] = str(lonlat_offset(lon, lat, radius, 0.0))
            row["west"] = str(lonlat_offset(lon, lat, -radius, 0.0))
            row["north_east"] = str(lonlat_offset(lon, lat, radius / 2.0, radius * 0.866))
            row["north_west"] = str(lonlat_offset(lon, lat, -radius / 2.0, radius * 0.866))
            row["south_east"] = str(lonlat_offset(lon, lat, radius / 2.0, -radius * 0.866))
            row["south_west"] = str(lonlat_offset(lon, lat, -radius / 2.0, -radius * 0.866))

            writer.writerow(row)


def write_hex_distances(path: Path, grid: HexGrid) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)

    xy = np.asarray(grid.xy, dtype=np.float32)
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1).astype(np.float32)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pickup_bin", "dropoff_bin", "straight_line_distance"],
        )
        writer.writeheader()

        for i in range(grid.num_cells):
            for j in range(grid.num_cells):
                if i == j:
                    continue
                writer.writerow(
                    {
                        "pickup_bin": i,
                        "dropoff_bin": j,
                        "straight_line_distance": float(dist[i, j]),  # 单位：km
                    }
                )

    return dist


def process_split(
    paths: list[Path],
    *,
    split_name: str,
    grid: HexGrid,
    assignment_grid: HexGrid,
    horizon_steps: int,
    step_minutes: int,
    start_hour: float,
    bounds: tuple[float, float, float, float],
    raw_csv_path: Path | None = None,
    average_by_day: bool = True,
    fare_base: float = 14.0,
    fare_per_km: float = 2.6,
    fare_per_minute: float = 0.5,
) -> tuple[dict[int, dict[str, object]], np.ndarray, dict[str, object]]:
    n = grid.num_cells
    selected_coord_to_idx = {coord: idx for idx, coord in enumerate(grid.coords)}

    ride_count = np.zeros((horizon_steps, n, n), dtype=np.float32)
    duration_sum = np.zeros_like(ride_count)
    duration_count = np.zeros_like(ride_count)
    distance_sum = np.zeros_like(ride_count)
    fare_sum = np.zeros_like(ride_count)
    origin_counts = np.zeros(n, dtype=np.float64)

    rows_seen = 0
    rows_used = 0
    rows_invalid = 0
    rows_outside_bounds = 0
    rows_outside_selected = 0
    active_days: set[str] = set()
    day_to_idx: dict[str, int] = {}

    raw_writer = None
    raw_file = None

    if raw_csv_path is not None:
        raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
        raw_file = raw_csv_path.open("w", newline="", encoding="utf-8")
        raw_writer = csv.DictWriter(
            raw_file,
            fieldnames=[
                "PU_time",
                "pickup_bin",
                "dropoff_bin",
                "weight",
                "trip_distance",
                "duration_seconds",
                "fare_amount",
            ],
        )
        raw_writer.writeheader()

    try:
        for trip in iter_trajectory_trips(paths):
            rows_seen += 1

            duration_minutes = float(trip.duration_minutes)
            if duration_minutes < 1.0 or duration_minutes > 180.0:
                rows_invalid += 1
                continue

            if (
                not valid_lonlat(trip.pickup_lon, trip.pickup_lat)
                or not valid_lonlat(trip.dropoff_lon, trip.dropoff_lat)
            ):
                rows_invalid += 1
                continue

            if (
                not inside_bounds(trip.pickup_lon, trip.pickup_lat, bounds)
                or not inside_bounds(trip.dropoff_lon, trip.dropoff_lat, bounds)
            ):
                rows_outside_bounds += 1
                continue

            minute = minute_of_horizon(
                trip.start,
                start_hour=start_hour,
                horizon_steps=horizon_steps,
                step_minutes=step_minutes,
            )
            if minute is None:
                continue

            step = int(minute // step_minutes)

            origin_full = int(
                assignment_grid.nearest_cells_lonlat(
                    np.asarray([trip.pickup_lon]),
                    np.asarray([trip.pickup_lat]),
                )[0]
            )
            dest_full = int(
                assignment_grid.nearest_cells_lonlat(
                    np.asarray([trip.dropoff_lon]),
                    np.asarray([trip.dropoff_lat]),
                )[0]
            )

            origin = selected_coord_to_idx.get(assignment_grid.coords[origin_full], -1)
            dest = selected_coord_to_idx.get(assignment_grid.coords[dest_full], -1)

            if origin < 0 or dest < 0:
                rows_outside_selected += 1
                continue

            distance_km = float(trip.distance_km)
            if not np.isfinite(distance_km) or distance_km <= 0:
                distance_km = float(np.linalg.norm(grid.xy[origin] - grid.xy[dest]))

            fare = trip_fare(distance_km, duration_minutes, fare_base, fare_per_km, fare_per_minute)
            duration_seconds = max(duration_minutes * 60.0, 60.0)

            ride_count[step, origin, dest] += 1.0
            duration_sum[step, origin, dest] += duration_minutes
            duration_count[step, origin, dest] += 1.0
            distance_sum[step, origin, dest] += distance_km
            fare_sum[step, origin, dest] += fare
            origin_counts[origin] += 1.0

            day = trip.start.date().isoformat()
            active_days.add(day)
            if day not in day_to_idx:
                day_to_idx[day] = len(day_to_idx)
            rows_used += 1

            if raw_writer is not None:
                horizon_start_seconds = int(round(start_hour * 3600))
                trip_seconds_of_day = trip.start.hour * 3600 + trip.start.minute * 60 + trip.start.second
                pu_time = day_to_idx[day] * 86_400 + max(trip_seconds_of_day - horizon_start_seconds, 0)
                raw_writer.writerow(
                    {
                        "PU_time": int(pu_time),
                        "pickup_bin": int(origin),
                        "dropoff_bin": int(dest),
                        "weight": 1,
                        "trip_distance": distance_km,
                        "duration_seconds": duration_seconds,
                        "fare_amount": fare,
                    }
                )
    finally:
        if raw_file is not None:
            raw_file.close()

    day_count = max(len(active_days), 1)

    if average_by_day:
        ride_count /= float(day_count)
        duration_sum /= float(day_count)
        duration_count /= float(day_count)
        distance_sum /= float(day_count)
        fare_sum /= float(day_count)

    distance_matrix_fallback = np.linalg.norm(
        grid.xy[:, None, :] - grid.xy[None, :, :],
        axis=-1,
    ).astype(np.float32)

    states: dict[int, dict[str, object]] = {}

    base_time = datetime(2016, 11, 1, int(start_hour), int((start_hour % 1) * 60))

    for step in range(horizon_steps):
        count = ride_count[step]

        duration_mean = np.divide(
            duration_sum[step],
            duration_count[step],
            out=np.zeros((n, n), dtype=np.float32),
            where=duration_count[step] > 0,
        )

        distance_mean = np.divide(
            distance_sum[step],
            count,
            out=distance_matrix_fallback.copy(),
            where=count > 0,
        )

        reward_mean = np.divide(
            fare_sum[step],
            count,
            out=np.zeros((n, n), dtype=np.float32),
            where=count > 0,
        )

        travel_units = np.ceil(duration_mean / float(step_minutes)).astype(np.float32)
        travel_units[(count > 0) & (travel_units <= 0)] = 1.0

        transition = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            total = float(count[i].sum())
            if total > 0:
                transition[i] = count[i] / total
            else:
                transition[i] = np.full(n, 1.0 / n, dtype=np.float32)

        states[step] = {
            "time": base_time + timedelta(minutes=step * step_minutes),
            "time_slice_start": base_time + timedelta(minutes=step * step_minutes - step_minutes // 2),
            "time_slice_end": base_time + timedelta(minutes=step * step_minutes + step_minutes // 2),
            "time_slice_duration": int(step_minutes),
            "time_unit_duration": int(step_minutes),
            "transition_matrix": transition.astype(np.float32),
            "ride_count_matrix": count.astype(np.float32),
            "pickup_vector": count.sum(axis=1).astype(np.float32),
            "dropoff_vector": count.sum(axis=0).astype(np.float32),
            "distance_matrix": distance_mean.astype(np.float32),
            "travel_time_matrix": travel_units.astype(np.float32),
            "reward_matrix": reward_mean.astype(np.float32),
            "geodesic_matrix": distance_matrix_fallback.astype(np.float32),
        }

    stats = {
        "split": split_name,
        "files": len(paths),
        "rows_seen": rows_seen,
        "rows_used": rows_used,
        "rows_invalid": rows_invalid,
        "rows_outside_bounds": rows_outside_bounds,
        "rows_outside_selected": rows_outside_selected,
        "days": day_count,
        "average_by_day": average_by_day,
    }

    return states, origin_counts, stats


def write_driver_distribution(path: Path, counts: np.ndarray, fleet_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    weights = counts.astype(np.float64)
    if float(weights.sum()) <= 0:
        weights = np.ones_like(weights, dtype=np.float64)
    weights = weights / weights.sum()

    driver_counts = np.floor(weights * fleet_size).astype(int)
    remainder = fleet_size - int(driver_counts.sum())
    if remainder > 0:
        order = np.argsort(-(weights * fleet_size - driver_counts))
        driver_counts[order[:remainder]] += 1

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hex_id", "driver_count"])
        writer.writeheader()
        for i, count in enumerate(driver_counts):
            writer.writerow({"hex_id": i, "driver_count": int(count)})

    # 兼容原版 MAMR envs/simulator.py 里读取 envs/drivers_d30.csv + counts_9 的逻辑。
    legacy = path.parent / "drivers_d30.csv"
    with legacy.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["counts_9"])
        writer.writeheader()
        for count in driver_counts:
            writer.writerow({"counts_9": int(count)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-data", type=Path, default=Path("data/mamr_chengdu"))
    parser.add_argument("--out-envs", type=Path, default=Path("data/mamr_chengdu_envs"))
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
    parser.add_argument("--bounds", default=",".join(str(x) for x in DEFAULT_BOUNDS))
    parser.add_argument("--no-average-by-day", action="store_true")
    parser.add_argument("--fare-base", type=float, default=14.0)
    parser.add_argument("--fare-per-km", type=float, default=2.6)
    parser.add_argument("--fare-per-minute", type=float, default=0.5)
    args = parser.parse_args()

    bounds = parse_bounds(args.bounds)

    all_files = collect_trajectory_files(args.raw_dir)
    if not all_files:
        raise FileNotFoundError(f"No trajectory files found under {args.raw_dir}")

    dated: list[tuple[Path, int]] = []
    for path in all_files:
        date = extract_chengdu_yyyymmdd(path)
        if date is not None:
            dated.append((path, date))

    train_paths = [path for path, date in dated if TRAIN_RANGE[0] <= date <= TRAIN_RANGE[1]]
    test_paths = [path for path, date in dated if TEST_RANGE[0] <= date <= TEST_RANGE[1]]

    print("all files:", len(all_files))
    print("dated files:", len(dated))
    print("train files:", len(train_paths))
    print("test files:", len(test_paths))

    if not train_paths:
        raise RuntimeError("train_paths is empty. Check file names/date parsing.")
    if not test_paths:
        raise RuntimeError("test_paths is empty. Check file names/date parsing.")

    assignment_grid = HexGrid.create_geographic_fixed(
        bounds,
        cell_width_km=args.cell_width_km,
        padding_km=args.grid_padding_km,
    )

    selected_coords = select_high_demand_trajectory_cells(
        train_paths,
        grid=assignment_grid,
        num_cells=args.num_cells,
        horizon_steps=args.horizon_steps,
        step_minutes=args.step_minutes,
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

    data_dir = args.out_data
    envs_dir = args.out_envs
    average_by_day = not args.no_average_by_day

    write_hex_attributes(data_dir / "hex_bins" / "hex_bin_attributes.csv", grid)
    write_hex_distances(data_dir / "hex_bins" / "hex_distances.csv", grid)

    train_states, train_origin_counts, train_stats = process_split(
        train_paths,
        split_name="train",
        grid=grid,
        assignment_grid=assignment_grid,
        horizon_steps=args.horizon_steps,
        step_minutes=args.step_minutes,
        start_hour=args.start_hour,
        bounds=bounds,
        raw_csv_path=data_dir / "raw_train.csv",
        average_by_day=average_by_day,
        fare_base=args.fare_base,
        fare_per_km=args.fare_per_km,
        fare_per_minute=args.fare_per_minute,
    )

    test_states, test_origin_counts, test_stats = process_split(
        test_paths,
        split_name="test",
        grid=grid,
        assignment_grid=assignment_grid,
        horizon_steps=args.horizon_steps,
        step_minutes=args.step_minutes,
        start_hour=args.start_hour,
        bounds=bounds,
        raw_csv_path=data_dir / "raw_test.csv",
        average_by_day=average_by_day,
        fare_base=args.fare_base,
        fare_per_km=args.fare_per_km,
        fare_per_minute=args.fare_per_minute,
    )

    save_pickle(data_dir / "city_states" / "city_states_train.dill", train_states)
    save_pickle(data_dir / "city_states" / "city_states_test.dill", test_states)

    # 默认 city_states.dill 指向训练集，方便你的 mamr_data.py 默认读取。
    save_pickle(data_dir / "city_states" / "city_states.dill", train_states)

    write_driver_distribution(data_dir / "driver_distribution.csv", train_origin_counts, args.fleet_size)
    write_driver_distribution(envs_dir / "driver_distribution.csv", train_origin_counts, args.fleet_size)

    # 原版 MAMR create_city_state.py 默认寻找 data/raw.csv。这里先放 train raw。
    # 如果你要原版脚本跑 test，需要手动改 DATA_DIR_origin。
    (data_dir / "raw.csv").write_text((data_dir / "raw_train.csv").read_text(encoding="utf-8"), encoding="utf-8")

    stats = {
        "source": "chengdu_trajectory_to_mamr_artifacts",
        "bounds": bounds,
        "num_cells": args.num_cells,
        "fleet_size": args.fleet_size,
        "horizon_steps": args.horizon_steps,
        "step_minutes": args.step_minutes,
        "start_hour": args.start_hour,
        "cell_width_km": args.cell_width_km,
        "fare_base": args.fare_base,
        "fare_per_km": args.fare_per_km,
        "fare_per_minute": args.fare_per_minute,
        "average_side_length_km": grid.average_side_length_km,
        "train": train_stats,
        "test": test_stats,
    }

    import json
    with (data_dir / "mamr_artifact_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("saved:", data_dir / "hex_bins" / "hex_bin_attributes.csv")
    print("saved:", data_dir / "hex_bins" / "hex_distances.csv")
    print("saved:", data_dir / "city_states" / "city_states_train.dill")
    print("saved:", data_dir / "city_states" / "city_states_test.dill")
    print("saved:", data_dir / "driver_distribution.csv")
    print("saved:", data_dir / "raw.csv")
    print("saved:", envs_dir / "drivers_d30.csv")
    print("train stats:", train_stats)
    print("test stats :", test_stats)


if __name__ == "__main__":
    main()
