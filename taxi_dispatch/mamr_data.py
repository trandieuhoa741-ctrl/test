from __future__ import annotations

import ast
import csv
import pickle
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data import EventTripDemand, TripEvent
from .env import EnvConfig
from .grid import HexGrid, lonlat_to_xy
from .mamr_env import MAMRDispatchEnv


MAMR_NEIGHBOR_COLUMNS = (
    "north_east_neighbor",
    "north_neighbor",
    "north_west_neighbor",
    "south_east_neighbor",
    "south_neighbor",
    "south_west_neighbor",
)

MILES_TO_KM = 1.609344
UNIX_TIMESTAMP_THRESHOLD = 1_000_000_000


@dataclass(frozen=True)
class MAMRDataBundle:
    """Converted MAMR preprocessing artifacts ready for the dispatch code path."""

    config: EnvConfig
    grid: HexGrid
    demand: EventTripDemand
    initial_taxi_distribution: np.ndarray | None
    hex_distance_matrix: np.ndarray
    hex_ids: tuple[int, ...]
    source_dir: Path
    raw_trips_path: Path | None = None


def load_mamr_preprocessed(
    data_dir: str | Path,
    *,
    city_states_path: str | Path | None = None,
    hex_attributes_path: str | Path | None = None,
    hex_distances_path: str | Path | None = None,
    driver_distribution_path: str | Path | None = None,
    raw_trips_path: str | Path | None = None,
    raw_trips_candidates: Sequence[str] = (),
    fleet_size: int = 6000,
    horizon_steps: int | None = None,
    step_minutes: int | None = None,
    max_wait_steps: int = 1,
    seed: int = 0,
    demand_scale: float = 1.0,
    distance_unit: str = "km",
    base_config: EnvConfig | None = None,
) -> MAMRDataBundle:
    """Read original MAMR preprocessing files and adapt them to this project.

    Required files are ``city_states.dill``, ``hex_bin_attributes.csv`` and
    ``hex_distances.csv``. ``driver_distribution.csv`` is used when present.
    Some MAMR releases store the initial distribution only in config, so it is
    intentionally optional here. ``raw_trips_candidates`` lets callers prefer
    per-order raw trip files while still falling back to aggregate city states
    when those files are absent.
    """

    root = Path(data_dir)
    attr_file = _resolve_file(
        root,
        hex_attributes_path,
        (
            "hex_bin_attributes.csv",
            "hex_bins/hex_bin_attributes.csv",
            "data/hex_bins/hex_bin_attributes.csv",
        ),
    )
    dist_file = _resolve_file(
        root,
        hex_distances_path,
        (
            "hex_distances.csv",
            "hex_bins/hex_distances.csv",
            "data/hex_bins/hex_distances.csv",
        ),
    )
    states_file = _resolve_file(
        root,
        city_states_path,
        (
            "city_states.dill",
            "city_states/city_states.dill",
            "data/city_states/city_states.dill",
        ),
    )
    driver_file = _resolve_optional_file(
        root,
        driver_distribution_path,
        (
            "driver_distribution.csv",
            "drivers_d30.csv",
            "envs/driver_distribution.csv",
            "envs/drivers_d30.csv",
        ),
    )
    raw_file = _resolve_optional_file(root, raw_trips_path, raw_trips_candidates)

    hex_rows = _read_hex_attributes(attr_file)
    hex_ids = tuple(int(row["hex_id"]) for row in hex_rows)
    id_to_idx = {hex_id: idx for idx, hex_id in enumerate(hex_ids)}
    hex_distance_matrix = _read_hex_distances(dist_file, hex_ids, id_to_idx, distance_unit=distance_unit)
    grid = _build_grid(hex_rows, id_to_idx, hex_distance_matrix)

    city_states = _ordered_city_states(_load_dill(states_file))
    if not city_states:
        raise ValueError(f"{states_file} does not contain any city states")

    inferred_step_minutes = _infer_step_minutes(city_states, step_minutes)
    limit = len(city_states) if horizon_steps is None or horizon_steps <= 0 else min(horizon_steps, len(city_states))
    selected_states = city_states[:limit]
    horizon_start_seconds = _infer_horizon_start_seconds(selected_states)
    if raw_file is not None:
        demand = _build_raw_trip_event_demand(
            raw_file,
            num_cells=grid.num_cells,
            horizon_steps=limit,
            step_minutes=inferred_step_minutes,
            demand_scale=demand_scale,
            horizon_start_seconds=horizon_start_seconds,
        )
    else:
        demand = _build_event_demand(
            selected_states,
            num_cells=grid.num_cells,
            step_minutes=inferred_step_minutes,
            demand_scale=demand_scale,
        )
    initial_distribution = None
    if driver_file is not None:
        initial_distribution = _read_driver_distribution(driver_file, hex_ids, id_to_idx)

    base = base_config or EnvConfig(
        fleet_size=int(fleet_size),
        horizon_steps=limit,
        step_minutes=inferred_step_minutes,
        max_wait_steps=max_wait_steps,
        demand_scale=demand_scale,
        seed=seed,
    )
    config = replace(
        base,
        num_cells=grid.num_cells,
        cell_width_km=grid.cell_width_km,
        fleet_size=int(fleet_size),
        horizon_steps=limit,
        step_minutes=inferred_step_minutes,
        max_wait_steps=max_wait_steps,
        demand_scale=demand_scale,
        seed=seed,
    )
    return MAMRDataBundle(
        config=config,
        grid=grid,
        demand=demand,
        initial_taxi_distribution=initial_distribution,
        hex_distance_matrix=hex_distance_matrix,
        hex_ids=hex_ids,
        source_dir=root,
        raw_trips_path=raw_file,
    )


def build_mamr_data_compatible_env(
    data_dir: str | Path,
    **kwargs: object,
) -> tuple[MAMRDispatchEnv, MAMRDataBundle]:
    bundle = load_mamr_preprocessed(data_dir, **kwargs)
    env = MAMRDispatchEnv.from_bundle(bundle)
    return env, bundle


def _resolve_file(root: Path, explicit: str | Path | None, candidates: Sequence[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of {', '.join(candidates)} under {root}")


def _resolve_optional_file(root: Path, explicit: str | Path | None, candidates: Sequence[str]) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return None


def _load_dill(path: Path) -> object:
    try:
        import dill  # type: ignore
    except ModuleNotFoundError:
        dill = None
    with path.open("rb") as f:
        if dill is not None:
            return dill.load(f)
        return pickle.load(f)


def _read_hex_attributes(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    required = {"hex_id", *MAMR_NEIGHBOR_COLUMNS, "east", "west"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def _read_hex_distances(
    path: Path,
    hex_ids: tuple[int, ...],
    id_to_idx: Mapping[int, int],
    *,
    distance_unit: str,
) -> np.ndarray:
    factor = _distance_factor(distance_unit)
    matrix = np.zeros((len(hex_ids), len(hex_ids)), dtype=np.float32)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"pickup_bin", "dropoff_bin", "straight_line_distance"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain pickup_bin, dropoff_bin and straight_line_distance")
        for row in reader:
            pickup = _parse_int(row["pickup_bin"])
            dropoff = _parse_int(row["dropoff_bin"])
            if pickup not in id_to_idx or dropoff not in id_to_idx:
                continue
            distance = float(row["straight_line_distance"]) * factor
            matrix[id_to_idx[pickup], id_to_idx[dropoff]] = max(distance, 0.0)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _distance_factor(unit: str) -> float:
    normalized = unit.lower()
    if normalized in {"mile", "miles", "mi"}:
        return MILES_TO_KM
    if normalized in {"km", "kilometer", "kilometers"}:
        return 1.0
    raise ValueError("distance_unit must be 'mile' or 'km'")


def _build_grid(
    hex_rows: list[dict[str, str]],
    id_to_idx: Mapping[int, int],
    hex_distance_matrix: np.ndarray,
) -> HexGrid:
    num_cells = len(hex_rows)
    neighbors = np.full((num_cells, 7), -1, dtype=np.int64)
    neighbors[:, 0] = np.arange(num_cells, dtype=np.int64)
    for idx, row in enumerate(hex_rows):
        for action, column in enumerate(MAMR_NEIGHBOR_COLUMNS, start=1):
            neighbor_id = _parse_optional_int(row.get(column, ""))
            if neighbor_id is not None and neighbor_id in id_to_idx:
                neighbors[idx, action] = id_to_idx[neighbor_id]

    lonlat = np.asarray([_hex_center(row) for row in hex_rows], dtype=np.float32)
    lon0 = float(np.mean(lonlat[:, 0]))
    lat0 = float(np.mean(lonlat[:, 1]))
    x, y = lonlat_to_xy(lonlat[:, 0], lonlat[:, 1], lon0, lat0)
    xy = np.stack([x, y], axis=1).astype(np.float32)
    cell_width_km = _estimate_cell_width(neighbors, hex_distance_matrix, xy)
    coords = _infer_axial_coords(neighbors)
    return HexGrid(
        coords=coords,
        neighbors=neighbors,
        xy=xy,
        cell_width_km=cell_width_km,
        lonlat=lonlat,
        projection_origin=(lon0, lat0),
    )


def _estimate_cell_width(neighbors: np.ndarray, distance_matrix: np.ndarray, xy: np.ndarray) -> float:
    values: list[float] = []
    for origin in range(neighbors.shape[0]):
        for destination in neighbors[origin, 1:]:
            if destination < 0:
                continue
            matrix_distance = float(distance_matrix[origin, int(destination)])
            if np.isfinite(matrix_distance) and matrix_distance > 0:
                values.append(matrix_distance)
            else:
                values.append(float(np.linalg.norm(xy[origin] - xy[int(destination)])))
    if values:
        return max(float(np.median(values)), 0.1)
    return 1.0


def _infer_axial_coords(neighbors: np.ndarray) -> tuple[tuple[int, int], ...]:
    deltas = ((1, -1), (0, -1), (-1, 0), (1, 0), (0, 1), (-1, 1))
    coords: list[tuple[int, int] | None] = [None] * neighbors.shape[0]
    component_offset = 0
    for start in range(neighbors.shape[0]):
        if coords[start] is not None:
            continue
        coords[start] = (component_offset, 0)
        queue = [start]
        while queue:
            origin = queue.pop(0)
            q, r = coords[origin] or (0, 0)
            for action, destination in enumerate(neighbors[origin, 1:], start=1):
                if destination < 0:
                    continue
                dst = int(destination)
                dq, dr = deltas[action - 1]
                candidate = (q + dq, r + dr)
                if coords[dst] is None:
                    coords[dst] = candidate
                    queue.append(dst)
        component_offset += 1000
    return tuple(coord if coord is not None else (idx, 0) for idx, coord in enumerate(coords))


def _hex_center(row: Mapping[str, str]) -> tuple[float, float]:
    if "center" in row and row["center"]:
        center = _parse_point(row["center"])
        return float(center[0]), float(center[1])
    west = _parse_point(row["west"])
    east = _parse_point(row["east"])
    return (float(west[0] + east[0]) / 2.0, float(west[1] + east[1]) / 2.0)


def _ordered_city_states(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, Mapping):
        return [value[key] for key in sorted(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise ValueError("city_states must be a mapping or sequence")


def _infer_step_minutes(city_states: Sequence[Mapping[str, object]], explicit: int | None) -> int:
    if explicit is not None and explicit > 0:
        return int(explicit)
    first = city_states[0]
    for key in ("time_unit_duration", "time_slice_duration", "step_minutes"):
        value = first.get(key)
        if value is not None:
            minutes = int(round(float(value)))
            if minutes > 0:
                return minutes
    return 10


def _infer_horizon_start_seconds(city_states: Sequence[Mapping[str, object]]) -> int:
    if not city_states:
        return 0
    first = city_states[0]
    for key in ("time", "timestamp", "time_slice_start"):
        value = first.get(key)
        if isinstance(value, datetime):
            return int(value.hour * 3600 + value.minute * 60 + value.second)
        if isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.strip())
            except ValueError:
                continue
            return int(parsed.hour * 3600 + parsed.minute * 60 + parsed.second)
    return 0


def _build_event_demand(
    city_states: Sequence[Mapping[str, object]],
    *,
    num_cells: int,
    step_minutes: int,
    demand_scale: float,
) -> EventTripDemand:
    horizon = len(city_states)
    od_counts = np.zeros((horizon, num_cells, num_cells), dtype=np.float32)
    mean_trip_minutes = np.zeros_like(od_counts, dtype=np.float32)
    events_by_step: list[tuple[TripEvent, ...]] = []
    trip_minutes_values: list[float] = []

    for step, state in enumerate(city_states):
        ride_counts = _matrix(state, "ride_count_matrix", (num_cells, num_cells))
        if demand_scale != 1.0:
            ride_counts = ride_counts * float(demand_scale)
        od_counts[step] = ride_counts

        travel_units = _optional_matrix(state, "travel_time_matrix", (num_cells, num_cells))
        if travel_units is not None:
            mean_trip_minutes[step] = np.maximum(travel_units, 0.0) * float(step_minutes)
            trip_minutes_values.extend(mean_trip_minutes[step][mean_trip_minutes[step] > 0].astype(float).tolist())

        step_events: list[TripEvent] = []
        event_counts = _largest_remainder_matrix(ride_counts)
        for origin in range(num_cells):
            for destination in range(num_cells):
                count = int(event_counts[origin, destination])
                if count <= 0:
                    continue
                offsets = _spread_offsets(count, step_minutes)
                trip_minutes = float(mean_trip_minutes[step, origin, destination])
                if trip_minutes <= 0:
                    trip_minutes = 0.0
                for offset in offsets:
                    step_events.append(
                        TripEvent(
                            origin=origin,
                            destination=destination,
                            minute_offset=offset,
                            trip_minutes=trip_minutes,
                        )
                    )
        events_by_step.append(tuple(step_events))

    od_probs = _od_probs_from_counts(od_counts)
    global_mean = float(np.mean(trip_minutes_values)) if trip_minutes_values else float(step_minutes)
    metadata = {
        "source": "mamr_city_states",
        "global_mean_trip_minutes": global_mean,
        "aggregate_city_states": True,
    }
    return EventTripDemand(
        rates=od_counts.sum(axis=2).astype(np.float32),
        od_probs=od_probs,
        mean_trip_minutes=mean_trip_minutes,
        events_by_day=(tuple(events_by_step),),
        stochastic=False,
        metadata=metadata,
    )


def _largest_remainder_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.maximum(matrix, 0.0)
    counts = np.floor(matrix).astype(np.int64)
    remainder_slots = max(int(np.rint(float(matrix.sum()))) - int(counts.sum()), 0)
    if remainder_slots <= 0 or matrix.size == 0:
        return counts
    remainders = (matrix - counts).reshape(-1)
    order = np.argsort(-remainders, kind="stable")
    flat = counts.reshape(-1)
    flat[order[:remainder_slots]] += 1
    return counts


def _build_raw_trip_event_demand(
    path: Path,
    *,
    num_cells: int,
    horizon_steps: int,
    step_minutes: int,
    demand_scale: float,
    horizon_start_seconds: int,
) -> EventTripDemand:
    day_steps: dict[int, list[list[TripEvent]]] = {}
    od_counts = np.zeros((horizon_steps, num_cells, num_cells), dtype=np.float32)
    trip_minutes_sum = np.zeros_like(od_counts, dtype=np.float32)
    trip_minutes_count = np.zeros_like(od_counts, dtype=np.float32)
    step_seconds = max(int(step_minutes) * 60, 1)
    raw_rows: list[tuple[int, int, int, int, float]] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"PU_time", "pickup_bin", "dropoff_bin"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain PU_time, pickup_bin and dropoff_bin")

        for row in reader:
            timestamp = _parse_timestamp_seconds(row["PU_time"])
            origin = _parse_int(row["pickup_bin"])
            destination = _parse_int(row["dropoff_bin"])
            if not (0 <= origin < num_cells and 0 <= destination < num_cells):
                continue

            event_count = int(round(_optional_float(row.get("weight"), 1.0) * float(demand_scale)))
            if event_count <= 0:
                continue

            trip_minutes = max(_optional_float(row.get("duration_seconds"), float(step_minutes) * 60.0) / 60.0, 1.0)
            raw_rows.append((timestamp, origin, destination, event_count, trip_minutes))

    raw_time_mode = _select_raw_time_mode(
        raw_rows,
        horizon_steps=horizon_steps,
        step_seconds=step_seconds,
        horizon_start_seconds=horizon_start_seconds,
    )

    for timestamp, origin, destination, event_count, trip_minutes in raw_rows:
        location = _raw_timestamp_location(
            timestamp,
            mode=raw_time_mode,
            horizon_steps=horizon_steps,
            step_seconds=step_seconds,
            horizon_start_seconds=horizon_start_seconds,
        )
        if location is None:
            continue
        day_key, step, seconds_of_episode = location
        minute_offset = int((seconds_of_episode - step * step_seconds) // 60)
        minute_offset = min(max(minute_offset, 0), max(int(step_minutes) - 1, 0))
        day_events = day_steps.setdefault(day_key, [[] for _ in range(horizon_steps)])
        for _ in range(event_count):
            day_events[step].append(
                TripEvent(
                    origin=origin,
                    destination=destination,
                    minute_offset=minute_offset,
                    trip_minutes=trip_minutes,
                )
            )
        od_counts[step, origin, destination] += float(event_count)
        trip_minutes_sum[step, origin, destination] += trip_minutes * float(event_count)
        trip_minutes_count[step, origin, destination] += float(event_count)

    if not day_steps:
        raise ValueError(f"{path} does not contain any trips inside the configured horizon")

    days = tuple(sorted(day_steps))
    events_by_day = tuple(
        tuple(tuple(sorted(step_events, key=lambda event: event.minute_offset)) for step_events in day_steps[day])
        for day in days
    )
    mean_trip_minutes = np.zeros_like(od_counts, dtype=np.float32)
    np.divide(
        trip_minutes_sum,
        trip_minutes_count,
        out=mean_trip_minutes,
        where=trip_minutes_count > 0,
    )
    averaged_counts = od_counts / float(len(days))
    global_mean = float(trip_minutes_sum.sum() / trip_minutes_count.sum()) if float(trip_minutes_count.sum()) > 0 else float(step_minutes)
    metadata = {
        "source": "mamr_raw_trips",
        "raw_trips_path": str(path),
        "raw_time_mode": raw_time_mode,
        "day_count": len(days),
        "days": _format_raw_days(days, raw_time_mode),
        "global_mean_trip_minutes": global_mean,
    }
    return EventTripDemand(
        rates=averaged_counts.sum(axis=2).astype(np.float32),
        od_probs=_od_probs_from_counts(averaged_counts),
        mean_trip_minutes=mean_trip_minutes,
        events_by_day=events_by_day,
        stochastic=False,
        metadata=metadata,
    )


def _select_raw_time_mode(
    raw_rows: Sequence[tuple[int, int, int, int, float]],
    *,
    horizon_steps: int,
    step_seconds: int,
    horizon_start_seconds: int,
) -> str:
    if not raw_rows:
        return "episode_relative"
    if max(timestamp for timestamp, *_rest in raw_rows) < UNIX_TIMESTAMP_THRESHOLD:
        return "episode_relative"

    candidates = ("unix_utc_day", "unix_wall_clock")
    counts: dict[str, int] = {}
    for mode in candidates:
        counts[mode] = sum(
            1
            for timestamp, *_rest in raw_rows
            if _raw_timestamp_location(
                timestamp,
                mode=mode,
                horizon_steps=horizon_steps,
                step_seconds=step_seconds,
                horizon_start_seconds=horizon_start_seconds,
            )
            is not None
        )
    return "unix_wall_clock" if counts["unix_wall_clock"] > counts["unix_utc_day"] else "unix_utc_day"


def _raw_timestamp_location(
    timestamp: int,
    *,
    mode: str,
    horizon_steps: int,
    step_seconds: int,
    horizon_start_seconds: int,
) -> tuple[int, int, int] | None:
    day_key = int(timestamp // 86_400)
    seconds = int(timestamp - day_key * 86_400)
    if mode == "unix_wall_clock":
        seconds -= int(horizon_start_seconds)
    elif mode not in {"episode_relative", "unix_utc_day"}:
        raise ValueError(f"unknown raw time mode: {mode}")

    if not (0 <= seconds < horizon_steps * step_seconds):
        return None
    step = int(seconds // step_seconds)
    return day_key, step, seconds


def _format_raw_days(days: Sequence[int], raw_time_mode: str) -> tuple[str, ...]:
    if raw_time_mode.startswith("unix_"):
        return tuple(datetime.fromtimestamp(day * 86_400, tz=timezone.utc).date().isoformat() for day in days)
    return tuple(str(day) for day in days)


def _od_probs_from_counts(od_counts: np.ndarray) -> np.ndarray:
    steps, num_cells, _ = od_counts.shape
    od_probs = np.zeros_like(od_counts, dtype=np.float32)
    global_dest = od_counts.sum(axis=(0, 1)).astype(np.float64)
    if float(global_dest.sum()) > 0:
        fallback = (global_dest / global_dest.sum()).astype(np.float32)
    else:
        fallback = np.full(num_cells, 1.0 / num_cells, dtype=np.float32)
    for step in range(steps):
        for origin in range(num_cells):
            total = float(od_counts[step, origin].sum())
            od_probs[step, origin] = od_counts[step, origin] / total if total > 0 else fallback
    return od_probs


def _read_driver_distribution(
    path: Path,
    hex_ids: tuple[int, ...],
    id_to_idx: Mapping[int, int],
) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        rows = list(reader)

    fields = set(reader.fieldnames)
    weights = np.zeros(len(hex_ids), dtype=np.float64)
    count_col = _first_existing(fields, ("driver_count", "drivers", "count", "num_drivers", "supply", "counts_9"))
    id_col = _first_existing(fields, ("hex_id", "hex_bin", "cell", "cell_id", "bin"))
    time_col = _first_existing(fields, ("t", "time", "step"))
    if id_col is not None and count_col is not None:
        first_time = rows[0].get(time_col) if rows and time_col is not None else None
        for row in rows:
            if first_time is not None and row.get(time_col) != first_time:
                continue
            hex_id = _parse_int(row[id_col])
            if hex_id in id_to_idx:
                weights[id_to_idx[hex_id]] += float(row[count_col])
    elif count_col is not None and len(rows) >= len(hex_ids):
        for idx, row in enumerate(rows[: len(hex_ids)]):
            weights[idx] = float(row[count_col])
    elif rows:
        row = rows[0]
        for hex_id in hex_ids:
            key = str(hex_id)
            if key in row and row[key] != "":
                weights[id_to_idx[hex_id]] = float(row[key])
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"{path} does not contain a positive driver distribution")
    return (weights / total).astype(np.float64)


def _first_existing(fields: set[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in fields:
            return candidate
    return None


def _matrix(state: Mapping[str, object], key: str, shape: tuple[int, int]) -> np.ndarray:
    if key not in state:
        raise ValueError(f"city state is missing {key}")
    value = np.asarray(state[key], dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {value.shape}")
    return np.maximum(value, 0.0)


def _optional_matrix(state: Mapping[str, object], key: str, shape: tuple[int, int]) -> np.ndarray | None:
    if key not in state:
        return None
    return _matrix(state, key, shape)


def _spread_offsets(count: int, step_minutes: int) -> list[int]:
    if count <= 0:
        return []
    last_minute = max(int(step_minutes) - 1, 0)
    if count == 1 or last_minute == 0:
        return [0] * count
    return np.linspace(0, last_minute, num=count, dtype=np.int64).astype(int).tolist()


def _parse_point(value: str | Sequence[float]) -> np.ndarray:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    arr = np.asarray(parsed, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"point must have two coordinates, got {value!r}")
    return arr


def _parse_int(value: object) -> int:
    return int(float(str(value).strip()))


def _parse_timestamp_seconds(value: object) -> int:
    return int(float(str(value).strip()))


def _optional_float(value: object, default: float) -> float:
    if value is None:
        return float(default)
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return float(default)
    return float(text)


def _parse_optional_int(value: object) -> int | None:
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        return _parse_int(text)
    except ValueError:
        return None


__all__ = [
    "MAMRDataBundle",
    "MAMR_NEIGHBOR_COLUMNS",
    "build_mamr_data_compatible_env",
    "load_mamr_preprocessed",
]
