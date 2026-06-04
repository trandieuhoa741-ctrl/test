from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import tarfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .baselines import diffusion_actions, park_actions, random_actions
from .config import parse_args_with_config
from .data import EmpiricalTripDemand, EventTripDemand, TabularDemand, TripEvent
from .env import EnvConfig, EpisodeMetrics, DispatchEnv
from .grid import DEFAULT_HEX_CENTER_SPACING_KM, HexGrid, lonlat_to_xy
from .plotting import plot_cell_cancellations
from .road_network import RoadCostMatrices, RoadNetwork


ORDER_ID = "订单ID"
START_TIME = "开始计费时间"
END_TIME = "结束计费时间"
PICKUP_LON = "上车位置经度"
PICKUP_LAT = "上车位置纬度"
DROPOFF_LON = "下车位置经度"
DROPOFF_LAT = "下车位置纬度"
RAW_COLUMN_ORDER = (ORDER_ID, START_TIME, END_TIME, PICKUP_LON, PICKUP_LAT, DROPOFF_LON, DROPOFF_LAT)


@dataclass
class ChengduDemandStats:
    raw_files: int
    rows_seen: int
    rows_used: int
    rows_duplicate: int
    rows_invalid: int
    rows_outside_bounds: int
    days: int
    bounds: tuple[float, float, float, float]
    bounds_quantiles: tuple[float, float]
    average_by_day: bool
    demand_scale: float
    grid_cell_width_km: float
    grid_average_side_length_km: float
    orders_per_episode: float
    global_mean_trip_minutes: float
    source: str = "chengdu_raw_orders"
    rows_outside_selected_cells: int = 0
    selected_cells: int = 0
    high_demand_min_orders_per_minute: float = 0.0
    mean_trip_speed_kmh: float = 0.0
    trajectory_points: int = 0
    peak_hotspot_multiplier: float = 1.0
    peak_hotspot_windows: tuple[tuple[float, float], ...] = ()
    peak_hotspot_cells: tuple[int, ...] = ()
    peak_hotspot_periods: tuple[dict[str, object], ...] = ()


@dataclass
class TrajectoryTrip:
    order_id: str
    driver_id: str
    start: datetime
    end: datetime
    pickup_lon: float
    pickup_lat: float
    dropoff_lon: float
    dropoff_lat: float
    points: int
    distance_km: float

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0

    @property
    def speed_kmh(self) -> float:
        hours = max(self.duration_minutes / 60.0, 1e-6)
        return self.distance_km / hours


DEFAULT_PEAK_HOTSPOT_PERIOD_NAMES: tuple[str, ...] = ("morning_peak", "midday_peak", "evening_peak")
DEFAULT_PEAK_HOTSPOT_PERIODS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("morning_peak", (7.0, 9.0)),
    ("midday_peak", (12.0, 14.0)),
    ("evening_peak", (17.0, 19.0)),
)
DEFAULT_PEAK_HOTSPOT_WINDOWS: tuple[tuple[float, float], ...] = tuple(
    window for _name, window in DEFAULT_PEAK_HOTSPOT_PERIODS
)
DEFAULT_PEAK_HOTSPOT_WINDOWS_TEXT = "morning_peak=7-9,midday_peak=12-14,evening_peak=17-19"


def parse_peak_hour_periods(
    value: str | Sequence[tuple[float, float]] | Sequence[tuple[str, tuple[float, float]]] | None,
) -> tuple[tuple[str, float, float], ...]:
    """Parse named clock-hour windows such as ``morning_peak=7-9,evening_peak=17-19``."""

    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        periods: list[tuple[str, float, float]] = []
        for index, item in enumerate(text.split(",")):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                name, window = item.split("=", 1)
                period_name = name.strip() or _default_peak_period_name(index)
            else:
                window = item
                period_name = _default_peak_period_name(index)
            if "-" not in window:
                raise ValueError(f"invalid peak hour window {item!r}; expected start-end")
            start_text, end_text = window.split("-", 1)
            periods.append((period_name, _parse_clock_hour(start_text), _parse_clock_hour(end_text)))
        return tuple(periods)

    periods = []
    for index, item in enumerate(value):
        if len(item) == 2 and isinstance(item[0], str):
            name = item[0]
            start, end = item[1]  # type: ignore[index]
        else:
            name = _default_peak_period_name(index)
            start, end = item  # type: ignore[misc]
        periods.append((str(name), float(start), float(end)))
    return tuple(periods)


def parse_peak_hour_windows(value: str | Sequence[tuple[float, float]] | None) -> tuple[tuple[float, float], ...]:
    """Parse clock-hour windows such as ``7-9,17:30-19``."""

    return tuple((start, end) for _name, start, end in parse_peak_hour_periods(value))


def parse_cell_indices(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        parts = [part.strip() for part in text.replace(";", ",").split(",")]
        return tuple(int(part) for part in parts if part)
    return tuple(int(cell) for cell in value)


def apply_peak_hotspot_boost_to_demand(
    demand: object,
    *,
    start_hour: float,
    step_minutes: int,
    peak_hotspot_multiplier: float = 1.0,
    peak_hotspot_windows: str | Sequence[tuple[float, float]] | None = DEFAULT_PEAK_HOTSPOT_WINDOWS,
    peak_hotspot_cells: str | Sequence[int] | None = None,
    peak_hotspot_top_cells: int = 10,
) -> tuple[object, tuple[int, ...], tuple[tuple[float, float], ...]]:
    """Increase demand only in selected cells and clock-time windows.

    This intentionally leaves any caller-provided initial taxi distribution
    untouched; only order rates/events are changed.
    """

    rates = getattr(demand, "rates", None)
    if rates is None:
        return demand, (), ()

    multiplier = float(peak_hotspot_multiplier)
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("peak_hotspot_multiplier must be a finite nonnegative value")
    if abs(multiplier - 1.0) < 1e-9:
        return demand, (), ()

    base_rates = np.asarray(rates, dtype=np.float32)
    if base_rates.ndim != 2:
        raise ValueError("demand rates must have shape (steps, cells)")

    periods = parse_peak_hour_periods(peak_hotspot_windows)
    if not periods:
        return demand, (), ()

    explicit_cells = parse_cell_indices(peak_hotspot_cells)
    period_specs: list[dict[str, object]] = []
    boosted_rates = base_rates.copy()
    boosted_cell_set: set[int] = set()
    boosted_windows: list[tuple[float, float]] = []
    for period_name, start, end in periods:
        step_indices = _peak_step_indices(
            horizon_steps=base_rates.shape[0],
            step_minutes=step_minutes,
            start_hour=start_hour,
            windows=((start, end),),
        )
        if not step_indices:
            continue
        peak_rates = base_rates[np.asarray(step_indices, dtype=np.int64)]
        cells = _select_peak_hotspot_cells(
            peak_rates,
            explicit_cells=explicit_cells,
            top_cells=peak_hotspot_top_cells,
        )
        if not cells:
            continue
        boosted_rates[np.ix_(step_indices, cells)] *= multiplier
        boosted_cell_set.update(cells)
        boosted_windows.append((float(start), float(end)))
        period_specs.append(
            {
                "period": str(period_name),
                "window": (float(start), float(end)),
                "steps": tuple(int(step) for step in step_indices),
                "cells": tuple(int(cell) for cell in cells),
            }
        )

    windows = tuple(boosted_windows)
    cells = tuple(sorted(boosted_cell_set))
    if not period_specs:
        return demand, (), tuple((float(start), float(end)) for _name, start, end in periods)

    metadata = _peak_hotspot_metadata(getattr(demand, "metadata", None), multiplier, period_specs)

    if isinstance(demand, EventTripDemand):
        events_by_day: Sequence[Sequence[Sequence[TripEvent]]] = demand.events_by_day
        for period in period_specs:
            events_by_day = _boost_ordered_events_by_day(
                events_by_day,
                period["steps"],  # type: ignore[arg-type]
                period["cells"],  # type: ignore[arg-type]
                multiplier,
            )
        return (
            EventTripDemand(
                rates=boosted_rates,
                od_probs=np.asarray(demand.od_probs, dtype=np.float32).copy(),
                mean_trip_minutes=np.asarray(demand.mean_trip_minutes, dtype=np.float32).copy(),
                events_by_day=events_by_day,
                stochastic=demand.stochastic,
                metadata=metadata,
            ),
            cells,
            windows,
        )
    if isinstance(demand, EmpiricalTripDemand):
        return (
            EmpiricalTripDemand(
                rates=boosted_rates,
                od_probs=np.asarray(demand.od_probs, dtype=np.float32).copy(),
                mean_trip_minutes=np.asarray(demand.mean_trip_minutes, dtype=np.float32).copy(),
                stochastic=demand.stochastic,
                metadata=metadata,
            ),
            cells,
            windows,
        )
    if isinstance(demand, TabularDemand):
        return (
            TabularDemand(
                rates=boosted_rates,
                od_probs=np.asarray(demand.od_probs, dtype=np.float32).copy(),
                stochastic=demand.stochastic,
            ),
            cells,
            windows,
        )
    return demand, cells, windows


def _parse_clock_hour(value: str) -> float:
    text = value.strip()
    if ":" not in text:
        return float(text)
    hour_text, minute_text = text.split(":", 1)
    return float(hour_text) + float(minute_text) / 60.0


def _default_peak_period_name(index: int) -> str:
    if 0 <= index < len(DEFAULT_PEAK_HOTSPOT_PERIOD_NAMES):
        return DEFAULT_PEAK_HOTSPOT_PERIOD_NAMES[index]
    return f"peak_{index + 1}"


def _peak_step_indices(
    *,
    horizon_steps: int,
    step_minutes: int,
    start_hour: float,
    windows: Sequence[tuple[float, float]],
) -> tuple[int, ...]:
    selected: list[int] = []
    for step in range(int(horizon_steps)):
        midpoint_hour = (float(start_hour) * 60.0 + (step + 0.5) * float(step_minutes)) / 60.0
        if _hour_in_windows(midpoint_hour, windows):
            selected.append(step)
    return tuple(selected)


def _hour_in_windows(hour: float, windows: Sequence[tuple[float, float]]) -> bool:
    hour_mod = hour % 24.0
    for start, end in windows:
        start_mod = float(start) % 24.0
        end_mod = float(end) % 24.0
        if abs(start_mod - end_mod) < 1e-9:
            continue
        if start_mod < end_mod:
            if start_mod <= hour_mod < end_mod:
                return True
        elif hour_mod >= start_mod or hour_mod < end_mod:
            return True
    return False


def _select_peak_hotspot_cells(
    rates: np.ndarray,
    *,
    explicit_cells: Sequence[int],
    top_cells: int,
) -> tuple[int, ...]:
    num_cells = int(rates.shape[1])
    if explicit_cells:
        return _normalize_cell_indices(explicit_cells, num_cells)

    count = min(max(int(top_cells), 0), num_cells)
    if count <= 0:
        return ()
    totals = np.nan_to_num(np.asarray(rates, dtype=np.float64).sum(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    if float(totals.sum()) <= 0.0:
        return tuple(range(count))
    ranked = np.argsort(-totals, kind="stable")
    return tuple(int(cell) for cell in ranked[:count])


def _normalize_cell_indices(cells: Sequence[int], num_cells: int) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for cell in cells:
        idx = int(cell)
        if not (0 <= idx < num_cells):
            raise ValueError(f"peak hotspot cell {idx} is outside [0, {num_cells})")
        if idx not in seen:
            normalized.append(idx)
            seen.add(idx)
    return tuple(normalized)


def _peak_hotspot_metadata(
    metadata: dict[str, object] | None,
    multiplier: float,
    period_specs: Sequence[dict[str, object]],
) -> dict[str, object]:
    meta = dict(metadata or {})
    windows = tuple(period["window"] for period in period_specs)
    cells = tuple(sorted({int(cell) for period in period_specs for cell in period["cells"]}))  # type: ignore[index]
    meta["peak_hotspot_multiplier"] = float(multiplier)
    meta["peak_hotspot_windows"] = tuple((float(start), float(end)) for start, end in windows)
    meta["peak_hotspot_cells"] = tuple(int(cell) for cell in cells)
    meta["peak_hotspot_periods"] = tuple(
        {
            "period": str(period["period"]),
            "window": tuple(float(value) for value in period["window"]),  # type: ignore[union-attr]
            "steps": tuple(int(step) for step in period["steps"]),  # type: ignore[union-attr]
            "cells": tuple(int(cell) for cell in period["cells"]),  # type: ignore[union-attr]
        }
        for period in period_specs
    )
    return meta


def _peak_hotspot_periods_from_demand(demand: object) -> tuple[dict[str, object], ...]:
    metadata = getattr(demand, "metadata", None)
    if not isinstance(metadata, dict):
        return ()
    periods = metadata.get("peak_hotspot_periods", ())
    if not isinstance(periods, (list, tuple)):
        return ()
    normalized: list[dict[str, object]] = []
    for period in periods:
        if isinstance(period, dict):
            normalized.append(dict(period))
    return tuple(normalized)


def _boost_ordered_events_by_day(
    events_by_day: Sequence[Sequence[Sequence[TripEvent]]],
    step_indices: Sequence[int],
    cells: Sequence[int],
    multiplier: float,
) -> tuple[tuple[tuple[TripEvent, ...], ...], ...]:
    step_set = set(int(step) for step in step_indices)
    cell_set = set(int(cell) for cell in cells)
    boosted_days: list[tuple[tuple[TripEvent, ...], ...]] = []
    for day_steps in events_by_day:
        boosted_steps: list[tuple[TripEvent, ...]] = []
        for step, step_events in enumerate(day_steps):
            ordered = tuple(sorted(step_events, key=lambda event: event.minute_offset))
            if step in step_set:
                boosted_steps.append(_boost_step_hotspot_events(ordered, cell_set, multiplier))
            else:
                boosted_steps.append(ordered)
        boosted_days.append(tuple(boosted_steps))
    return tuple(boosted_days)


def _boost_step_hotspot_events(
    events: Sequence[TripEvent],
    cells: set[int],
    multiplier: float,
) -> tuple[TripEvent, ...]:
    if not events or not cells:
        return tuple(sorted(events, key=lambda event: event.minute_offset))
    boosted = [event for event in events if int(event.origin) in cells]
    if not boosted:
        return tuple(sorted(events, key=lambda event: event.minute_offset))
    unchanged = [event for event in events if int(event.origin) not in cells]
    scaled = _scale_step_events(boosted, multiplier)
    return tuple(sorted([*unchanged, *scaled], key=lambda event: event.minute_offset))


def build_chengdu_env(
    raw_paths: str | Path | Sequence[str | Path] = "data/2016年11月成都滴滴订单数据",
    config: EnvConfig | None = None,
    start_hour: float = 6.0,
    bounds: tuple[float, float, float, float] | None = None,
    bounds_quantiles: tuple[float, float] = (0.01, 0.99),
    bounds_samples: int = 200_000,
    grid_padding: float = 0.08,
    average_by_day: bool = True,
    deduplicate: bool = True,
    filter_to_bounds: bool = True,
    stochastic: bool = False,
    demand_scale: float = 1.0,
    seed: int = 0,
    fixed_cell_width_km: float | None = None,
    grid_padding_km: float = 2.5,
    high_demand_min_orders_per_minute: float = 0.0,
    peak_hotspot_multiplier: float = 1.0,
    peak_hotspot_windows: str | Sequence[tuple[float, float]] | None = DEFAULT_PEAK_HOTSPOT_WINDOWS,
    peak_hotspot_cells: str | Sequence[int] | None = None,
    peak_hotspot_top_cells: int = 10,
    road_network: RoadNetwork | None = None,
) -> tuple[DispatchEnv, ChengduDemandStats]:
    """Build a regional dispatch simulator from Chengdu raw order CSV files."""

    config = config or EnvConfig(num_cells=142, fleet_size=6_000, horizon_steps=108, step_minutes=10)
    paths = resolve_raw_paths(raw_paths)
    if bounds is None:
        bounds = infer_chengdu_bounds(
            paths,
            quantiles=bounds_quantiles,
            max_samples=bounds_samples,
            deduplicate=deduplicate,
            seed=seed,
        )
    assignment_grid: HexGrid | None = None
    if fixed_cell_width_km is not None:
        assignment_grid = HexGrid.create_geographic_fixed(bounds, cell_width_km=fixed_cell_width_km, padding_km=grid_padding_km)
        selected_coords = select_high_demand_order_cells(
            paths,
            grid=assignment_grid,
            num_cells=config.num_cells,
            horizon_steps=config.horizon_steps,
            step_minutes=config.step_minutes,
            start_hour=start_hour,
            bounds=bounds,
            average_by_day=average_by_day,
            deduplicate=deduplicate,
            filter_to_bounds=filter_to_bounds,
            min_orders_per_minute=high_demand_min_orders_per_minute,
        )
        grid = HexGrid.from_axial_coords(
            selected_coords,
            cell_width_km=assignment_grid.cell_width_km,
            projection_origin=assignment_grid.projection_origin,
        )
    else:
        grid = HexGrid.create_geographic(config.num_cells, bounds, padding=grid_padding)
    config = replace(config, cell_width_km=grid.cell_width_km)
    demand, stats = build_chengdu_demand(
        paths,
        grid=grid,
        horizon_steps=config.horizon_steps,
        step_minutes=config.step_minutes,
        start_hour=start_hour,
        bounds=bounds,
        bounds_quantiles=bounds_quantiles,
        average_by_day=average_by_day,
        deduplicate=deduplicate,
        filter_to_bounds=filter_to_bounds,
        stochastic=stochastic,
        demand_scale=demand_scale,
        assignment_grid=assignment_grid,
        peak_hotspot_multiplier=peak_hotspot_multiplier,
        peak_hotspot_windows=peak_hotspot_windows,
        peak_hotspot_cells=peak_hotspot_cells,
        peak_hotspot_top_cells=peak_hotspot_top_cells,
    )
    stats.high_demand_min_orders_per_minute = float(high_demand_min_orders_per_minute if assignment_grid is not None else 0.0)
    return DispatchEnv(config, demand=demand, grid=grid, road_network=road_network), stats


def build_chengdu_trajectory_env(
    trajectory_paths: str | Path | Sequence[str | Path] = "data/2016年成都滴滴轨迹数据",
    config: EnvConfig | None = None,
    start_hour: float = 6.0,
    bounds: tuple[float, float, float, float] | None = None,
    bounds_quantiles: tuple[float, float] = (0.01, 0.99),
    bounds_samples: int = 200_000,
    grid_padding_km: float = 2.5,
    average_by_day: bool = True,
    filter_to_bounds: bool = True,
    stochastic: bool = False,
    demand_scale: float = 1.0,
    high_demand_min_orders_per_minute: float = 1.0,
    peak_hotspot_multiplier: float = 1.0,
    peak_hotspot_windows: str | Sequence[tuple[float, float]] | None = DEFAULT_PEAK_HOTSPOT_WINDOWS,
    peak_hotspot_cells: str | Sequence[int] | None = None,
    peak_hotspot_top_cells: int = 10,
    road_network: RoadNetwork | None = None,
    seed: int = 0,
) -> tuple[DispatchEnv, ChengduDemandStats]:
    """Build the paper-style Chengdu simulator by extracting trips from GPS trajectories."""

    config = config or EnvConfig(
        num_cells=142,
        cell_width_km=DEFAULT_HEX_CENTER_SPACING_KM,
        fleet_size=6_000,
        horizon_steps=108,
        step_minutes=10,
    )
    paths = resolve_trajectory_paths(trajectory_paths)
    if bounds is None:
        bounds = infer_trajectory_bounds(
            paths,
            quantiles=bounds_quantiles,
            max_samples=bounds_samples,
            seed=seed,
        )
    assignment_grid = HexGrid.create_geographic_fixed(bounds, cell_width_km=config.cell_width_km, padding_km=grid_padding_km)
    selected_coords = select_high_demand_trajectory_cells(
        paths,
        grid=assignment_grid,
        num_cells=config.num_cells,
        horizon_steps=config.horizon_steps,
        step_minutes=config.step_minutes,
        start_hour=start_hour,
        bounds=bounds,
        average_by_day=average_by_day,
        filter_to_bounds=filter_to_bounds,
        min_orders_per_minute=high_demand_min_orders_per_minute,
    )
    grid = HexGrid.from_axial_coords(
        selected_coords,
        cell_width_km=assignment_grid.cell_width_km,
        projection_origin=assignment_grid.projection_origin,
    )
    config = replace(config, cell_width_km=grid.cell_width_km)
    demand, stats, initial_distribution = build_chengdu_trajectory_demand(
        paths,
        raw_file_count=len(paths),
        grid=grid,
        assignment_grid=assignment_grid,
        horizon_steps=config.horizon_steps,
        step_minutes=config.step_minutes,
        start_hour=start_hour,
        bounds=bounds,
        bounds_quantiles=bounds_quantiles,
        average_by_day=average_by_day,
        filter_to_bounds=filter_to_bounds,
        stochastic=stochastic,
        demand_scale=demand_scale,
        high_demand_min_orders_per_minute=high_demand_min_orders_per_minute,
        peak_hotspot_multiplier=peak_hotspot_multiplier,
        peak_hotspot_windows=peak_hotspot_windows,
        peak_hotspot_cells=peak_hotspot_cells,
        peak_hotspot_top_cells=peak_hotspot_top_cells,
    )
    env = DispatchEnv(
        config,
        demand=demand,
        grid=grid,
        initial_taxi_distribution=initial_distribution,
        road_network=road_network,
    )
    return env, stats


def build_chengdu_demand(
    raw_paths: Sequence[str | Path],
    grid: HexGrid,
    horizon_steps: int,
    step_minutes: int,
    start_hour: float,
    bounds: tuple[float, float, float, float],
    bounds_quantiles: tuple[float, float] = (0.01, 0.99),
    average_by_day: bool = True,
    deduplicate: bool = True,
    filter_to_bounds: bool = True,
    stochastic: bool = False,
    demand_scale: float = 1.0,
    batch_size: int = 8192,
    duration_clip_minutes: tuple[float, float] = (1.0, 180.0),
    assignment_grid: HexGrid | None = None,
    peak_hotspot_multiplier: float = 1.0,
    peak_hotspot_windows: str | Sequence[tuple[float, float]] | None = DEFAULT_PEAK_HOTSPOT_WINDOWS,
    peak_hotspot_cells: str | Sequence[int] | None = None,
    peak_hotspot_top_cells: int = 10,
) -> tuple[EmpiricalTripDemand, ChengduDemandStats]:
    """Aggregate Chengdu raw orders into step/cell OD demand and trip durations."""

    paths = [Path(path) for path in raw_paths]
    num_cells = grid.num_cells
    od_counts = np.zeros((horizon_steps, num_cells, num_cells), dtype=np.float32)
    trip_minutes_sum = np.zeros_like(od_counts, dtype=np.float32)
    trip_minutes_count = np.zeros_like(od_counts, dtype=np.float32)

    rows_seen = 0
    rows_used = 0
    rows_duplicate = 0
    rows_invalid = 0
    rows_outside_bounds = 0
    rows_outside_selected_cells = 0
    active_days: set[str] = set()
    seen_orders: set[tuple[str, ...]] = set()
    selected_coord_to_idx = {coord: idx for idx, coord in enumerate(grid.coords)}

    batch_steps: list[int] = []
    batch_pickup_lon: list[float] = []
    batch_pickup_lat: list[float] = []
    batch_dropoff_lon: list[float] = []
    batch_dropoff_lat: list[float] = []
    batch_duration: list[float] = []

    def flush_batch() -> None:
        nonlocal rows_outside_selected_cells
        if not batch_steps:
            return
        steps = np.asarray(batch_steps, dtype=np.int64)
        pickup_lon = np.asarray(batch_pickup_lon, dtype=np.float64)
        pickup_lat = np.asarray(batch_pickup_lat, dtype=np.float64)
        dropoff_lon = np.asarray(batch_dropoff_lon, dtype=np.float64)
        dropoff_lat = np.asarray(batch_dropoff_lat, dtype=np.float64)
        durations = np.asarray(batch_duration, dtype=np.float32)

        cell_grid = assignment_grid or grid
        origins = cell_grid.nearest_cells_lonlat(pickup_lon, pickup_lat)
        destinations = cell_grid.nearest_cells_lonlat(dropoff_lon, dropoff_lat)
        if assignment_grid is not None:
            origin_selected = np.asarray(
                [selected_coord_to_idx.get(assignment_grid.coords[int(cell)], -1) for cell in origins],
                dtype=np.int64,
            )
            destination_selected = np.asarray(
                [selected_coord_to_idx.get(assignment_grid.coords[int(cell)], -1) for cell in destinations],
                dtype=np.int64,
            )
            selected_mask = (origin_selected >= 0) & (destination_selected >= 0)
            dropped = int(np.count_nonzero(~selected_mask))
            if dropped:
                rows_outside_selected_cells += dropped
            origins = origin_selected[selected_mask]
            destinations = destination_selected[selected_mask]
            steps = steps[selected_mask]
            durations = durations[selected_mask]
            if steps.size == 0:
                batch_steps.clear()
                batch_pickup_lon.clear()
                batch_pickup_lat.clear()
                batch_dropoff_lon.clear()
                batch_dropoff_lat.clear()
                batch_duration.clear()
                return
        np.add.at(od_counts, (steps, origins, destinations), 1.0)
        np.add.at(trip_minutes_sum, (steps, origins, destinations), durations)
        np.add.at(trip_minutes_count, (steps, origins, destinations), 1.0)

        batch_steps.clear()
        batch_pickup_lon.clear()
        batch_pickup_lat.clear()
        batch_dropoff_lon.clear()
        batch_dropoff_lat.clear()
        batch_duration.clear()

    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                continue
            indices = _column_indices(header)
            for row in reader:
                rows_seen += 1
                try:
                    if deduplicate:
                        order_key = _dedupe_key(row, indices)
                        if order_key in seen_orders:
                            rows_duplicate += 1
                            continue
                        seen_orders.add(order_key)

                    start = datetime.fromisoformat(row[indices[START_TIME]])
                    end = datetime.fromisoformat(row[indices[END_TIME]])
                    duration = (end - start).total_seconds() / 60.0
                    if duration < duration_clip_minutes[0] or duration > duration_clip_minutes[1]:
                        rows_invalid += 1
                        continue

                    pickup_lon = float(row[indices[PICKUP_LON]])
                    pickup_lat = float(row[indices[PICKUP_LAT]])
                    dropoff_lon = float(row[indices[DROPOFF_LON]])
                    dropoff_lat = float(row[indices[DROPOFF_LAT]])
                    if not _valid_lonlat(pickup_lon, pickup_lat) or not _valid_lonlat(dropoff_lon, dropoff_lat):
                        rows_invalid += 1
                        continue
                    if filter_to_bounds and (
                        not _inside_bounds(pickup_lon, pickup_lat, bounds)
                        or not _inside_bounds(dropoff_lon, dropoff_lat, bounds)
                    ):
                        rows_outside_bounds += 1
                        continue

                    step = _step_of_day(start, start_hour=start_hour, horizon_steps=horizon_steps, step_minutes=step_minutes)
                    if step is None:
                        continue

                    active_days.add(start.date().isoformat())
                    batch_steps.append(step)
                    batch_pickup_lon.append(pickup_lon)
                    batch_pickup_lat.append(pickup_lat)
                    batch_dropoff_lon.append(dropoff_lon)
                    batch_dropoff_lat.append(dropoff_lat)
                    batch_duration.append(duration)
                    rows_used += 1

                    if len(batch_steps) >= batch_size:
                        flush_batch()
                except (IndexError, KeyError, TypeError, ValueError):
                    rows_invalid += 1

    flush_batch()

    day_count = max(len(active_days), 1)
    if average_by_day:
        od_counts /= float(day_count)
    if demand_scale != 1.0:
        od_counts *= float(demand_scale)

    metadata = {
        "source": "chengdu_raw_orders",
        "days": day_count,
        "rows_used": rows_used,
        "average_by_day": average_by_day,
        "demand_scale": demand_scale,
    }
    demand = EmpiricalTripDemand.from_counts(
        od_counts=od_counts,
        trip_minutes_sum=trip_minutes_sum,
        trip_minutes_count=trip_minutes_count,
        stochastic=stochastic,
        metadata=metadata,
    )
    demand, boosted_cells, boosted_windows = apply_peak_hotspot_boost_to_demand(
        demand,
        start_hour=start_hour,
        step_minutes=step_minutes,
        peak_hotspot_multiplier=peak_hotspot_multiplier,
        peak_hotspot_windows=peak_hotspot_windows,
        peak_hotspot_cells=peak_hotspot_cells,
        peak_hotspot_top_cells=peak_hotspot_top_cells,
    )
    stats = ChengduDemandStats(
        raw_files=len(paths),
        rows_seen=rows_seen,
        rows_used=rows_used,
        rows_duplicate=rows_duplicate,
        rows_invalid=rows_invalid,
        rows_outside_bounds=rows_outside_bounds,
        days=day_count,
        bounds=tuple(float(x) for x in bounds),
        bounds_quantiles=tuple(float(x) for x in bounds_quantiles),
        average_by_day=average_by_day,
        demand_scale=float(demand_scale),
        grid_cell_width_km=float(grid.cell_width_km),
        grid_average_side_length_km=float(grid.average_side_length_km),
        orders_per_episode=float(demand.rates.sum()),
        global_mean_trip_minutes=float(demand.metadata["global_mean_trip_minutes"] if demand.metadata else 0.0),
        rows_outside_selected_cells=rows_outside_selected_cells,
        selected_cells=grid.num_cells,
        high_demand_min_orders_per_minute=0.0,
        peak_hotspot_multiplier=float(peak_hotspot_multiplier),
        peak_hotspot_windows=boosted_windows,
        peak_hotspot_cells=boosted_cells,
        peak_hotspot_periods=_peak_hotspot_periods_from_demand(demand),
    )
    return demand, stats


def select_high_demand_order_cells(
    raw_paths: Sequence[str | Path],
    grid: HexGrid,
    num_cells: int,
    horizon_steps: int,
    step_minutes: int,
    start_hour: float,
    bounds: tuple[float, float, float, float],
    average_by_day: bool = True,
    deduplicate: bool = True,
    filter_to_bounds: bool = True,
    min_orders_per_minute: float = 0.0,
    batch_size: int = 8192,
) -> tuple[tuple[int, int], ...]:
    counts = np.zeros(grid.num_cells, dtype=np.float64)
    active_days: set[str] = set()
    seen_orders: set[tuple[str, ...]] = set()
    batch_lon: list[float] = []
    batch_lat: list[float] = []

    def flush_batch() -> None:
        if not batch_lon:
            return
        cells = grid.nearest_cells_lonlat(
            np.asarray(batch_lon, dtype=np.float64),
            np.asarray(batch_lat, dtype=np.float64),
        )
        np.add.at(counts, cells, 1.0)
        batch_lon.clear()
        batch_lat.clear()

    for path in [Path(path) for path in raw_paths]:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                continue
            indices = _column_indices(header)
            for row in reader:
                try:
                    if deduplicate:
                        order_key = _dedupe_key(row, indices)
                        if order_key in seen_orders:
                            continue
                        seen_orders.add(order_key)
                    start = datetime.fromisoformat(row[indices[START_TIME]])
                    step = _step_of_day(start, start_hour=start_hour, horizon_steps=horizon_steps, step_minutes=step_minutes)
                    if step is None:
                        continue
                    lon = float(row[indices[PICKUP_LON]])
                    lat = float(row[indices[PICKUP_LAT]])
                    if not _valid_lonlat(lon, lat):
                        continue
                    if filter_to_bounds and not _inside_bounds(lon, lat, bounds):
                        continue
                    active_days.add(start.date().isoformat())
                    batch_lon.append(lon)
                    batch_lat.append(lat)
                    if len(batch_lon) >= batch_size:
                        flush_batch()
                except (IndexError, KeyError, TypeError, ValueError):
                    continue

    flush_batch()
    day_count = max(len(active_days), 1)
    ranking_counts = counts / float(day_count) if average_by_day else counts
    threshold = float(min_orders_per_minute) * float(horizon_steps * step_minutes)
    eligible = np.flatnonzero(ranking_counts >= threshold)
    if eligible.size < num_cells:
        eligible = np.flatnonzero(ranking_counts > 0)
    if eligible.size < num_cells:
        eligible = np.arange(grid.num_cells)
    ranked = sorted(eligible.tolist(), key=lambda idx: (-ranking_counts[idx], grid.coords[idx]))
    selected = ranked[:num_cells]
    if len(selected) < num_cells:
        raise ValueError(f"could not select {num_cells} demand cells from the fixed grid")
    return tuple(grid.coords[idx] for idx in selected)


def select_high_demand_trajectory_cells(
    trajectory_paths: Sequence[str | Path],
    grid: HexGrid,
    num_cells: int,
    horizon_steps: int,
    step_minutes: int,
    start_hour: float,
    bounds: tuple[float, float, float, float],
    average_by_day: bool = True,
    filter_to_bounds: bool = True,
    min_orders_per_minute: float = 1.0,
) -> tuple[tuple[int, int], ...]:
    counts = np.zeros(grid.num_cells, dtype=np.float64)
    active_days: set[str] = set()

    for trip in iter_trajectory_trips(trajectory_paths):
        step = _step_of_day(trip.start, start_hour=start_hour, horizon_steps=horizon_steps, step_minutes=step_minutes)
        if step is None:
            continue
        if filter_to_bounds and not _inside_bounds(trip.pickup_lon, trip.pickup_lat, bounds):
            continue
        if not _valid_lonlat(trip.pickup_lon, trip.pickup_lat):
            continue
        cell = int(grid.nearest_cells_lonlat(np.asarray([trip.pickup_lon]), np.asarray([trip.pickup_lat]))[0])
        counts[cell] += 1.0
        active_days.add(trip.start.date().isoformat())

    day_count = max(len(active_days), 1)
    ranking_counts = counts / float(day_count) if average_by_day else counts
    threshold = float(min_orders_per_minute) * float(horizon_steps * step_minutes)
    eligible = np.flatnonzero(ranking_counts >= threshold)
    if eligible.size < num_cells:
        eligible = np.flatnonzero(ranking_counts > 0)
    if eligible.size < num_cells:
        eligible = np.arange(grid.num_cells)
    ranked = sorted(eligible.tolist(), key=lambda idx: (-ranking_counts[idx], grid.coords[idx]))
    selected = ranked[:num_cells]
    if len(selected) < num_cells:
        raise ValueError(f"could not select {num_cells} trajectory demand cells from the fixed grid")
    return tuple(grid.coords[idx] for idx in selected)


def build_chengdu_trajectory_demand(
    trajectory_paths: Sequence[str | Path],
    raw_file_count: int,
    grid: HexGrid,
    assignment_grid: HexGrid,
    horizon_steps: int,
    step_minutes: int,
    start_hour: float,
    bounds: tuple[float, float, float, float],
    bounds_quantiles: tuple[float, float] = (0.01, 0.99),
    average_by_day: bool = True,
    filter_to_bounds: bool = True,
    stochastic: bool = False,
    demand_scale: float = 1.0,
    high_demand_min_orders_per_minute: float = 1.0,
    duration_clip_minutes: tuple[float, float] = (1.0, 180.0),
    peak_hotspot_multiplier: float = 1.0,
    peak_hotspot_windows: str | Sequence[tuple[float, float]] | None = DEFAULT_PEAK_HOTSPOT_WINDOWS,
    peak_hotspot_cells: str | Sequence[int] | None = None,
    peak_hotspot_top_cells: int = 10,
) -> tuple[EmpiricalTripDemand, ChengduDemandStats, np.ndarray | None]:
    num_cells = grid.num_cells
    od_counts = np.zeros((horizon_steps, num_cells, num_cells), dtype=np.float32)
    trip_minutes_sum = np.zeros_like(od_counts, dtype=np.float32)
    trip_minutes_count = np.zeros_like(od_counts, dtype=np.float32)
    initial_counts = np.zeros(num_cells, dtype=np.float64)
    selected_coord_to_idx = {coord: idx for idx, coord in enumerate(grid.coords)}

    rows_seen = 0
    rows_used = 0
    rows_invalid = 0
    rows_outside_bounds = 0
    rows_outside_selected_cells = 0
    trajectory_points = 0
    active_days: set[str] = set()
    speed_sum = 0.0
    speed_count = 0

    for trip in iter_trajectory_trips(trajectory_paths):
        rows_seen += 1
        trajectory_points += trip.points
        duration = trip.duration_minutes
        if duration < duration_clip_minutes[0] or duration > duration_clip_minutes[1]:
            rows_invalid += 1
            continue
        if (
            not _valid_lonlat(trip.pickup_lon, trip.pickup_lat)
            or not _valid_lonlat(trip.dropoff_lon, trip.dropoff_lat)
        ):
            rows_invalid += 1
            continue
        if filter_to_bounds and (
            not _inside_bounds(trip.pickup_lon, trip.pickup_lat, bounds)
            or not _inside_bounds(trip.dropoff_lon, trip.dropoff_lat, bounds)
        ):
            rows_outside_bounds += 1
            continue
        step = _step_of_day(trip.start, start_hour=start_hour, horizon_steps=horizon_steps, step_minutes=step_minutes)
        if step is None:
            continue

        origin_full = int(assignment_grid.nearest_cells_lonlat(np.asarray([trip.pickup_lon]), np.asarray([trip.pickup_lat]))[0])
        dest_full = int(assignment_grid.nearest_cells_lonlat(np.asarray([trip.dropoff_lon]), np.asarray([trip.dropoff_lat]))[0])
        origin = selected_coord_to_idx.get(assignment_grid.coords[origin_full], -1)
        destination = selected_coord_to_idx.get(assignment_grid.coords[dest_full], -1)
        if origin < 0 or destination < 0:
            rows_outside_selected_cells += 1
            continue

        od_counts[step, origin, destination] += 1.0
        trip_minutes_sum[step, origin, destination] += float(duration)
        trip_minutes_count[step, origin, destination] += 1.0
        if step == 0:
            initial_counts[origin] += 1.0
        active_days.add(trip.start.date().isoformat())
        rows_used += 1
        if np.isfinite(trip.speed_kmh) and trip.speed_kmh > 0:
            speed_sum += float(trip.speed_kmh)
            speed_count += 1

    day_count = max(len(active_days), 1)
    if average_by_day:
        od_counts /= float(day_count)
        trip_minutes_sum /= float(day_count)
        trip_minutes_count /= float(day_count)
        initial_counts /= float(day_count)
    if demand_scale != 1.0:
        od_counts *= float(demand_scale)

    metadata = {
        "source": "chengdu_trajectory_orders",
        "days": day_count,
        "rows_used": rows_used,
        "average_by_day": average_by_day,
        "demand_scale": demand_scale,
    }
    demand = EmpiricalTripDemand.from_counts(
        od_counts=od_counts,
        trip_minutes_sum=trip_minutes_sum,
        trip_minutes_count=trip_minutes_count,
        stochastic=stochastic,
        metadata=metadata,
    )
    demand, boosted_cells, boosted_windows = apply_peak_hotspot_boost_to_demand(
        demand,
        start_hour=start_hour,
        step_minutes=step_minutes,
        peak_hotspot_multiplier=peak_hotspot_multiplier,
        peak_hotspot_windows=peak_hotspot_windows,
        peak_hotspot_cells=peak_hotspot_cells,
        peak_hotspot_top_cells=peak_hotspot_top_cells,
    )
    stats = ChengduDemandStats(
        raw_files=raw_file_count,
        rows_seen=rows_seen,
        rows_used=rows_used,
        rows_duplicate=0,
        rows_invalid=rows_invalid,
        rows_outside_bounds=rows_outside_bounds,
        days=day_count,
        bounds=tuple(float(x) for x in bounds),
        bounds_quantiles=tuple(float(x) for x in bounds_quantiles),
        average_by_day=average_by_day,
        demand_scale=float(demand_scale),
        grid_cell_width_km=float(grid.cell_width_km),
        grid_average_side_length_km=float(grid.average_side_length_km),
        orders_per_episode=float(demand.rates.sum()),
        global_mean_trip_minutes=float(demand.metadata["global_mean_trip_minutes"] if demand.metadata else 0.0),
        source="chengdu_trajectory_orders",
        rows_outside_selected_cells=rows_outside_selected_cells,
        selected_cells=grid.num_cells,
        high_demand_min_orders_per_minute=float(high_demand_min_orders_per_minute),
        mean_trip_speed_kmh=float(speed_sum / speed_count) if speed_count else 0.0,
        trajectory_points=trajectory_points,
        peak_hotspot_multiplier=float(peak_hotspot_multiplier),
        peak_hotspot_windows=boosted_windows,
        peak_hotspot_cells=boosted_cells,
        peak_hotspot_periods=_peak_hotspot_periods_from_demand(demand),
    )
    return demand, stats, initial_counts if float(initial_counts.sum()) > 0 else None


def build_chengdu_trajectory_event_demand(
    trajectory_paths: Sequence[str | Path],
    raw_file_count: int,
    grid: HexGrid,
    assignment_grid: HexGrid,
    horizon_steps: int,
    step_minutes: int,
    start_hour: float,
    bounds: tuple[float, float, float, float],
    bounds_quantiles: tuple[float, float] = (0.01, 0.99),
    average_by_day: bool = True,
    filter_to_bounds: bool = True,
    stochastic: bool = False,
    demand_scale: float = 1.0,
    high_demand_min_orders_per_minute: float = 1.0,
    duration_clip_minutes: tuple[float, float] = (1.0, 180.0),
    peak_hotspot_multiplier: float = 1.0,
    peak_hotspot_windows: str | Sequence[tuple[float, float]] | None = DEFAULT_PEAK_HOTSPOT_WINDOWS,
    peak_hotspot_cells: str | Sequence[int] | None = None,
    peak_hotspot_top_cells: int = 10,
) -> tuple[EventTripDemand, ChengduDemandStats, np.ndarray | None]:
    """Build ordered one-minute trajectory demand on a fixed selected grid."""

    num_cells = grid.num_cells
    od_counts = np.zeros((horizon_steps, num_cells, num_cells), dtype=np.float32)
    trip_minutes_sum = np.zeros_like(od_counts, dtype=np.float32)
    trip_minutes_count = np.zeros_like(od_counts, dtype=np.float32)
    initial_counts = np.zeros(num_cells, dtype=np.float64)
    selected_coord_to_idx = {coord: idx for idx, coord in enumerate(grid.coords)}

    rows_seen = 0
    rows_used = 0
    rows_invalid = 0
    rows_outside_bounds = 0
    rows_outside_selected_cells = 0
    trajectory_points = 0
    active_days: set[str] = set()
    speed_sum = 0.0
    speed_count = 0
    events_by_day: dict[str, list[list[TripEvent]]] = {}

    for trip in iter_trajectory_trips(trajectory_paths):
        rows_seen += 1
        trajectory_points += trip.points
        duration = trip.duration_minutes
        if duration < duration_clip_minutes[0] or duration > duration_clip_minutes[1]:
            rows_invalid += 1
            continue
        if (
            not _valid_lonlat(trip.pickup_lon, trip.pickup_lat)
            or not _valid_lonlat(trip.dropoff_lon, trip.dropoff_lat)
        ):
            rows_invalid += 1
            continue
        if filter_to_bounds and (
            not _inside_bounds(trip.pickup_lon, trip.pickup_lat, bounds)
            or not _inside_bounds(trip.dropoff_lon, trip.dropoff_lat, bounds)
        ):
            rows_outside_bounds += 1
            continue

        minute = _minute_of_horizon(
            trip.start,
            start_hour=start_hour,
            horizon_steps=horizon_steps,
            step_minutes=step_minutes,
        )
        if minute is None:
            continue
        step = int(minute // step_minutes)
        minute_offset = int(minute % step_minutes)

        origin_full = int(assignment_grid.nearest_cells_lonlat(np.asarray([trip.pickup_lon]), np.asarray([trip.pickup_lat]))[0])
        dest_full = int(assignment_grid.nearest_cells_lonlat(np.asarray([trip.dropoff_lon]), np.asarray([trip.dropoff_lat]))[0])
        origin = selected_coord_to_idx.get(assignment_grid.coords[origin_full], -1)
        destination = selected_coord_to_idx.get(assignment_grid.coords[dest_full], -1)
        if origin < 0 or destination < 0:
            rows_outside_selected_cells += 1
            continue

        pickup_xy = grid.project_lonlat(np.asarray([trip.pickup_lon]), np.asarray([trip.pickup_lat]))[0]
        dropoff_xy = grid.project_lonlat(np.asarray([trip.dropoff_lon]), np.asarray([trip.dropoff_lat]))[0]
        event = TripEvent(
            origin=int(origin),
            destination=int(destination),
            minute_offset=minute_offset,
            trip_minutes=float(duration),
            pickup_xy=(float(pickup_xy[0]), float(pickup_xy[1])),
            dropoff_xy=(float(dropoff_xy[0]), float(dropoff_xy[1])),
        )

        day = trip.start.date().isoformat()
        day_steps = events_by_day.setdefault(day, [[] for _ in range(horizon_steps)])
        day_steps[step].append(event)
        active_days.add(day)

        od_counts[step, origin, destination] += 1.0
        trip_minutes_sum[step, origin, destination] += float(duration)
        trip_minutes_count[step, origin, destination] += 1.0
        if step == 0:
            initial_counts[origin] += 1.0
        rows_used += 1
        if np.isfinite(trip.speed_kmh) and trip.speed_kmh > 0:
            speed_sum += float(trip.speed_kmh)
            speed_count += 1

    day_count = max(len(active_days), 1)
    if average_by_day:
        od_counts /= float(day_count)
        trip_minutes_sum /= float(day_count)
        trip_minutes_count /= float(day_count)
        initial_counts /= float(day_count)
    if demand_scale != 1.0:
        od_counts *= float(demand_scale)

    metadata = {
        "source": "chengdu_trajectory_event_orders",
        "days": day_count,
        "rows_used": rows_used,
        "average_by_day": average_by_day,
        "demand_scale": demand_scale,
        "event_stream": True,
    }
    aggregate = EmpiricalTripDemand.from_counts(
        od_counts=od_counts,
        trip_minutes_sum=trip_minutes_sum,
        trip_minutes_count=trip_minutes_count,
        stochastic=stochastic,
        metadata=metadata,
    )
    ordered_events = _scaled_ordered_events_by_day(events_by_day, demand_scale)
    demand = EventTripDemand(
        rates=aggregate.rates,
        od_probs=aggregate.od_probs,
        mean_trip_minutes=aggregate.mean_trip_minutes,
        events_by_day=ordered_events,
        stochastic=stochastic,
        metadata=aggregate.metadata,
    )
    demand, boosted_cells, boosted_windows = apply_peak_hotspot_boost_to_demand(
        demand,
        start_hour=start_hour,
        step_minutes=step_minutes,
        peak_hotspot_multiplier=peak_hotspot_multiplier,
        peak_hotspot_windows=peak_hotspot_windows,
        peak_hotspot_cells=peak_hotspot_cells,
        peak_hotspot_top_cells=peak_hotspot_top_cells,
    )
    stats = ChengduDemandStats(
        raw_files=raw_file_count,
        rows_seen=rows_seen,
        rows_used=rows_used,
        rows_duplicate=0,
        rows_invalid=rows_invalid,
        rows_outside_bounds=rows_outside_bounds,
        days=day_count,
        bounds=tuple(float(x) for x in bounds),
        bounds_quantiles=tuple(float(x) for x in bounds_quantiles),
        average_by_day=average_by_day,
        demand_scale=float(demand_scale),
        grid_cell_width_km=float(grid.cell_width_km),
        grid_average_side_length_km=float(grid.average_side_length_km),
        orders_per_episode=float(demand.rates.sum()),
        global_mean_trip_minutes=float(demand.metadata["global_mean_trip_minutes"] if demand.metadata else 0.0),
        source="chengdu_trajectory_event_orders",
        rows_outside_selected_cells=rows_outside_selected_cells,
        selected_cells=grid.num_cells,
        high_demand_min_orders_per_minute=float(high_demand_min_orders_per_minute),
        mean_trip_speed_kmh=float(speed_sum / speed_count) if speed_count else 0.0,
        trajectory_points=trajectory_points,
        peak_hotspot_multiplier=float(peak_hotspot_multiplier),
        peak_hotspot_windows=boosted_windows,
        peak_hotspot_cells=boosted_cells,
        peak_hotspot_periods=_peak_hotspot_periods_from_demand(demand),
    )
    return demand, stats, initial_counts if float(initial_counts.sum()) > 0 else None


def build_chengdu_trajectory_od_matrices(
    trajectory_paths: Sequence[str | Path],
    grid: HexGrid,
    assignment_grid: HexGrid | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    filter_to_bounds: bool = True,
    duration_clip_minutes: tuple[float, float] = (1.0, 180.0),
    default_speed_kmph: float = 30.0,
    min_samples_per_od: int = 1,
) -> tuple[RoadCostMatrices, dict[str, object]]:
    """Build cell OD distance/time matrices from historical trajectory trips.

    The caller controls which files are passed in. For leak-free experiments,
    pass only the training split files, e.g. 2016-11-08 through 2016-11-30.
    """

    if assignment_grid is None:
        assignment_grid = grid
    selected_coord_to_idx = {coord: idx for idx, coord in enumerate(grid.coords)}

    num_cells = grid.num_cells
    distance_sum = np.zeros((num_cells, num_cells), dtype=np.float64)
    time_sum = np.zeros((num_cells, num_cells), dtype=np.float64)
    sample_count = np.zeros((num_cells, num_cells), dtype=np.int64)

    rows_seen = 0
    rows_used = 0
    rows_invalid = 0
    rows_outside_bounds = 0
    rows_outside_selected_cells = 0
    trajectory_points = 0
    active_days: set[str] = set()

    for trip in iter_trajectory_trips(trajectory_paths):
        rows_seen += 1
        trajectory_points += trip.points
        duration = float(trip.duration_minutes)
        if duration < duration_clip_minutes[0] or duration > duration_clip_minutes[1]:
            rows_invalid += 1
            continue
        if (
            not _valid_lonlat(trip.pickup_lon, trip.pickup_lat)
            or not _valid_lonlat(trip.dropoff_lon, trip.dropoff_lat)
        ):
            rows_invalid += 1
            continue
        if filter_to_bounds and bounds is not None and (
            not _inside_bounds(trip.pickup_lon, trip.pickup_lat, bounds)
            or not _inside_bounds(trip.dropoff_lon, trip.dropoff_lat, bounds)
        ):
            rows_outside_bounds += 1
            continue

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
        destination = selected_coord_to_idx.get(assignment_grid.coords[dest_full], -1)
        if origin < 0 or destination < 0:
            rows_outside_selected_cells += 1
            continue

        distance_km = float(trip.distance_km)
        if not np.isfinite(distance_km) or distance_km <= 0.0:
            pickup_xy = grid.project_lonlat(np.asarray([trip.pickup_lon]), np.asarray([trip.pickup_lat]))[0]
            dropoff_xy = grid.project_lonlat(np.asarray([trip.dropoff_lon]), np.asarray([trip.dropoff_lat]))[0]
            distance_km = float(np.linalg.norm(pickup_xy - dropoff_xy))

        if not np.isfinite(distance_km) or distance_km < 0.0:
            rows_invalid += 1
            continue

        distance_sum[origin, destination] += distance_km
        time_sum[origin, destination] += duration
        sample_count[origin, destination] += 1
        active_days.add(trip.start.date().isoformat())
        rows_used += 1

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
        "source": "chengdu_train_trajectory_od_matrices",
        "rows_seen": rows_seen,
        "rows_used": rows_used,
        "rows_invalid": rows_invalid,
        "rows_outside_bounds": rows_outside_bounds,
        "rows_outside_selected_cells": rows_outside_selected_cells,
        "trajectory_points": trajectory_points,
        "days": len(active_days),
        "num_cells": num_cells,
        "observed_od_pairs": int((observed_mask & off_diagonal).sum()),
        "fallback_distance_count": fallback_count,
        "fallback_time_count": fallback_count,
        "fallback_distance_ratio": float(fallback_count / max(num_cells * num_cells - num_cells, 1)),
        "fallback_time_ratio": float(fallback_count / max(num_cells * num_cells - num_cells, 1)),
        "mean_observed_speed_kmph": float(observed_speed_kmph),
        "min_samples_per_od": min_samples,
        "duration_clip_minutes": tuple(float(x) for x in duration_clip_minutes),
    }
    return matrices, stats


def _scaled_ordered_events_by_day(
    events_by_day: dict[str, list[list[TripEvent]]],
    demand_scale: float,
) -> tuple[tuple[tuple[TripEvent, ...], ...], ...]:
    scale = max(float(demand_scale), 0.0)
    return tuple(
        _scale_day_steps(events_by_day[day], scale)
        for day in sorted(events_by_day)
    )


def _scale_day_steps(day_steps: list[list[TripEvent]], scale: float) -> tuple[tuple[TripEvent, ...], ...]:
    ordered_steps = [tuple(sorted(step_events, key=lambda event: event.minute_offset)) for step_events in day_steps]
    if abs(scale - 1.0) < 1e-9:
        return tuple(ordered_steps)

    flat: list[tuple[int, TripEvent]] = [
        (step, event)
        for step, step_events in enumerate(ordered_steps)
        for event in step_events
    ]
    if not flat:
        return tuple(ordered_steps)

    target = int(np.floor(len(flat) * scale + 0.5))
    scaled_steps: list[list[TripEvent]] = [[] for _ in day_steps]
    if target <= 0:
        return tuple(tuple() for _ in day_steps)
    if target <= len(flat):
        indices = np.linspace(0, len(flat) - 1, num=target, dtype=np.int64)
        for index in indices:
            step, event = flat[int(index)]
            scaled_steps[step].append(event)
    else:
        base_repeats = int(np.floor(scale))
        for step, event in flat:
            scaled_steps[step].extend([event] * base_repeats)
        extras = target - base_repeats * len(flat)
        for step, event in flat[:extras]:
            scaled_steps[step].append(event)
    return tuple(tuple(sorted(step_events, key=lambda event: event.minute_offset)) for step_events in scaled_steps)


def _scale_step_events(events: list[TripEvent], scale: float) -> tuple[TripEvent, ...]:
    ordered = tuple(sorted(events, key=lambda event: event.minute_offset))
    if not ordered or abs(scale - 1.0) < 1e-9:
        return ordered

    target = int(np.floor(len(ordered) * scale + 0.5))
    if target <= 0:
        return ()
    if target <= len(ordered):
        if target == len(ordered):
            return ordered
        indices = np.linspace(0, len(ordered) - 1, num=target, dtype=np.int64)
        return tuple(ordered[int(index)] for index in indices)

    base_repeats = int(np.floor(scale))
    repeated: list[TripEvent] = []
    for event in ordered:
        repeated.extend([event] * base_repeats)
    extras = target - len(repeated)
    for event in ordered[:extras]:
        repeated.append(event)
    return tuple(sorted(repeated, key=lambda event: event.minute_offset))


def infer_chengdu_bounds(
    raw_paths: Sequence[str | Path],
    quantiles: tuple[float, float] = (0.01, 0.99),
    max_samples: int = 200_000,
    deduplicate: bool = True,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Infer robust lon/lat bounds from pickup and dropoff points."""

    paths = [Path(path) for path in raw_paths]
    q_low, q_high = quantiles
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")

    rng = np.random.default_rng(seed)
    samples: list[tuple[float, float]] = []
    points_seen = 0
    seen_orders: set[tuple[str, ...]] = set()

    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                continue
            indices = _column_indices(header)
            for row in reader:
                try:
                    if deduplicate:
                        order_key = _dedupe_key(row, indices)
                        if order_key in seen_orders:
                            continue
                        seen_orders.add(order_key)
                    points = (
                        (float(row[indices[PICKUP_LON]]), float(row[indices[PICKUP_LAT]])),
                        (float(row[indices[DROPOFF_LON]]), float(row[indices[DROPOFF_LAT]])),
                    )
                except (IndexError, KeyError, TypeError, ValueError):
                    continue

                for lon, lat in points:
                    if not _valid_lonlat(lon, lat):
                        continue
                    points_seen += 1
                    if len(samples) < max_samples:
                        samples.append((lon, lat))
                    else:
                        replace_at = int(rng.integers(0, points_seen))
                        if replace_at < max_samples:
                            samples[replace_at] = (lon, lat)

    if not samples:
        raise ValueError("could not infer bounds from raw files")

    arr = np.asarray(samples, dtype=np.float64)
    lon_min, lat_min = np.quantile(arr, q_low, axis=0)
    lon_max, lat_max = np.quantile(arr, q_high, axis=0)
    return (float(lon_min), float(lat_min), float(lon_max), float(lat_max))


def infer_trajectory_bounds(
    trajectory_paths: Sequence[str | Path],
    quantiles: tuple[float, float] = (0.01, 0.99),
    max_samples: int = 200_000,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    q_low, q_high = quantiles
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")

    rng = np.random.default_rng(seed)
    samples: list[tuple[float, float]] = []
    points_seen = 0
    for path in trajectory_paths:
        for row in iter_trajectory_rows(path):
            try:
                lon = float(row["lon"])
                lat = float(row["lat"])
            except (KeyError, TypeError, ValueError):
                continue
            if not _valid_lonlat(lon, lat):
                continue
            points_seen += 1
            if len(samples) < max_samples:
                samples.append((lon, lat))
            else:
                replace_at = int(rng.integers(0, points_seen))
                if replace_at < max_samples:
                    samples[replace_at] = (lon, lat)

    if not samples:
        raise ValueError("could not infer bounds from trajectory files")
    arr = np.asarray(samples, dtype=np.float64)
    lon_min, lat_min = np.quantile(arr, q_low, axis=0)
    lon_max, lat_max = np.quantile(arr, q_high, axis=0)
    return (float(lon_min), float(lat_min), float(lon_max), float(lat_max))


def iter_trajectory_trips(trajectory_paths: Sequence[str | Path]) -> Iterable[TrajectoryTrip]:
    for path in [Path(path) for path in trajectory_paths]:
        trips: dict[str, dict[str, object]] = {}
        for row in iter_trajectory_rows(path):
            try:
                order_id = str(row["order_id"])
                driver_id = str(row["driver_id"])
                if not order_id:
                    continue
                ts = datetime.fromisoformat(str(row["time"]))
                lon = float(row["lon"])
                lat = float(row["lat"])
            except (KeyError, TypeError, ValueError):
                continue
            if not _valid_lonlat(lon, lat):
                continue

            record = trips.get(order_id)
            if record is None:
                trips[order_id] = {
                    "order_id": order_id,
                    "driver_id": driver_id,
                    "start": ts,
                    "end": ts,
                    "pickup_lon": lon,
                    "pickup_lat": lat,
                    "dropoff_lon": lon,
                    "dropoff_lat": lat,
                    "last_time": ts,
                    "last_lon": lon,
                    "last_lat": lat,
                    "points": 1,
                    "distance_km": 0.0,
                }
                continue

            record["points"] = int(record["points"]) + 1
            last_time = record["last_time"]
            last_lon = float(record["last_lon"])
            last_lat = float(record["last_lat"])
            if isinstance(last_time, datetime) and ts >= last_time:
                record["distance_km"] = float(record["distance_km"]) + _point_distance_km(last_lon, last_lat, lon, lat)
                record["last_time"] = ts
                record["last_lon"] = lon
                record["last_lat"] = lat

            if ts < record["start"]:
                record["start"] = ts
                record["pickup_lon"] = lon
                record["pickup_lat"] = lat
            if ts > record["end"]:
                record["end"] = ts
                record["dropoff_lon"] = lon
                record["dropoff_lat"] = lat

        for record in trips.values():
            start = record["start"]
            end = record["end"]
            if not isinstance(start, datetime) or not isinstance(end, datetime) or end <= start:
                continue
            yield TrajectoryTrip(
                order_id=str(record["order_id"]),
                driver_id=str(record["driver_id"]),
                start=start,
                end=end,
                pickup_lon=float(record["pickup_lon"]),
                pickup_lat=float(record["pickup_lat"]),
                dropoff_lon=float(record["dropoff_lon"]),
                dropoff_lat=float(record["dropoff_lat"]),
                points=int(record["points"]),
                distance_km=float(record["distance_km"]),
            )


def iter_trajectory_rows(path: str | Path) -> Iterable[dict[str, str]]:
    path = Path(path)
    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the real trajectory archive. "
            "Fetch the LFS files before preprocessing."
        )
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            yield from _iter_trajectory_csv_reader(f)
        return

    if path.name.endswith(".tar.gz") or path.suffix == ".tgz" or path.suffix == ".tar":
        mode = "r:gz" if path.name.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(path, mode) as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".csv"):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                with io.TextIOWrapper(extracted, encoding="utf-8-sig", newline="") as f:
                    yield from _iter_trajectory_csv_reader(f)
        return

    raise ValueError(f"unsupported trajectory file: {path}")


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(128)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _iter_trajectory_csv_reader(f: Iterable[str]) -> Iterable[dict[str, str]]:
    reader = csv.reader(f)
    try:
        header = next(reader)
    except StopIteration:
        return
    indices = _trajectory_column_indices(header)
    for row in reader:
        try:
            yield {
                "driver_id": row[indices["driver_id"]],
                "order_id": row[indices["order_id"]],
                "time": row[indices["time"]],
                "lon": row[indices["lon"]],
                "lat": row[indices["lat"]],
            }
        except IndexError:
            continue


def _trajectory_column_indices(header: Sequence[str]) -> dict[str, int]:
    header_map = {name: idx for idx, name in enumerate(header)}
    aliases = {
        "driver_id": ("司机ID", "driver_id", "taxi_id", "vehicle_id"),
        "order_id": ("订单ID", "order_id", "trip_id"),
        "time": ("GPS时间", "gps_time", "time", "timestamp"),
        "lon": ("轨迹点经度", "lon", "longitude"),
        "lat": ("轨迹点纬度", "lat", "latitude"),
    }
    result: dict[str, int] = {}
    for key, names in aliases.items():
        for name in names:
            if name in header_map:
                result[key] = header_map[name]
                break
        else:
            raise ValueError("trajectory CSV must contain driver, order, time, lon and lat columns")
    return result


def _point_distance_km(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    origin_lon = (lon_a + lon_b) / 2.0
    origin_lat = (lat_a + lat_b) / 2.0
    x, y = lonlat_to_xy(np.asarray([lon_a, lon_b]), np.asarray([lat_a, lat_b]), origin_lon, origin_lat)
    return float(np.hypot(float(x[1] - x[0]), float(y[1] - y[0])))


def simulate_policy(
    env: DispatchEnv,
    method: str,
    seed: int = 0,
) -> tuple[EpisodeMetrics, DispatchEnv]:
    rng = np.random.default_rng(seed)

    def policy(cur_env: DispatchEnv, observations: np.ndarray, state: np.ndarray) -> np.ndarray:
        if method == "park":
            return park_actions(cur_env)
        if method == "random":
            return random_actions(cur_env, rng)
        if method == "diffusion":
            return diffusion_actions(cur_env)
        raise ValueError(f"unknown policy: {method}")

    env.reset(seed)
    done = False
    reward = 0.0
    while not done:
        _obs, _state, critic_rewards, _actor_rewards, done = env.advance(policy)
        reward += float(np.sum(critic_rewards))
    return env.metrics(reward), env


def write_grid(path: str | Path, grid: HexGrid) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "q", "r", "x_km", "y_km", "lon", "lat"])
        for idx, ((q, r), xy) in enumerate(zip(grid.coords, grid.xy)):
            lon = ""
            lat = ""
            if grid.lonlat is not None:
                lon = f"{float(grid.lonlat[idx, 0]):.8f}"
                lat = f"{float(grid.lonlat[idx, 1]):.8f}"
            writer.writerow([idx, q, r, f"{float(xy[0]):.4f}", f"{float(xy[1]):.4f}", lon, lat])


def resolve_raw_paths(raw_paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(raw_paths, (str, Path)):
        raw_items: Iterable[str | Path] = [raw_paths]
    else:
        raw_items = raw_paths

    paths: list[Path] = []
    for raw in raw_items:
        raw_str = str(raw)
        path = Path(raw_str)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.csv")))
        elif path.exists():
            paths.append(path)
        else:
            matches = sorted(Path(match) for match in glob.glob(raw_str))
            paths.extend(match for match in matches if match.is_file())

    unique_paths = sorted(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise FileNotFoundError(f"no raw CSV files found for {raw_paths}")
    return unique_paths


def resolve_trajectory_paths(raw_paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(raw_paths, (str, Path)):
        raw_items: Iterable[str | Path] = [raw_paths]
    else:
        raw_items = raw_paths

    paths: list[Path] = []
    for raw in raw_items:
        raw_str = str(raw)
        path = Path(raw_str)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.tar.gz")))
            paths.extend(sorted(path.rglob("*.tgz")))
            paths.extend(
                sorted(
                    csv_path
                    for csv_path in path.rglob("*.csv")
                    if "轨迹" in str(csv_path) or "trajectory" in str(csv_path).lower() or "gps" in str(csv_path).lower()
                )
            )
        elif path.exists():
            paths.append(path)
        else:
            matches = sorted(Path(match) for match in glob.glob(raw_str))
            paths.extend(match for match in matches if match.is_file())

    unique_paths = sorted(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise FileNotFoundError(f"no trajectory CSV/tar.gz files found for {raw_paths}")
    return unique_paths


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = EnvConfig(
        num_cells=args.cells,
        fleet_size=args.taxis,
        horizon_steps=args.horizon_steps,
        step_minutes=args.step_minutes,
        max_wait_steps=args.max_wait_steps,
        demand_scale=args.demand_scale,
        seed=args.seed,
        travel_speed_kmph=args.road_speed_kmph,
        relocation_cost_weight=args.relocation_cost_weight,
        future_demand_steps=args.future_demand_steps,
        region_value_weight=_region_value_weight_arg(args),
        future_value_weight=_region_value_weight_arg(args),
        future_gap_weight=args.future_gap_weight,
        dispatch_intensity_weight=args.dispatch_intensity_weight,
        trip_time_weight=args.trip_time_weight,
        service_revenue_weight=args.service_revenue_weight,
        fare_base=args.fare_base,
        fare_per_km=args.fare_per_km,
        fare_per_minute=args.fare_per_minute,
        wait_time_penalty_weight=args.wait_time_penalty_weight,
        wait_time_tolerance_minutes=args.wait_time_tolerance_minutes,
        wait_time_penalty_exponent=args.wait_time_penalty_exponent,
        cancellation_penalty=args.cancellation_penalty,
        future_imbalance_weight=args.future_imbalance_weight,
        idle_time_penalty_weight=args.idle_time_penalty_weight,
        execution_mismatch_penalty=args.execution_mismatch_penalty,
        same_step_reposition_service=args.same_step_reposition_service,
        reposition_before_assignment=args.reposition_before_assignment,
        pickup_scope="origin",
        matching_mode="greedy",
    )
    bounds = _parse_bounds(args.bounds) if args.bounds else None
    bounds_quantiles = _parse_quantiles(args.bounds_quantiles)
    road_network = _load_road_network_from_args(args)
    if road_network is None:
        message = "no road network loaded; using Euclidean/grid-distance fallback routing."
        if args.require_road_costs:
            raise SystemExit(
                "ERROR: " + message + " Pass --road-nodes/--road-edges, --road-network, or --download-road-network."
            )
        print("WARNING: " + message)
    base_env, stats = build_chengdu_env(
        raw_paths=args.raw,
        config=config,
        start_hour=args.start_hour,
        bounds=bounds,
        bounds_quantiles=bounds_quantiles,
        bounds_samples=args.bounds_samples,
        grid_padding=args.grid_padding,
        average_by_day=args.average_by_day,
        deduplicate=args.deduplicate,
        filter_to_bounds=not args.keep_outside_bounds,
        stochastic=args.stochastic_demand,
        demand_scale=args.demand_scale,
        peak_hotspot_multiplier=args.peak_hotspot_multiplier,
        peak_hotspot_windows=args.peak_hotspot_windows,
        peak_hotspot_cells=args.peak_hotspot_cells,
        peak_hotspot_top_cells=args.peak_hotspot_top_cells,
        road_network=road_network,
        seed=args.seed,
    )

    methods = ["park", "random", "diffusion"] if args.policy == "all" else [args.policy]
    summaries: list[dict[str, object]] = []
    write_grid(out_dir / "chengdu_grid.csv", base_env.grid)
    (out_dir / "chengdu_demand_stats.json").write_text(
        json.dumps(asdict(stats), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for idx, method in enumerate(methods):
        env = DispatchEnv(
            base_env.config,
            demand=base_env.demand,
            grid=base_env.grid,
            road_network=getattr(base_env, "road_network", None),
        )
        metrics, env = simulate_policy(env, method=method, seed=args.seed + idx * 10_000)
        row = {"policy": method, **asdict(metrics)}
        summaries.append(row)
        env.save_cell_cancellations(out_dir / f"{method}_cell_cancellations.csv")
        if args.plots:
            plot_cell_cancellations(env, out_dir / f"{method}_cell_cancellations.png")
        print(
            f"{method:9s} response={metrics.response_rate:.3f} "
            f"wait={metrics.response_time_seconds:.1f}s occupied={metrics.occupied_rate:.3f} "
            f"orders={metrics.orders} cancellations={metrics.cancellations}"
        )

    _write_summary(out_dir / "summary.csv", summaries)
    print(
        f"built Chengdu simulator from {stats.raw_files} file(s), "
        f"{stats.days} day(s), {stats.orders_per_episode:.0f} orders/episode; wrote {out_dir}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and run a Chengdu taxi dispatch simulator from raw order CSVs.")
    parser.add_argument(
        "--raw",
        nargs="+",
        default=["data/2016年11月成都滴滴订单数据"],
        help="Raw CSV file(s), directory, or glob pattern.",
    )
    parser.add_argument("--policy", choices=["all", "park", "random", "diffusion"], default="diffusion")
    parser.add_argument("--cells", type=int, default=142)
    parser.add_argument("--taxis", type=int, default=6_000)
    parser.add_argument("--horizon-steps", type=int, default=108)
    parser.add_argument("--step-minutes", type=int, default=10)
    parser.add_argument("--start-hour", type=float, default=6.0)
    parser.add_argument("--max-wait-steps", type=int, default=1)
    parser.add_argument("--demand-scale", type=float, default=1.8)
    parser.add_argument(
        "--same-step-reposition-service",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow repositioned taxis to become available before the current decision step ends.",
    )
    parser.add_argument(
        "--reposition-before-assignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply regional repositioning before the current step's minute-level assignment window.",
    )
    parser.add_argument("--relocation-cost-weight", type=float, default=0.5)
    parser.add_argument("--future-demand-steps", type=int, default=0)
    parser.add_argument("--region-value-weight", type=float, default=None)
    parser.add_argument("--future-value-weight", type=float, default=None, help="Deprecated alias for --region-value-weight.")
    parser.add_argument("--future-gap-weight", type=float, default=0.25)
    parser.add_argument("--dispatch-intensity-weight", type=float, default=0.25)
    parser.add_argument("--trip-time-weight", type=float, default=0.05)
    parser.add_argument("--service-revenue-weight", type=float, default=0.05)
    parser.add_argument("--fare-base", type=float, default=14.0)
    parser.add_argument("--fare-per-km", type=float, default=2.6)
    parser.add_argument("--fare-per-minute", type=float, default=0.5)
    parser.add_argument("--wait-time-penalty-weight", type=float, default=0.1)
    parser.add_argument("--wait-time-tolerance-minutes", type=float, default=5.0)
    parser.add_argument("--wait-time-penalty-exponent", type=float, default=1.0)
    parser.add_argument("--cancellation-penalty", type=float, default=3.0)
    parser.add_argument("--future-imbalance-weight", type=float, default=0.5)
    parser.add_argument("--idle-time-penalty-weight", type=float, default=0.01)
    parser.add_argument("--execution-mismatch-penalty", type=float, default=0.25)
    parser.add_argument("--bounds", default="", help="Optional lon_min,lat_min,lon_max,lat_max.")
    parser.add_argument("--bounds-quantiles", default="0.01,0.99", help="Robust bounds quantiles when --bounds is absent.")
    parser.add_argument("--bounds-samples", type=int, default=200_000)
    parser.add_argument("--grid-padding", type=float, default=0.08)
    parser.add_argument("--average-by-day", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-outside-bounds", action="store_true")
    parser.add_argument("--stochastic-demand", action="store_true")
    parser.add_argument("--peak-hotspot-multiplier", type=float, default=1.0)
    parser.add_argument("--peak-hotspot-windows", default=DEFAULT_PEAK_HOTSPOT_WINDOWS_TEXT)
    parser.add_argument("--peak-hotspot-cells", default="")
    parser.add_argument("--peak-hotspot-top-cells", type=int, default=10)
    parser.add_argument("--road-network", default="", help="Optional cached road graph GraphML path.")
    parser.add_argument("--road-nodes", default="", help="Optional road node CSV with node,lon,lat columns.")
    parser.add_argument("--road-edges", default="", help="Optional road edge CSV with u,v,length_m columns.")
    parser.add_argument("--download-road-network", action="store_true")
    parser.add_argument("--road-network-cache", default="data/processed/chengdu_road.graphml")
    parser.add_argument("--road-place", default="Chengdu, Sichuan, China")
    parser.add_argument("--road-network-type", default="drive")
    parser.add_argument("--travel-speed-kmph", dest="road_speed_kmph", type=float, default=30.0)
    parser.add_argument("--road-speed-kmph", dest="road_speed_kmph", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--require-road-costs", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/chengdu_sim")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    run(parse_args_with_config(build_parser))


def _column_indices(header: Sequence[str]) -> dict[str, int]:
    header_map = {name: idx for idx, name in enumerate(header)}
    if all(name in header_map for name in RAW_COLUMN_ORDER):
        return {name: header_map[name] for name in RAW_COLUMN_ORDER}
    if len(header) >= len(RAW_COLUMN_ORDER):
        return {name: idx for idx, name in enumerate(RAW_COLUMN_ORDER)}
    raise ValueError("raw order CSV must contain the Chengdu order columns")


def _dedupe_key(row: Sequence[str], indices: dict[str, int]) -> tuple[str, str]:
    return (row[indices[ORDER_ID]], row[indices[START_TIME]])


def _step_of_day(start: datetime, start_hour: float, horizon_steps: int, step_minutes: int) -> int | None:
    minute = _minute_of_horizon(start, start_hour=start_hour, horizon_steps=horizon_steps, step_minutes=step_minutes)
    if minute is None:
        return None
    return int(minute // step_minutes)


def _minute_of_horizon(start: datetime, start_hour: float, horizon_steps: int, step_minutes: int) -> int | None:
    minute = start.hour * 60.0 + start.minute + start.second / 60.0
    window_start = start_hour * 60.0
    rel = (minute - window_start) % 1440.0
    if rel >= horizon_steps * step_minutes:
        return None
    return int(rel)


def _valid_lonlat(lon: float, lat: float) -> bool:
    return np.isfinite(lon) and np.isfinite(lat) and 70.0 <= lon <= 140.0 and 15.0 <= lat <= 55.0


def _inside_bounds(lon: float, lat: float, bounds: tuple[float, float, float, float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bounds
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def _parse_bounds(value: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bounds must have four comma-separated floats")
    return (parts[0], parts[1], parts[2], parts[3])


def _parse_quantiles(value: str) -> tuple[float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--bounds-quantiles must have two comma-separated floats")
    return (parts[0], parts[1])


def _load_road_network_from_args(args: argparse.Namespace) -> RoadNetwork | None:
    if args.road_nodes or args.road_edges:
        if not args.road_nodes or not args.road_edges:
            raise ValueError("--road-nodes and --road-edges must be provided together")
        return RoadNetwork.from_csv(args.road_nodes, args.road_edges, default_speed_kmph=args.road_speed_kmph)
    if args.road_network:
        return RoadNetwork.from_graphml(args.road_network, default_speed_kmph=args.road_speed_kmph)
    if args.download_road_network:
        return RoadNetwork.load_or_download(
            cache_path=args.road_network_cache,
            place=args.road_place,
            network_type=args.road_network_type,
            default_speed_kmph=args.road_speed_kmph,
            download=True,
        )
    cache = Path(args.road_network_cache)
    if cache.exists():
        return RoadNetwork.from_graphml(cache, default_speed_kmph=args.road_speed_kmph)
    return None


def _region_value_weight_arg(args: argparse.Namespace) -> float:
    value = getattr(args, "region_value_weight", None)
    if value is not None:
        return float(value)
    legacy = getattr(args, "future_value_weight", None)
    if legacy is not None:
        return float(legacy)
    return 0.25


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
