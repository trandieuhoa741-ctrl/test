from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _largest_remainder_counts(rates: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(rates, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    counts = np.floor(values).astype(np.int64)
    target_total = int(np.rint(float(values.sum())))
    remainder_slots = max(target_total - int(counts.sum()), 0)
    if remainder_slots <= 0 or values.size == 0:
        return counts

    remainders = values - counts
    order = np.argsort(-remainders, kind="stable")
    counts[order[:remainder_slots]] += 1
    return counts


def _metadata_with_global_trip_mean(
    metadata: dict[str, object] | None,
    mean_trip_minutes: np.ndarray,
) -> dict[str, object]:
    meta = dict(metadata or {})
    current = meta.get("global_mean_trip_minutes")
    if current is not None:
        try:
            value = float(current)
        except (TypeError, ValueError):
            value = 0.0
        if np.isfinite(value) and value > 0.0:
            meta["global_mean_trip_minutes"] = value
            return meta

    values = np.asarray(mean_trip_minutes, dtype=np.float32)
    valid = values[np.isfinite(values) & (values > 0)]
    meta["global_mean_trip_minutes"] = float(valid.mean()) if valid.size else 15.0
    return meta


@dataclass
class TabularDemand:
    """Demand adapter for preprocessed real OD tables.

    Expected aggregate CSV columns by default:
    ``step,origin,destination,count``.
    """

    rates: np.ndarray
    od_probs: np.ndarray
    stochastic: bool = False

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        num_cells: int,
        horizon_steps: int,
        step_col: str = "step",
        origin_col: str = "origin",
        destination_col: str = "destination",
        count_col: str = "count",
        stochastic: bool = False,
    ) -> "TabularDemand":
        rates = np.zeros((horizon_steps, num_cells), dtype=np.float32)
        od_counts = np.zeros((horizon_steps, num_cells, num_cells), dtype=np.float32)

        with Path(path).open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row[step_col])
                origin = int(row[origin_col])
                destination = int(row[destination_col])
                count = float(row[count_col]) if count_col in row and row[count_col] != "" else 1.0
                if not (0 <= step < horizon_steps and 0 <= origin < num_cells and 0 <= destination < num_cells):
                    continue
                rates[step, origin] += count
                od_counts[step, origin, destination] += count

        od_probs = np.zeros_like(od_counts)
        uniform = np.full(num_cells, 1.0 / num_cells, dtype=np.float32)
        for step in range(horizon_steps):
            for origin in range(num_cells):
                total = float(od_counts[step, origin].sum())
                od_probs[step, origin] = od_counts[step, origin] / total if total > 0 else uniform
        return cls(rates=rates, od_probs=od_probs, stochastic=stochastic)

    def generate(self, step: int, rng: np.random.Generator) -> np.ndarray:
        if self.stochastic:
            return rng.poisson(self.rates[step]).astype(np.int64)
        return _largest_remainder_counts(self.rates[step])

    def sample_destinations(self, origin: int, step: int, n: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.asarray([], dtype=np.int64)
        return rng.choice(self.od_probs.shape[1], size=n, p=self.od_probs[step, origin]).astype(np.int64)


@dataclass(frozen=True)
class TripEvent:
    """One real-order event inside a simulator time step."""

    origin: int
    destination: int
    minute_offset: int
    trip_minutes: float
    pickup_xy: tuple[float, float] | None = None
    dropoff_xy: tuple[float, float] | None = None


@dataclass
class EventTripDemand:
    """Demand model backed by ordered per-day trip events.

    ``events_by_day`` has shape ``day -> step -> TripEvent``. At reset time the
    environment selects one empirical day, so repeated episodes preserve the
    one-minute order stream without collapsing all training days into one huge
    synthetic day.
    """

    rates: np.ndarray
    od_probs: np.ndarray
    mean_trip_minutes: np.ndarray
    events_by_day: tuple[tuple[tuple[TripEvent, ...], ...], ...]
    stochastic: bool = False
    metadata: dict[str, object] | None = None
    _day_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.metadata = _metadata_with_global_trip_mean(self.metadata, self.mean_trip_minutes)

    def reset_episode(self, rng: np.random.Generator, episode_index: int | None = None) -> None:
        if not self.events_by_day:
            self._day_index = 0
            return
        if episode_index is not None:
            self._day_index = int(episode_index) % len(self.events_by_day)
            return
        self._day_index = int(rng.integers(0, len(self.events_by_day)))

    @property
    def episode_count(self) -> int:
        return len(self.events_by_day)

    def generate(self, step: int, rng: np.random.Generator) -> np.ndarray:
        counts = np.zeros(self.rates.shape[1], dtype=np.int64)
        for event in self.sample_requests(step, rng):
            counts[event.origin] += 1
        return counts

    def sample_requests(self, step: int, rng: np.random.Generator) -> tuple[TripEvent, ...]:
        if not self.events_by_day:
            return ()
        day_events = self.events_by_day[self._day_index]
        if not (0 <= step < len(day_events)):
            return ()
        events = day_events[step]
        if not self.stochastic or not events:
            return events

        sampled: list[TripEvent] = []
        for event in events:
            sampled.extend([event] * int(rng.poisson(1.0)))
        return tuple(sampled)

    def sample_destinations(self, origin: int, step: int, n: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.asarray([], dtype=np.int64)
        return rng.choice(self.od_probs.shape[1], size=n, p=self.od_probs[step, origin]).astype(np.int64)

    def sample_trip_minutes(
        self,
        origin: int,
        step: int,
        destinations: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        dest = np.asarray(destinations, dtype=np.int64)
        if dest.size == 0:
            return np.asarray([], dtype=np.float32)

        values = self.mean_trip_minutes[step, origin, dest].astype(np.float32)
        fallback = float((self.metadata or {}).get("global_mean_trip_minutes", 15.0))
        mask = (~np.isfinite(values)) | (values <= 0)
        if mask.any():
            values[mask] = fallback
        if self.stochastic:
            shape = 8.0
            values = rng.gamma(shape=shape, scale=np.maximum(values, 1.0) / shape).astype(np.float32)
        return np.maximum(values, 1.0).astype(np.float32)


@dataclass
class EmpiricalTripDemand:
    """Demand model backed by empirical OD counts and trip durations."""

    rates: np.ndarray
    od_probs: np.ndarray
    mean_trip_minutes: np.ndarray
    stochastic: bool = False
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        self.metadata = _metadata_with_global_trip_mean(self.metadata, self.mean_trip_minutes)

    @classmethod
    def from_counts(
        cls,
        od_counts: np.ndarray,
        trip_minutes_sum: np.ndarray,
        trip_minutes_count: np.ndarray,
        stochastic: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> "EmpiricalTripDemand":
        if od_counts.ndim != 3:
            raise ValueError("od_counts must have shape (steps, origins, destinations)")
        if trip_minutes_sum.shape != od_counts.shape or trip_minutes_count.shape != od_counts.shape:
            raise ValueError("trip duration arrays must match od_counts")

        rates = od_counts.sum(axis=2).astype(np.float32)
        steps, num_cells, _ = od_counts.shape
        od_probs = np.zeros_like(od_counts, dtype=np.float32)
        global_dest = od_counts.sum(axis=(0, 1)).astype(np.float64)
        if float(global_dest.sum()) <= 0:
            global_dest_probs = np.full(num_cells, 1.0 / num_cells, dtype=np.float32)
        else:
            global_dest_probs = (global_dest / global_dest.sum()).astype(np.float32)

        for step in range(steps):
            for origin in range(num_cells):
                total = float(od_counts[step, origin].sum())
                od_probs[step, origin] = od_counts[step, origin] / total if total > 0 else global_dest_probs

        mean_trip_minutes = np.zeros_like(od_counts, dtype=np.float32)
        np.divide(
            trip_minutes_sum,
            trip_minutes_count,
            out=mean_trip_minutes,
            where=trip_minutes_count > 0,
        )

        global_mean = 15.0
        if float(trip_minutes_count.sum()) > 0:
            global_mean = float(trip_minutes_sum.sum() / trip_minutes_count.sum())
        meta = dict(metadata or {})
        meta.setdefault("global_mean_trip_minutes", global_mean)
        return cls(
            rates=rates,
            od_probs=od_probs,
            mean_trip_minutes=mean_trip_minutes,
            stochastic=stochastic,
            metadata=meta,
        )

    def generate(self, step: int, rng: np.random.Generator) -> np.ndarray:
        if self.stochastic:
            return rng.poisson(self.rates[step]).astype(np.int64)
        return _largest_remainder_counts(self.rates[step])

    def sample_destinations(self, origin: int, step: int, n: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.asarray([], dtype=np.int64)
        return rng.choice(self.od_probs.shape[1], size=n, p=self.od_probs[step, origin]).astype(np.int64)

    def sample_trip_minutes(
        self,
        origin: int,
        step: int,
        destinations: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        dest = np.asarray(destinations, dtype=np.int64)
        if dest.size == 0:
            return np.asarray([], dtype=np.float32)

        values = self.mean_trip_minutes[step, origin, dest].astype(np.float32)
        fallback = float((self.metadata or {}).get("global_mean_trip_minutes", 15.0))
        mask = (~np.isfinite(values)) | (values <= 0)
        if mask.any():
            values[mask] = fallback
        if self.stochastic:
            shape = 8.0
            values = rng.gamma(shape=shape, scale=np.maximum(values, 1.0) / shape).astype(np.float32)
        return np.maximum(values, 1.0).astype(np.float32)
