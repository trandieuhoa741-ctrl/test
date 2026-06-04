from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, radians, sqrt
from typing import Iterable

import numpy as np


DEFAULT_HEX_AVERAGE_SIDE_KM = 1.22
DEFAULT_HEX_CENTER_SPACING_KM = sqrt(3.0) * DEFAULT_HEX_AVERAGE_SIDE_KM

AXIAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)


@dataclass(frozen=True)
class HexGrid:
    """Irregular axial hex grid with action-ordered six-neighbor lookup.

    Action 0 is always "stay"; actions 1..6 follow ``AXIAL_DIRECTIONS``.
    Missing neighbors are represented with -1, preserving a compact seven-action
    interface for stay plus six neighboring movement directions.
    ``cell_width_km`` is the distance between adjacent hex centers; the regular
    hex side length is ``cell_width_km / sqrt(3)``.
    """

    coords: tuple[tuple[int, int], ...]
    neighbors: np.ndarray
    xy: np.ndarray
    cell_width_km: float
    lonlat: np.ndarray | None = None
    projection_origin: tuple[float, float] | None = None

    @classmethod
    def create(cls, num_cells: int = 49, cell_width_km: float = DEFAULT_HEX_CENTER_SPACING_KM) -> "HexGrid":
        if num_cells < 1:
            raise ValueError("num_cells must be positive")

        radius = 0
        while 1 + 3 * radius * (radius + 1) < num_cells:
            radius += 1

        coords = sorted(_axial_disk(radius), key=lambda c: (_hex_distance(c), c[0], c[1]))
        selected = tuple(coords[:num_cells])
        coord_to_idx = {coord: idx for idx, coord in enumerate(selected)}

        neighbors = np.full((num_cells, 7), -1, dtype=np.int64)
        for idx, (q, r) in enumerate(selected):
            neighbors[idx, 0] = idx
            for action, (dq, dr) in enumerate(AXIAL_DIRECTIONS, start=1):
                neighbors[idx, action] = coord_to_idx.get((q + dq, r + dr), -1)

        xy = np.asarray([_axial_to_xy(coord, cell_width_km) for coord in selected], dtype=np.float32)
        return cls(coords=selected, neighbors=neighbors, xy=xy, cell_width_km=float(cell_width_km))

    @classmethod
    def from_axial_coords(
        cls,
        coords: Iterable[tuple[int, int]],
        cell_width_km: float = DEFAULT_HEX_CENTER_SPACING_KM,
        xy_offset: tuple[float, float] = (0.0, 0.0),
        projection_origin: tuple[float, float] | None = None,
    ) -> "HexGrid":
        selected = tuple(sorted(dict.fromkeys(coords), key=lambda c: (_hex_distance(c), c[0], c[1])))
        if not selected:
            raise ValueError("coords must not be empty")

        coord_to_idx = {coord: idx for idx, coord in enumerate(selected)}
        neighbors = np.full((len(selected), 7), -1, dtype=np.int64)
        for idx, (q, r) in enumerate(selected):
            neighbors[idx, 0] = idx
            for action, (dq, dr) in enumerate(AXIAL_DIRECTIONS, start=1):
                neighbors[idx, action] = coord_to_idx.get((q + dq, r + dr), -1)

        offset = np.asarray(xy_offset, dtype=np.float32)
        xy = np.asarray([_axial_to_xy(coord, cell_width_km) for coord in selected], dtype=np.float32) + offset
        lonlat = None
        if projection_origin is not None:
            lon0, lat0 = projection_origin
            lonlat = xy_to_lonlat_array(xy, lon0, lat0).astype(np.float32)
        return cls(
            coords=selected,
            neighbors=neighbors,
            xy=xy.astype(np.float32),
            cell_width_km=float(cell_width_km),
            lonlat=lonlat,
            projection_origin=projection_origin,
        )

    @classmethod
    def create_geographic_fixed(
        cls,
        bounds: tuple[float, float, float, float],
        cell_width_km: float = DEFAULT_HEX_CENTER_SPACING_KM,
        padding_km: float = 2.5,
    ) -> "HexGrid":
        """Create a fixed-width hex overlay covering a lon/lat bounding box."""

        lon_min, lat_min, lon_max, lat_max = bounds
        if lon_min >= lon_max or lat_min >= lat_max:
            raise ValueError("bounds must be (lon_min, lat_min, lon_max, lat_max)")
        if cell_width_km <= 0:
            raise ValueError("cell_width_km must be positive")

        lon0 = (lon_min + lon_max) / 2.0
        lat0 = (lat_min + lat_max) / 2.0
        x_min, y_min = lonlat_to_xy(lon_min, lat_min, lon0, lat0)
        x_max, y_max = lonlat_to_xy(lon_max, lat_max, lon0, lat0)
        x_min = float(x_min) - padding_km
        x_max = float(x_max) + padding_km
        y_min = float(y_min) - padding_km
        y_max = float(y_max) + padding_km

        max_extent = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max), cell_width_km)
        radius = int(ceil(max_extent / cell_width_km * 2.5)) + 4
        coords: list[tuple[int, int]] = []
        for coord in _axial_disk(radius):
            x, y = _axial_to_xy(coord, cell_width_km)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                coords.append(coord)
        return cls.from_axial_coords(
            coords,
            cell_width_km=cell_width_km,
            projection_origin=(float(lon0), float(lat0)),
        )

    @classmethod
    def create_geographic(
        cls,
        num_cells: int,
        bounds: tuple[float, float, float, float],
        padding: float = 0.08,
    ) -> "HexGrid":
        """Create a hex grid whose centers cover a lon/lat bounding box.

        ``bounds`` is ``(lon_min, lat_min, lon_max, lat_max)``. Coordinates
        are projected to local kilometers with an equirectangular projection,
        which is accurate enough for city-scale dispatch simulation.
        """

        lon_min, lat_min, lon_max, lat_max = bounds
        if lon_min >= lon_max or lat_min >= lat_max:
            raise ValueError("bounds must be (lon_min, lat_min, lon_max, lat_max)")

        lon0 = (lon_min + lon_max) / 2.0
        lat0 = (lat_min + lat_max) / 2.0
        x_min, y_min = lonlat_to_xy(lon_min, lat_min, lon0, lat0)
        x_max, y_max = lonlat_to_xy(lon_max, lat_max, lon0, lat0)
        target_width = max(abs(float(x_max - x_min)), 0.5) * (1.0 + 2.0 * padding)
        target_height = max(abs(float(y_max - y_min)), 0.5) * (1.0 + 2.0 * padding)

        unit_grid = cls.create(num_cells=num_cells, cell_width_km=1.0)
        span_x = max(float(np.ptp(unit_grid.xy[:, 0])), 1.0)
        span_y = max(float(np.ptp(unit_grid.xy[:, 1])), 1.0)
        cell_width_km = max(target_width / span_x, target_height / span_y)

        grid = cls.create(num_cells=num_cells, cell_width_km=cell_width_km)
        xy = grid.xy.copy()
        target_center = np.asarray([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=np.float32)
        xy += target_center - xy.mean(axis=0)
        lonlat = xy_to_lonlat_array(xy, lon0, lat0).astype(np.float32)
        return cls(
            coords=grid.coords,
            neighbors=grid.neighbors,
            xy=xy.astype(np.float32),
            cell_width_km=float(cell_width_km),
            lonlat=lonlat,
            projection_origin=(float(lon0), float(lat0)),
        )

    @property
    def num_cells(self) -> int:
        return len(self.coords)

    @property
    def average_side_length_km(self) -> float:
        return float(self.cell_width_km) / sqrt(3.0)

    def valid_actions(self, cell: int) -> list[int]:
        return [action for action, dst in enumerate(self.neighbors[cell]) if dst >= 0]

    def local_cells(self, cell: int) -> list[int]:
        return [int(dst) for dst in self.neighbors[cell] if dst >= 0]

    def distance(self, a: int, b: int) -> float:
        return float(np.linalg.norm(self.xy[a] - self.xy[b]))

    def project_lonlat(self, lon: np.ndarray | float, lat: np.ndarray | float) -> np.ndarray:
        if self.projection_origin is None:
            raise ValueError("this grid was not created with geographic bounds")
        lon0, lat0 = self.projection_origin
        x, y = lonlat_to_xy(lon, lat, lon0, lat0)
        return np.stack([x, y], axis=-1).astype(np.float32)

    def nearest_cells_xy(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points_xy must have shape (n, 2)")
        diff = points[:, None, :] - self.xy[None, :, :]
        return np.argmin(np.sum(diff * diff, axis=2), axis=1).astype(np.int64)

    def nearest_cells_lonlat(self, lon: np.ndarray | float, lat: np.ndarray | float) -> np.ndarray:
        return self.nearest_cells_xy(self.project_lonlat(lon, lat))

    def iter_edges(self) -> Iterable[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        for i in range(self.num_cells):
            for j in self.neighbors[i, 1:]:
                if j < 0:
                    continue
                edge = tuple(sorted((i, int(j))))
                if edge not in seen:
                    seen.add(edge)
                    yield edge


def _axial_disk(radius: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for q in range(-radius, radius + 1):
        r_min = max(-radius, -q - radius)
        r_max = min(radius, -q + radius)
        for r in range(r_min, r_max + 1):
            coords.append((q, r))
    return coords


def _hex_distance(coord: tuple[int, int]) -> int:
    q, r = coord
    return max(abs(q), abs(r), abs(-q - r))


def _axial_to_xy(coord: tuple[int, int], width: float) -> tuple[float, float]:
    q, r = coord
    return width * (q + 0.5 * r), width * (sqrt(3.0) * 0.5 * r)


def lonlat_to_xy(
    lon: np.ndarray | float,
    lat: np.ndarray | float,
    origin_lon: float,
    origin_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    km_per_lon = 111.320 * cos(radians(origin_lat))
    x = (lon_arr - origin_lon) * km_per_lon
    y = (lat_arr - origin_lat) * 110.574
    return x, y


def xy_to_lonlat_array(points_xy: np.ndarray, origin_lon: float, origin_lat: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    km_per_lon = 111.320 * cos(radians(origin_lat))
    lon = points[:, 0] / km_per_lon + origin_lon
    lat = points[:, 1] / 110.574 + origin_lat
    return np.stack([lon, lat], axis=1)
