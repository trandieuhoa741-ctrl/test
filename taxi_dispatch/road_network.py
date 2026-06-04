from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np

from .grid import HexGrid, lonlat_to_xy, xy_to_lonlat_array

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised only when scipy is absent
    cKDTree = None


@dataclass(frozen=True)
class RoadNode:
    node_id: str
    lon: float
    lat: float
    x: float
    y: float


@dataclass(frozen=True)
class RoadCostMatrices:
    """Cell-to-cell road costs precomputed on a fixed selected grid."""

    distance_km: np.ndarray
    time_minutes: np.ndarray
    cell_nodes: tuple[str, ...] = ()
    fallback_distance_count: int = 0
    fallback_time_count: int = 0

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            distance_km=np.asarray(self.distance_km, dtype=np.float32),
            time_minutes=np.asarray(self.time_minutes, dtype=np.float32),
            cell_nodes=np.asarray(self.cell_nodes, dtype=str),
            fallback_distance_count=np.asarray(self.fallback_distance_count, dtype=np.int64),
            fallback_time_count=np.asarray(self.fallback_time_count, dtype=np.int64),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "RoadCostMatrices":
        with np.load(path, allow_pickle=False) as data:
            distance_km = np.asarray(data["distance_km"], dtype=np.float32)
            time_minutes = np.asarray(data["time_minutes"], dtype=np.float32)
            cell_nodes = tuple(str(node) for node in data["cell_nodes"]) if "cell_nodes" in data.files else ()
            fallback_distance_count = (
                int(np.asarray(data["fallback_distance_count"]).item()) if "fallback_distance_count" in data.files else 0
            )
            fallback_time_count = (
                int(np.asarray(data["fallback_time_count"]).item()) if "fallback_time_count" in data.files else 0
            )
        return cls(
            distance_km=distance_km,
            time_minutes=time_minutes,
            cell_nodes=cell_nodes,
            fallback_distance_count=fallback_distance_count,
            fallback_time_count=fallback_time_count,
        )


class RoadNetwork:
    """Road graph wrapper used by the lower routing layer.

    Edge attributes are normalized to ``length_km`` and ``travel_time_min``.
    GraphML exported by OSMnx is supported when ``networkx`` can read it.
    Downloading from OSM requires the optional ``osmnx`` package.
    """

    def __init__(self, graph: nx.MultiDiGraph | nx.DiGraph, default_speed_kmph: float = 30.0) -> None:
        if graph.number_of_nodes() == 0:
            raise ValueError("road graph must contain at least one node")
        self.graph = nx.relabel_nodes(nx.MultiDiGraph(graph), lambda node: str(node), copy=True)
        self.default_speed_kmph = float(default_speed_kmph)
        self._normalize_graph()
        self.nodes = self._read_nodes()
        self.node_ids = [node.node_id for node in self.nodes]
        self.node_xy = np.asarray([(node.x, node.y) for node in self.nodes], dtype=np.float32)
        self._node_kdtree = cKDTree(self.node_xy) if cKDTree is not None else None

    @classmethod
    def from_csv(
        cls,
        nodes_path: str | Path,
        edges_path: str | Path,
        directed: bool = True,
        default_speed_kmph: float = 30.0,
    ) -> "RoadNetwork":
        graph: nx.MultiDiGraph = nx.MultiDiGraph()
        with Path(nodes_path).open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = str(row.get("node") or row.get("id") or row.get("node_id"))
                lon = float(row.get("lon") or row.get("x") or row.get("longitude"))
                lat = float(row.get("lat") or row.get("y") or row.get("latitude"))
                graph.add_node(node_id, x=lon, y=lat)

        with Path(edges_path).open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u = str(row.get("u") or row.get("source") or row.get("from"))
                v = str(row.get("v") or row.get("target") or row.get("to"))
                length_m = float(row.get("length_m") or row.get("length") or row.get("distance_m") or 0.0)
                speed = float(row.get("speed_kmph") or row.get("speed") or default_speed_kmph)
                attrs = {"length": length_m, "speed_kph": speed}
                graph.add_edge(u, v, **attrs)
                if not directed:
                    graph.add_edge(v, u, **attrs)
        return cls(graph, default_speed_kmph=default_speed_kmph)

    @classmethod
    def from_graphml(cls, path: str | Path, default_speed_kmph: float = 30.0) -> "RoadNetwork":
        graph = nx.read_graphml(path)
        return cls(graph, default_speed_kmph=default_speed_kmph)

    @classmethod
    def download_osm(
        cls,
        place: str = "Chengdu, Sichuan, China",
        network_type: str = "drive",
        cache_path: str | Path | None = None,
        default_speed_kmph: float = 30.0,
    ) -> "RoadNetwork":
        try:
            import osmnx as ox
        except ImportError as exc:
            raise ImportError("OSM download requires the optional osmnx package: python -m pip install osmnx") from exc

        graph = ox.graph_from_place(place, network_type=network_type, simplify=True)
        graph = ox.add_edge_speeds(graph, fallback=default_speed_kmph)
        graph = ox.add_edge_travel_times(graph)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(graph, filepath=str(cache_path))
        return cls(graph, default_speed_kmph=default_speed_kmph)

    @classmethod
    def load_or_download(
        cls,
        cache_path: str | Path,
        place: str = "Chengdu, Sichuan, China",
        network_type: str = "drive",
        default_speed_kmph: float = 30.0,
        download: bool = False,
    ) -> "RoadNetwork":
        path = Path(cache_path)
        if path.exists():
            return cls.from_graphml(path, default_speed_kmph=default_speed_kmph)
        if not download:
            raise FileNotFoundError(f"road network cache not found: {path}")
        return cls.download_osm(place=place, network_type=network_type, cache_path=path, default_speed_kmph=default_speed_kmph)

    def nearest_node(self, lon: float, lat: float) -> str:
        return self.nearest_nodes(np.asarray([lon], dtype=np.float64), np.asarray([lat], dtype=np.float64))[0]

    def nearest_nodes(self, lon: np.ndarray, lat: np.ndarray) -> list[str]:
        x, y = lonlat_to_xy(lon, lat, self.origin_lon, self.origin_lat)
        points = np.stack([x, y], axis=1).astype(np.float32)
        if points.size == 0:
            return []
        if self._node_kdtree is not None:
            _distance, idx = self._node_kdtree.query(points)
            idx = np.asarray(idx, dtype=np.int64)
        else:
            idx = self._nearest_node_indices_no_tree(points)
        return [self.node_ids[int(i)] for i in idx]

    def nearest_grid_xy(self, points_xy: np.ndarray, projection_origin: tuple[float, float]) -> list[str]:
        lonlat = xy_to_lonlat_array(points_xy, projection_origin[0], projection_origin[1])
        return self.nearest_nodes(lonlat[:, 0], lonlat[:, 1])

    def shortest_path_distance_km(self, source: str, target: str) -> float:
        return self._shortest_path(source, target, "length_km")

    def shortest_path_time_minutes(self, source: str, target: str) -> float:
        return self._shortest_path(source, target, "travel_time_min")

    def distance_matrix_km(self, sources: Iterable[str], targets: Iterable[str]) -> np.ndarray:
        return self._matrix(sources, targets, "length_km")

    def time_matrix_minutes(self, sources: Iterable[str], targets: Iterable[str]) -> np.ndarray:
        return self._matrix(sources, targets, "travel_time_min")

    def cell_cost_matrices(
        self,
        grid: HexGrid,
        *,
        fallback_speed_kmph: float | None = None,
    ) -> RoadCostMatrices:
        """Precompute cell-center shortest-path distance and time matrices.

        The matrices are keyed by the already-selected grid order, so train and
        test data can share one road-cost cache without remapping.
        """

        if grid.projection_origin is None:
            raise ValueError("road cell matrices require a geographic grid with projection_origin")

        cell_nodes = tuple(self.nearest_grid_xy(grid.xy, grid.projection_origin))
        distance_km = self.distance_matrix_km(cell_nodes, cell_nodes)
        time_minutes = self.time_matrix_minutes(cell_nodes, cell_nodes)

        fallback_distance = np.linalg.norm(
            np.asarray(grid.xy, dtype=np.float32)[:, None, :]
            - np.asarray(grid.xy, dtype=np.float32)[None, :, :],
            axis=2,
        ).astype(np.float32)
        speed = float(fallback_speed_kmph if fallback_speed_kmph is not None else self.default_speed_kmph)
        fallback_time = fallback_distance / max(speed, 1e-6) * 60.0

        fallback_distance_count = _bad_matrix_count(distance_km)
        fallback_time_count = _bad_matrix_count(time_minutes)
        distance_km = _fill_bad_matrix_values(distance_km, fallback_distance)
        time_minutes = _fill_bad_matrix_values(time_minutes, fallback_time)
        np.fill_diagonal(distance_km, 0.0)
        np.fill_diagonal(time_minutes, 0.0)
        return RoadCostMatrices(
            distance_km=distance_km.astype(np.float32),
            time_minutes=time_minutes.astype(np.float32),
            cell_nodes=cell_nodes,
            fallback_distance_count=fallback_distance_count,
            fallback_time_count=fallback_time_count,
        )

    def _nearest_node_indices_no_tree(self, points: np.ndarray) -> np.ndarray:
        indices = np.empty(points.shape[0], dtype=np.int64)
        for i, point in enumerate(points):
            diff = self.node_xy - point[None, :]
            dist2 = np.einsum("ij,ij->i", diff, diff)
            indices[i] = int(np.argmin(dist2))
        return indices

    def _matrix(self, sources: Iterable[str], targets: Iterable[str], weight: str) -> np.ndarray:
        source_list = [str(source) for source in sources]
        target_list = [str(target) for target in targets]
        values = np.full((len(source_list), len(target_list)), np.inf, dtype=np.float32)
        for i, source in enumerate(source_list):
            lengths = nx.single_source_dijkstra_path_length(self.graph, source, weight=weight)
            for j, target in enumerate(target_list):
                if target in lengths:
                    values[i, j] = float(lengths[target])
        return values

    @lru_cache(maxsize=200_000)
    def _shortest_path(self, source: str, target: str, weight: str) -> float:
        try:
            return float(nx.shortest_path_length(self.graph, source, target, weight=weight))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    def _read_nodes(self) -> list[RoadNode]:
        lons: list[float] = []
        lats: list[float] = []
        raw_nodes: list[tuple[str, float, float]] = []
        for node_id, attrs in self.graph.nodes(data=True):
            lon = _float_attr(attrs, ("x", "lon", "longitude"))
            lat = _float_attr(attrs, ("y", "lat", "latitude"))
            if lon is None or lat is None:
                continue
            node_key = str(node_id)
            raw_nodes.append((node_key, lon, lat))
            lons.append(lon)
            lats.append(lat)
        if not raw_nodes:
            raise ValueError("road graph nodes must include lon/lat attributes")

        self.origin_lon = float(np.mean(lons))
        self.origin_lat = float(np.mean(lats))
        nodes: list[RoadNode] = []
        for node_id, lon, lat in raw_nodes:
            x, y = lonlat_to_xy(lon, lat, self.origin_lon, self.origin_lat)
            nodes.append(RoadNode(node_id=node_id, lon=lon, lat=lat, x=float(x), y=float(y)))
        return nodes

    def _normalize_graph(self) -> None:
        for _u, _v, _key, attrs in self.graph.edges(keys=True, data=True):
            length_m = _float_attr(attrs, ("length", "length_m", "distance_m"))
            if length_m is None or length_m <= 0:
                length_m = 1.0
            speed_kmph = _float_attr(attrs, ("speed_kph", "speed_kmph", "speed"))
            if speed_kmph is None or speed_kmph <= 0:
                speed_kmph = self.default_speed_kmph
            travel_time_seconds = _float_attr(attrs, ("travel_time", "travel_time_s", "time_s"))
            length_km = float(length_m) / 1000.0
            attrs["length_km"] = length_km
            attrs["travel_time_min"] = (
                float(travel_time_seconds) / 60.0
                if travel_time_seconds is not None and travel_time_seconds > 0
                else length_km / max(speed_kmph, 1e-6) * 60.0
            )


def _float_attr(attrs: dict[str, object], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name not in attrs:
            continue
        value = attrs[name]
        if isinstance(value, list):
            value = value[0] if value else None
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _fill_bad_matrix_values(matrix: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32).copy()
    fallback_values = np.asarray(fallback, dtype=np.float32)
    bad = (~np.isfinite(values)) | (values < 0)
    values[bad] = fallback_values[bad]
    return values


def _bad_matrix_count(matrix: np.ndarray) -> int:
    values = np.asarray(matrix, dtype=np.float32)
    return int(((~np.isfinite(values)) | (values < 0)).sum())


def load_road_cost_matrices(path: str | Path) -> RoadCostMatrices:
    return RoadCostMatrices.from_npz(path)
