from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable

import numpy as np


@dataclass
class RouteAssignment:
    taxi_id: int
    target_cell: int
    cost: float


@dataclass
class OrderMatch:
    order_index: int
    taxi_id: int
    pickup_distance: float


SparseCandidateEdge = tuple[int, int, float] | tuple[int, int, float, float]
FilteredSparseEdge = tuple[int, int, float, float]


@dataclass
class _Edge:
    to: int
    rev: int
    cap: int
    cost: int


class _MinCostFlow:
    def __init__(self, n: int) -> None:
        self.graph: list[list[_Edge]] = [[] for _ in range(n)]

    def add_edge(self, src: int, dst: int, cap: int, cost: int) -> _Edge:
        fwd = _Edge(dst, len(self.graph[dst]), cap, cost)
        rev = _Edge(src, len(self.graph[src]), 0, -cost)
        self.graph[src].append(fwd)
        self.graph[dst].append(rev)
        return fwd

    def solve(self, source: int, sink: int, max_flow: int) -> tuple[int, int]:
        n = len(self.graph)
        flow = 0
        cost = 0
        potential = [0] * n

        while flow < max_flow:
            dist = [10**18] * n
            prev_v = [-1] * n
            prev_e = [-1] * n
            dist[source] = 0
            heap: list[tuple[int, int]] = [(0, source)]

            while heap:
                cur_dist, v = heappop(heap)
                if cur_dist != dist[v]:
                    continue
                for edge_idx, edge in enumerate(self.graph[v]):
                    if edge.cap <= 0:
                        continue
                    nd = cur_dist + edge.cost + potential[v] - potential[edge.to]
                    if nd < dist[edge.to]:
                        dist[edge.to] = nd
                        prev_v[edge.to] = v
                        prev_e[edge.to] = edge_idx
                        heappush(heap, (nd, edge.to))

            if prev_v[sink] == -1:
                break

            for v in range(n):
                if dist[v] < 10**18:
                    potential[v] += dist[v]

            add = max_flow - flow
            v = sink
            while v != source:
                edge = self.graph[prev_v[v]][prev_e[v]]
                add = min(add, edge.cap)
                v = prev_v[v]

            v = sink
            while v != source:
                edge = self.graph[prev_v[v]][prev_e[v]]
                edge.cap -= add
                self.graph[v][edge.rev].cap += add
                cost += add * edge.cost
                v = prev_v[v]
            flow += add

        return flow, cost


def assign_taxis_min_cost(
    taxi_ids: Iterable[int],
    taxi_xy: np.ndarray,
    target_quotas: dict[int, int],
    target_xy: np.ndarray,
    cost_scale: int = 1000,
) -> list[RouteAssignment]:
    """Assign unit-capacity taxis to target cell quotas with min-cost max-flow."""

    taxi_ids = list(taxi_ids)
    targets = [(cell, quota) for cell, quota in target_quotas.items() if quota > 0]
    total_quota = sum(quota for _, quota in targets)
    if not taxi_ids or not targets or total_quota <= 0:
        return []

    max_flow = min(len(taxi_ids), total_quota)
    n_taxis = len(taxi_ids)
    n_targets = len(targets)
    source = n_taxis + n_targets
    sink = source + 1
    flow_graph = _MinCostFlow(sink + 1)

    for i in range(n_taxis):
        flow_graph.add_edge(source, i, 1, 0)

    tracked_edges: dict[tuple[int, int], _Edge] = {}
    for i, taxi_id in enumerate(taxi_ids):
        for j, (cell, _) in enumerate(targets):
            dist = float(np.linalg.norm(taxi_xy[taxi_id] - target_xy[cell]))
            edge = flow_graph.add_edge(i, n_taxis + j, 1, int(round(dist * cost_scale)))
            tracked_edges[(i, j)] = edge

    for j, (_, quota) in enumerate(targets):
        flow_graph.add_edge(n_taxis + j, sink, quota, 0)

    flow, _ = flow_graph.solve(source, sink, max_flow)
    if flow == 0:
        return []

    assignments: list[RouteAssignment] = []
    for (i, j), edge in tracked_edges.items():
        if edge.cap == 0:
            taxi_id = taxi_ids[i]
            target_cell = targets[j][0]
            cost = float(np.linalg.norm(taxi_xy[taxi_id] - target_xy[target_cell]))
            assignments.append(RouteAssignment(taxi_id=taxi_id, target_cell=target_cell, cost=cost))
    return assignments


def assign_taxis_min_cost_from_costs(
    taxi_ids: Iterable[int],
    target_quotas: dict[int, int],
    cost_matrix: np.ndarray,
    cost_scale: int = 1000,
) -> list[RouteAssignment]:
    """Assign taxis to target quotas from a taxi-by-target-cell cost matrix."""

    taxi_ids = list(taxi_ids)
    targets = [(cell, quota) for cell, quota in target_quotas.items() if quota > 0]
    total_quota = sum(quota for _, quota in targets)
    if not taxi_ids or not targets or total_quota <= 0:
        return []

    costs = np.asarray(cost_matrix, dtype=np.float32)
    if costs.shape != (len(taxi_ids), len(targets)):
        raise ValueError(f"cost_matrix must have shape {(len(taxi_ids), len(targets))}")

    max_flow = min(len(taxi_ids), total_quota)
    n_taxis = len(taxi_ids)
    n_targets = len(targets)
    source = n_taxis + n_targets
    sink = source + 1
    flow_graph = _MinCostFlow(sink + 1)

    for i in range(n_taxis):
        flow_graph.add_edge(source, i, 1, 0)

    tracked_edges: dict[tuple[int, int], _Edge] = {}
    for i in range(n_taxis):
        for j in range(n_targets):
            cost = float(costs[i, j])
            if not np.isfinite(cost):
                continue
            edge = flow_graph.add_edge(i, n_taxis + j, 1, int(round(cost * cost_scale)))
            tracked_edges[(i, j)] = edge

    for j, (_, quota) in enumerate(targets):
        flow_graph.add_edge(n_taxis + j, sink, quota, 0)

    flow, _ = flow_graph.solve(source, sink, max_flow)
    if flow == 0:
        return []

    assignments: list[RouteAssignment] = []
    for (i, j), edge in tracked_edges.items():
        if edge.cap == 0:
            assignments.append(
                RouteAssignment(
                    taxi_id=taxi_ids[i],
                    target_cell=targets[j][0],
                    cost=float(costs[i, j]),
                )
            )
    return assignments


def match_orders_min_cost(
    order_xy: np.ndarray,
    taxi_ids: Iterable[int],
    taxi_xy: np.ndarray,
    max_pickup_distance_km: float = 2.5,
    cost_matrix: np.ndarray | None = None,
    cost_scale: int = 1000,
) -> list[OrderMatch]:
    """Max-cardinality, min-pickup-distance order/taxi matching."""

    orders = np.asarray(order_xy, dtype=np.float32)
    taxi_ids = list(taxi_ids)
    if orders.size == 0 or not taxi_ids:
        return []
    if orders.ndim != 2 or orders.shape[1] != 2:
        raise ValueError("order_xy must have shape (n, 2)")

    n_orders = orders.shape[0]
    n_taxis = len(taxi_ids)
    source = n_orders + n_taxis
    sink = source + 1
    flow_graph = _MinCostFlow(sink + 1)

    for i in range(n_orders):
        flow_graph.add_edge(source, i, 1, 0)

    tracked_edges: dict[tuple[int, int], _Edge] = {}
    max_dist = float(max_pickup_distance_km)
    costs = None if cost_matrix is None else np.asarray(cost_matrix, dtype=np.float32)
    if costs is not None and costs.shape != (n_orders, n_taxis):
        raise ValueError(f"cost_matrix must have shape {(n_orders, n_taxis)}")

    for i in range(n_orders):
        for j, taxi_id in enumerate(taxi_ids):
            dist = float(costs[i, j]) if costs is not None else float(np.linalg.norm(orders[i] - taxi_xy[taxi_id]))
            if dist > max_dist + 1e-6:
                continue
            if not np.isfinite(dist):
                continue
            edge = flow_graph.add_edge(i, n_orders + j, 1, int(round(dist * cost_scale)))
            tracked_edges[(i, j)] = edge

    for j in range(n_taxis):
        flow_graph.add_edge(n_orders + j, sink, 1, 0)

    max_flow = min(n_orders, n_taxis)
    flow, _ = flow_graph.solve(source, sink, max_flow)
    if flow == 0:
        return []

    matches: list[OrderMatch] = []
    for (order_index, taxi_index), edge in tracked_edges.items():
        if edge.cap == 0:
            taxi_id = taxi_ids[taxi_index]
            pickup_distance = (
                float(costs[order_index, taxi_index])
                if costs is not None
                else float(np.linalg.norm(orders[order_index] - taxi_xy[taxi_id]))
            )
            matches.append(OrderMatch(order_index=order_index, taxi_id=taxi_id, pickup_distance=pickup_distance))
    return matches


def match_orders_min_cost_sparse(
    n_orders: int,
    taxi_ids: Iterable[int],
    candidate_edges: Iterable[SparseCandidateEdge],
    max_pickup_distance_km: float = 2.5,
    cost_scale: int = 1000,
) -> list[OrderMatch]:
    """Max-cardinality, min-cost matching from sparse order/taxi candidate edges.

    ``candidate_edges`` contains either ``(order_index, taxi_id,
    pickup_distance)`` rows or ``(order_index, taxi_id, pickup_distance,
    match_cost)`` rows. The optional fourth value lets callers optimize a
    shaped matching cost while preserving true pickup distance for filtering
    and service accounting.
    """

    taxi_ids = list(taxi_ids)
    if n_orders <= 0 or not taxi_ids:
        return []

    taxi_to_idx = {taxi_id: idx for idx, taxi_id in enumerate(taxi_ids)}
    max_dist = float(max_pickup_distance_km)
    filtered_edges = _filter_sparse_candidate_edges(
        n_orders=n_orders,
        taxi_to_idx=taxi_to_idx,
        candidate_edges=candidate_edges,
        max_dist=max_dist,
    )
    if not filtered_edges:
        return []

    max_real_cost = max(edge[3] for edge in filtered_edges)
    invalid_cost = (max(max_dist, max_real_cost, 1.0) + 1.0) * max(n_orders, len(taxi_ids), 1) + 1.0
    hungarian_matches = _match_orders_sparse_hungarian(
        n_orders=n_orders,
        taxi_ids=taxi_ids,
        taxi_to_idx=taxi_to_idx,
        filtered_edges=filtered_edges,
        invalid_cost=invalid_cost,
    )
    if hungarian_matches is not None:
        return hungarian_matches

    return _match_orders_sparse_min_cost_flow(
        n_orders=n_orders,
        taxi_ids=taxi_ids,
        filtered_edges=filtered_edges,
        cost_scale=cost_scale,
    )


def match_orders_greedy_sparse(
    n_orders: int,
    taxi_ids: Iterable[int],
    candidate_edges: Iterable[SparseCandidateEdge],
    max_pickup_distance_km: float = 2.5,
) -> list[OrderMatch]:
    """Greedy order/taxi matching over sparse pickup candidates.

    Orders are considered in their waiting-list order. Each order takes the
    currently available taxi with the lowest edge cost, without the global
    reassignment used by max-cardinality min-cost matching.
    """

    taxi_ids = list(taxi_ids)
    if n_orders <= 0 or not taxi_ids:
        return []

    taxi_to_idx = {taxi_id: idx for idx, taxi_id in enumerate(taxi_ids)}
    filtered_edges = _filter_sparse_candidate_edges(
        n_orders=n_orders,
        taxi_to_idx=taxi_to_idx,
        candidate_edges=candidate_edges,
        max_dist=float(max_pickup_distance_km),
    )
    if not filtered_edges:
        return []

    by_order: list[list[tuple[float, float, int]]] = [[] for _ in range(n_orders)]
    for order_index, taxi_index, pickup_distance, match_cost in filtered_edges:
        by_order[order_index].append((float(match_cost), float(pickup_distance), int(taxi_index)))

    used_taxis: set[int] = set()
    matches: list[OrderMatch] = []
    for order_index, edges in enumerate(by_order):
        for _match_cost, pickup_distance, taxi_index in sorted(edges):
            if taxi_index in used_taxis:
                continue
            used_taxis.add(taxi_index)
            matches.append(
                OrderMatch(
                    order_index=order_index,
                    taxi_id=taxi_ids[taxi_index],
                    pickup_distance=pickup_distance,
                )
            )
            break
    return matches


def _filter_sparse_candidate_edges(
    *,
    n_orders: int,
    taxi_to_idx: dict[int, int],
    candidate_edges: Iterable[SparseCandidateEdge],
    max_dist: float,
) -> list[FilteredSparseEdge]:
    best: dict[tuple[int, int], tuple[float, float]] = {}
    min_match_cost = 0.0
    for edge in candidate_edges:
        if len(edge) not in (3, 4):
            raise ValueError("candidate_edges rows must have 3 or 4 values")
        order_index = int(edge[0])
        if not (0 <= order_index < n_orders):
            continue
        taxi_index = taxi_to_idx.get(int(edge[1]))
        if taxi_index is None:
            continue
        pickup_distance = float(edge[2])
        if not np.isfinite(pickup_distance) or pickup_distance > max_dist + 1e-6:
            continue
        match_cost = float(edge[3]) if len(edge) == 4 else pickup_distance
        if not np.isfinite(match_cost):
            continue
        key = (order_index, taxi_index)
        previous = best.get(key)
        if previous is None or match_cost < previous[1]:
            best[key] = (pickup_distance, match_cost)
            min_match_cost = min(min_match_cost, match_cost)

    shift = -min_match_cost if min_match_cost < 0.0 else 0.0
    return [
        (order_index, taxi_index, pickup_distance, match_cost + shift)
        for (order_index, taxi_index), (pickup_distance, match_cost) in best.items()
    ]


def _match_orders_sparse_hungarian(
    *,
    n_orders: int,
    taxi_ids: list[int],
    taxi_to_idx: dict[int, int],
    filtered_edges: list[FilteredSparseEdge],
    invalid_cost: float,
) -> list[OrderMatch] | None:
    try:
        from scipy.sparse import csr_matrix  # type: ignore
        from scipy.sparse.csgraph import min_weight_full_bipartite_matching  # type: ignore
    except Exception:
        return None

    n_taxis = len(taxi_ids)
    epsilon = 1e-6
    if not filtered_edges:
        return []

    rows_arr = np.asarray([edge[0] for edge in filtered_edges], dtype=np.int64)
    cols_arr = np.asarray([edge[1] for edge in filtered_edges], dtype=np.int64)
    data_arr = np.asarray([edge[3] for edge in filtered_edges], dtype=np.float64) + epsilon
    pickup_lookup = {(edge[0], edge[1]): edge[2] for edge in filtered_edges}

    dummy_rows = np.arange(n_orders, dtype=np.int64)
    rows_arr = np.concatenate([rows_arr, dummy_rows])
    cols_arr = np.concatenate([cols_arr, n_taxis + dummy_rows])
    data_arr = np.concatenate([data_arr, np.full(n_orders, float(invalid_cost), dtype=np.float64)])

    costs = csr_matrix((data_arr, (rows_arr, cols_arr)), shape=(n_orders, n_taxis + n_orders), dtype=np.float64)
    try:
        row_indices, col_indices = min_weight_full_bipartite_matching(costs)
    except Exception:
        return None

    matches: list[OrderMatch] = []
    for order_index, taxi_index in zip(row_indices, col_indices):
        if taxi_index >= n_taxis:
            continue
        match_cost = float(costs[int(order_index), int(taxi_index)]) - epsilon
        if match_cost < 0 or match_cost >= invalid_cost - epsilon:
            continue
        pickup_distance = float(pickup_lookup[(int(order_index), int(taxi_index))])
        matches.append(
            OrderMatch(
                order_index=int(order_index),
                taxi_id=taxi_ids[int(taxi_index)],
                pickup_distance=pickup_distance,
            )
        )
    return matches


def _match_orders_sparse_min_cost_flow(
    *,
    n_orders: int,
    taxi_ids: list[int],
    filtered_edges: list[FilteredSparseEdge],
    cost_scale: int,
) -> list[OrderMatch]:
    n_taxis = len(taxi_ids)
    source = n_orders + n_taxis
    sink = source + 1
    flow_graph = _MinCostFlow(sink + 1)

    for i in range(n_orders):
        flow_graph.add_edge(source, i, 1, 0)

    tracked_edges: dict[tuple[int, int], _Edge] = {}
    tracked_pickup_distances: dict[tuple[int, int], float] = {}
    for order_index, taxi_index, pickup_distance, match_cost in filtered_edges:
        key = (order_index, taxi_index)
        if key in tracked_pickup_distances:
            continue
        edge = flow_graph.add_edge(order_index, n_orders + taxi_index, 1, int(round(match_cost * cost_scale)))
        tracked_edges[key] = edge
        tracked_pickup_distances[key] = pickup_distance

    for j in range(n_taxis):
        flow_graph.add_edge(n_orders + j, sink, 1, 0)

    max_flow = min(n_orders, n_taxis)
    flow, _ = flow_graph.solve(source, sink, max_flow)
    if flow == 0:
        return []

    matches: list[OrderMatch] = []
    for (order_index, taxi_index), edge in tracked_edges.items():
        if edge.cap == 0:
            matches.append(
                OrderMatch(
                    order_index=order_index,
                    taxi_id=taxi_ids[taxi_index],
                    pickup_distance=tracked_pickup_distances[(order_index, taxi_index)],
                )
            )
    return matches
