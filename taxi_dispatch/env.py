from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Callable, Protocol
import warnings

import numpy as np

from .grid import DEFAULT_HEX_CENTER_SPACING_KM, HexGrid
from .road_network import RoadNetwork
from .routing import (
    OrderMatch,
    SparseCandidateEdge,
    assign_taxis_min_cost,
    assign_taxis_min_cost_from_costs,
    match_orders_greedy_sparse,
    match_orders_min_cost_sparse,
)


REGIONAL_STATE_FEATURES = (
    "current_orders",
    "idle_taxis",
    "incoming_taxis",
    "future_demand",
    "supply_demand_gap",
    "observed_gap",
    "future_need",
    "match_rate",
    "reject_rate",
    "idle_time",
    "empty_distance",
    "execution_bias",
)
DEMAND_FEATURE = 0
IDLE_SUPPLY_FEATURE = 1
INCOMING_SUPPLY_FEATURE = 2
FUTURE_DEMAND_FEATURE = 3
SUPPLY_DEMAND_GAP_FEATURE = 4
OBSERVED_GAP_FEATURE = 5
CURRENT_NEED_FEATURE = OBSERVED_GAP_FEATURE
FUTURE_NEED_FEATURE = 6
MATCH_RATE_FEATURE = 7
REJECT_RATE_FEATURE = 8
IDLE_TIME_FEATURE = 9
EMPTY_DISTANCE_FEATURE = 10
EXECUTION_BIAS_FEATURE = 11

DEFAULT_REWARD_MATCH_WEIGHT = 1.0
DEFAULT_REWARD_REPO_BALANCE_WEIGHT = 0.1
DEFAULT_REWARD_REPO_COST_WEIGHT = 0.05
DEFAULT_REWARD_ACTION_MOVE_COST = 0.05
DEFAULT_REWARD_INCOMING_DISCOUNT = 0.3
DEFAULT_REWARD_REPO_FUTURE_DEMAND_WEIGHT = 1.0
DEFAULT_REWARD_DISTANCE_COST_WEIGHT = 0.0
DEFAULT_REWARD_TIME_COST_WEIGHT = 0.0
REWARD_CLIP_VALUE = 5.0


class DemandModel(Protocol):
    rates: np.ndarray

    def generate(self, step: int, rng: np.random.Generator) -> np.ndarray: ...

    def sample_destinations(self, origin: int, step: int, n: int, rng: np.random.Generator) -> np.ndarray: ...


@dataclass
class EnvConfig:
    num_cells: int = 49
    cell_width_km: float = DEFAULT_HEX_CENTER_SPACING_KM
    fleet_size: int = 600
    horizon_steps: int = 72
    step_minutes: int = 10
    max_wait_steps: int = 1
    taxi_to_idle_weight: float = 0.5
    demand_scale: float = 1.8
    seed: int = 0
    invalid_action_penalty: float = 8.0
    pickup_radius_km: float = 1.5
    pickup_scope: str = "origin_and_neighbor"
    same_step_reposition_service: bool = True
    reposition_before_assignment: bool = False
    travel_speed_kmph: float = 30.0
    relocation_cost_weight: float = 0.5
    future_demand_steps: int = 0
    region_value_weight: float | None = None
    future_value_weight: float = 0.25
    future_gap_weight: float = 0.25
    dispatch_intensity_weight: float = 0.25
    trip_time_weight: float = 0.05
    service_revenue_weight: float = 0.05
    fare_base: float = 14.0
    fare_per_km: float = 2.6
    fare_per_minute: float = 0.5
    wait_time_penalty_weight: float = 0.1
    wait_time_tolerance_minutes: float = 5.0
    wait_time_penalty_exponent: float = 1.0
    cancellation_penalty: float = 3.0
    future_imbalance_weight: float = 0.5
    idle_time_penalty_weight: float = 0.01
    execution_mismatch_penalty: float = 0.25
    matching_mode: str = "value_guided"
    matching_time_scale_minutes: float | None = None
    state_feature_clip: float = 1.0e6

    def resolved_region_value_weight(self) -> float:
        if self.region_value_weight is not None:
            return float(self.region_value_weight)
        return float(self.future_value_weight)


@dataclass
class EpisodeMetrics:
    reward: float
    response_rate: float
    response_time_seconds: float
    cancellation_rate: float
    occupied_rate: float
    normalized_gmv: float
    orders: int
    served_orders: int
    cancellations: int
    repositioned: int


PolicyOutput = np.ndarray | dict[str, np.ndarray] | tuple[object, ...]


@dataclass
class _OrderRequest:
    origin: int
    destination: int
    created_step: int
    created_minute: int
    pickup_xy: np.ndarray
    dropoff_xy: np.ndarray | None = None
    trip_minutes: float | None = None
    pickup_node: str | None = None
    dropoff_node: str | None = None


class SyntheticDemand:
    """Small, deterministic OD generator used when the Shanghai data is absent."""

    def __init__(self, rates: np.ndarray, od_probs: np.ndarray) -> None:
        self.rates = rates
        self.od_probs = od_probs

    @classmethod
    def create(cls, grid: HexGrid, config: EnvConfig) -> "SyntheticDemand":
        xy = grid.xy
        center = xy.mean(axis=0)
        dist = np.linalg.norm(xy - center, axis=1)
        max_dist = max(float(dist.max()), 1.0)
        center_score = np.exp(-((dist / max_dist) ** 2) * 5.0)
        outer_score = dist / max_dist
        airport_idx = int(np.argmax(xy[:, 0] - 0.35 * xy[:, 1]))
        airport_score = np.exp(-np.linalg.norm(xy - xy[airport_idx], axis=1) / (max_dist * 0.35))

        rates = np.zeros((config.horizon_steps, grid.num_cells), dtype=np.float32)
        od_probs = np.zeros((config.horizon_steps, grid.num_cells, grid.num_cells), dtype=np.float32)

        for t in range(config.horizon_steps):
            frac = t / max(config.horizon_steps - 1, 1)
            morning = np.exp(-((frac - 0.13) / 0.15) ** 2)
            midday = np.exp(-((frac - 0.48) / 0.22) ** 2)
            evening = np.exp(-((frac - 0.78) / 0.15) ** 2)

            origin_score = (
                0.28
                + 1.15 * morning * outer_score
                + 0.95 * midday * center_score
                + 1.35 * evening * center_score
                + 0.75 * airport_score * (morning + 0.6 * evening)
            )
            rates[t] = config.demand_scale * origin_score * max(config.fleet_size / grid.num_cells / 2.8, 1.0)

            if frac < 0.35:
                dest_score = 0.25 + 1.9 * center_score + 0.25 * airport_score
            elif frac < 0.65:
                dest_score = 0.25 + 0.95 * center_score + 0.55 * outer_score + 0.35 * airport_score
            else:
                dest_score = 0.25 + 1.75 * outer_score + 0.55 * airport_score

            for origin in range(grid.num_cells):
                local = dest_score.copy()
                local[origin] *= 0.2
                total = float(local.sum())
                od_probs[t, origin] = local / total if total > 0 else np.full(grid.num_cells, 1.0 / grid.num_cells)

        return cls(rates=rates, od_probs=od_probs)

    def generate(self, step: int, rng: np.random.Generator) -> np.ndarray:
        return rng.poisson(self.rates[step]).astype(np.int64)

    def sample_destinations(self, origin: int, step: int, n: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.asarray([], dtype=np.int64)
        return rng.choice(self.od_probs.shape[1], size=n, p=self.od_probs[step, origin]).astype(np.int64)


class DispatchEnv:
    """Regional dispatch simulator with MAMR-style action and observation shapes."""

    action_dim = 7
    state_feature_names = REGIONAL_STATE_FEATURES
    demand_feature_index = DEMAND_FEATURE
    idle_supply_feature_index = IDLE_SUPPLY_FEATURE
    incoming_supply_feature_index = INCOMING_SUPPLY_FEATURE
    future_demand_feature_index = FUTURE_DEMAND_FEATURE
    supply_demand_gap_feature_index = SUPPLY_DEMAND_GAP_FEATURE
    current_need_feature_index = CURRENT_NEED_FEATURE
    observed_gap_feature_index = OBSERVED_GAP_FEATURE
    future_gap_feature_index = OBSERVED_GAP_FEATURE
    future_need_feature_index = FUTURE_NEED_FEATURE
    match_rate_feature_index = MATCH_RATE_FEATURE
    reject_rate_feature_index = REJECT_RATE_FEATURE
    idle_time_feature_index = IDLE_TIME_FEATURE
    empty_distance_feature_index = EMPTY_DISTANCE_FEATURE
    execution_bias_feature_index = EXECUTION_BIAS_FEATURE

    def __init__(
        self,
        config: EnvConfig | None = None,
        demand: DemandModel | None = None,
        grid: HexGrid | None = None,
        initial_taxi_distribution: np.ndarray | None = None,
        road_network: RoadNetwork | None = None,
        hex_distance_matrix: np.ndarray | None = None,
        road_distance_matrix: np.ndarray | None = None,
        road_time_matrix: np.ndarray | None = None,
    ) -> None:
        self.config = config or EnvConfig()
        self.grid = grid or HexGrid.create(self.config.num_cells, self.config.cell_width_km)
        if self.grid.num_cells != self.config.num_cells:
            raise ValueError("grid.num_cells must match config.num_cells")
        self.demand = demand or SyntheticDemand.create(self.grid, self.config)
        self.initial_taxi_distribution = self._normalize_initial_distribution(initial_taxi_distribution)
        self.road_network = road_network
        if self.road_network is not None and self.grid.projection_origin is None:
            raise ValueError("road_network requires a geographic grid with projection_origin")
        self.hex_distance_matrix = self._normalize_hex_distance_matrix(hex_distance_matrix)
        self.road_distance_matrix = self._normalize_road_cost_matrix(road_distance_matrix, "road_distance_matrix")
        self.road_time_matrix = self._normalize_road_cost_matrix(road_time_matrix, "road_time_matrix")

        self.grid_number = self.grid.num_cells
        self.state_feature_dim = len(self.state_feature_names)
        self.state_dim = self.grid_number * self.state_feature_dim
        self.observation_dim = self.action_dim * self.state_feature_dim
        self.feature_scale = max(self.config.fleet_size / max(self.grid_number, 1), 1.0)
        self.feature_scales = self._state_feature_scales()
        self.reward_scale = self.feature_scale
        self.available_actions = (self.grid.neighbors >= 0).astype(np.float32)

        self.rng = np.random.default_rng(self.config.seed)
        self.step_index = 0
        self.current_minute = 0
        self.done = False
        self.waiting_orders: list[_OrderRequest] = []
        self.idle_by_cell: list[list[int]] = [[] for _ in range(self.grid_number)]
        self.taxi_cell = np.full(self.config.fleet_size, -1, dtype=np.int64)
        self.taxi_xy = np.zeros((self.config.fleet_size, 2), dtype=np.float32)
        self.taxi_node: list[str | None] = [None for _ in range(self.config.fleet_size)]
        self.cell_road_nodes = self._cell_road_nodes()
        self.action_costs = self._action_costs()
        self.busy_arrivals: list[list[tuple[int, int, int]]] = []
        self.reposition_arrivals: list[list[tuple[int, int]]] = []
        self._state_raw = np.zeros((self.grid_number, self.state_feature_dim), dtype=np.float32)
        self.region_value_guidance = np.zeros(self.grid_number, dtype=np.float32)
        self.region_value_target = np.zeros(self.grid_number, dtype=np.float32)
        self.observed_gap_target = np.zeros(self.grid_number, dtype=np.float32)
        self.future_need_target = np.zeros(self.grid_number, dtype=np.float32)
        self.future_gap_guidance = np.zeros(self.grid_number, dtype=np.float32)
        self.dispatch_intensity_guidance = np.ones(self.grid_number, dtype=np.float32)
        self.upper_guidance_active = False
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.observations = np.zeros((self.grid_number, self.observation_dim), dtype=np.float32)
        self.raw_reward = np.zeros(self.grid_number, dtype=np.float32)
        self.actor_rewards = np.zeros((self.grid_number, self.action_dim), dtype=np.float32)
        self.reward_debug: dict[str, float] = {}
        self.cell_orders = np.zeros(self.grid_number, dtype=np.int64)
        self.cell_served_orders = np.zeros(self.grid_number, dtype=np.int64)
        self.cell_cancellations = np.zeros(self.grid_number, dtype=np.int64)
        self._step_service_revenue = np.zeros(self.grid_number, dtype=np.float32)
        self._step_service_distance = np.zeros(self.grid_number, dtype=np.float32)
        self._step_service_minutes = np.zeros(self.grid_number, dtype=np.float32)
        self._step_wait_minutes = np.zeros(self.grid_number, dtype=np.float32)
        self._step_wait_penalty_minutes = np.zeros(self.grid_number, dtype=np.float32)
        self._step_cancelled_orders = np.zeros(self.grid_number, dtype=np.float32)
        self._step_relocation_cost = np.zeros(self.grid_number, dtype=np.float32)
        self._step_order_requests = np.zeros(self.grid_number, dtype=np.float32)
        self._step_matched_orders = np.zeros(self.grid_number, dtype=np.float32)
        self._step_idle_minutes = np.zeros(self.grid_number, dtype=np.float32)
        self._step_empty_distance = np.zeros(self.grid_number, dtype=np.float32)
        self._step_planned_dispatch = np.zeros(self.grid_number, dtype=np.float32)
        self._step_actual_dispatch = np.zeros(self.grid_number, dtype=np.float32)
        self._step_decision_demand = np.zeros(self.grid_number, dtype=np.float32)
        self._feedback_match_rate = np.zeros(self.grid_number, dtype=np.float32)
        self._feedback_reject_rate = np.zeros(self.grid_number, dtype=np.float32)
        self._feedback_idle_time = np.zeros(self.grid_number, dtype=np.float32)
        self._feedback_empty_distance = np.zeros(self.grid_number, dtype=np.float32)
        self._feedback_execution_bias = np.zeros(self.grid_number, dtype=np.float32)
        self._repo_reward_supply_before: np.ndarray | None = None
        self._repo_reward_demand_before: np.ndarray | None = None
        self.reset()

    def reset(self, seed: int | None = None, episode_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        reset_demand = getattr(self.demand, "reset_episode", None)
        if callable(reset_demand):
            try:
                reset_demand(self.rng, episode_index=episode_index)
            except TypeError:
                reset_demand(self.rng)
        self.step_index = 0
        self.current_minute = 0
        self.done = False
        self.waiting_orders = []
        self.cell_orders.fill(0)
        self.cell_served_orders.fill(0)
        self.cell_cancellations.fill(0)
        self._reset_feedback_state()

        self.total_orders = 0
        self.served_orders = 0
        self.cancelled_orders = 0
        self.response_time_seconds = 0.0
        self.occupied_minutes = 0.0
        self.gmv = 0.0
        self.repositioned = 0
        self.reposition_cost = 0.0
        self._reset_step_reward_components()

        event_slots = self.config.horizon_steps * self.config.step_minutes + 32 * self.config.step_minutes + 1
        self.busy_arrivals = [[] for _ in range(event_slots)]
        self.reposition_arrivals = [[] for _ in range(event_slots)]
        self.idle_by_cell = [[] for _ in range(self.grid_number)]

        if self.initial_taxi_distribution is None:
            weights = self.demand.rates.mean(axis=0) + 0.15
            weights = weights / weights.sum()
        else:
            weights = self.initial_taxi_distribution
        initial_cells = self.rng.choice(self.grid_number, size=self.config.fleet_size, p=weights)
        for taxi_id, cell in enumerate(initial_cells):
            self._set_taxi_idle_at_cell(taxi_id, int(cell))

        self._update_state_and_rewards()
        return self.observations.copy(), self.state.copy()

    def step(self, actions: np.ndarray | list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """Legacy tick API.

        Deprecated because it applies relocation actions before generating the
        current step's orders. Use :meth:`advance` for training and evaluation
        so the policy observes current demand before deciding.
        """

        warnings.warn(
            "DispatchEnv.step() is deprecated because actions are applied before "
            "current-step demand is generated; use DispatchEnv.advance(policy) instead.",
            FutureWarning,
            stacklevel=2,
        )
        if self.done:
            raise RuntimeError("step() called after the episode finished; call reset().")

        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape != (self.grid_number, self.action_dim):
            raise ValueError(f"actions must have shape {(self.grid_number, self.action_dim)}")

        self.current_minute = self._step_start_minute()
        self._complete_arrivals()
        self._reset_step_reward_components()
        self._apply_repositioning(action_array)
        self._generate_orders()
        self._update_state_and_rewards(demand_cutoff_minute=self._step_end_minute())
        self._capture_step_decision_demand()
        self._run_assignment_window()
        self._finalize_execution_feedback()

        self.step_index += 1
        self.done = self.step_index >= self.config.horizon_steps
        self._update_state_and_rewards()
        return (
            self.observations.copy(),
            self.state.copy(),
            self.raw_reward.copy(),
            self.actor_rewards.copy(),
            self.done,
        )

    def advance(
        self,
        policy: Callable[["DispatchEnv", np.ndarray, np.ndarray], PolicyOutput],
        reposition_policy: Callable[["DispatchEnv", np.ndarray, np.ndarray], PolicyOutput] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """Run one simulator tick with the paper-style event order.

        The policy sees newly generated demand for the current time step. The
        environment first settles order assignment with the shared matching
        rule, then applies the policy's repositioning action to taxis that
        remain idle for future steps. When ``reposition_policy`` is provided,
        ``policy`` is used only for pre-assignment matching guidance and the
        repositioning action is sampled from a refreshed post-assignment state.
        """

        if self.done:
            raise RuntimeError("advance() called after the episode finished; call reset().")

        self.current_minute = self._step_start_minute()
        self._complete_arrivals()
        self._reset_step_reward_components()
        self._generate_orders()
        self._update_state_and_rewards(demand_cutoff_minute=self._step_end_minute())
        self._capture_step_decision_demand()
        action_array = self._parse_policy_output(policy(self, self.observations.copy(), self.state.copy()))
        if action_array.shape != (self.grid_number, self.action_dim):
            raise ValueError(f"policy actions must have shape {(self.grid_number, self.action_dim)}")
        if bool(getattr(self.config, "reposition_before_assignment", False)):
            self._apply_repositioning(action_array)
            self._run_assignment_window()
        else:
            self._run_assignment_window()
            if reposition_policy is not None:
                self._update_state_and_rewards(demand_cutoff_minute=self._step_end_minute())
                action_array = self._parse_policy_output(
                    reposition_policy(self, self.observations.copy(), self.state.copy())
                )
                if action_array.shape != (self.grid_number, self.action_dim):
                    raise ValueError(f"reposition policy actions must have shape {(self.grid_number, self.action_dim)}")
            self._apply_repositioning(action_array)
        self._finalize_execution_feedback()

        self.step_index += 1
        self.done = self.step_index >= self.config.horizon_steps
        self._update_state_and_rewards()
        return (
            self.observations.copy(),
            self.state.copy(),
            self.raw_reward.copy(),
            self.actor_rewards.copy(),
            self.done,
        )

    def metrics(self, episode_reward: float = 0.0) -> EpisodeMetrics:
        orders = max(self.total_orders, 1)
        served_orders = max(self.served_orders, 1)
        elapsed_minutes = max(self.config.horizon_steps * self.config.step_minutes, 1)
        fleet_minutes = max(self.config.fleet_size * elapsed_minutes, 1)
        return EpisodeMetrics(
            reward=float(episode_reward),
            response_rate=float(self.served_orders / orders),
            response_time_seconds=float(self.response_time_seconds / served_orders),
            cancellation_rate=float(self.cancelled_orders / orders),
            occupied_rate=float(self.occupied_minutes / fleet_minutes),
            normalized_gmv=float(self.gmv / fleet_minutes * 60.0),
            orders=int(self.total_orders),
            served_orders=int(self.served_orders),
            cancellations=int(self.cancelled_orders),
            repositioned=int(self.repositioned),
        )

    def regional_state_raw(self) -> np.ndarray:
        return self._state_raw.copy()

    def auxiliary_targets(self) -> dict[str, np.ndarray]:
        """Targets for the currently observed regional shortage signal.

        The training loop attaches ``region_value`` separately from realized
        matching guidance. Future-pressure supervision for a transition is
        built from the next decision state's observed gap and optional
        lookahead demand pressure, not directly from this current-state helper.
        """

        observed_gap = np.asarray(self.observed_gap_target, dtype=np.float32).copy()
        future_need = np.asarray(self.future_need_target, dtype=np.float32).copy()
        return {
            "observed_gap": observed_gap,
            "future_need": future_need,
            "dispatch_intensity": np.clip(observed_gap, 0.0, 1.0).astype(np.float32),
        }

    def _reset_step_reward_components(self) -> None:
        self._step_service_revenue.fill(0.0)
        self._step_service_distance.fill(0.0)
        self._step_service_minutes.fill(0.0)
        self._step_wait_minutes.fill(0.0)
        self._step_wait_penalty_minutes.fill(0.0)
        self._step_cancelled_orders.fill(0.0)
        self._step_relocation_cost.fill(0.0)
        self._step_order_requests.fill(0.0)
        self._step_matched_orders.fill(0.0)
        self._step_idle_minutes.fill(0.0)
        self._step_empty_distance.fill(0.0)
        self._step_planned_dispatch.fill(0.0)
        self._step_actual_dispatch.fill(0.0)
        self._step_decision_demand.fill(0.0)
        self._repo_reward_supply_before = None
        self._repo_reward_demand_before = None
        self.reward_debug = {}
        self._reset_upper_guidance()

    def _reset_upper_guidance(self) -> None:
        self.region_value_guidance.fill(0.0)
        self.future_gap_guidance.fill(0.0)
        self.dispatch_intensity_guidance.fill(1.0)
        self.upper_guidance_active = False

    def _reset_feedback_state(self) -> None:
        self._feedback_match_rate.fill(0.0)
        self._feedback_reject_rate.fill(0.0)
        self._feedback_idle_time.fill(0.0)
        self._feedback_empty_distance.fill(0.0)
        self._feedback_execution_bias.fill(0.0)

    def cancellation_rate_by_cell(self) -> np.ndarray:
        denom = np.maximum(self.cell_orders, 1)
        return self.cell_cancellations / denom

    def response_rate_by_cell(self) -> np.ndarray:
        denom = np.maximum(self.cell_orders, 1)
        return self.cell_served_orders / denom

    def _step_start_minute(self) -> int:
        return int(self.step_index * self.config.step_minutes)

    def _step_end_minute(self) -> int:
        return self._step_start_minute() + max(int(self.config.step_minutes), 1)

    def _run_assignment_window(self) -> None:
        start_minute = self._step_start_minute()
        for offset in range(max(int(self.config.step_minutes), 1)):
            self.current_minute = start_minute + offset
            self._complete_arrivals()
            self._accumulate_idle_feedback()
            self._match_orders()
            self._age_and_cancel_orders()
        self.current_minute = self._step_end_minute()
        self._cancel_unsettled_step_orders()

    def _complete_arrivals(self) -> None:
        if self.current_minute >= len(self.busy_arrivals):
            return
        for taxi_id, cell, _duration in self.busy_arrivals[self.current_minute]:
            self._set_taxi_idle_at_cell(taxi_id, cell)
        self.busy_arrivals[self.current_minute].clear()

        for taxi_id, cell in self.reposition_arrivals[self.current_minute]:
            self._set_taxi_idle_at_cell(taxi_id, cell)
        self.reposition_arrivals[self.current_minute].clear()

    def _apply_repositioning(self, actions: np.ndarray) -> None:
        self._capture_reposition_reward_before()
        for origin in range(self.grid_number):
            idle_count = len(self.idle_by_cell[origin])
            if idle_count <= 0:
                continue

            probs = self._valid_action_probs(origin, actions[origin])
            action_counts = self._integer_action_counts(idle_count, probs)
            target_quotas: dict[int, int] = {}
            for action in range(1, self.action_dim):
                target = int(self.grid.neighbors[origin, action])
                if target < 0:
                    continue
                count = int(action_counts[action])
                if count > 0:
                    target_quotas[target] = target_quotas.get(target, 0) + count
            if not target_quotas:
                continue
            self._step_planned_dispatch[origin] += float(sum(target_quotas.values()))

            taxi_ids = self.idle_by_cell[origin]
            if self.road_time_matrix is None and self.road_network is None:
                assignments = assign_taxis_min_cost(
                    taxi_ids=taxi_ids,
                    taxi_xy=self.taxi_xy,
                    target_quotas=target_quotas,
                    target_xy=self.grid.xy,
                )
            else:
                target_cells = [cell for cell, quota in target_quotas.items() if quota > 0]
                assignments = assign_taxis_min_cost_from_costs(
                    taxi_ids=taxi_ids,
                    target_quotas=target_quotas,
                    cost_matrix=self._road_reposition_costs(taxi_ids, target_cells),
                )
            if not assignments:
                continue

            assigned_ids = {assignment.taxi_id for assignment in assignments}
            self.idle_by_cell[origin] = [tid for tid in self.idle_by_cell[origin] if tid not in assigned_ids]
            self._step_actual_dispatch[origin] += float(len(assignments))
            for assignment in assignments:
                if self.road_time_matrix is None and self.road_network is None:
                    travel_minutes = max(1, int(ceil(self._distance_to_travel_minutes(assignment.cost))))
                else:
                    travel_minutes = max(1, int(ceil(assignment.cost)))
                arrival_minute = self._reposition_arrival_minute(travel_minutes)
                self._ensure_event_slot(arrival_minute)
                self.reposition_arrivals[arrival_minute].append((assignment.taxi_id, assignment.target_cell))
                self.taxi_cell[assignment.taxi_id] = -1
                self.taxi_node[assignment.taxi_id] = None
                self.repositioned += 1
                self.reposition_cost += float(travel_minutes)
                self._step_relocation_cost[origin] += float(travel_minutes)
                self._step_empty_distance[origin] += float(
                    self._cell_to_cell_distance(origin, assignment.target_cell)
                )

    def _reposition_arrival_minute(self, travel_minutes: int) -> int:
        physical_arrival_minute = self.current_minute + max(int(travel_minutes), 1)
        if not bool(getattr(self.config, "same_step_reposition_service", True)):
            return max(physical_arrival_minute, self._step_end_minute())
        return physical_arrival_minute

    def _generate_orders(self) -> None:
        request_sampler = getattr(self.demand, "sample_requests", None)
        if callable(request_sampler):
            self._generate_event_orders(tuple(request_sampler(self.step_index, self.rng)))
            return

        demand = self.demand.generate(self.step_index, self.rng)
        self.total_orders += int(demand.sum())
        self.cell_orders += demand
        created_minute = self._step_start_minute()
        for origin, count in enumerate(demand):
            n = int(count)
            if n <= 0:
                continue
            self._step_order_requests[origin] += float(n)
            destinations = self.demand.sample_destinations(origin, self.step_index, n, self.rng)
            for destination in destinations:
                pickup_xy = self._sample_position(int(origin))
                self.waiting_orders.append(
                    _OrderRequest(
                        origin=int(origin),
                        destination=int(destination),
                        created_step=self.step_index,
                        created_minute=created_minute,
                        pickup_xy=pickup_xy,
                        dropoff_xy=self._sample_position(int(destination)),
                        pickup_node=None,
                    )
                )
                self.waiting_orders[-1].pickup_node = self._road_node_for_xy(self.waiting_orders[-1].pickup_xy)
                self.waiting_orders[-1].dropoff_node = self._road_node_for_xy(self.waiting_orders[-1].dropoff_xy)

    def _generate_event_orders(self, events: tuple[object, ...]) -> None:
        if not events:
            return
        step_start = self._step_start_minute()
        for event in events:
            origin = int(_event_value(event, "origin"))
            destination = int(_event_value(event, "destination"))
            minute_offset = int(_event_value(event, "minute_offset", 0))
            minute_offset = min(max(minute_offset, 0), max(int(self.config.step_minutes) - 1, 0))

            pickup_xy_value = _event_value(event, "pickup_xy", None)
            dropoff_xy_value = _event_value(event, "dropoff_xy", None)
            pickup_xy = (
                np.asarray(pickup_xy_value, dtype=np.float32)
                if pickup_xy_value is not None
                else self._sample_position(origin)
            )
            dropoff_xy = (
                np.asarray(dropoff_xy_value, dtype=np.float32)
                if dropoff_xy_value is not None
                else self._sample_position(destination)
            )
            trip_minutes_value = _event_value(event, "trip_minutes", None)
            trip_minutes = None if trip_minutes_value is None else float(trip_minutes_value)

            order = _OrderRequest(
                origin=origin,
                destination=destination,
                created_step=self.step_index,
                created_minute=step_start + minute_offset,
                pickup_xy=pickup_xy,
                dropoff_xy=dropoff_xy,
                trip_minutes=trip_minutes,
            )
            order.pickup_node = self._road_node_for_xy(order.pickup_xy)
            order.dropoff_node = self._road_node_for_xy(order.dropoff_xy)
            self.waiting_orders.append(order)
            self.total_orders += 1
            self.cell_orders[origin] += 1
            self._step_order_requests[origin] += 1.0

    def _match_orders(self) -> None:
        if not self.waiting_orders:
            return
        active_indices = [
            idx for idx, order in enumerate(self.waiting_orders) if order.created_minute <= self.current_minute
        ]
        if not active_indices:
            return

        if self._uses_value_guided_matching():
            matched_order_indices = self._match_orders_value_guided(active_indices)
        else:
            matched_order_indices = self._match_orders_nearest_first(active_indices)

        if not matched_order_indices:
            return

        self.waiting_orders = [order for idx, order in enumerate(self.waiting_orders) if idx not in matched_order_indices]

    def _uses_value_guided_matching(self) -> bool:
        return self._matching_mode() == "value_guided" and bool(self.upper_guidance_active)

    def _match_orders_nearest_first(self, active_indices: list[int]) -> set[int]:
        matched_order_indices: set[int] = set()
        for order_index in active_indices:
            order = self.waiting_orders[order_index]
            match = self._nearest_idle_match_for_order(order)
            if match is None:
                continue
            taxi_id, pickup_distance = match
            self._remove_idle_taxi(taxi_id)
            dispatched = self._dispatch_order_service(
                order=order,
                taxi_id=taxi_id,
                pickup_distance=pickup_distance,
            )
            if not dispatched:
                self._restore_idle_taxi(taxi_id)
            matched_order_indices.add(order_index)
        return matched_order_indices

    def _match_orders_value_guided(self, active_indices: list[int]) -> set[int]:
        active_orders = [self.waiting_orders[idx] for idx in active_indices]
        matched_active_positions: set[int] = set()
        for pickup_scope in self._matching_pickup_scopes():
            scoped_positions = [idx for idx in range(len(active_orders)) if idx not in matched_active_positions]
            if not scoped_positions:
                break
            scoped_orders = [active_orders[idx] for idx in scoped_positions]
            matches = self._match_active_orders(scoped_orders, pickup_scope=pickup_scope)
            if not matches:
                continue

            matched_taxis = {match.taxi_id for match in matches}
            for taxi_id in matched_taxis:
                self._remove_idle_taxi(taxi_id)
            for match in matches:
                active_position = scoped_positions[match.order_index]
                matched_active_positions.add(active_position)
                order = self.waiting_orders[active_indices[active_position]]
                dispatched = self._dispatch_order_service(
                    order=order,
                    taxi_id=match.taxi_id,
                    pickup_distance=match.pickup_distance,
                )
                if not dispatched:
                    self._restore_idle_taxi(match.taxi_id)
        return {active_indices[idx] for idx in matched_active_positions}

    def _nearest_idle_match_for_order(self, order: _OrderRequest) -> tuple[int, float] | None:
        for pickup_scope in self._matching_pickup_scopes():
            match = self._nearest_idle_match_in_scope(order, pickup_scope)
            if match is not None:
                return match
        return None

    def _nearest_idle_match_in_scope(self, order: _OrderRequest, pickup_scope: str) -> tuple[int, float] | None:
        best: tuple[float, float, int] | None = None
        for taxi_id in self._pickup_taxis(order.origin, pickup_scope):
            pickup_distance = self._pickup_distance_for_taxi(taxi_id, order)
            if not np.isfinite(pickup_distance):
                continue
            pickup_minutes = self._road_pickup_minutes(taxi_id, order, pickup_distance)
            if not np.isfinite(pickup_minutes):
                pickup_minutes = self._distance_to_travel_minutes(pickup_distance)
            if not self._is_serviceable_pickup_candidate(order, pickup_distance, pickup_minutes):
                continue
            candidate = (float(pickup_distance), float(pickup_minutes), int(taxi_id))
            if best is None or candidate < best:
                best = candidate
        if best is None:
            return None
        return best[2], best[0]

    def _pickup_distance_for_taxi(self, taxi_id: int, order: _OrderRequest) -> float:
        source_cell = int(self.taxi_cell[taxi_id])
        if source_cell < 0:
            return float("inf")
        same_cell_distance = None
        uses_cell_or_road_costs = (
            self.road_network is not None
            or self.road_distance_matrix is not None
            or self.road_time_matrix is not None
            or self.hex_distance_matrix is not None
        )
        if source_cell == int(order.origin) and uses_cell_or_road_costs:
            same_cell_distance = self._same_cell_pickup_distance(taxi_id, order)
        if self.road_network is not None and self.taxi_node[taxi_id] is not None and order.pickup_node is not None:
            distance = self.road_network.shortest_path_distance_km(self.taxi_node[taxi_id], order.pickup_node)
            if np.isfinite(distance):
                if same_cell_distance is not None:
                    return max(float(distance), same_cell_distance)
                return float(distance)
        if self.road_distance_matrix is not None or self.hex_distance_matrix is not None:
            distance = self._cell_to_cell_distance(source_cell, order.origin)
            if np.isfinite(distance):
                if same_cell_distance is not None:
                    return max(float(distance), same_cell_distance)
                return float(distance)
        distance = float(np.linalg.norm(self.taxi_xy[taxi_id] - order.pickup_xy))
        if np.isfinite(distance):
            if same_cell_distance is not None:
                return max(distance, same_cell_distance)
            return distance
        return float(self._cell_to_cell_distance(source_cell, order.origin))

    def _same_cell_pickup_distance(self, taxi_id: int, order: _OrderRequest) -> float:
        distance = float(np.linalg.norm(self.taxi_xy[taxi_id] - order.pickup_xy))
        if not np.isfinite(distance):
            distance = 0.0
        return max(distance, self._same_cell_pickup_floor_km())

    def _same_cell_pickup_floor_km(self) -> float:
        max_pickup = self._max_pickup_distance_km()
        if not np.isfinite(max_pickup):
            max_pickup = float("inf")
        # Matrix diagonals are centroid self-loops, not door-to-door pickup costs.
        floor = max(float(self.config.cell_width_km) * 0.35, 0.0)
        return float(min(floor, max_pickup))

    def _remove_idle_taxi(self, taxi_id: int) -> None:
        source_cell = int(self.taxi_cell[taxi_id])
        if 0 <= source_cell < self.grid_number:
            self.idle_by_cell[source_cell] = [tid for tid in self.idle_by_cell[source_cell] if tid != taxi_id]
            return
        for cell in range(self.grid_number):
            if taxi_id in self.idle_by_cell[cell]:
                self.idle_by_cell[cell] = [tid for tid in self.idle_by_cell[cell] if tid != taxi_id]
                return

    def _restore_idle_taxi(self, taxi_id: int) -> None:
        source_cell = int(self.taxi_cell[taxi_id])
        if not (0 <= source_cell < self.grid_number):
            return
        if taxi_id not in self.idle_by_cell[source_cell]:
            self.idle_by_cell[source_cell].append(taxi_id)

    def _match_active_orders(self, active_orders: list[_OrderRequest], pickup_scope: str) -> list[OrderMatch]:
        taxi_ids, candidate_edges = self._pickup_candidate_edges(active_orders, pickup_scope=pickup_scope)
        if not taxi_ids:
            return []

        matching_mode = self._matching_mode()
        match_fn = match_orders_greedy_sparse if matching_mode == "greedy" else match_orders_min_cost_sparse
        return match_fn(
            n_orders=len(active_orders),
            taxi_ids=taxi_ids,
            candidate_edges=candidate_edges,
            max_pickup_distance_km=self._max_pickup_distance_km(),
        )

    def _dispatch_service(self, origin: int, source_cell: int, taxi_ids: list[int], age: int) -> None:
        n = len(taxi_ids)
        destinations = self.demand.sample_destinations(origin, self.step_index, n, self.rng)
        trip_minutes_samples = self._sample_trip_minutes(origin, destinations)
        pickup_minutes = self._cell_to_cell_time(source_cell, origin)
        for taxi_id, dest_cell, trip_minutes in zip(taxi_ids, destinations, trip_minutes_samples):
            dest_cell = int(dest_cell)
            trip_distance = max(self._cell_to_cell_distance(origin, dest_cell), self.config.cell_width_km * 0.6)
            service_minutes = max(1, int(ceil(pickup_minutes + trip_minutes)))
            arrival_minute = self.current_minute + service_minutes
            self._ensure_event_slot(arrival_minute)
            self.busy_arrivals[arrival_minute].append((taxi_id, dest_cell, service_minutes))
            self.taxi_cell[taxi_id] = -1
            self.taxi_node[taxi_id] = None

            response_seconds = age * self.config.step_minutes * 60.0 + pickup_minutes * 60.0
            response_minutes = float(response_seconds / 60.0)
            self.response_time_seconds += response_seconds
            self.served_orders += 1
            self.cell_served_orders[origin] += 1
            self._step_matched_orders[origin] += 1.0
            remaining_minutes = max(self.config.horizon_steps * self.config.step_minutes - self.current_minute, 0)
            self.occupied_minutes += min(service_minutes, remaining_minutes)
            fare = self._trip_fare(trip_distance, float(trip_minutes))
            self.gmv += fare
            self._step_service_revenue[origin] += float(fare)
            self._step_service_distance[origin] += float(trip_distance)
            self._step_service_minutes[origin] += float(trip_minutes)
            self._step_wait_minutes[origin] += response_minutes
            self._step_wait_penalty_minutes[origin] += self._quality_wait_penalty_minutes(response_minutes)

    def _dispatch_order_service(
        self,
        order: _OrderRequest,
        taxi_id: int,
        pickup_distance: float,
    ) -> bool:
        source_cell = int(self.taxi_cell[taxi_id])
        pickup_minutes = self._road_pickup_minutes(taxi_id, order, pickup_distance)
        if not self._is_responsive_pickup(order, pickup_minutes):
            self._record_order_cancellation(order)
            return False
        trip_distance = self._road_trip_distance(order)
        trip_minutes = self._road_trip_minutes(order)
        service_minutes = max(1, int(ceil(pickup_minutes + trip_minutes)))
        arrival_minute = self.current_minute + service_minutes
        self._ensure_event_slot(arrival_minute)
        self.busy_arrivals[arrival_minute].append((taxi_id, order.destination, service_minutes))
        self.taxi_cell[taxi_id] = -1
        self.taxi_node[taxi_id] = None

        wait_minutes = max(self.current_minute - order.created_minute, 0)
        response_minutes = float(wait_minutes + pickup_minutes)
        response_seconds = response_minutes * 60.0
        self.response_time_seconds += response_seconds
        self.served_orders += 1
        self.cell_served_orders[order.origin] += 1
        self._step_matched_orders[order.origin] += 1.0
        if 0 <= source_cell < self.grid_number and np.isfinite(pickup_distance):
            self._step_empty_distance[order.origin] += float(pickup_distance)
        remaining_minutes = max(self.config.horizon_steps * self.config.step_minutes - self.current_minute, 0)
        self.occupied_minutes += min(service_minutes, remaining_minutes)
        fare = self._trip_fare(trip_distance, trip_minutes)
        self.gmv += fare
        self._step_service_revenue[order.origin] += float(fare)
        self._step_service_distance[order.origin] += float(trip_distance)
        self._step_service_minutes[order.origin] += float(trip_minutes)
        self._step_wait_minutes[order.origin] += response_minutes
        self._step_wait_penalty_minutes[order.origin] += self._quality_wait_penalty_minutes(response_minutes)
        return True

    def _age_and_cancel_orders(self) -> None:
        if not self.waiting_orders:
            return
        kept: list[_OrderRequest] = []
        max_wait_minutes = self._max_response_minutes()
        for order in self.waiting_orders:
            if order.created_minute > self.current_minute:
                kept.append(order)
            elif self.current_minute - order.created_minute >= max_wait_minutes:
                self._record_order_cancellation(order)
            else:
                kept.append(order)
        self.waiting_orders = kept

    def _record_order_cancellation(self, order: _OrderRequest) -> None:
        self.cancelled_orders += 1
        self.cell_cancellations[order.origin] += 1
        self._step_cancelled_orders[order.origin] += 1.0

    def _cancel_unsettled_step_orders(self) -> None:
        if not self.waiting_orders:
            return
        cutoff_minute = self._step_end_minute()
        kept: list[_OrderRequest] = []
        for order in self.waiting_orders:
            if order.created_minute < cutoff_minute:
                self._record_order_cancellation(order)
            else:
                kept.append(order)
        self.waiting_orders = kept

    def _max_response_minutes(self) -> float:
        return float(max(int(self.config.max_wait_steps * self.config.step_minutes), 1))

    def _max_pickup_distance_km(self) -> float:
        max_distance = float(self.config.pickup_radius_km)
        if not np.isfinite(max_distance):
            return float("inf")
        return max(max_distance, 0.0)

    def _response_minutes(self, order: _OrderRequest, pickup_minutes: float) -> float:
        if not np.isfinite(pickup_minutes):
            return float("inf")
        wait_minutes = max(float(self.current_minute - order.created_minute), 0.0)
        return wait_minutes + max(float(pickup_minutes), 0.0)

    def _is_responsive_pickup(self, order: _OrderRequest, pickup_minutes: float) -> bool:
        return self._response_minutes(order, pickup_minutes) <= self._max_response_minutes()

    def _is_serviceable_pickup_candidate(
        self,
        order: _OrderRequest,
        pickup_distance: float,
        pickup_minutes: float,
    ) -> bool:
        if not np.isfinite(pickup_distance):
            return False
        if pickup_distance > self._max_pickup_distance_km() + 1e-6:
            return False
        if not np.isfinite(pickup_minutes):
            pickup_minutes = self._distance_to_travel_minutes(pickup_distance)
        return self._is_responsive_pickup(order, pickup_minutes)

    def _update_state_and_rewards(self, demand_cutoff_minute: int | None = None) -> None:
        demand = np.zeros(self.grid_number, dtype=np.float32)
        for order in self.waiting_orders:
            if self._order_visible_in_state(order, demand_cutoff_minute):
                demand[order.origin] += 1.0
        demand = self._sanitize_feature_vector(demand, nonnegative=True)
        idle_supply = self._sanitize_feature_vector(
            np.asarray([len(x) for x in self.idle_by_cell], dtype=np.float32),
            nonnegative=True,
        )
        incoming_supply = self._sanitize_feature_vector(self._incoming_supply_by_cell(), nonnegative=True)
        future_demand = self._sanitize_feature_vector(self._future_demand_by_cell(), nonnegative=True)
        taxi_to_idle_weight = self._finite_scalar(self.config.taxi_to_idle_weight, default=0.0, nonnegative=True)
        supply = self._sanitize_feature_vector(idle_supply + taxi_to_idle_weight * incoming_supply, nonnegative=True)
        expected_future_supply = supply
        supply_demand_gap = self._sanitize_feature_vector(supply - demand)
        current_need = self._sanitize_feature_vector(np.maximum(demand - supply, 0.0), nonnegative=True)
        future_need = self._sanitize_feature_vector(np.maximum(future_demand - expected_future_supply, 0.0), nonnegative=True)
        self.observed_gap_target = (current_need / self.feature_scale).astype(np.float32)
        self.future_need_target = (future_need / self.feature_scale).astype(np.float32)

        feedback_match_rate = np.clip(self._sanitize_feature_vector(self._feedback_match_rate, nonnegative=True), 0.0, 1.0)
        feedback_reject_rate = np.clip(self._sanitize_feature_vector(self._feedback_reject_rate, nonnegative=True), 0.0, 1.0)
        self._state_raw = self._sanitize_feature_matrix(
            np.stack(
                [
                    demand,
                    idle_supply,
                    incoming_supply,
                    future_demand,
                    supply_demand_gap,
                    current_need,
                    future_need,
                    feedback_match_rate,
                    feedback_reject_rate,
                    self._sanitize_feature_vector(self._feedback_idle_time, nonnegative=True),
                    self._sanitize_feature_vector(self._feedback_empty_distance, nonnegative=True),
                    self._sanitize_feature_vector(self._feedback_execution_bias),
                ],
                axis=1,
            )
        )
        scaled_state = self._sanitize_feature_matrix(
            self._state_raw / self._safe_feature_scales(),
            clip=self._state_feature_clip(),
        ).astype(np.float32)
        self.state = scaled_state.reshape(-1)
        self.observations = np.zeros((self.grid_number, self.observation_dim), dtype=np.float32)
        for cell in range(self.grid_number):
            chunks: list[np.ndarray] = []
            for dst in self.grid.neighbors[cell]:
                if dst < 0:
                    chunks.append(np.zeros(self.state_feature_dim, dtype=np.float32))
                else:
                    chunks.append(scaled_state[int(dst)])
            self.observations[cell] = np.concatenate(chunks).astype(np.float32)

        reward_supply = self._reward_supply_from_components(idle_supply, incoming_supply)
        repo_demand = self._repo_reward_demand_for_update(demand, future_demand)
        self.raw_reward, self.actor_rewards, self.reward_debug = self._stage_reward_outputs(
            supply=reward_supply,
            demand=repo_demand,
        )

    def _order_visible_in_state(self, order: _OrderRequest, demand_cutoff_minute: int | None) -> bool:
        if demand_cutoff_minute is None:
            return order.created_minute <= self.current_minute
        return order.created_minute < int(demand_cutoff_minute)

    def _stage_reward_outputs(
        self,
        *,
        supply: np.ndarray,
        demand: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        match_profit, match_profit_norm = self._match_reward_components()
        repo_before, repo_after, repo_improve = self._repo_balance_reward_components(supply, demand)
        repo_cost = self._repo_cost_component()

        match_weight = self._reward_param("reward_match_weight", DEFAULT_REWARD_MATCH_WEIGHT)
        repo_balance_weight = self._reward_param(
            "reward_repo_balance_weight",
            DEFAULT_REWARD_REPO_BALANCE_WEIGHT,
        )
        repo_cost_weight = self._reward_param("reward_repo_cost_weight", DEFAULT_REWARD_REPO_COST_WEIGHT)

        local_reward = match_weight * match_profit_norm + repo_balance_weight * repo_improve - repo_cost_weight * repo_cost
        local_reward = np.nan_to_num(
            local_reward,
            nan=0.0,
            posinf=REWARD_CLIP_VALUE,
            neginf=-REWARD_CLIP_VALUE,
        )
        local_reward = np.clip(local_reward, -REWARD_CLIP_VALUE, REWARD_CLIP_VALUE).astype(np.float32)
        global_reward = float(np.mean(local_reward)) if local_reward.size else 0.0

        critic_rewards = 0.75 * local_reward + 0.25 * global_reward
        critic_rewards = np.nan_to_num(
            critic_rewards,
            nan=0.0,
            posinf=REWARD_CLIP_VALUE,
            neginf=-REWARD_CLIP_VALUE,
        )
        critic_rewards = np.clip(critic_rewards, -REWARD_CLIP_VALUE, REWARD_CLIP_VALUE).astype(np.float32)

        actor_rewards = self._actor_rewards_from_local_reward(local_reward, global_reward)
        debug = {
            "match_profit_mean": float(np.mean(match_profit)) if match_profit.size else 0.0,
            "match_profit_norm_mean": float(np.mean(match_profit_norm)) if match_profit_norm.size else 0.0,
            "repo_balance_before_mean": float(np.mean(repo_before)) if repo_before.size else 0.0,
            "repo_balance_after_mean": float(np.mean(repo_after)) if repo_after.size else 0.0,
            "repo_balance_improve_mean": float(np.mean(repo_improve)) if repo_improve.size else 0.0,
            "repo_cost_mean": float(np.mean(repo_cost)) if repo_cost.size else 0.0,
            "local_reward_mean": float(np.mean(local_reward)) if local_reward.size else 0.0,
            "global_reward": global_reward,
            "critic_reward_mean": float(np.mean(critic_rewards)) if critic_rewards.size else 0.0,
            "actor_reward_mean": float(np.mean(actor_rewards)) if actor_rewards.size else 0.0,
        }
        return critic_rewards, actor_rewards, debug

    def _match_reward_components(self) -> tuple[np.ndarray, np.ndarray]:
        order_denominator = np.maximum(
            np.maximum(self._step_order_requests, self._step_matched_orders),
            1.0,
        ).astype(np.float32)

        matched_total = float(np.sum(self._step_matched_orders))
        revenue_total = float(np.sum(self._step_service_revenue))
        fallback_distance = max(float(self.config.cell_width_km), 1.0)
        fallback_fare = self._trip_fare(
            fallback_distance,
            self._distance_to_travel_minutes(fallback_distance),
        )
        fare_scale = revenue_total / matched_total if matched_total > 0.0 else fallback_fare
        fare_scale = max(self._finite_scalar(fare_scale, default=fallback_fare, nonnegative=True), 1.0)

        distance_cost_weight = self._reward_param(
            "reward_match_distance_cost_weight",
            DEFAULT_REWARD_DISTANCE_COST_WEIGHT,
        )
        time_cost_weight = self._reward_param(
            "reward_match_time_cost_weight",
            DEFAULT_REWARD_TIME_COST_WEIGHT,
        )
        travel_cost = distance_cost_weight * self._step_service_distance + time_cost_weight * self._step_service_minutes
        match_profit = self._step_service_revenue - travel_cost
        match_profit = np.nan_to_num(
            match_profit,
            nan=0.0,
            posinf=REWARD_CLIP_VALUE,
            neginf=-REWARD_CLIP_VALUE,
        ).astype(np.float32)
        gmv_scale = np.maximum(order_denominator * fare_scale, 1.0).astype(np.float32)
        match_profit_norm = match_profit / gmv_scale
        match_profit_norm = np.nan_to_num(
            match_profit_norm,
            nan=0.0,
            posinf=REWARD_CLIP_VALUE,
            neginf=-REWARD_CLIP_VALUE,
        ).astype(np.float32)
        return match_profit, match_profit_norm

    def _repo_balance_reward_components(self, supply_after: np.ndarray, demand_after: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        supply_after = self._sanitize_feature_vector(supply_after, nonnegative=True)
        demand_after = self._sanitize_feature_vector(demand_after, nonnegative=True)
        repo_after = self._neighbor_balance_mean(supply_after, demand_after)
        if (
            self._repo_reward_supply_before is not None
            and self._repo_reward_demand_before is not None
            and self._repo_reward_supply_before.shape == supply_after.shape
            and self._repo_reward_demand_before.shape == demand_after.shape
        ):
            repo_before = self._neighbor_balance_mean(
                self._repo_reward_supply_before,
                self._repo_reward_demand_before,
            )
        else:
            repo_before = repo_after.copy()
        repo_improve = repo_after - repo_before
        repo_improve = np.nan_to_num(repo_improve, nan=0.0, posinf=REWARD_CLIP_VALUE, neginf=-REWARD_CLIP_VALUE)
        return repo_before.astype(np.float32), repo_after.astype(np.float32), repo_improve.astype(np.float32)

    def _repo_cost_component(self) -> np.ndarray:
        repo_cost_scale = max(float(getattr(self.config, "step_minutes", 1)), 1.0)
        repo_cost = self._step_relocation_cost / repo_cost_scale
        return np.nan_to_num(repo_cost, nan=0.0, posinf=REWARD_CLIP_VALUE, neginf=0.0).astype(np.float32)

    def _actor_rewards_from_local_reward(self, local_reward: np.ndarray, global_reward: float) -> np.ndarray:
        actor_rewards = np.zeros((self.grid_number, self.action_dim), dtype=np.float32)
        action_move_cost = self._normalized_action_move_costs()
        invalid_penalty = -min(self._finite_scalar(self.config.invalid_action_penalty, default=REWARD_CLIP_VALUE, nonnegative=True), REWARD_CLIP_VALUE)
        for src in range(self.grid_number):
            for action in range(self.action_dim):
                dst = int(self.grid.neighbors[src, action])
                if dst < 0:
                    actor_rewards[src, action] = invalid_penalty
                elif action == 0:
                    actor_rewards[src, action] = 0.8 * float(local_reward[src]) + 0.2 * global_reward
                else:
                    actor_rewards[src, action] = (
                        0.6 * float(local_reward[dst])
                        + 0.2 * float(local_reward[src])
                        + 0.2 * global_reward
                        - float(action_move_cost[src, action])
                    )
        actor_rewards = np.nan_to_num(
            actor_rewards,
            nan=0.0,
            posinf=REWARD_CLIP_VALUE,
            neginf=-REWARD_CLIP_VALUE,
        )
        return np.clip(actor_rewards, -REWARD_CLIP_VALUE, REWARD_CLIP_VALUE).astype(np.float32)

    def _normalized_action_move_costs(self) -> np.ndarray:
        action_costs = np.asarray(self.action_costs, dtype=np.float32)
        finite_move_costs = action_costs[:, 1:][np.isfinite(action_costs[:, 1:]) & (action_costs[:, 1:] > 0.0)]
        max_cost = float(np.max(finite_move_costs)) if finite_move_costs.size else 1.0
        max_cost = max(self._finite_scalar(max_cost, default=1.0, nonnegative=True), 1.0)
        move_weight = self._reward_param("reward_action_move_cost", DEFAULT_REWARD_ACTION_MOVE_COST)
        normalized = np.nan_to_num(action_costs / max_cost, nan=0.0, posinf=0.0, neginf=0.0)
        normalized[:, 0] = 0.0
        return (move_weight * np.maximum(normalized, 0.0)).astype(np.float32)

    def _capture_reposition_reward_before(self) -> None:
        idle = self._sanitize_feature_vector(
            np.asarray([len(x) for x in self.idle_by_cell], dtype=np.float32),
            nonnegative=True,
        )
        incoming = self._sanitize_feature_vector(self._incoming_supply_by_cell(), nonnegative=True)
        supply = self._reward_supply_from_components(idle, incoming)
        demand = self._repo_reward_demand_from_components(
            self._step_decision_demand,
            self._future_demand_by_cell(),
        )
        self._repo_reward_supply_before = supply.copy()
        self._repo_reward_demand_before = demand.copy()

    def _repo_reward_demand_for_update(self, current_demand: np.ndarray, future_demand: np.ndarray) -> np.ndarray:
        if self._repo_reward_demand_before is not None and self._repo_reward_demand_before.shape == current_demand.shape:
            return self._sanitize_feature_vector(self._repo_reward_demand_before, nonnegative=True)
        decision_demand = self._sanitize_feature_vector(self._step_decision_demand, nonnegative=True)
        if not np.any(decision_demand > 0.0):
            decision_demand = self._sanitize_feature_vector(current_demand, nonnegative=True)
        return self._repo_reward_demand_from_components(decision_demand, future_demand)

    def _repo_reward_demand_from_components(
        self,
        decision_demand: np.ndarray,
        future_demand: np.ndarray,
    ) -> np.ndarray:
        decision = self._sanitize_feature_vector(decision_demand, nonnegative=True)
        future = self._sanitize_feature_vector(future_demand, nonnegative=True)
        future_weight = self._reward_param(
            "reward_repo_future_demand_weight",
            DEFAULT_REWARD_REPO_FUTURE_DEMAND_WEIGHT,
        )
        return self._sanitize_feature_vector(decision + future_weight * future, nonnegative=True)

    def _reward_supply_from_components(self, idle: np.ndarray, incoming: np.ndarray) -> np.ndarray:
        incoming_discount = self._reward_param("reward_incoming_discount", DEFAULT_REWARD_INCOMING_DISCOUNT)
        supply = idle + incoming_discount * incoming
        return self._sanitize_feature_vector(supply, nonnegative=True)

    def _neighbor_balance_mean(self, supply: np.ndarray, demand: np.ndarray) -> np.ndarray:
        balance = self._balance_degree(supply, demand)
        neighborhood_balance = np.zeros(self.grid_number, dtype=np.float32)
        for cell in range(self.grid_number):
            values: list[float] = []
            for dst in self.grid.neighbors[cell]:
                dst = int(dst)
                if 0 <= dst < self.grid_number:
                    values.append(float(balance[dst]))
            neighborhood_balance[cell] = float(np.mean(values)) if values else 0.0
        return np.nan_to_num(neighborhood_balance, nan=0.0, posinf=REWARD_CLIP_VALUE, neginf=0.0).astype(np.float32)

    def _balance_degree(self, supply: np.ndarray, demand: np.ndarray) -> np.ndarray:
        supply = self._sanitize_feature_vector(supply, nonnegative=True)
        demand = self._sanitize_feature_vector(demand, nonnegative=True)
        denominator = np.maximum(np.maximum(supply, demand), 1.0)
        balance = np.minimum(supply, demand) / denominator
        return np.nan_to_num(balance, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    def _reward_param(self, name: str, default: float) -> float:
        return self._finite_scalar(getattr(self.config, name, default), default=default, nonnegative=True)

    def _state_feature_clip(self) -> float:
        return self._finite_scalar(self.config.state_feature_clip, default=1.0e6, nonnegative=True)

    def _quality_wait_penalty_minutes(self, response_minutes: float) -> float:
        response = self._finite_scalar(response_minutes, default=0.0, nonnegative=True)
        tolerance = self._finite_scalar(
            getattr(self.config, "wait_time_tolerance_minutes", 0.0),
            default=0.0,
            nonnegative=True,
        )
        excess = max(response - tolerance, 0.0)
        exponent = self._finite_scalar(
            getattr(self.config, "wait_time_penalty_exponent", 1.0),
            default=1.0,
            nonnegative=True,
        )
        if excess <= 0.0:
            return 0.0
        if exponent <= 1.0 + 1e-9:
            return excess
        scale = max(tolerance, float(self.config.step_minutes), 1.0)
        return float(excess * (excess / scale) ** (exponent - 1.0))

    def _trip_fare(self, trip_distance_km: float, trip_minutes: float) -> float:
        distance = self._finite_scalar(trip_distance_km, default=0.0, nonnegative=True)
        minutes = self._finite_scalar(trip_minutes, default=0.0, nonnegative=True)
        base = self._finite_scalar(self.config.fare_base, default=0.0, nonnegative=True)
        per_km = self._finite_scalar(self.config.fare_per_km, default=0.0, nonnegative=True)
        per_minute = self._finite_scalar(self.config.fare_per_minute, default=0.0, nonnegative=True)
        return float(base + per_km * distance + per_minute * minutes)

    def _finite_scalar(self, value: float, default: float = 0.0, nonnegative: bool = False) -> float:
        result = float(value)
        if not np.isfinite(result):
            result = float(default)
        if nonnegative:
            result = max(result, 0.0)
        return result

    def _sanitize_feature_vector(self, values: object, nonnegative: bool = False) -> np.ndarray:
        clip = self._state_feature_clip()
        arr = np.asarray(values, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=clip, neginf=-clip)
        arr = np.clip(arr, -clip, clip).astype(np.float32)
        if nonnegative:
            arr = np.maximum(arr, 0.0).astype(np.float32)
        return arr

    def _sanitize_feature_matrix(self, values: object, clip: float | None = None) -> np.ndarray:
        limit = self._state_feature_clip() if clip is None else max(float(clip), 1.0)
        arr = np.asarray(values, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=limit, neginf=-limit)
        return np.clip(arr, -limit, limit).astype(np.float32)

    def _incoming_supply_by_cell(self, include_reposition: bool = True) -> np.ndarray:
        supply = np.zeros(self.grid_number, dtype=np.float32)
        if not self.busy_arrivals:
            return supply

        start_minute = self._step_start_minute()
        end_minute = start_minute + max(int(self.config.step_minutes), 1)
        for minute in range(start_minute, min(end_minute, len(self.busy_arrivals))):
            for _taxi_id, cell, _duration in self.busy_arrivals[minute]:
                if 0 <= cell < self.grid_number:
                    supply[cell] += 1.0
        if include_reposition:
            for minute in range(start_minute, min(end_minute, len(self.reposition_arrivals))):
                for _taxi_id, cell in self.reposition_arrivals[minute]:
                    if 0 <= cell < self.grid_number:
                        supply[cell] += 1.0
        return supply

    def _future_demand_by_cell(self) -> np.ndarray:
        rates = getattr(self.demand, "rates", None)
        if rates is None:
            return np.zeros(self.grid_number, dtype=np.float32)
        values = np.asarray(rates, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.grid_number:
            return np.zeros(self.grid_number, dtype=np.float32)
        lookahead = max(int(self.config.future_demand_steps), 0)
        if lookahead <= 0:
            return np.zeros(self.grid_number, dtype=np.float32)
        if values.shape[0] == 0:
            return np.zeros(self.grid_number, dtype=np.float32)
        start = self.step_index + 1
        end = min(start + lookahead, values.shape[0])
        if start >= values.shape[0]:
            window = np.repeat(values[-1:, :], lookahead, axis=0)
            return window.sum(axis=0).astype(np.float32)
        window = values[start:end]
        missing = lookahead - window.shape[0]
        if missing > 0:
            padding = np.repeat(values[-1:, :], missing, axis=0)
            window = np.concatenate([window, padding], axis=0)
        return window.sum(axis=0).astype(np.float32)

    @property
    def future_cell_value(self) -> np.ndarray:
        return self.region_value_guidance

    @future_cell_value.setter
    def future_cell_value(self, values: object) -> None:
        self.region_value_guidance = np.asarray(values, dtype=np.float32)

    def _state_feature_scales(self) -> np.ndarray:
        scales = np.ones(self.state_feature_dim, dtype=np.float32)
        count_features = [
            DEMAND_FEATURE,
            IDLE_SUPPLY_FEATURE,
            INCOMING_SUPPLY_FEATURE,
            FUTURE_DEMAND_FEATURE,
            SUPPLY_DEMAND_GAP_FEATURE,
            CURRENT_NEED_FEATURE,
            FUTURE_NEED_FEATURE,
            IDLE_TIME_FEATURE,
            EXECUTION_BIAS_FEATURE,
        ]
        for feature in count_features:
            scales[feature] = self.feature_scale
        scales[EMPTY_DISTANCE_FEATURE] = max(self.feature_scale * self.config.cell_width_km, 1.0)
        scales = np.nan_to_num(scales, nan=1.0, posinf=self._state_feature_clip(), neginf=1.0)
        return np.maximum(scales, 1.0e-6).astype(np.float32)

    def _safe_feature_scales(self) -> np.ndarray:
        scales = np.asarray(self.feature_scales, dtype=np.float32)
        scales = np.nan_to_num(scales, nan=1.0, posinf=self._state_feature_clip(), neginf=1.0)
        return np.maximum(scales, 1.0e-6).astype(np.float32)

    def _capture_step_decision_demand(self) -> None:
        self._step_decision_demand = self._state_raw[:, DEMAND_FEATURE].astype(np.float32).copy()

    def _accumulate_idle_feedback(self) -> None:
        self._step_idle_minutes += np.asarray([len(x) for x in self.idle_by_cell], dtype=np.float32)

    def _finalize_execution_feedback(self) -> None:
        attempted = np.maximum(self._step_decision_demand, self._step_order_requests)
        denom = np.maximum(attempted, 1.0)
        self._feedback_match_rate = (self._step_matched_orders / denom).astype(np.float32)
        self._feedback_reject_rate = (self._step_cancelled_orders / denom).astype(np.float32)
        self._feedback_idle_time = (
            self._step_idle_minutes / max(float(self.config.step_minutes), 1.0)
        ).astype(np.float32)
        self._feedback_match_rate = np.clip(self._sanitize_feature_vector(self._feedback_match_rate, nonnegative=True), 0.0, 1.0)
        self._feedback_reject_rate = np.clip(self._sanitize_feature_vector(self._feedback_reject_rate, nonnegative=True), 0.0, 1.0)
        self._feedback_idle_time = self._sanitize_feature_vector(self._feedback_idle_time, nonnegative=True)
        self._feedback_empty_distance = self._sanitize_feature_vector(self._step_empty_distance, nonnegative=True)
        self._feedback_execution_bias = self._sanitize_feature_vector(self._step_planned_dispatch - self._step_actual_dispatch)

    def _parse_policy_output(self, output: object) -> np.ndarray:
        self._reset_upper_guidance()
        region_value = None
        future_pressure = None
        future_gap = None
        dispatch_intensity = None
        actions = output
        if isinstance(output, dict):
            actions = output.get("actions", output.get("action"))
            region_value = output.get("region_value", output.get("future_value"))
            future_pressure = output.get("future_pressure", output.get("pressure"))
            future_gap = output.get("future_gap", output.get("gap"))
            dispatch_intensity = output.get("dispatch_intensity", output.get("intensity"))
        elif isinstance(output, tuple):
            if len(output) == 0:
                raise ValueError("policy output tuple must include actions")
            actions = output[0]
            if len(output) > 1:
                region_value = output[1]
            if len(output) > 2:
                future_gap = output[2]
            if len(output) > 3:
                dispatch_intensity = output[3]
        if actions is None:
            raise ValueError("policy output must include actions")
        if future_pressure is not None:
            pressure = self._coerce_cell_vector(future_pressure, "future_pressure")
            if future_gap is None:
                future_gap = pressure
            if dispatch_intensity is None:
                dispatch_intensity = np.clip(pressure, 0.0, 1.0).astype(np.float32)
        self._apply_upper_guidance(region_value, future_gap, dispatch_intensity)
        return np.asarray(actions, dtype=np.float32)

    def _apply_upper_guidance(
        self,
        region_value: object | None,
        future_gap: object | None,
        dispatch_intensity: object | None,
    ) -> None:
        if region_value is not None:
            values = self._coerce_cell_vector(region_value, "region_value")
            self.region_value_guidance = self._standardize_region_value_guidance(values)
            self.upper_guidance_active = True
        if future_gap is not None:
            gaps = self._coerce_cell_vector(future_gap, "future_gap")
            self.future_gap_guidance = np.clip(gaps, 0.0, 1.0).astype(np.float32)
            self.upper_guidance_active = True
        if dispatch_intensity is not None:
            intensity = self._coerce_cell_vector(dispatch_intensity, "dispatch_intensity")
            self.dispatch_intensity_guidance = np.clip(intensity, 0.0, 1.0).astype(np.float32)
            self.upper_guidance_active = True

    def _coerce_cell_vector(self, values: object, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape == (self.grid_number, 1):
            arr = arr[:, 0]
        if arr.shape != (self.grid_number,):
            raise ValueError(f"{name} must have shape {(self.grid_number,)}")
        arr = arr.copy()
        arr[~np.isfinite(arr)] = 0.0
        return arr.astype(np.float32)

    def _standardize_region_value_guidance(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros(self.grid_number, dtype=np.float32)
        mean = float(finite.mean())
        std = float(finite.std())
        if std < 1e-6:
            return np.zeros(self.grid_number, dtype=np.float32)
        standardized = (arr - mean) / std
        return np.clip(standardized, -3.0, 3.0).astype(np.float32)

    def _candidate_match_edge(
        self,
        order_index: int,
        taxi_id: int,
        pickup_distance: float,
        pickup_time: float,
        order: _OrderRequest,
    ) -> tuple[int, int, float] | tuple[int, int, float, float]:
        if self._matching_mode() != "value_guided" or not self.upper_guidance_active:
            cost = float(pickup_time) if np.isfinite(pickup_time) else float(pickup_distance)
            return (order_index, taxi_id, float(pickup_distance), cost)

        alpha = float(self.config.trip_time_weight)
        beta = float(self.config.resolved_region_value_weight())
        if alpha == 0.0 and beta == 0.0:
            cost = float(pickup_time) if np.isfinite(pickup_time) else float(pickup_distance)
            return (order_index, taxi_id, float(pickup_distance), cost)
        destination_value = 0.0
        if 0 <= order.destination < self.grid_number:
            destination_value = float(self.region_value_guidance[order.destination])
        source_value = 0.0
        if 0 <= taxi_id < len(self.taxi_cell):
            source_cell = int(self.taxi_cell[taxi_id])
            if 0 <= source_cell < self.grid_number:
                source_value = float(self.region_value_guidance[source_cell])
        marginal_value = destination_value - source_value
        time_scale = self._matching_time_scale_minutes()
        pickup_time_cost = self._finite_scalar(pickup_time, default=0.0, nonnegative=True) / time_scale
        trip_time = self._road_trip_minutes(order) if alpha != 0.0 else 0.0
        trip_time_cost = self._finite_scalar(trip_time, default=0.0, nonnegative=True) / time_scale
        match_cost = (
            pickup_time_cost
            + alpha * trip_time_cost
            - beta * marginal_value
        )
        return (order_index, taxi_id, float(pickup_distance), match_cost)

    def _matching_time_scale_minutes(self) -> float:
        configured = self.config.matching_time_scale_minutes
        if configured is None:
            configured = float(self.config.step_minutes)
        return max(self._finite_scalar(float(configured), default=float(self.config.step_minutes), nonnegative=True), 1.0e-6)

    def _matching_mode(self) -> str:
        mode = str(getattr(self.config, "matching_mode", "value_guided")).strip().lower().replace("-", "_")
        aliases = {
            "greed": "greedy",
            "greedy": "greedy",
            "greedy_matching": "greedy",
            "min_cost": "min_cost",
            "min_cost_matching": "min_cost",
            "mincost": "min_cost",
            "value_guided": "value_guided",
            "value_guided_matching": "value_guided",
        }
        if mode not in aliases:
            raise ValueError("matching_mode must be 'greedy', 'min_cost', or 'value_guided'")
        return aliases[mode]

    def _matching_pickup_scopes(self) -> tuple[str, ...]:
        scope = str(getattr(self.config, "pickup_scope", "origin_and_neighbor")).strip().lower().replace("-", "_")
        aliases = {
            "origin": ("origin",),
            "same_cell": ("origin",),
            "neighbor": ("neighbor",),
            "origin_and_neighbor": ("origin", "neighbor"),
            "local": ("origin", "neighbor"),
        }
        if scope not in aliases:
            raise ValueError("pickup_scope must be 'origin', 'neighbor', or 'origin_and_neighbor'")
        return aliases[scope]

    def _valid_action_probs(self, origin: int, action: np.ndarray) -> np.ndarray:
        probs = np.asarray(action, dtype=np.float32).copy()
        probs[~np.isfinite(probs)] = 0.0
        probs = np.maximum(probs, 0.0)
        probs[self.available_actions[origin] <= 0.0] = 0.0
        total = float(probs.sum())
        if total <= 0:
            probs[:] = 0.0
            probs[0] = 1.0
            return probs
        return probs / total

    def _integer_action_counts(self, unit_count: int, probs: np.ndarray) -> np.ndarray:
        """Convert action probabilities to per-taxi integer counts.

        MAMR applies one discrete action per idle driver. The learning policy
        emits aggregate cell-level proportions, so this largest-remainder
        allocation keeps the aggregate interface while avoiding floor rounding
        that can swallow all moves when a cell has only a few idle taxis.
        """

        counts = np.zeros(self.action_dim, dtype=np.int64)
        if unit_count <= 0:
            return counts

        expected = np.asarray(probs, dtype=np.float64) * int(unit_count)
        counts = np.floor(expected).astype(np.int64)
        missing = int(unit_count) - int(counts.sum())
        if missing <= 0:
            return counts

        fractional = expected - counts
        order = np.lexsort((np.arange(self.action_dim), -fractional))
        for action_idx in order[:missing]:
            counts[int(action_idx)] += 1
        return counts

    def _action_costs(self) -> np.ndarray:
        costs = np.zeros((self.grid_number, self.action_dim), dtype=np.float32)
        for origin in range(self.grid_number):
            for action, destination in enumerate(self.grid.neighbors[origin]):
                if action == 0:
                    continue
                if destination < 0:
                    costs[origin, action] = np.inf
                    continue
                costs[origin, action] = self._cell_to_cell_time(origin, int(destination))
        return costs

    def _cell_to_cell_distance(self, origin: int, destination: int) -> float:
        if self.road_distance_matrix is not None:
            distance = float(self.road_distance_matrix[origin, destination])
            if np.isfinite(distance) and (distance > 0 or origin == destination):
                return distance
        if self.road_network is not None:
            origin_node = self.cell_road_nodes[origin]
            destination_node = self.cell_road_nodes[destination]
            if origin_node is not None and destination_node is not None:
                distance = self.road_network.shortest_path_distance_km(origin_node, destination_node)
                if np.isfinite(distance) and distance > 0:
                    return float(distance)
        if self.hex_distance_matrix is not None:
            distance = float(self.hex_distance_matrix[origin, destination])
            if np.isfinite(distance) and distance > 0:
                return distance
        return self.grid.distance(origin, destination)

    def _cell_to_cell_time(self, origin: int, destination: int) -> float:
        if self.road_time_matrix is not None:
            minutes = float(self.road_time_matrix[origin, destination])
            if np.isfinite(minutes) and (minutes > 0 or origin == destination):
                return minutes
        if self.road_network is not None:
            origin_node = self.cell_road_nodes[origin]
            destination_node = self.cell_road_nodes[destination]
            if origin_node is not None and destination_node is not None:
                minutes = self.road_network.shortest_path_time_minutes(origin_node, destination_node)
                if np.isfinite(minutes) and minutes >= 0:
                    return float(minutes)
        distance = self._cell_to_cell_distance(origin, destination)
        return self._distance_to_travel_minutes(distance)

    def _cell_road_nodes(self) -> list[str | None]:
        if self.road_network is None or self.grid.projection_origin is None:
            return [None for _ in range(self.grid_number)]
        return self.road_network.nearest_grid_xy(self.grid.xy, self.grid.projection_origin)

    def _set_taxi_idle_at_cell(self, taxi_id: int, cell: int) -> None:
        self.taxi_cell[taxi_id] = cell
        xy = self._sample_position(cell)
        self.taxi_xy[taxi_id] = xy
        self.taxi_node[taxi_id] = self._road_node_for_xy(xy)
        self.idle_by_cell[cell].append(taxi_id)

    def _road_node_for_xy(self, xy: np.ndarray) -> str | None:
        if self.road_network is None or self.grid.projection_origin is None:
            return None
        return self.road_network.nearest_grid_xy(np.asarray([xy], dtype=np.float32), self.grid.projection_origin)[0]

    def _road_reposition_costs(self, taxi_ids: list[int], target_cells: list[int]) -> np.ndarray:
        if self.road_time_matrix is not None:
            costs = np.full((len(taxi_ids), len(target_cells)), np.inf, dtype=np.float32)
            for i, taxi_id in enumerate(taxi_ids):
                source_cell = int(self.taxi_cell[taxi_id])
                if source_cell < 0:
                    continue
                for j, target_cell in enumerate(target_cells):
                    costs[i, j] = float(self._cell_to_cell_time(source_cell, int(target_cell)))
            return costs
        if self.road_network is None:
            raise RuntimeError("road reposition costs requested without road_network or road_time_matrix")
        source_nodes = [self.taxi_node[taxi_id] or self.cell_road_nodes[int(self.taxi_cell[taxi_id])] for taxi_id in taxi_ids]
        target_nodes = [self.cell_road_nodes[cell] for cell in target_cells]
        if any(node is None for node in source_nodes) or any(node is None for node in target_nodes):
            return np.full((len(taxi_ids), len(target_cells)), np.inf, dtype=np.float32)
        return self.road_network.time_matrix_minutes(source_nodes, target_nodes)

    def _road_pickup_distances(self, taxi_ids: list[int], orders: list[_OrderRequest] | None = None) -> np.ndarray:
        if self.road_network is None:
            raise RuntimeError("road pickup distances requested without road_network")
        active_orders = self.waiting_orders if orders is None else orders
        order_nodes = [order.pickup_node for order in active_orders]
        taxi_nodes = [self.taxi_node[taxi_id] for taxi_id in taxi_ids]
        if any(node is None for node in order_nodes) or any(node is None for node in taxi_nodes):
            return np.full((len(order_nodes), len(taxi_ids)), np.inf, dtype=np.float32)
        return self.road_network.distance_matrix_km(taxi_nodes, order_nodes).T

    def _pickup_candidate_edges(
        self,
        active_orders: list[_OrderRequest],
        pickup_scope: str = "origin_and_neighbor",
    ) -> tuple[list[int], list[SparseCandidateEdge]]:
        if not active_orders:
            return [], []
        if self.road_time_matrix is not None:
            return self._matrix_pickup_candidate_edges(active_orders, pickup_scope=pickup_scope)
        if self.road_network is not None:
            return self._road_pickup_candidate_edges(active_orders, pickup_scope=pickup_scope)

        taxi_ids: list[int] = []
        seen_taxis: set[int] = set()
        candidate_edges: list[SparseCandidateEdge] = []
        orders_by_origin: dict[int, list[int]] = {}

        for order_index, order in enumerate(active_orders):
            if 0 <= order.origin < self.grid_number:
                orders_by_origin.setdefault(order.origin, []).append(order_index)

        for origin, order_indices in orders_by_origin.items():
            local_taxis = self._pickup_taxis(origin, pickup_scope)
            if not local_taxis:
                continue

            local_array = np.asarray(local_taxis, dtype=np.int64)
            order_points = np.asarray([active_orders[idx].pickup_xy for idx in order_indices], dtype=np.float32)
            pickup_costs = np.linalg.norm(order_points[:, None, :] - self.taxi_xy[local_array][None, :, :], axis=2)

            for order_row, taxi_col in np.ndindex(pickup_costs.shape):
                taxi_id = int(local_array[int(taxi_col)])
                cost = float(pickup_costs[int(order_row), int(taxi_col)])
                if not np.isfinite(cost):
                    continue
                order_index = order_indices[int(order_row)]
                pickup_minutes = self._distance_to_travel_minutes(cost)
                if not self._is_serviceable_pickup_candidate(
                    active_orders[order_index],
                    cost,
                    pickup_minutes,
                ):
                    continue
                if taxi_id not in seen_taxis:
                    seen_taxis.add(taxi_id)
                    taxi_ids.append(taxi_id)
                candidate_edges.append(
                    self._candidate_match_edge(order_index, taxi_id, cost, pickup_minutes, active_orders[order_index])
                )

        return taxi_ids, candidate_edges

    def _matrix_pickup_candidate_edges(
        self,
        active_orders: list[_OrderRequest],
        pickup_scope: str = "origin_and_neighbor",
    ) -> tuple[list[int], list[SparseCandidateEdge]]:
        taxi_ids: list[int] = []
        seen_taxis: set[int] = set()
        candidate_edges: list[SparseCandidateEdge] = []
        for order_index, order in enumerate(active_orders):
            local_taxis = self._pickup_taxis(order.origin, pickup_scope)
            if not local_taxis:
                continue
            for taxi_id in local_taxis:
                source_cell = int(self.taxi_cell[taxi_id])
                if source_cell < 0:
                    continue
                pickup_distance = self._pickup_distance_for_taxi(taxi_id, order)
                if not np.isfinite(pickup_distance):
                    continue
                pickup_minutes = self._road_pickup_minutes(taxi_id, order, pickup_distance)
                if not np.isfinite(pickup_minutes):
                    continue
                if not self._is_serviceable_pickup_candidate(order, pickup_distance, pickup_minutes):
                    continue
                if taxi_id not in seen_taxis:
                    seen_taxis.add(taxi_id)
                    taxi_ids.append(taxi_id)
                candidate_edges.append(
                    self._candidate_match_edge(
                        order_index,
                        taxi_id,
                        float(pickup_distance),
                        float(pickup_minutes),
                        order,
                    )
                )
        return taxi_ids, candidate_edges

    def _road_pickup_candidate_edges(
        self,
        active_orders: list[_OrderRequest],
        pickup_scope: str = "origin_and_neighbor",
    ) -> tuple[list[int], list[SparseCandidateEdge]]:
        taxi_ids: list[int] = []
        seen_taxis: set[int] = set()
        candidate_edges: list[SparseCandidateEdge] = []
        for order_index, order in enumerate(active_orders):
            local_taxis = self._pickup_taxis(order.origin, pickup_scope)
            if not local_taxis:
                continue
            order_candidate_ids = list(local_taxis)
            pickup_costs = self._road_pickup_distances(order_candidate_ids, [order])[0]
            taxi_nodes = [self.taxi_node[taxi_id] for taxi_id in order_candidate_ids]
            if order.pickup_node is not None and all(node is not None for node in taxi_nodes):
                pickup_minutes = self.road_network.time_matrix_minutes(taxi_nodes, [order.pickup_node])[:, 0]
            else:
                pickup_minutes = self._distance_array_to_travel_minutes(pickup_costs)
            for taxi_id, pickup_cost, pickup_minute in zip(order_candidate_ids, pickup_costs, pickup_minutes):
                cost = float(pickup_cost)
                if not np.isfinite(cost):
                    continue
                match_cost = float(pickup_minute)
                if not np.isfinite(match_cost):
                    match_cost = self._distance_to_travel_minutes(cost)
                source_cell = int(self.taxi_cell[taxi_id])
                if source_cell == int(order.origin):
                    cost = max(cost, self._same_cell_pickup_distance(taxi_id, order))
                    match_cost = max(match_cost, self._distance_to_travel_minutes(cost))
                if not self._is_serviceable_pickup_candidate(order, cost, match_cost):
                    continue
                if taxi_id not in seen_taxis:
                    seen_taxis.add(taxi_id)
                    taxi_ids.append(taxi_id)
                candidate_edges.append(self._candidate_match_edge(order_index, taxi_id, cost, match_cost, order))
        return taxi_ids, candidate_edges

    def _pickup_taxis(self, origin: int, pickup_scope: str) -> list[int]:
        if not (0 <= origin < self.grid_number):
            return []
        scope = str(pickup_scope).strip().lower()
        if scope == "origin":
            return list(self.idle_by_cell[origin])
        if scope == "neighbor":
            taxi_ids: list[int] = []
            for action in range(1, self.action_dim):
                cell = int(self.grid.neighbors[origin, action])
                if cell >= 0:
                    taxi_ids.extend(self.idle_by_cell[cell])
            return taxi_ids
        if scope not in {"origin_and_neighbor", "local"}:
            raise ValueError("pickup_scope must be 'origin', 'neighbor', or 'origin_and_neighbor'")
        taxi_ids: list[int] = []
        for cell in self.grid.local_cells(origin):
            taxi_ids.extend(self.idle_by_cell[cell])
        return taxi_ids

    def _road_pickup_minutes(self, taxi_id: int, order: _OrderRequest, pickup_distance: float) -> float:
        if self.road_time_matrix is not None:
            source_cell = int(self.taxi_cell[taxi_id])
            if source_cell >= 0:
                minutes = self._cell_to_cell_time(source_cell, order.origin)
                if np.isfinite(minutes):
                    if source_cell == int(order.origin):
                        return max(float(minutes), self._distance_to_travel_minutes(pickup_distance))
                    return float(minutes)
        if self.road_network is None or self.taxi_node[taxi_id] is None or order.pickup_node is None:
            return self._distance_to_travel_minutes(pickup_distance)
        minutes = self.road_network.shortest_path_time_minutes(self.taxi_node[taxi_id], order.pickup_node)
        if np.isfinite(minutes):
            source_cell = int(self.taxi_cell[taxi_id])
            if source_cell == int(order.origin):
                return max(float(minutes), self._distance_to_travel_minutes(pickup_distance))
            return float(minutes)
        return self._distance_to_travel_minutes(pickup_distance)

    def _distance_to_travel_minutes(self, distance_km: float) -> float:
        speed = self._finite_scalar(float(self.config.travel_speed_kmph), default=30.0, nonnegative=True)
        return float(distance_km) / max(speed, 1.0e-6) * 60.0

    def _distance_array_to_travel_minutes(self, distance_km: np.ndarray) -> np.ndarray:
        values = np.asarray(distance_km, dtype=np.float32)
        speed = self._finite_scalar(float(self.config.travel_speed_kmph), default=30.0, nonnegative=True)
        return values / max(speed, 1.0e-6) * 60.0

    def _road_trip_distance(self, order: _OrderRequest) -> float:
        if self.road_distance_matrix is not None:
            distance = self._cell_to_cell_distance(order.origin, order.destination)
            if np.isfinite(distance) and distance >= 0:
                return max(float(distance), self.config.cell_width_km * 0.6)
        if self.road_network is None or order.pickup_node is None:
            matrix_distance = self._cell_to_cell_distance(order.origin, order.destination)
            if np.isfinite(matrix_distance) and matrix_distance > 0:
                return max(float(matrix_distance), self.config.cell_width_km * 0.6)
            if order.dropoff_xy is not None:
                return max(float(np.linalg.norm(order.pickup_xy - order.dropoff_xy)), self.config.cell_width_km * 0.6)
            return max(self.grid.distance(order.origin, order.destination), self.config.cell_width_km * 0.6)
        dest_node = order.dropoff_node or self.cell_road_nodes[order.destination]
        if dest_node is None:
            return max(self.grid.distance(order.origin, order.destination), self.config.cell_width_km * 0.6)
        distance = self.road_network.shortest_path_distance_km(order.pickup_node, dest_node)
        if np.isfinite(distance) and distance > 0:
            return float(distance)
        return max(self.grid.distance(order.origin, order.destination), self.config.cell_width_km * 0.6)

    def _road_trip_minutes(self, order: _OrderRequest) -> float:
        if int(order.origin) == int(order.destination):
            minutes = self._empirical_trip_minutes(order)
            if np.isfinite(minutes) and minutes > 0:
                return float(minutes)
            return self._same_cell_trip_minutes_fallback(order)
        if self.road_time_matrix is not None:
            minutes = self._cell_to_cell_time(order.origin, order.destination)
            if np.isfinite(minutes) and minutes >= 0:
                return max(float(minutes), 1.0)
        if self.road_network is not None and order.pickup_node is not None:
            dest_node = order.dropoff_node or self.cell_road_nodes[order.destination]
            if dest_node is not None:
                minutes = self.road_network.shortest_path_time_minutes(order.pickup_node, dest_node)
                if np.isfinite(minutes) and minutes > 0:
                    return float(minutes)
        minutes = self._empirical_trip_minutes(order)
        if np.isfinite(minutes) and minutes > 0:
            return float(minutes)
        return float(self._sample_trip_minutes(order.origin, np.asarray([order.destination], dtype=np.int64))[0])

    def _empirical_trip_minutes(self, order: _OrderRequest) -> float:
        if order.trip_minutes is not None and np.isfinite(order.trip_minutes) and order.trip_minutes > 0:
            return float(order.trip_minutes)
        sampler = getattr(self.demand, "sample_trip_minutes", None)
        if callable(sampler):
            values = sampler(order.origin, self.step_index, np.asarray([order.destination], dtype=np.int64), self.rng)
            values = np.asarray(values, dtype=np.float32)
            if values.shape == (1,) and np.isfinite(values[0]) and values[0] > 0:
                return float(np.maximum(values[0], 1.0))
        return float("nan")

    def _same_cell_trip_minutes_fallback(self, order: _OrderRequest) -> float:
        if order.dropoff_xy is not None:
            distance = float(np.linalg.norm(order.pickup_xy - order.dropoff_xy))
            if np.isfinite(distance) and distance > 0:
                return max(self._distance_to_travel_minutes(distance), 1.0)
        distance = max(float(self.config.cell_width_km) * 0.6, 0.0)
        return max(self._distance_to_travel_minutes(distance), 1.0)

    def _sample_position(self, cell: int) -> np.ndarray:
        jitter = self.rng.normal(0.0, self.config.cell_width_km * 0.12, size=2)
        return (self.grid.xy[cell] + jitter).astype(np.float32)

    def _sample_trip_minutes(self, origin: int, destinations: np.ndarray) -> np.ndarray:
        if self.road_time_matrix is not None:
            dest = np.asarray(destinations, dtype=np.int64)
            values = self.road_time_matrix[int(origin), dest].astype(np.float32)
            fallback = np.asarray(
                [
                    self._same_cell_sample_trip_minutes_fallback()
                    if int(origin) == int(dest_cell)
                    else self._distance_to_travel_minutes(self._cell_to_cell_distance(int(origin), int(dest_cell)))
                    for dest_cell in dest
                ],
                dtype=np.float32,
            )
            same_cell = dest == int(origin)
            sampler = getattr(self.demand, "sample_trip_minutes", None)
            if callable(sampler) and np.any(same_cell):
                empirical = sampler(origin, self.step_index, dest[same_cell], self.rng)
                empirical = np.asarray(empirical, dtype=np.float32)
                if empirical.shape == values[same_cell].shape:
                    valid = np.isfinite(empirical) & (empirical > 0)
                    same_indices = np.flatnonzero(same_cell)
                    values[same_indices[valid]] = empirical[valid]
            mask = (~np.isfinite(values)) | (values < 0) | (same_cell & (values <= 0))
            if mask.any():
                values[mask] = fallback[mask]
            return np.maximum(values, 1.0).astype(np.float32)

        sampler = getattr(self.demand, "sample_trip_minutes", None)
        if callable(sampler):
            values = sampler(origin, self.step_index, destinations, self.rng)
            values = np.asarray(values, dtype=np.float32)
            if values.shape == destinations.shape:
                return np.maximum(values, 1.0)

        fallback: list[float] = []
        for dest_cell in destinations:
            trip_distance = max(self.grid.distance(origin, int(dest_cell)), self.config.cell_width_km * 0.6)
            fallback.append(7.0 + 3.2 * trip_distance + float(self.rng.gamma(shape=1.5, scale=1.5)))
        return np.asarray(fallback, dtype=np.float32)

    def _same_cell_sample_trip_minutes_fallback(self) -> float:
        distance = max(float(self.config.cell_width_km) * 0.6, 0.0)
        return max(self._distance_to_travel_minutes(distance), 1.0)

    def _ensure_event_slot(self, minute: int) -> None:
        if minute < len(self.busy_arrivals):
            return
        missing = minute + 1 - len(self.busy_arrivals)
        self.busy_arrivals.extend([] for _ in range(missing))
        self.reposition_arrivals.extend([] for _ in range(missing))

    def _normalize_initial_distribution(self, distribution: np.ndarray | None) -> np.ndarray | None:
        if distribution is None:
            return None
        weights = np.asarray(distribution, dtype=np.float64)
        if weights.shape != (self.grid.num_cells,):
            raise ValueError(f"initial_taxi_distribution must have shape {(self.grid.num_cells,)}")
        weights = np.maximum(weights, 0.0)
        total = float(weights.sum())
        if total <= 0:
            return None
        return (weights / total).astype(np.float64)

    def _normalize_hex_distance_matrix(self, matrix: np.ndarray | None) -> np.ndarray | None:
        if matrix is None:
            return None
        distances = np.asarray(matrix, dtype=np.float32)
        expected_shape = (self.grid.num_cells, self.grid.num_cells)
        if distances.shape != expected_shape:
            raise ValueError(f"hex_distance_matrix must have shape {expected_shape}")
        distances = distances.copy()
        distances[~np.isfinite(distances)] = 0.0
        distances = np.maximum(distances, 0.0)
        np.fill_diagonal(distances, 0.0)
        return distances

    def _normalize_road_cost_matrix(self, matrix: np.ndarray | None, name: str) -> np.ndarray | None:
        if matrix is None:
            return None
        costs = np.asarray(matrix, dtype=np.float32)
        expected_shape = (self.grid.num_cells, self.grid.num_cells)
        if costs.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        costs = costs.copy()
        costs[costs < 0] = np.inf
        np.fill_diagonal(costs, 0.0)
        return costs

    def save_cell_cancellations(self, path: str | Path) -> None:
        path = Path(path)
        rates = self.cancellation_rate_by_cell()
        response_rates = self.response_rate_by_cell()
        lines = ["cell,q,r,x,y,orders,served_orders,cancellations,response_rate,cancellation_rate"]
        for i, ((q, r), xy) in enumerate(zip(self.grid.coords, self.grid.xy)):
            lines.append(
                f"{i},{q},{r},{xy[0]:.4f},{xy[1]:.4f},{self.cell_orders[i]},"
                f"{self.cell_served_orders[i]},{self.cell_cancellations[i]},"
                f"{response_rates[i]:.6f},{rates[i]:.6f}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _event_value(event: object, key: str, default: object = None) -> object:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)
