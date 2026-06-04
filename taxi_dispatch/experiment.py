from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .baselines import diffusion_actions, park_actions, random_actions
from .chengdu import DEFAULT_PEAK_HOTSPOT_WINDOWS_TEXT
from .config import parse_args_with_config
from .env import DemandModel, EnvConfig, EpisodeMetrics, DispatchEnv
from .grid import DEFAULT_HEX_CENTER_SPACING_KM, HexGrid
from .plotting import plot_cell_cancellations, plot_training_curves
from .road_network import RoadCostMatrices, RoadNetwork, load_road_cost_matrices
from .fv_bicoord import FVBiCoordAgent, TRAINING_ARCHITECTURES, build_road_time_adjacency


PolicyOutput = np.ndarray | dict[str, np.ndarray] | tuple[object, ...]
PolicyFn = Callable[[DispatchEnv, np.ndarray, np.ndarray], PolicyOutput]
EnvClass = type[DispatchEnv]

MAMR_TRAIN_RAW_CANDIDATES = (
    "raw_train.csv",
    "train_raw.csv",
    "trips_train.csv",
    "raw_trips_train.csv",
    "data/raw_train.csv",
    "data/train_raw.csv",
    "data/trips_train.csv",
    "data/raw_trips_train.csv",
    "raw.csv",
    "data/raw.csv",
)
MAMR_TEST_RAW_CANDIDATES = (
    "raw_test.csv",
    "test_raw.csv",
    "trips_test.csv",
    "raw_trips_test.csv",
    "data/raw_test.csv",
    "data/test_raw.csv",
    "data/trips_test.csv",
    "data/raw_trips_test.csv",
)
CHENGDU_TRAJECTORY_TRAIN_RANGE = (20161108, 20161130)
CHENGDU_TRAJECTORY_TEST_RANGE = (20161101, 20161107)
_CHECKPOINT_METRIC_NAMES = (
    "reward",
    "response_rate",
    "response_time_seconds",
    "cancellation_rate",
    "occupied_rate",
    "normalized_gmv",
    "orders",
    "served_orders",
    "cancellations",
    "actor_loss",
    "critic_loss",
    "auxiliary_loss",
    "total_loss",
    "policy_entropy",
    "entropy_loss",
    "total_grad_norm",
    "encoder_grad_norm",
    "actor_head_grad_norm",
    "critic_head_grad_norm",
    "auxiliary_head_grad_norm",
    "local_head_grad_norm",
    "global_head_grad_norm",
)


def _supply_sufficiency_action_mask_from_env(
    env: DispatchEnv,
    available_actions: np.ndarray,
    *,
    source_surplus_threshold: float = 0.0,
    target_shortage_threshold: float = 0.0,
    global_shortage_threshold: float = 0.0,
) -> np.ndarray:
    mask = np.asarray(available_actions, dtype=np.float32).copy()
    raw = np.asarray(env.regional_state_raw(), dtype=np.float32)

    idle = np.nan_to_num(
        raw[:, env.idle_supply_feature_index],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    incoming_index = getattr(env, "incoming_supply_feature_index", None)
    if incoming_index is None:
        incoming = np.zeros_like(idle)
    else:
        incoming = np.nan_to_num(
            raw[:, incoming_index],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    need = np.nan_to_num(
        raw[:, env.current_need_feature_index],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    supply = np.maximum(idle + incoming, 0.0)
    need = np.maximum(need, 0.0)
    shortage = np.maximum(need - supply, 0.0)
    surplus = np.maximum(supply - need, 0.0)

    if float(np.sum(shortage)) <= float(global_shortage_threshold):
        mask[:, :] = 0.0
        mask[:, 0] = 1.0
        return mask

    neighbors = env.grid.neighbors
    num_cells, action_dim = mask.shape
    for src in range(num_cells):
        mask[src, 0] = 1.0
        if surplus[src] <= float(source_surplus_threshold):
            mask[src, 1:] = 0.0
            continue

        for action in range(1, action_dim):
            if mask[src, action] <= 0.0:
                continue
            dst = int(neighbors[src, action])
            if dst < 0 or dst >= num_cells:
                mask[src, action] = 0.0
                continue
            if shortage[dst] <= float(target_shortage_threshold):
                mask[src, action] = 0.0

        if not np.any(mask[src] > 0.0):
            mask[src, 0] = 1.0

    return mask


def _spatiotemporal_pressure_from_shortage(
    env: DispatchEnv,
    shortage: np.ndarray,
    *,
    pressure_temperature_minutes: float,
    max_neighbor_time_minutes: float,
    pressure_clip: float,
) -> np.ndarray:
    shortage = np.maximum(np.nan_to_num(np.asarray(shortage, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    num_cells = int(shortage.shape[0])
    road_time_matrix = getattr(env, "road_time_matrix", None)
    if road_time_matrix is None:
        road_time_matrix = getattr(env, "_road_time_matrix", None)

    if road_time_matrix is not None:
        travel_time = np.asarray(road_time_matrix, dtype=np.float32)
        if travel_time.shape == (num_cells, num_cells):
            valid = np.isfinite(travel_time)
            valid &= travel_time > 0.0
            valid &= travel_time <= float(max_neighbor_time_minutes)
            kernel = np.zeros_like(travel_time, dtype=np.float32)
            kernel[valid] = np.exp(-travel_time[valid] / max(float(pressure_temperature_minutes), 1.0e-6))
            np.fill_diagonal(kernel, 0.0)
            denom = kernel.sum(axis=1) + 1.0e-6
            pressure = (kernel @ shortage) / denom
            pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(pressure, 0.0, float(pressure_clip)).astype(np.float32)

    kernel = np.zeros((num_cells, num_cells), dtype=np.float32)
    neighbors = np.asarray(env.grid.neighbors, dtype=np.int64)
    for src in range(num_cells):
        for dst in neighbors[src, 1:]:
            if 0 <= int(dst) < num_cells:
                kernel[src, int(dst)] = 1.0
    denom = kernel.sum(axis=1) + 1.0e-6
    pressure = (kernel @ shortage) / denom
    pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(pressure, 0.0, float(pressure_clip)).astype(np.float32)


def _dynamic_soft_expansion_action_mask_from_env(
    env: DispatchEnv,
    available_actions: np.ndarray,
    *,
    source_surplus_threshold: float = 0.0,
    target_shortage_threshold: float = 0.0,
    global_shortage_threshold: float = 0.0,
    pressure_strength: float = 0.2,
    pressure_temperature_minutes: float = 10.0,
    max_neighbor_time_minutes: float = 30.0,
    pressure_clip: float = 10.0,
) -> np.ndarray:
    mask = np.asarray(available_actions, dtype=np.float32).copy()
    raw = np.asarray(env.regional_state_raw(), dtype=np.float32)

    idle = np.nan_to_num(
        raw[:, env.idle_supply_feature_index],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    incoming_index = getattr(env, "incoming_supply_feature_index", None)
    if incoming_index is None:
        incoming = np.zeros_like(idle)
    else:
        incoming = np.nan_to_num(
            raw[:, incoming_index],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    need = np.nan_to_num(
        raw[:, env.current_need_feature_index],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    supply = np.maximum(idle + incoming, 0.0)
    need = np.maximum(need, 0.0)
    shortage = np.maximum(need - supply, 0.0)
    surplus = np.maximum(supply - need, 0.0)

    if float(np.sum(shortage)) <= float(global_shortage_threshold):
        mask[:, :] = 0.0
        mask[:, 0] = 1.0
        return mask

    pressure = _spatiotemporal_pressure_from_shortage(
        env,
        shortage,
        pressure_temperature_minutes=pressure_temperature_minutes,
        max_neighbor_time_minutes=max_neighbor_time_minutes,
        pressure_clip=pressure_clip,
    )
    effective_surplus = surplus + float(pressure_strength) * pressure
    effective_shortage = shortage + float(pressure_strength) * pressure

    neighbors = env.grid.neighbors
    num_cells, action_dim = mask.shape
    for src in range(num_cells):
        mask[src, 0] = 1.0
        if effective_surplus[src] <= float(source_surplus_threshold):
            mask[src, 1:] = 0.0
            continue

        for action in range(1, action_dim):
            if mask[src, action] <= 0.0:
                continue
            dst = int(neighbors[src, action])
            if dst < 0 or dst >= num_cells:
                mask[src, action] = 0.0
                continue
            if effective_shortage[dst] <= float(target_shortage_threshold):
                mask[src, action] = 0.0

        if not np.any(mask[src] > 0.0):
            mask[src, 0] = 1.0

    return mask


def _response_budget_pressure_from_env(
    env: DispatchEnv,
    *,
    pressure_temperature_minutes: float,
    pressure_max_neighbor_time_minutes: float,
    pressure_clip: float,
) -> np.ndarray:
    raw = np.asarray(env.regional_state_raw(), dtype=np.float32)
    idle = np.nan_to_num(
        raw[:, env.idle_supply_feature_index],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    incoming_index = getattr(env, "incoming_supply_feature_index", None)
    if incoming_index is None:
        incoming = np.zeros_like(idle)
    else:
        incoming = np.nan_to_num(
            raw[:, incoming_index],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    need = np.nan_to_num(
        raw[:, env.current_need_feature_index],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    supply = np.maximum(idle + incoming, 0.0)
    shortage = np.maximum(np.maximum(need, 0.0) - supply, 0.0)
    return _spatiotemporal_pressure_from_shortage(
        env,
        shortage,
        pressure_temperature_minutes=pressure_temperature_minutes,
        max_neighbor_time_minutes=pressure_max_neighbor_time_minutes,
        pressure_clip=pressure_clip,
    )


def _apply_response_protected_move_budget_from_env(
    env: DispatchEnv,
    policy_output: PolicyOutput,
    *,
    incoming_discount: float = 0.3,
    safety_buffer: float = 1.0,
    min_idle_to_move: float = 1.0,
    pressure: np.ndarray | None = None,
    pressure_strength: float = 0.0,
    ratio_temperature: float = 0.05,
    ratio_tolerance: float = 0.0,
    ratio_margin: float | None = None,
    imbalance_eps: float = 1.0,
) -> PolicyOutput:
    """Project Actor output with a soft supply-demand imbalance-ratio gate."""
    if isinstance(policy_output, dict):
        output = dict(policy_output)
        actions = np.asarray(output["actions"], dtype=np.float32).copy()
    else:
        output = None
        actions = np.asarray(policy_output, dtype=np.float32).copy()

    raw = np.asarray(env.regional_state_raw(), dtype=np.float32)
    idle = np.nan_to_num(
        raw[:, env.idle_supply_feature_index],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    incoming_index = getattr(env, "incoming_supply_feature_index", None)
    if incoming_index is None:
        incoming = np.zeros_like(idle)
    else:
        incoming = np.nan_to_num(
            raw[:, incoming_index],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    need = np.nan_to_num(
        raw[:, env.current_need_feature_index],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    idle = np.maximum(idle, 0.0)
    incoming = np.maximum(incoming, 0.0)
    need = np.maximum(need, 0.0)
    _ = (safety_buffer, min_idle_to_move, pressure, pressure_strength)

    supply = np.maximum(idle + float(incoming_discount) * incoming, 0.0)
    eps = max(float(imbalance_eps), 1.0e-6)
    imbalance_ratio = np.abs(need - supply) / (need + eps)
    imbalance_ratio = np.nan_to_num(imbalance_ratio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    neighbors = np.asarray(env.grid.neighbors, dtype=np.int64)
    ratio_gate = np.ones_like(actions, dtype=np.float32)
    ratio_gate[:, 1:] = 0.0
    valid_improvements: list[float] = []
    valid_scaled_improvements: list[float] = []
    valid_expected_fluxes: list[float] = []
    valid_gates: list[float] = []
    num_cells, action_dim = actions.shape
    temperature = max(float(ratio_temperature), 1.0e-6)
    margin = -float(ratio_tolerance) if ratio_margin is None else float(ratio_margin)
    for src in range(num_cells):
        for action in range(1, action_dim):
            if src >= neighbors.shape[0] or action >= neighbors.shape[1]:
                continue
            dst = int(neighbors[src, action])
            if dst < 0 or dst >= num_cells or env.available_actions[src, action] <= 0.0:
                continue
            raw_prob = float(
                np.nan_to_num(
                    actions[src, action],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
            if raw_prob <= 1.0e-8:
                ratio_gate[src, action] = 0.0
                continue
            expected_flux = max(float(idle[src]) * raw_prob, 0.0)
            valid_expected_fluxes.append(expected_flux)
            max_flux = float(idle[src])
            if max_flux <= 0.0:
                ratio_gate[src, action] = 0.0
                continue
            flux = float(np.clip(expected_flux, 1.0, max_flux))

            before = float(imbalance_ratio[src] + imbalance_ratio[dst])
            src_supply_after = max(float(supply[src]) - flux, 0.0)
            dst_supply_after = float(supply[dst]) + flux
            src_after = abs(float(need[src]) - src_supply_after) / (float(need[src]) + eps)
            dst_after = abs(float(need[dst]) - dst_supply_after) / (float(need[dst]) + eps)
            after = src_after + dst_after
            improvement = before - after
            pair_need = 0.5 * (float(need[src]) + float(need[dst]))
            scaled_improvement = improvement * np.sqrt(pair_need + 1.0)
            scaled = np.clip((scaled_improvement - margin) / temperature, -60.0, 60.0)
            gate = float(1.0 / (1.0 + np.exp(-scaled)))
            ratio_gate[src, action] = np.float32(gate)
            valid_improvements.append(float(improvement))
            valid_scaled_improvements.append(float(scaled_improvement))
            valid_gates.append(gate)

    move_action_rate_before_gate = float(np.clip(actions[:, 1:].sum(axis=1), 0.0, 1.0).mean())
    actions[:, 1:] *= ratio_gate[:, 1:]
    move_sum = np.clip(actions[:, 1:].sum(axis=1), 0.0, 1.0)
    actions[:, 0] = 1.0 - move_sum

    actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
    actions = np.maximum(actions, 0.0)
    actions *= env.available_actions

    row_sum = actions.sum(axis=1, keepdims=True)
    bad_rows = row_sum[:, 0] <= 0.0
    if np.any(bad_rows):
        actions[bad_rows] = 0.0
        actions[bad_rows, 0] = 1.0
        row_sum = actions.sum(axis=1, keepdims=True)
    actions = actions / np.maximum(row_sum, 1.0e-6)
    move_action_rate_after_gate = float(np.clip(actions[:, 1:].sum(axis=1), 0.0, 1.0).mean())

    debug = {
        "mean_imbalance_ratio": float(np.mean(imbalance_ratio)) if imbalance_ratio.size else 0.0,
        "mean_ratio_gate": float(np.mean(valid_gates)) if valid_gates else 0.0,
        "move_action_rate_before_gate": move_action_rate_before_gate,
        "move_action_rate_after_gate": move_action_rate_after_gate,
        "ratio_improvement_mean": float(np.mean(valid_improvements)) if valid_improvements else 0.0,
        "expected_flux_mean": float(np.mean(valid_expected_fluxes)) if valid_expected_fluxes else 0.0,
        "expected_flux_max": float(np.max(valid_expected_fluxes)) if valid_expected_fluxes else 0.0,
        "scaled_ratio_improvement_mean": (
            float(np.mean(valid_scaled_improvements)) if valid_scaled_improvements else 0.0
        ),
    }
    setattr(env, "response_budget_debug", debug)

    if output is not None:
        output["actions"] = actions.astype(np.float32)
        output["response_budget_debug"] = debug
        return output
    return actions.astype(np.float32)


def train_fv_bicoord(
    config: EnvConfig,
    episodes: int,
    hidden_dim: int,
    actor_lr: float,
    critic_lr: float,
    tau: float,
    gamma: float,
    device: str,
    seed: int,
    out_dir: Path,
    road_time_weight: float,
    temporal_window: int,
    graph_temperature: float,
    attention_heads: int,
    symmetric_adjacency: bool,
    region_value_loss_weight: float,
    future_gap_loss_weight: float,
    future_demand_loss_weight: float,
    intensity_loss_weight: float,
    actor_future_demand_weight: float = 0.0,
    actor_region_value_weight: float = 0.0,
    future_pressure_loss_weight: float | None = None,
    actor_future_pressure_weight: float | None = None,
    entropy_coef: float = 0.01,
    training_architecture: str = "full",
    park_mask_surplus_threshold: float | None = None,
    park_mask_neighbor_need_threshold: float = 0.0,
    use_local_global_heads: bool = True,
    use_future_pressure_head: bool = True,
    use_supply_sufficiency_gate: bool = False,
    source_surplus_threshold: float = 0.0,
    target_shortage_threshold: float = 0.0,
    global_shortage_threshold: float = 0.0,
    use_dynamic_soft_expansion_gate: bool = False,
    dynamic_source_surplus_threshold: float = 0.0,
    dynamic_target_shortage_threshold: float = 0.0,
    dynamic_global_shortage_threshold: float = 0.0,
    pressure_gate_strength: float = 0.2,
    pressure_temperature_minutes: float = 10.0,
    pressure_max_neighbor_time_minutes: float = 30.0,
    pressure_clip: float = 10.0,
    use_response_protected_move_budget: bool = False,
    response_budget_incoming_discount: float = 0.3,
    response_budget_safety_buffer: float = 1.0,
    response_budget_min_idle_to_move: float = 1.0,
    response_budget_pressure_strength: float = 0.2,
    local_residual_scale: float = 0.1,
    global_bias_scale: float = 0.1,
    demand: DemandModel | None = None,
    grid: HexGrid | None = None,
    initial_taxi_distribution: np.ndarray | None = None,
    road_network: RoadNetwork | None = None,
    hex_distance_matrix: np.ndarray | None = None,
    road_distance_matrix: np.ndarray | None = None,
    road_time_matrix: np.ndarray | None = None,
    env_cls: EnvClass = DispatchEnv,
    live_history_path: Path | None = None,
    live_epoch_history_path: Path | None = None,
) -> tuple[FVBiCoordAgent, list[dict[str, float]]]:
    config = _fv_bicoord_config(config)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_cls(
        config,
        demand=demand,
        grid=grid,
        initial_taxi_distribution=initial_taxi_distribution,
        road_network=road_network,
        hex_distance_matrix=hex_distance_matrix,
        road_distance_matrix=road_distance_matrix,
        road_time_matrix=road_time_matrix,
    )
    adjacency = build_road_time_adjacency(
        env.grid,
        road_time_matrix=road_time_matrix,
        hex_distance_matrix=hex_distance_matrix,
        step_minutes=env.config.step_minutes,
        temperature=graph_temperature,
        symmetric=symmetric_adjacency,
    )
    agent = FVBiCoordAgent(
        agent_n=env.grid_number,
        feature_dim=env.state_feature_dim,
        hidden_dim=hidden_dim,
        action_dim=env.action_dim,
        adjacency=adjacency,
        action_costs=env.action_costs,
        action_destinations=env.grid.neighbors,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        tau=tau,
        gamma=gamma,
        road_time_weight=road_time_weight,
        temporal_window=temporal_window,
        attention_heads=attention_heads,
        region_value_loss_weight=region_value_loss_weight,
        future_pressure_loss_weight=future_pressure_loss_weight,
        future_gap_loss_weight=future_gap_loss_weight,
        future_demand_loss_weight=future_demand_loss_weight,
        intensity_loss_weight=intensity_loss_weight,
        actor_future_pressure_weight=actor_future_pressure_weight,
        actor_future_demand_weight=actor_future_demand_weight,
        actor_region_value_weight=actor_region_value_weight,
        entropy_coef=entropy_coef,
        training_architecture=training_architecture,
        park_mask_surplus_threshold=park_mask_surplus_threshold,
        park_mask_neighbor_need_threshold=park_mask_neighbor_need_threshold,
        use_local_global_heads=use_local_global_heads,
        use_future_pressure_head=use_future_pressure_head,
        local_residual_scale=local_residual_scale,
        global_bias_scale=global_bias_scale,
        future_gap_feature_index=env.observed_gap_feature_index,
        supply_demand_gap_feature_index=env.supply_demand_gap_feature_index,
        idle_supply_feature_index=env.idle_supply_feature_index,
        current_need_feature_index=env.current_need_feature_index,
        incoming_supply_feature_index=getattr(env, "incoming_supply_feature_index", None),
        device=device,
    )

    history: list[dict[str, float]] = []
    epoch_size = _demand_episode_count(env.demand)
    if epoch_size > 1:
        print(f"fv_bicoord epoch_size={epoch_size} episodes", flush=True)
    for episode in range(episodes):
        _observations, state = env.reset(seed + episode, episode_index=episode)
        temporal_history = [_state_matrix(env, state)]
        done = False
        episode_reward = 0.0
        transitions: list[dict[str, object]] = []
        pending_transition: dict[str, object] | None = None

        while not done:
            guidance_state_matrix: np.ndarray | None = None
            guidance_sequence: np.ndarray | None = None
            guidance_future_pressure_target: np.ndarray | None = None
            guidance_region_value_target: np.ndarray | None = None
            action_state_matrix: np.ndarray | None = None
            action_sequence: np.ndarray | None = None
            action_available_actions: np.ndarray | None = None
            two_phase_decision = not bool(getattr(env.config, "reposition_before_assignment", False))

            def guidance_policy(cur_env: DispatchEnv, _obs: np.ndarray, st: np.ndarray) -> PolicyOutput:
                nonlocal guidance_state_matrix, guidance_sequence, guidance_future_pressure_target
                nonlocal guidance_region_value_target
                nonlocal action_state_matrix, action_sequence, action_available_actions
                guidance_state_matrix = _state_matrix(cur_env, st)
                guidance_sequence = _temporal_sequence(
                    temporal_history,
                    guidance_state_matrix,
                    agent.temporal_window,
                )
                guidance_available_actions = agent.available_actions_for_sequence(
                    guidance_sequence,
                    cur_env.available_actions,
                )
                guidance_future_pressure_target = _future_pressure_target_from_env(cur_env)
                guidance_region_value_target = _destination_value_targets_from_state_matrix(
                    cur_env,
                    guidance_state_matrix,
                )["region_value"]
                if not two_phase_decision:
                    action_state_matrix = guidance_state_matrix
                    action_sequence = guidance_sequence
                    action_available_actions = guidance_available_actions
                return agent.take_decision(guidance_sequence, guidance_available_actions)

            def reposition_policy(cur_env: DispatchEnv, _obs: np.ndarray, st: np.ndarray) -> PolicyOutput:
                nonlocal action_state_matrix, action_sequence, action_available_actions
                action_state_matrix = _state_matrix(cur_env, st)
                action_sequence = _temporal_sequence(
                    temporal_history,
                    action_state_matrix,
                    agent.temporal_window,
                )
                available = cur_env.available_actions.copy()
                action_available_actions = agent.available_actions_for_sequence(
                    action_sequence,
                    available,
                )
                decision = agent.take_decision(action_sequence, action_available_actions)
                if use_response_protected_move_budget:
                    pressure = (
                        _response_budget_pressure_from_env(
                            cur_env,
                            pressure_temperature_minutes=pressure_temperature_minutes,
                            pressure_max_neighbor_time_minutes=pressure_max_neighbor_time_minutes,
                            pressure_clip=pressure_clip,
                        )
                        if use_dynamic_soft_expansion_gate
                        else None
                    )
                    decision = _apply_response_protected_move_budget_from_env(
                        cur_env,
                        decision,
                        incoming_discount=response_budget_incoming_discount,
                        safety_buffer=response_budget_safety_buffer,
                        min_idle_to_move=response_budget_min_idle_to_move,
                        pressure=pressure,
                        pressure_strength=response_budget_pressure_strength,
                    )
                return decision

            _observations, state, critic_rewards, actor_rewards, done = env.advance(
                guidance_policy,
                reposition_policy=reposition_policy if two_phase_decision else None,
            )
            if (
                guidance_state_matrix is None
                or guidance_sequence is None
                or action_state_matrix is None
                or action_sequence is None
            ):
                raise RuntimeError("advance() did not call both Bi-STAR policy phases")

            if pending_transition is not None:
                pending_transition["next_sequence"] = action_sequence.copy()
                pending_transition["next_available_actions"] = (
                    action_available_actions.copy()
                    if action_available_actions is not None
                    else env.available_actions.copy()
                )
                pending_targets = dict(pending_transition.get("auxiliary_targets") or {})
                _merge_auxiliary_targets(
                    pending_targets,
                    _future_pressure_targets_from_next_state_matrix(env, action_state_matrix),
                )
                pending_transition["auxiliary_targets"] = pending_targets
                transitions.append(pending_transition)

            current_auxiliary_targets: dict[str, np.ndarray] = {}
            if guidance_region_value_target is not None:
                current_auxiliary_targets["region_value"] = guidance_region_value_target.copy()
            if guidance_future_pressure_target is not None:
                current_auxiliary_targets["future_pressure"] = guidance_future_pressure_target.copy()
            current_available_actions = (
                action_available_actions.copy()
                if action_available_actions is not None
                else env.available_actions.copy()
            )
            current_transition: dict[str, object] = {
                "sequence": action_sequence.copy(),
                "auxiliary_sequence": guidance_sequence.copy(),
                "critic_rewards": critic_rewards.copy(),
                "actor_rewards": actor_rewards.copy(),
                "actor_cell_weights": _actor_cell_weights_from_state_matrix(env, action_state_matrix),
                "available_actions": current_available_actions,
                "done": bool(done),
                "auxiliary_targets": current_auxiliary_targets,
            }
            if done:
                current_transition["next_sequence"] = action_sequence.copy()
                current_transition["next_available_actions"] = current_available_actions.copy()
                transitions.append(current_transition)
                pending_transition = None
            else:
                pending_transition = current_transition

            temporal_history.append(action_state_matrix.copy())
            episode_reward += float(np.sum(critic_rewards))

        actor_loss, critic_loss = agent.update(transitions)
        metrics = env.metrics(episode_reward)
        epoch = episode // epoch_size + 1
        row = {
            "episode": float(episode + 1),
            "epoch": float(epoch),
            "epoch_episode": float(episode % epoch_size + 1),
            "epoch_size": float(epoch_size),
            "reward": metrics.reward,
            "response_rate": metrics.response_rate,
            "response_time_seconds": metrics.response_time_seconds,
            "cancellation_rate": metrics.cancellation_rate,
            "occupied_rate": metrics.occupied_rate,
            "normalized_gmv": metrics.normalized_gmv,
            "orders": float(metrics.orders),
            "served_orders": float(metrics.served_orders),
            "cancellations": float(metrics.cancellations),
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "auxiliary_loss": float(agent.last_auxiliary_loss),
            "total_loss": float(agent.last_total_loss),
            "policy_entropy": float(agent.last_policy_entropy),
            "entropy_loss": float(agent.last_entropy_loss),
            "total_grad_norm": float(agent.last_total_grad_norm),
            "encoder_grad_norm": float(agent.last_encoder_grad_norm),
            "actor_head_grad_norm": float(agent.last_actor_head_grad_norm),
            "critic_head_grad_norm": float(agent.last_critic_head_grad_norm),
            "auxiliary_head_grad_norm": float(agent.last_auxiliary_head_grad_norm),
            "local_head_grad_norm": float(agent.last_local_head_grad_norm),
            "global_head_grad_norm": float(agent.last_global_head_grad_norm),
        }
        history.append(row)
        if live_history_path is not None:
            _write_history(history, live_history_path)
        if live_epoch_history_path is not None:
            _write_history(aggregate_training_epochs(history), live_epoch_history_path)
        print(
            f"fv_bicoord episode {episode + 1:03d}/{episodes} "
            f"reward={metrics.reward:.2f} response={metrics.response_rate:.3f} "
            f"wait={metrics.response_time_seconds:.1f}s actor_loss={actor_loss:.4f} "
            f"aux_loss={agent.last_auxiliary_loss:.4f}",
            flush=True,
        )
        if epoch_size > 1 and ((episode + 1) % epoch_size == 0 or episode + 1 == episodes):
            epoch_start_index = (epoch - 1) * epoch_size
            epoch_rows = history[epoch_start_index : episode + 1]
            epoch_row = _aggregate_training_epoch_rows(
                epoch_rows,
                epoch=int(epoch),
                episode_start=epoch_start_index + 1,
                episode_end=episode + 1,
                epoch_size=epoch_size,
            )
            partial = "" if int(epoch_row["episodes"]) == epoch_size else " partial"
            print(
                f"fv_bicoord epoch {epoch:03d}{partial} "
                f"episodes={int(epoch_row['episode_start'])}-{int(epoch_row['episode_end'])} "
                f"reward={epoch_row['reward']:.2f} response={epoch_row['response_rate']:.3f} "
                f"wait={epoch_row['response_time_seconds']:.1f}s "
                f"cancellations={epoch_row['cancellations']:.1f}",
                flush=True,
            )

    final_metadata = {
        "checkpoint_type": "final",
        "episodes": int(episodes),
        "episode": int(history[-1]["episode"]) if history else int(episodes),
        "metrics": {
            name: float(history[-1][name])
            for name in _CHECKPOINT_METRIC_NAMES
            if history and name in history[-1]
        },
    }
    agent.save(out_dir / "fv_bicoord_model", metadata=final_metadata)
    _write_checkpoint_metadata(out_dir / "fv_bicoord_final_checkpoint.json", final_metadata)
    if history:
        print(
            "fv_bicoord final checkpoint "
            f"episode={int(history[-1]['episode'])}",
            flush=True,
        )
    return agent, history


def aggregate_training_epochs(history: list[dict[str, float]]) -> list[dict[str, float]]:
    if not history:
        return []
    epochs: list[dict[str, float]] = []
    start = 0
    while start < len(history):
        epoch = int(history[start].get("epoch", start + 1))
        epoch_size = int(max(history[start].get("epoch_size", 1.0), 1.0))
        end = start
        while end < len(history) and int(history[end].get("epoch", epoch)) == epoch:
            end += 1
        rows = history[start:end]
        episode_start = int(rows[0].get("episode", start + 1))
        episode_end = int(rows[-1].get("episode", end))
        epochs.append(
            _aggregate_training_epoch_rows(
                rows,
                epoch=epoch,
                episode_start=episode_start,
                episode_end=episode_end,
                epoch_size=epoch_size,
            )
        )
        start = end
    return epochs


def _aggregate_training_epoch_rows(
    rows: list[dict[str, float]],
    *,
    epoch: int,
    episode_start: int,
    episode_end: int,
    epoch_size: int,
) -> dict[str, float]:
    total_orders = sum(max(float(row.get("orders", 0.0)), 0.0) for row in rows)
    total_served = sum(max(float(row.get("served_orders", 0.0)), 0.0) for row in rows)
    total_cancellations = sum(max(float(row.get("cancellations", 0.0)), 0.0) for row in rows)
    response_rate = total_served / total_orders if total_orders > 0.0 else 0.0
    cancellation_rate = total_cancellations / total_orders if total_orders > 0.0 else 0.0
    return {
        "epoch": float(epoch),
        "episode_start": float(episode_start),
        "episode_end": float(episode_end),
        "episodes": float(len(rows)),
        "epoch_size": float(epoch_size),
        "reward": _mean_training_value(rows, "reward"),
        "response_rate": float(response_rate),
        "response_time_seconds": _weighted_training_value(rows, "response_time_seconds", "served_orders"),
        "cancellation_rate": float(cancellation_rate),
        "occupied_rate": _mean_training_value(rows, "occupied_rate"),
        "normalized_gmv": _mean_training_value(rows, "normalized_gmv"),
        "orders": float(total_orders / max(len(rows), 1)),
        "served_orders": float(total_served / max(len(rows), 1)),
        "cancellations": float(total_cancellations / max(len(rows), 1)),
        "actor_loss": _mean_training_value(rows, "actor_loss"),
        "critic_loss": _mean_training_value(rows, "critic_loss"),
        "auxiliary_loss": _mean_training_value(rows, "auxiliary_loss"),
        "total_loss": _mean_training_value(rows, "total_loss"),
        "policy_entropy": _mean_training_value(rows, "policy_entropy"),
        "entropy_loss": _mean_training_value(rows, "entropy_loss"),
        "total_grad_norm": _mean_training_value(rows, "total_grad_norm"),
        "encoder_grad_norm": _mean_training_value(rows, "encoder_grad_norm"),
        "actor_head_grad_norm": _mean_training_value(rows, "actor_head_grad_norm"),
        "critic_head_grad_norm": _mean_training_value(rows, "critic_head_grad_norm"),
        "auxiliary_head_grad_norm": _mean_training_value(rows, "auxiliary_head_grad_norm"),
        "local_head_grad_norm": _mean_training_value(rows, "local_head_grad_norm"),
        "global_head_grad_norm": _mean_training_value(rows, "global_head_grad_norm"),
    }


def _mean_training_value(rows: list[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    if not values:
        return 0.0
    return float(np.mean(values))


def _weighted_training_value(rows: list[dict[str, float]], key: str, weight_key: str) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for row in rows:
        if key not in row:
            continue
        weight = max(float(row.get(weight_key, 0.0)), 0.0)
        value = float(row[key])
        if not np.isfinite(value) or not np.isfinite(weight):
            continue
        weighted_sum += value * weight
        total_weight += weight
    if total_weight <= 0.0:
        return _mean_training_value(rows, key)
    return float(weighted_sum / total_weight)


def _write_checkpoint_metadata(metadata_path: Path, metadata: dict[str, object]) -> None:
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_fv_bicoord(
    config: EnvConfig,
    agent: FVBiCoordAgent,
    episodes: int,
    seed: int,
    demand: DemandModel | None = None,
    grid: HexGrid | None = None,
    initial_taxi_distribution: np.ndarray | None = None,
    road_network: RoadNetwork | None = None,
    hex_distance_matrix: np.ndarray | None = None,
    road_distance_matrix: np.ndarray | None = None,
    road_time_matrix: np.ndarray | None = None,
    env_cls: EnvClass = DispatchEnv,
    use_supply_sufficiency_gate: bool = False,
    source_surplus_threshold: float = 0.0,
    target_shortage_threshold: float = 0.0,
    global_shortage_threshold: float = 0.0,
    use_dynamic_soft_expansion_gate: bool = False,
    dynamic_source_surplus_threshold: float = 0.0,
    dynamic_target_shortage_threshold: float = 0.0,
    dynamic_global_shortage_threshold: float = 0.0,
    pressure_gate_strength: float = 0.2,
    pressure_temperature_minutes: float = 10.0,
    pressure_max_neighbor_time_minutes: float = 30.0,
    pressure_clip: float = 10.0,
    use_response_protected_move_budget: bool = False,
    response_budget_incoming_discount: float = 0.3,
    response_budget_safety_buffer: float = 1.0,
    response_budget_min_idle_to_move: float = 1.0,
    response_budget_pressure_strength: float = 0.2,
) -> tuple[EpisodeMetrics, DispatchEnv]:
    config = _fv_bicoord_config(config)
    total_reward = 0.0
    total_orders = 0.0
    total_served_orders = 0.0
    total_cancellations = 0.0
    total_response_time_seconds = 0.0
    total_occupied_minutes = 0.0
    total_gmv = 0.0
    total_fleet_minutes = 0.0
    total_repositioned = 0.0
    last_env: DispatchEnv | None = None
    for episode in range(episodes):
        env = env_cls(
            config,
            demand=demand,
            grid=grid,
            initial_taxi_distribution=initial_taxi_distribution,
            road_network=road_network,
            hex_distance_matrix=hex_distance_matrix,
            road_distance_matrix=road_distance_matrix,
            road_time_matrix=road_time_matrix,
        )
        _observations, state = env.reset(seed + episode, episode_index=episode)
        temporal_history = [_state_matrix(env, state)]
        done = False
        reward = 0.0

        while not done:
            two_phase_decision = not bool(getattr(env.config, "reposition_before_assignment", False))

            def guidance_policy(cur_env: DispatchEnv, _obs: np.ndarray, st: np.ndarray) -> PolicyOutput:
                state_matrix = _state_matrix(cur_env, st)
                sequence = _temporal_sequence(temporal_history, state_matrix, agent.temporal_window)
                if not two_phase_decision:
                    temporal_history.append(state_matrix.copy())
                available_actions = agent.available_actions_for_sequence(sequence, cur_env.available_actions)
                return agent.take_decision(sequence, available_actions)

            def reposition_policy(cur_env: DispatchEnv, _obs: np.ndarray, st: np.ndarray) -> PolicyOutput:
                state_matrix = _state_matrix(cur_env, st)
                sequence = _temporal_sequence(temporal_history, state_matrix, agent.temporal_window)
                temporal_history.append(state_matrix.copy())
                available = cur_env.available_actions.copy()
                available_actions = agent.available_actions_for_sequence(sequence, available)
                decision = agent.take_decision(sequence, available_actions)
                if use_response_protected_move_budget:
                    pressure = (
                        _response_budget_pressure_from_env(
                            cur_env,
                            pressure_temperature_minutes=pressure_temperature_minutes,
                            pressure_max_neighbor_time_minutes=pressure_max_neighbor_time_minutes,
                            pressure_clip=pressure_clip,
                        )
                        if use_dynamic_soft_expansion_gate
                        else None
                    )
                    decision = _apply_response_protected_move_budget_from_env(
                        cur_env,
                        decision,
                        incoming_discount=response_budget_incoming_discount,
                        safety_buffer=response_budget_safety_buffer,
                        min_idle_to_move=response_budget_min_idle_to_move,
                        pressure=pressure,
                        pressure_strength=response_budget_pressure_strength,
                    )
                return decision

            if two_phase_decision:
                _observations, state, critic_rewards, _actor_rewards, done = env.advance(
                    guidance_policy,
                    reposition_policy=reposition_policy,
                )
            else:
                _observations, state, critic_rewards, _actor_rewards, done = env.advance(guidance_policy)
            reward += float(np.sum(critic_rewards))

        elapsed_minutes = max(env.config.horizon_steps * env.config.step_minutes, 1)
        fleet_minutes = max(env.config.fleet_size * elapsed_minutes, 1)
        total_reward += reward
        total_orders += float(env.total_orders)
        total_served_orders += float(env.served_orders)
        total_cancellations += float(env.cancelled_orders)
        total_response_time_seconds += float(env.response_time_seconds)
        total_occupied_minutes += float(env.occupied_minutes)
        total_gmv += float(env.gmv)
        total_fleet_minutes += float(fleet_minutes)
        total_repositioned += float(env.repositioned)
        last_env = env

    episode_count = max(episodes, 1)
    order_denominator = max(total_orders, 1.0)
    served_denominator = max(total_served_orders, 1.0)
    fleet_denominator = max(total_fleet_minutes, 1.0)
    aggregated = EpisodeMetrics(
        reward=float(total_reward / episode_count),
        response_rate=float(total_served_orders / order_denominator),
        response_time_seconds=float(total_response_time_seconds / served_denominator),
        cancellation_rate=float(total_cancellations / order_denominator),
        occupied_rate=float(total_occupied_minutes / fleet_denominator),
        normalized_gmv=float(total_gmv / fleet_denominator * 60.0),
        orders=float(total_orders / episode_count),
        served_orders=float(total_served_orders / episode_count),
        cancellations=float(total_cancellations / episode_count),
        repositioned=float(total_repositioned / episode_count),
    )
    return aggregated, last_env if last_env is not None else env_cls(
        config,
        demand=demand,
        grid=grid,
        initial_taxi_distribution=initial_taxi_distribution,
        road_network=road_network,
        hex_distance_matrix=hex_distance_matrix,
        road_distance_matrix=road_distance_matrix,
        road_time_matrix=road_time_matrix,
    )


def evaluate_policy(
    config: EnvConfig,
    policy: PolicyFn,
    episodes: int,
    seed: int,
    demand: DemandModel | None = None,
    grid: HexGrid | None = None,
    initial_taxi_distribution: np.ndarray | None = None,
    road_network: RoadNetwork | None = None,
    hex_distance_matrix: np.ndarray | None = None,
    road_distance_matrix: np.ndarray | None = None,
    road_time_matrix: np.ndarray | None = None,
    env_cls: EnvClass = DispatchEnv,
) -> tuple[EpisodeMetrics, DispatchEnv]:
    total_reward = 0.0
    total_orders = 0.0
    total_served_orders = 0.0
    total_cancellations = 0.0
    total_response_time_seconds = 0.0
    total_occupied_minutes = 0.0
    total_gmv = 0.0
    total_fleet_minutes = 0.0
    total_repositioned = 0.0
    last_env: DispatchEnv | None = None
    for episode in range(episodes):
        env = env_cls(
            config,
            demand=demand,
            grid=grid,
            initial_taxi_distribution=initial_taxi_distribution,
            road_network=road_network,
            hex_distance_matrix=hex_distance_matrix,
            road_distance_matrix=road_distance_matrix,
            road_time_matrix=road_time_matrix,
        )
        observations, state = env.reset(seed + episode, episode_index=episode)
        done = False
        reward = 0.0
        while not done:
            observations, state, critic_rewards, _actor_rewards, done = env.advance(policy)
            reward += float(np.sum(critic_rewards))
        elapsed_minutes = max(env.config.horizon_steps * env.config.step_minutes, 1)
        fleet_minutes = max(env.config.fleet_size * elapsed_minutes, 1)
        total_reward += reward
        total_orders += float(env.total_orders)
        total_served_orders += float(env.served_orders)
        total_cancellations += float(env.cancelled_orders)
        total_response_time_seconds += float(env.response_time_seconds)
        total_occupied_minutes += float(env.occupied_minutes)
        total_gmv += float(env.gmv)
        total_fleet_minutes += float(fleet_minutes)
        total_repositioned += float(env.repositioned)
        last_env = env

    episode_count = max(episodes, 1)
    order_denominator = max(total_orders, 1.0)
    served_denominator = max(total_served_orders, 1.0)
    fleet_denominator = max(total_fleet_minutes, 1.0)
    aggregated = EpisodeMetrics(
        reward=float(total_reward / episode_count),
        response_rate=float(total_served_orders / order_denominator),
        response_time_seconds=float(total_response_time_seconds / served_denominator),
        cancellation_rate=float(total_cancellations / order_denominator),
        occupied_rate=float(total_occupied_minutes / fleet_denominator),
        normalized_gmv=float(total_gmv / fleet_denominator * 60.0),
        orders=float(total_orders / episode_count),
        served_orders=float(total_served_orders / episode_count),
        cancellations=float(total_cancellations / episode_count),
        repositioned=float(total_repositioned / episode_count),
    )
    return aggregated, last_env if last_env is not None else env_cls(
        config,
        demand=demand,
        grid=grid,
        initial_taxi_distribution=initial_taxi_distribution,
        road_network=road_network,
        hex_distance_matrix=hex_distance_matrix,
        road_distance_matrix=road_distance_matrix,
        road_time_matrix=road_time_matrix,
    )


def run(args: argparse.Namespace) -> None:
    default_out = {
        "synthetic": "outputs/dispatch_synthetic",
        "chengdu": "outputs/dispatch_chengdu",
        "chengdu-trajectory": "outputs/dispatch_chengdu_trajectory",
        "mamr": "outputs/dispatch_mamr",
    }[args.real_demand]
    out_dir = Path(args.out or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = EnvConfig(
        num_cells=args.cells,
        cell_width_km=args.cell_width_km,
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
        matching_mode="value_guided",
    )
    demand: DemandModel | None = None
    grid: HexGrid | None = None
    initial_taxi_distribution: np.ndarray | None = None
    eval_config = config
    eval_demand: DemandModel | None = None
    eval_grid: HexGrid | None = None
    eval_initial_taxi_distribution: np.ndarray | None = None
    road_network: RoadNetwork | None = _load_road_network(
        args,
        use_default_cache=args.real_demand in {"chengdu", "chengdu-trajectory"},
    )
    hex_distance_matrix: np.ndarray | None = None
    eval_hex_distance_matrix: np.ndarray | None = None
    road_cost_matrices = _load_road_cost_matrices(args)
    road_distance_matrix = None if road_cost_matrices is None else road_cost_matrices.distance_km
    road_time_matrix = None if road_cost_matrices is None else road_cost_matrices.time_minutes
    eval_road_distance_matrix = road_distance_matrix
    eval_road_time_matrix = road_time_matrix
    env_cls: EnvClass = DispatchEnv
    if road_network is None and road_time_matrix is None and args.real_demand in {"chengdu", "chengdu-trajectory"}:
        message = "no road network loaded; using Euclidean/grid-distance fallback routing."
        if args.require_road_costs:
            raise SystemExit(
                "ERROR: " + message + " Pass --road-matrix, --road-network, --road-nodes/--road-edges, "
                "or --download-road-network."
            )
        print("WARNING: " + message)

    if args.real_demand == "mamr":
        from .mamr_data import load_mamr_preprocessed
        from .mamr_env import MAMRDispatchEnv

        env_cls = MAMRDispatchEnv
        mamr_root = Path(args.mamr_data)
        legacy_city_states = _resolve_mamr_optional_path(mamr_root, args.mamr_city_states, ())
        train_city_states = (
            _resolve_mamr_optional_path(mamr_root, args.mamr_train_city_states, ())
            or legacy_city_states
            or _resolve_mamr_optional_path(
                mamr_root,
                "",
                (
                    "city_states/city_states_train.dill",
                    "data/city_states/city_states_train.dill",
                    "city_states_train.dill",
                ),
            )
            or _resolve_mamr_optional_path(
                mamr_root,
                "",
                (
                    "city_states.dill",
                    "city_states/city_states.dill",
                    "data/city_states/city_states.dill",
                ),
            )
        )
        test_city_states = (
            _resolve_mamr_optional_path(mamr_root, args.mamr_test_city_states, ())
            or _resolve_mamr_optional_path(
                mamr_root,
                "",
                (
                    "city_states/city_states_test.dill",
                    "data/city_states/city_states_test.dill",
                    "city_states_test.dill",
                ),
            )
            or legacy_city_states
            or train_city_states
        )
        train_raw_trips = _resolve_mamr_optional_path(mamr_root, args.mamr_train_raw, ())
        test_raw_trips = _resolve_mamr_optional_path(mamr_root, args.mamr_test_raw, ())

        train_bundle = load_mamr_preprocessed(
            args.mamr_data,
            city_states_path=train_city_states,
            hex_attributes_path=args.mamr_hex_attributes or None,
            hex_distances_path=args.mamr_hex_distances or None,
            driver_distribution_path=args.mamr_driver_distribution or None,
            raw_trips_path=train_raw_trips,
            raw_trips_candidates=MAMR_TRAIN_RAW_CANDIDATES,
            fleet_size=args.taxis,
            horizon_steps=args.horizon_steps,
            step_minutes=args.step_minutes,
            max_wait_steps=args.max_wait_steps,
            seed=args.seed,
            demand_scale=args.demand_scale,
            distance_unit=args.mamr_distance_unit,
            base_config=config,
        )
        eval_bundle = load_mamr_preprocessed(
            args.mamr_data,
            city_states_path=test_city_states,
            hex_attributes_path=args.mamr_hex_attributes or None,
            hex_distances_path=args.mamr_hex_distances or None,
            driver_distribution_path=args.mamr_driver_distribution or None,
            raw_trips_path=test_raw_trips,
            raw_trips_candidates=MAMR_TEST_RAW_CANDIDATES,
            fleet_size=args.taxis,
            horizon_steps=args.horizon_steps,
            step_minutes=args.step_minutes,
            max_wait_steps=args.max_wait_steps,
            seed=args.seed,
            demand_scale=args.demand_scale,
            distance_unit=args.mamr_distance_unit,
            base_config=config,
        )
        config = train_bundle.config
        demand = train_bundle.demand
        grid = train_bundle.grid
        initial_taxi_distribution = train_bundle.initial_taxi_distribution
        hex_distance_matrix = train_bundle.hex_distance_matrix
        eval_config = eval_bundle.config
        eval_demand = eval_bundle.demand
        eval_grid = eval_bundle.grid
        eval_initial_taxi_distribution = train_bundle.initial_taxi_distribution
        eval_hex_distance_matrix = eval_bundle.hex_distance_matrix
        if road_cost_matrices is None and road_network is not None:
            road_cost_matrices = road_network.cell_cost_matrices(grid, fallback_speed_kmph=args.road_speed_kmph)
            road_distance_matrix = road_cost_matrices.distance_km
            road_time_matrix = road_cost_matrices.time_minutes
            eval_road_distance_matrix = road_distance_matrix
            eval_road_time_matrix = road_time_matrix
        print(
            f"loaded MAMR-compatible data: train_steps={config.horizon_steps} "
            f"eval_steps={eval_config.horizon_steps} step_minutes={config.step_minutes} "
            f"eval_days={_demand_episode_count(eval_demand)}"
        )
        train_metadata = getattr(demand, "metadata", None) or {}
        eval_metadata = getattr(eval_demand, "metadata", None) or {}
        _write_mamr_data_sources(
            out_dir / "mamr_data_sources.json",
            train_bundle=train_bundle,
            eval_bundle=eval_bundle,
            train_city_states=train_city_states,
            eval_city_states=test_city_states,
        )
        if train_metadata.get("source") == "mamr_raw_trips":
            print(
                f"loaded MAMR train raw events: raw_time_mode={train_metadata.get('raw_time_mode')} "
                f"raw_days={train_metadata.get('day_count')}"
            )
        else:
            print("WARNING: MAMR train demand is using aggregate city-state matrices; no train raw-trip file was found.")
        if eval_metadata.get("source") == "mamr_raw_trips":
            print(
                f"loaded MAMR eval raw events: raw_time_mode={eval_metadata.get('raw_time_mode')} "
                f"raw_days={eval_metadata.get('day_count')}"
            )
        elif eval_metadata.get("aggregate_city_states") and _demand_episode_count(eval_demand) == 1:
            print(
                "WARNING: MAMR eval demand is a single aggregate city-state day; "
                "add raw_test.csv under --mamr-data or pass --mamr-test-raw for per-day raw-event evaluation."
            )
        else:
            print("WARNING: MAMR eval demand is using aggregate city-state matrices; no eval raw-trip file was found.")

    if args.real_demand in {"chengdu", "chengdu-trajectory"}:
        from .chengdu import (
            build_chengdu_env,
            build_chengdu_trajectory_env,
            build_chengdu_trajectory_event_demand,
            resolve_trajectory_paths,
            write_grid,
        )

        bounds = _parse_bounds(args.bounds) if args.bounds else None
        eval_stats = None
        if args.real_demand == "chengdu-trajectory":
            trajectory_paths, eval_trajectory_paths, trajectory_split_info = _resolve_chengdu_trajectory_split(
                args,
                resolve_trajectory_paths,
            )
            base_env, stats = build_chengdu_trajectory_env(
                trajectory_paths=trajectory_paths,
                config=config,
                start_hour=args.start_hour,
                bounds=bounds,
                bounds_quantiles=_parse_quantiles(args.bounds_quantiles),
                bounds_samples=args.bounds_samples,
                grid_padding_km=args.grid_padding_km,
                average_by_day=args.average_by_day,
                filter_to_bounds=not args.keep_outside_bounds,
                stochastic=args.stochastic_demand,
                demand_scale=args.demand_scale,
                high_demand_min_orders_per_minute=args.min_orders_per_minute,
                peak_hotspot_multiplier=args.peak_hotspot_multiplier,
                peak_hotspot_windows=args.peak_hotspot_windows,
                peak_hotspot_cells=args.peak_hotspot_cells,
                peak_hotspot_top_cells=args.peak_hotspot_top_cells,
                road_network=road_network,
                seed=args.seed,
            )
            eval_bounds = tuple(float(x) for x in stats.bounds)
            assignment_grid = HexGrid.create_geographic_fixed(
                eval_bounds,
                cell_width_km=base_env.config.cell_width_km,
                padding_km=args.grid_padding_km,
            )
            eval_demand, eval_stats, _eval_initial_distribution = build_chengdu_trajectory_event_demand(
                eval_trajectory_paths,
                raw_file_count=len(eval_trajectory_paths),
                grid=base_env.grid,
                assignment_grid=assignment_grid,
                horizon_steps=base_env.config.horizon_steps,
                step_minutes=base_env.config.step_minutes,
                start_hour=args.start_hour,
                bounds=eval_bounds,
                bounds_quantiles=_parse_quantiles(args.bounds_quantiles),
                average_by_day=args.average_by_day,
                filter_to_bounds=not args.keep_outside_bounds,
                stochastic=args.stochastic_demand,
                demand_scale=args.demand_scale,
                high_demand_min_orders_per_minute=args.min_orders_per_minute,
                peak_hotspot_multiplier=args.peak_hotspot_multiplier,
                peak_hotspot_windows=args.peak_hotspot_windows,
                peak_hotspot_cells=args.peak_hotspot_cells,
                peak_hotspot_top_cells=args.peak_hotspot_top_cells,
            )
        else:
            base_env, stats = build_chengdu_env(
                raw_paths=args.raw,
                config=config,
                start_hour=args.start_hour,
                bounds=bounds,
                bounds_quantiles=_parse_quantiles(args.bounds_quantiles),
                bounds_samples=args.bounds_samples,
                grid_padding=args.grid_padding,
                average_by_day=args.average_by_day,
                deduplicate=args.deduplicate,
                filter_to_bounds=not args.keep_outside_bounds,
                stochastic=args.stochastic_demand,
                demand_scale=args.demand_scale,
                seed=args.seed,
                fixed_cell_width_km=args.cell_width_km if args.paper_grid else None,
                high_demand_min_orders_per_minute=args.min_orders_per_minute if args.paper_grid else 0.0,
                peak_hotspot_multiplier=args.peak_hotspot_multiplier,
                peak_hotspot_windows=args.peak_hotspot_windows,
                peak_hotspot_cells=args.peak_hotspot_cells,
                peak_hotspot_top_cells=args.peak_hotspot_top_cells,
                road_network=road_network,
            )
        config = base_env.config
        demand = base_env.demand
        grid = base_env.grid
        initial_taxi_distribution = getattr(base_env, "initial_taxi_distribution", None)
        if eval_demand is not None:
            eval_config = config
            eval_grid = grid
            eval_initial_taxi_distribution = initial_taxi_distribution
        road_network = getattr(base_env, "road_network", road_network)
        if road_cost_matrices is None and road_network is not None and grid is not None:
            road_cost_matrices = road_network.cell_cost_matrices(grid, fallback_speed_kmph=args.road_speed_kmph)
            road_distance_matrix = road_cost_matrices.distance_km
            road_time_matrix = road_cost_matrices.time_minutes
            eval_road_distance_matrix = road_distance_matrix
            eval_road_time_matrix = road_time_matrix
        write_grid(out_dir / "chengdu_grid.csv", grid)
        (out_dir / "chengdu_demand_stats.json").write_text(
            json.dumps(asdict(stats), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if eval_stats is not None:
            (out_dir / "chengdu_eval_demand_stats.json").write_text(
                json.dumps(asdict(eval_stats), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (out_dir / "chengdu_trajectory_split_stats.json").write_text(
                json.dumps(
                    {
                        **trajectory_split_info,
                        "train_source": stats.source,
                        "eval_source": eval_stats.source,
                        "train_rows_used": int(stats.rows_used),
                        "eval_rows_used": int(eval_stats.rows_used),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"loaded Chengdu trajectory demand: train_source={stats.source} "
                f"eval_source={eval_stats.source} train_files={trajectory_split_info['train_files']} "
                f"eval_files={trajectory_split_info['test_files']} eval_days={_demand_episode_count(eval_demand)}"
            )

    if args.real_demand != "mamr" and eval_demand is None:
        eval_config = config
        eval_demand = demand
        eval_grid = grid
        eval_initial_taxi_distribution = initial_taxi_distribution
        eval_hex_distance_matrix = hex_distance_matrix
        eval_road_distance_matrix = road_distance_matrix
        eval_road_time_matrix = road_time_matrix

    eval_episodes = args.eval_episodes
    if eval_episodes is None:
        eval_episodes = _demand_episode_count(eval_demand) if args.real_demand in {"mamr", "chengdu-trajectory"} else 5
    eval_episodes = max(int(eval_episodes), 1)

    methods = ["park", "random", "diffusion", "fv_bicoord"] if args.method == "all" else [args.method]
    summaries: list[dict[str, float | str]] = []
    trained_fv_agent: FVBiCoordAgent | None = None
    baseline_eval_config = _baseline_config(eval_config)
    park_eval_config = replace(
        baseline_eval_config,
        pickup_scope="origin",
        same_step_reposition_service=args.same_step_reposition_service,
        reposition_before_assignment=args.reposition_before_assignment,
    )
    random_eval_config = replace(
        baseline_eval_config,
        pickup_scope="origin",
        same_step_reposition_service=args.same_step_reposition_service,
        reposition_before_assignment=args.reposition_before_assignment,
    )
    diffusion_eval_config = replace(
        baseline_eval_config,
        pickup_scope="origin",
        same_step_reposition_service=args.same_step_reposition_service,
        reposition_before_assignment=args.reposition_before_assignment,
    )

    if "fv_bicoord" in methods:
        trained_fv_agent, fv_history = train_fv_bicoord(
            config=_fv_bicoord_config(config, matching_mode=args.fv_matching_mode),
            episodes=args.episodes,
            hidden_dim=args.hidden_dim,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            tau=args.tau,
            gamma=args.gamma,
            device=args.device,
            seed=args.seed + 70_000,
            out_dir=out_dir,
            road_time_weight=args.fv_road_time_weight,
            temporal_window=args.fv_temporal_window,
            graph_temperature=args.fv_graph_temperature,
            attention_heads=args.fv_attention_heads,
            symmetric_adjacency=args.fv_symmetric_adjacency,
            region_value_loss_weight=_region_value_loss_weight_arg(args),
            future_pressure_loss_weight=_future_pressure_loss_weight_arg(args),
            future_gap_loss_weight=args.future_gap_loss_weight,
            future_demand_loss_weight=args.future_demand_loss_weight,
            intensity_loss_weight=args.intensity_loss_weight,
            actor_future_pressure_weight=_actor_future_pressure_weight_arg(args),
            actor_future_demand_weight=args.actor_future_demand_weight,
            actor_region_value_weight=args.actor_region_value_weight,
            entropy_coef=args.fv_entropy_coef,
            training_architecture=args.fv_training_architecture,
            park_mask_surplus_threshold=args.fv_park_mask_surplus_threshold,
            park_mask_neighbor_need_threshold=args.fv_park_mask_neighbor_need_threshold,
            use_local_global_heads=args.fv_use_local_global_heads,
            use_future_pressure_head=args.fv_use_future_pressure_head,
            use_supply_sufficiency_gate=args.fv_use_supply_sufficiency_gate,
            source_surplus_threshold=args.fv_source_surplus_threshold,
            target_shortage_threshold=args.fv_target_shortage_threshold,
            global_shortage_threshold=args.fv_global_shortage_threshold,
            use_dynamic_soft_expansion_gate=args.fv_use_dynamic_soft_expansion_gate,
            dynamic_source_surplus_threshold=args.fv_dynamic_source_surplus_threshold,
            dynamic_target_shortage_threshold=args.fv_dynamic_target_shortage_threshold,
            dynamic_global_shortage_threshold=args.fv_dynamic_global_shortage_threshold,
            pressure_gate_strength=args.fv_pressure_gate_strength,
            pressure_temperature_minutes=args.fv_pressure_temperature_minutes,
            pressure_max_neighbor_time_minutes=args.fv_pressure_max_neighbor_time_minutes,
            pressure_clip=args.fv_pressure_clip,
            use_response_protected_move_budget=args.fv_use_response_protected_move_budget,
            response_budget_incoming_discount=args.fv_response_budget_incoming_discount,
            response_budget_safety_buffer=args.fv_response_budget_safety_buffer,
            response_budget_min_idle_to_move=args.fv_response_budget_min_idle_to_move,
            response_budget_pressure_strength=args.fv_response_budget_pressure_strength,
            local_residual_scale=args.fv_local_residual_scale,
            global_bias_scale=args.fv_global_bias_scale,
            demand=demand,
            grid=grid,
            initial_taxi_distribution=initial_taxi_distribution,
            road_network=road_network,
            hex_distance_matrix=hex_distance_matrix,
            road_distance_matrix=road_distance_matrix,
            road_time_matrix=road_time_matrix,
            env_cls=env_cls,
            live_history_path=out_dir / "fv_bicoord_training_history.csv",
            live_epoch_history_path=out_dir / "fv_bicoord_training_epochs.csv",
        )
        _write_history(fv_history, out_dir / "fv_bicoord_training_history.csv")
        _write_history(aggregate_training_epochs(fv_history), out_dir / "fv_bicoord_training_epochs.csv")
        if args.plots:
            plot_training_curves(fv_history, out_dir / "fv_bicoord_training_curves.png")
            plot_training_curves(
                aggregate_training_epochs(fv_history),
                out_dir / "fv_bicoord_training_epoch_curves.png",
            )

    for method in methods:
        if method == "park":
            metrics, env = evaluate_policy(
                park_eval_config,
                lambda env, obs, state: park_actions(env),
                eval_episodes,
                args.seed + 10_000,
                demand=eval_demand,
                grid=eval_grid,
                initial_taxi_distribution=eval_initial_taxi_distribution,
                road_network=road_network,
                hex_distance_matrix=eval_hex_distance_matrix,
                road_distance_matrix=eval_road_distance_matrix,
                road_time_matrix=eval_road_time_matrix,
                env_cls=env_cls,
            )
        elif method == "random":
            rng = np.random.default_rng(args.seed + 20_000)
            metrics, env = evaluate_policy(
                random_eval_config,
                lambda env, obs, state: random_actions(env, rng),
                eval_episodes,
                args.seed + 30_000,
                demand=eval_demand,
                grid=eval_grid,
                initial_taxi_distribution=eval_initial_taxi_distribution,
                road_network=road_network,
                hex_distance_matrix=eval_hex_distance_matrix,
                road_distance_matrix=eval_road_distance_matrix,
                road_time_matrix=eval_road_time_matrix,
                env_cls=env_cls,
            )
        elif method == "diffusion":
            metrics, env = evaluate_policy(
                diffusion_eval_config,
                lambda env, obs, state: diffusion_actions(env),
                eval_episodes,
                args.seed + 40_000,
                demand=eval_demand,
                grid=eval_grid,
                initial_taxi_distribution=eval_initial_taxi_distribution,
                road_network=road_network,
                hex_distance_matrix=eval_hex_distance_matrix,
                road_distance_matrix=eval_road_distance_matrix,
                road_time_matrix=eval_road_time_matrix,
                env_cls=env_cls,
            )
        elif method == "fv_bicoord":
            if trained_fv_agent is None:
                raise RuntimeError("Bi-STAR evaluation requested without a trained agent")
            metrics, env = evaluate_fv_bicoord(
                _fv_bicoord_config(eval_config, matching_mode=args.fv_matching_mode),
                trained_fv_agent,
                eval_episodes,
                args.seed + 80_000,
                demand=eval_demand,
                grid=eval_grid,
                initial_taxi_distribution=eval_initial_taxi_distribution,
                road_network=road_network,
                hex_distance_matrix=eval_hex_distance_matrix,
                road_distance_matrix=eval_road_distance_matrix,
                road_time_matrix=eval_road_time_matrix,
                env_cls=env_cls,
                use_supply_sufficiency_gate=args.fv_use_supply_sufficiency_gate,
                source_surplus_threshold=args.fv_source_surplus_threshold,
                target_shortage_threshold=args.fv_target_shortage_threshold,
                global_shortage_threshold=args.fv_global_shortage_threshold,
                use_dynamic_soft_expansion_gate=args.fv_use_dynamic_soft_expansion_gate,
                dynamic_source_surplus_threshold=args.fv_dynamic_source_surplus_threshold,
                dynamic_target_shortage_threshold=args.fv_dynamic_target_shortage_threshold,
                dynamic_global_shortage_threshold=args.fv_dynamic_global_shortage_threshold,
                pressure_gate_strength=args.fv_pressure_gate_strength,
                pressure_temperature_minutes=args.fv_pressure_temperature_minutes,
                pressure_max_neighbor_time_minutes=args.fv_pressure_max_neighbor_time_minutes,
                pressure_clip=args.fv_pressure_clip,
                use_response_protected_move_budget=args.fv_use_response_protected_move_budget,
                response_budget_incoming_discount=args.fv_response_budget_incoming_discount,
                response_budget_safety_buffer=args.fv_response_budget_safety_buffer,
                response_budget_min_idle_to_move=args.fv_response_budget_min_idle_to_move,
                response_budget_pressure_strength=args.fv_response_budget_pressure_strength,
            )
        else:
            raise ValueError(f"unknown method: {method}")

        row = {"method": method, **asdict(metrics)}
        summaries.append(row)
        print(
            f"{method:9s} response={metrics.response_rate:.3f} "
            f"wait={metrics.response_time_seconds:.1f}s occupied={metrics.occupied_rate:.3f} "
            f"Normalized GMV={metrics.normalized_gmv:.3f}"
        )
        env.save_cell_cancellations(out_dir / f"{method}_cell_cancellations.csv")
        if args.plots:
            plot_cell_cancellations(env, out_dir / f"{method}_cell_cancellations.png")

    _write_summary(summaries, out_dir / "summary.csv")
    print(f"wrote results to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a regional taxi dispatch experiment.")
    parser.add_argument("--method", choices=["all", "park", "random", "diffusion", "fv_bicoord"], default="all")
    parser.add_argument("--real-demand", choices=["synthetic", "chengdu", "chengdu-trajectory", "mamr"], default="synthetic")
    parser.add_argument(
        "--raw",
        nargs="+",
        default=["data/2016年11月成都滴滴订单数据"],
        help="Chengdu raw CSV file(s), directory, or glob pattern.",
    )
    parser.add_argument(
        "--trajectory",
        nargs="+",
        default=["data/2016年成都滴滴轨迹数据"],
        help="Chengdu trajectory CSV/tar.gz file(s), directory, or glob pattern.",
    )
    parser.add_argument(
        "--train-trajectory",
        nargs="+",
        default=None,
        help="Explicit Chengdu trajectory training file(s), directory, or glob pattern.",
    )
    parser.add_argument(
        "--test-trajectory",
        nargs="+",
        default=None,
        help="Explicit Chengdu trajectory evaluation file(s), directory, or glob pattern.",
    )
    parser.add_argument("--episodes", type=int, default=20, help="learning-policy training episodes")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--cells", type=int, default=142)
    parser.add_argument("--taxis", type=int, default=6000)
    parser.add_argument("--horizon-steps", type=int, default=108)
    parser.add_argument("--step-minutes", type=int, default=10)
    parser.add_argument("--max-wait-steps", type=int, default=1)
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
    parser.add_argument(
        "--cell-width-km",
        type=float,
        default=DEFAULT_HEX_CENTER_SPACING_KM,
        help="Adjacent hex-center spacing in km; default corresponds to average hex side length 1.22 km.",
    )
    parser.add_argument("--demand-scale", type=float, default=1.8)
    parser.add_argument("--travel-speed-kmph", dest="road_speed_kmph", type=float, default=30.0)
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
    parser.add_argument("--start-hour", type=float, default=6.0)
    parser.add_argument("--bounds", default="", help="Optional Chengdu lon_min,lat_min,lon_max,lat_max.")
    parser.add_argument("--bounds-quantiles", default="0.01,0.99", help="Robust Chengdu bounds quantiles when --bounds is absent.")
    parser.add_argument("--bounds-samples", type=int, default=200_000)
    parser.add_argument("--grid-padding", type=float, default=0.08)
    parser.add_argument("--grid-padding-km", type=float, default=2.5)
    parser.add_argument("--paper-grid", action="store_true", help="Use fixed-width high-demand cells for Chengdu order data.")
    parser.add_argument("--min-orders-per-minute", type=float, default=1.0)
    parser.add_argument(
        "--peak-hotspot-multiplier",
        type=float,
        default=1.0,
        help="Multiply orders in selected hotspot cells during peak clock-hour windows.",
    )
    parser.add_argument(
        "--peak-hotspot-windows",
        default=DEFAULT_PEAK_HOTSPOT_WINDOWS_TEXT,
        help="Comma-separated peak windows, e.g. morning_peak=7-9,midday_peak=12-14,evening_peak=17-19.",
    )
    parser.add_argument(
        "--peak-hotspot-cells",
        default="",
        help="Optional comma-separated selected-grid cell indices to boost; defaults to top-demand cells.",
    )
    parser.add_argument("--peak-hotspot-top-cells", type=int, default=10)
    parser.add_argument("--road-network", default="", help="Optional cached road graph GraphML path.")
    parser.add_argument("--road-nodes", default="", help="Optional road node CSV with node,lon,lat columns.")
    parser.add_argument("--road-edges", default="", help="Optional road edge CSV with u,v,length_m columns.")
    parser.add_argument("--download-road-network", action="store_true", help="Download OSM road graph when --road-network is absent.")
    parser.add_argument("--road-network-cache", default="data/processed/chengdu_road.graphml")
    parser.add_argument("--road-matrix", default="", help="Optional precomputed cell road matrix .npz.")
    parser.add_argument("--road-place", default="Chengdu, Sichuan, China")
    parser.add_argument("--road-network-type", default="drive")
    parser.add_argument("--road-speed-kmph", dest="road_speed_kmph", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--require-road-costs", action="store_true")
    parser.add_argument("--mamr-data", default="data/mamr_chengdu", help="Directory containing MAMR preprocessing files.")
    parser.add_argument("--mamr-city-states", default="", help="Optional explicit city_states.dill path.")
    parser.add_argument("--mamr-train-city-states", default="", help="Optional explicit MAMR train city_states.dill path.")
    parser.add_argument("--mamr-test-city-states", default="", help="Optional explicit MAMR test city_states.dill path.")
    parser.add_argument("--mamr-hex-attributes", default="", help="Optional explicit hex_bin_attributes.csv path.")
    parser.add_argument("--mamr-hex-distances", default="", help="Optional explicit hex_distances.csv path.")
    parser.add_argument("--mamr-driver-distribution", default="", help="Optional explicit driver_distribution.csv path.")
    parser.add_argument(
        "--mamr-train-raw",
        default="",
        help="Optional MAMR train raw-trip CSV path; auto-discovered from --mamr-data when omitted.",
    )
    parser.add_argument(
        "--mamr-test-raw",
        default="",
        help="Optional MAMR test raw-trip CSV path; auto-discovered from --mamr-data when omitted.",
    )
    parser.add_argument("--mamr-distance-unit", choices=["mile", "km"], default="km")
    parser.add_argument("--average-by-day", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-outside-bounds", action="store_true")
    parser.add_argument("--stochastic-demand", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--fv-temporal-window", type=int, default=3)
    parser.add_argument("--fv-road-time-weight", type=float, default=0.2)
    parser.add_argument("--fv-graph-temperature", type=float, default=10.0)
    parser.add_argument("--fv-attention-heads", type=int, default=4)
    parser.add_argument("--fv-symmetric-adjacency", action="store_true")
    parser.add_argument("--fv-training-architecture", choices=TRAINING_ARCHITECTURES, default="full")
    parser.add_argument(
        "--fv-matching-mode",
        choices=["value_guided", "min_cost", "greedy"],
        default="value_guided",
        help="Lower-layer matching mode used by the FV/Bi-STAR policy.",
    )
    parser.add_argument("--fv-use-local-global-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fv-use-future-pressure-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fv-use-supply-sufficiency-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fv-source-surplus-threshold", type=float, default=0.0)
    parser.add_argument("--fv-target-shortage-threshold", type=float, default=0.0)
    parser.add_argument("--fv-global-shortage-threshold", type=float, default=0.0)
    parser.add_argument("--fv-use-dynamic-soft-expansion-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fv-dynamic-source-surplus-threshold", type=float, default=0.0)
    parser.add_argument("--fv-dynamic-target-shortage-threshold", type=float, default=0.0)
    parser.add_argument("--fv-dynamic-global-shortage-threshold", type=float, default=0.0)
    parser.add_argument("--fv-pressure-gate-strength", type=float, default=0.2)
    parser.add_argument("--fv-pressure-temperature-minutes", type=float, default=10.0)
    parser.add_argument("--fv-pressure-max-neighbor-time-minutes", type=float, default=30.0)
    parser.add_argument("--fv-pressure-clip", type=float, default=10.0)
    parser.add_argument("--fv-use-response-protected-move-budget", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fv-response-budget-incoming-discount", type=float, default=0.3)
    parser.add_argument("--fv-response-budget-safety-buffer", type=float, default=1.0)
    parser.add_argument("--fv-response-budget-min-idle-to-move", type=float, default=1.0)
    parser.add_argument("--fv-response-budget-pressure-strength", type=float, default=0.2)
    parser.add_argument("--fv-local-residual-scale", type=float, default=0.1)
    parser.add_argument("--fv-global-bias-scale", type=float, default=0.1)
    parser.add_argument("--fv-entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--fv-park-mask-surplus-threshold",
        type=float,
        default=None,
        help="Disable stay in oversupplied cells above this scaled supply-demand gap when a neighbor has unmet demand.",
    )
    parser.add_argument("--fv-park-mask-neighbor-need-threshold", type=float, default=0.0)
    parser.add_argument("--region-value-loss-weight", type=float, default=None)
    parser.add_argument("--future-value-loss-weight", type=float, default=None, help="Deprecated alias for --region-value-loss-weight.")
    parser.add_argument("--future-pressure-loss-weight", type=float, default=None)
    parser.add_argument("--future-gap-loss-weight", type=float, default=0.0)
    parser.add_argument("--future-demand-loss-weight", type=float, default=0.0)
    parser.add_argument("--intensity-loss-weight", type=float, default=0.0)
    parser.add_argument("--actor-future-pressure-weight", type=float, default=None)
    parser.add_argument("--actor-future-demand-weight", type=float, default=0.0)
    parser.add_argument("--actor-region-value-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    run(parse_args_with_config(build_parser))


def _state_matrix(env: DispatchEnv, state: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=np.float32).reshape(env.grid_number, env.state_feature_dim)


def _actor_cell_weights_from_state_matrix(env: DispatchEnv, state_matrix: np.ndarray) -> np.ndarray:
    state = np.asarray(state_matrix, dtype=np.float32)
    if not (0 <= env.idle_supply_feature_index < state.shape[1]):
        return np.ones(env.grid_number, dtype=np.float32)
    weights = np.nan_to_num(
        state[:, env.idle_supply_feature_index],
        nan=0.0,
        posinf=1.0e6,
        neginf=0.0,
    )
    weights = np.maximum(weights, 0.0).astype(np.float32)
    if not np.any(weights > 0.0):
        return np.ones(env.grid_number, dtype=np.float32)
    return weights


def _temporal_sequence(history: list[np.ndarray], current: np.ndarray, temporal_window: int) -> np.ndarray:
    frames = [np.asarray(frame, dtype=np.float32) for frame in history]
    frames.append(np.asarray(current, dtype=np.float32))
    window = max(int(temporal_window), 1)
    if len(frames) < window:
        pad = [frames[0] for _ in range(window - len(frames))]
        frames = pad + frames
    return np.stack(frames[-window:], axis=0).astype(np.float32)


def _future_pressure_targets_from_next_state_matrix(env: DispatchEnv, next_state_matrix: np.ndarray) -> dict[str, np.ndarray]:
    next_observed_gap = np.maximum(
        np.asarray(next_state_matrix, dtype=np.float32)[:, env.observed_gap_feature_index],
        0.0,
    ).astype(np.float32)
    return {"future_pressure": next_observed_gap.copy()}


def _destination_value_targets_from_state_matrix(env: DispatchEnv, state_matrix: np.ndarray) -> dict[str, np.ndarray]:
    state = np.asarray(state_matrix, dtype=np.float32)
    fallback = _current_region_pressure_from_state_matrix(env, state)
    arrival_steps = _destination_arrival_steps(env, state)
    if arrival_steps is None:
        destination_value = fallback
    else:
        destination_value = _arrival_pressure_targets(env, fallback, arrival_steps)
    destination_value[~np.isfinite(destination_value)] = 0.0
    return {"region_value": destination_value}


def _current_region_pressure_from_state_matrix(env: DispatchEnv, state_matrix: np.ndarray) -> np.ndarray:
    observed_gap = np.maximum(state_matrix[:, env.observed_gap_feature_index], 0.0)
    future_need = np.zeros_like(observed_gap, dtype=np.float32)
    if 0 <= env.future_need_feature_index < state_matrix.shape[1]:
        future_need = np.maximum(state_matrix[:, env.future_need_feature_index], 0.0)
    return np.maximum(observed_gap, future_need).astype(np.float32)


def _destination_arrival_steps(env: DispatchEnv, state_matrix: np.ndarray) -> np.ndarray | None:
    arrival_minutes = _destination_arrival_minutes(env, state_matrix)
    if arrival_minutes is None:
        return None
    step_minutes = max(int(getattr(env.config, "step_minutes", 10)), 1)
    safe_minutes = np.where(np.isfinite(arrival_minutes), np.maximum(arrival_minutes, 0.0), 0.0)
    steps = np.ceil(safe_minutes / float(step_minutes)).astype(np.int64)
    return np.maximum(steps, 0)


def _destination_arrival_minutes(env: DispatchEnv, state_matrix: np.ndarray) -> np.ndarray | None:
    cells = int(env.grid_number)
    minutes = np.full(cells, np.nan, dtype=np.float32)
    for candidate in (
        _mean_order_trip_minutes_to_destination(env),
        _mean_reposition_minutes_to_destination(env, state_matrix),
    ):
        if candidate is None:
            continue
        values = np.asarray(candidate, dtype=np.float32)
        valid = np.isfinite(values) & (values >= 0.0) & ~np.isfinite(minutes)
        minutes[valid] = values[valid]

    exact_order_minutes = _visible_order_trip_minutes_to_destination(env)
    if exact_order_minutes is not None:
        valid = np.isfinite(exact_order_minutes) & (exact_order_minutes > 0.0)
        minutes[valid] = exact_order_minutes[valid]

    if not np.isfinite(minutes).any():
        return None
    return minutes


def _visible_order_trip_minutes_to_destination(env: DispatchEnv) -> np.ndarray | None:
    orders = getattr(env, "waiting_orders", ())
    if not orders:
        return None
    cells = int(env.grid_number)
    totals = np.zeros(cells, dtype=np.float64)
    counts = np.zeros(cells, dtype=np.float64)
    step_end = getattr(env, "_step_end_minute", None)
    cutoff_minute = (
        int(step_end())
        if callable(step_end)
        else int(getattr(env, "current_minute", 0)) + max(int(getattr(env.config, "step_minutes", 10)), 1)
    )
    for order in orders:
        destination = int(getattr(order, "destination", -1))
        if not (0 <= destination < cells):
            continue
        if int(getattr(order, "created_minute", 0)) >= cutoff_minute:
            continue
        trip_minutes = getattr(order, "trip_minutes", None)
        if trip_minutes is None or not np.isfinite(float(trip_minutes)) or float(trip_minutes) <= 0.0:
            continue
        totals[destination] += float(trip_minutes)
        counts[destination] += 1.0
    if not np.any(counts > 0.0):
        return None
    result = np.full(cells, np.nan, dtype=np.float32)
    valid = counts > 0.0
    result[valid] = (totals[valid] / counts[valid]).astype(np.float32)
    return result


def _mean_order_trip_minutes_to_destination(env: DispatchEnv) -> np.ndarray | None:
    mean_trip_minutes = getattr(env.demand, "mean_trip_minutes", None)
    od_probs = getattr(env.demand, "od_probs", None)
    rates = getattr(env.demand, "rates", None)
    if mean_trip_minutes is None or od_probs is None or rates is None:
        return None
    trip = np.asarray(mean_trip_minutes, dtype=np.float32)
    od = np.asarray(od_probs, dtype=np.float32)
    demand = np.asarray(rates, dtype=np.float32)
    cells = int(env.grid_number)
    if trip.ndim != 3 or od.ndim != 3 or demand.ndim != 2:
        return None
    if trip.shape[1:] != (cells, cells) or od.shape[1:] != (cells, cells) or demand.shape[1] != cells:
        return None
    max_step = min(trip.shape[0], od.shape[0], demand.shape[0]) - 1
    if max_step < 0:
        return None
    step = int(np.clip(int(getattr(env, "step_index", 0)), 0, max_step))
    weights = np.maximum(demand[step], 0.0)[:, None] * np.maximum(od[step], 0.0)
    valid = np.isfinite(trip[step]) & (trip[step] > 0.0) & (weights > 0.0)
    weighted_sum = np.where(valid, weights * trip[step], 0.0).sum(axis=0)
    weight_sum = np.where(valid, weights, 0.0).sum(axis=0)
    result = np.full(cells, np.nan, dtype=np.float32)
    has_valid = weight_sum > 0.0
    result[has_valid] = (weighted_sum[has_valid] / weight_sum[has_valid]).astype(np.float32)

    has_order_weight = weights.sum(axis=0) > 0.0
    missing_with_orders = has_order_weight & ~has_valid
    if np.any(missing_with_orders):
        result[missing_with_orders] = _global_mean_trip_minutes(env)
    return result if np.isfinite(result).any() else None


def _mean_reposition_minutes_to_destination(env: DispatchEnv, state_matrix: np.ndarray) -> np.ndarray | None:
    action_costs = getattr(env, "action_costs", None)
    neighbors = getattr(env.grid, "neighbors", None)
    if action_costs is None or neighbors is None:
        return None
    costs = np.asarray(action_costs, dtype=np.float32)
    destinations = np.asarray(neighbors, dtype=np.int64)
    cells = int(env.grid_number)
    if costs.shape[0] != cells or destinations.shape[0] != cells:
        return None
    if not (0 <= env.idle_supply_feature_index < state_matrix.shape[1]):
        return None
    idle_weights = np.maximum(state_matrix[:, env.idle_supply_feature_index], 0.0)
    totals = np.zeros(cells, dtype=np.float64)
    weights = np.zeros(cells, dtype=np.float64)
    action_count = min(costs.shape[1], destinations.shape[1])
    for origin in range(cells):
        origin_weight = float(idle_weights[origin])
        if origin_weight <= 0.0:
            continue
        for action in range(1, action_count):
            destination = int(destinations[origin, action])
            if not (0 <= destination < cells):
                continue
            minutes = float(costs[origin, action])
            if not np.isfinite(minutes) or minutes < 0.0:
                continue
            totals[destination] += origin_weight * minutes
            weights[destination] += origin_weight
    if not np.any(weights > 0.0):
        return None
    result = np.full(cells, np.nan, dtype=np.float32)
    valid = weights > 0.0
    result[valid] = (totals[valid] / weights[valid]).astype(np.float32)
    return result


def _arrival_pressure_targets(env: DispatchEnv, fallback: np.ndarray, arrival_steps: np.ndarray) -> np.ndarray:
    rates = getattr(env.demand, "rates", None)
    if rates is None:
        return fallback.copy()
    demand = np.asarray(rates, dtype=np.float32)
    cells = int(env.grid_number)
    if demand.ndim != 2 or demand.shape[1] != cells or demand.shape[0] == 0:
        return fallback.copy()
    scale = max(float(getattr(env, "feature_scale", 1.0)), 1.0e-6)
    current_step = int(getattr(env, "step_index", 0))
    target = np.asarray(fallback, dtype=np.float32).copy()
    for cell in range(cells):
        delta = int(arrival_steps[cell])
        if delta <= 0:
            continue
        step = min(max(current_step + delta, 0), demand.shape[0] - 1)
        pressure = float(demand[step, cell]) / scale
        if np.isfinite(pressure):
            target[cell] = max(pressure, 0.0)
    return target.astype(np.float32)


def _global_mean_trip_minutes(env: DispatchEnv) -> float:
    metadata = getattr(env.demand, "metadata", None) or {}
    value = float(metadata.get("global_mean_trip_minutes", float(getattr(env.config, "step_minutes", 10))))
    if not np.isfinite(value) or value <= 0.0:
        return float(max(int(getattr(env.config, "step_minutes", 10)), 1))
    return value


def _future_pressure_target_from_env(env: DispatchEnv) -> np.ndarray:
    future_demand = env._future_demand_by_cell()
    scale = max(float(getattr(env, "feature_scale", 1.0)), 1.0e-6)
    target = np.asarray(future_demand, dtype=np.float32) / scale
    target[~np.isfinite(target)] = 0.0
    return np.maximum(target, 0.0).astype(np.float32)


def _merge_auxiliary_targets(targets: dict[str, np.ndarray], updates: dict[str, np.ndarray]) -> None:
    for key, value in updates.items():
        incoming = np.asarray(value, dtype=np.float32)
        if key == "future_pressure" and key in targets:
            targets[key] = np.maximum(np.asarray(targets[key], dtype=np.float32), incoming).astype(np.float32)
        else:
            targets[key] = incoming.copy()


def _baseline_config(config: EnvConfig) -> EnvConfig:
    return replace(
        config,
        future_demand_steps=0,
        region_value_weight=0.0,
        future_value_weight=0.0,
        future_gap_weight=0.0,
        dispatch_intensity_weight=0.0,
        trip_time_weight=0.0,
        future_imbalance_weight=0.0,
        matching_mode="greedy",
    )


def _fv_bicoord_config(config: EnvConfig, matching_mode: str | None = None) -> EnvConfig:
    if matching_mode is None:
        matching_mode = getattr(config, "matching_mode", "value_guided")
    return replace(config, pickup_scope="origin", matching_mode=matching_mode)


def _region_value_weight_arg(args: argparse.Namespace) -> float:
    value = getattr(args, "region_value_weight", None)
    if value is not None:
        return float(value)
    legacy = getattr(args, "future_value_weight", None)
    if legacy is not None:
        return float(legacy)
    return 0.25


def _region_value_loss_weight_arg(args: argparse.Namespace) -> float:
    value = getattr(args, "region_value_loss_weight", None)
    if value is not None:
        return float(value)
    legacy = getattr(args, "future_value_loss_weight", None)
    if legacy is not None:
        return float(legacy)
    return 0.0


def _future_pressure_loss_weight_arg(args: argparse.Namespace) -> float:
    value = getattr(args, "future_pressure_loss_weight", None)
    overrides = set(getattr(args, "config_overrides", ()))
    if value is not None and (
        "future_pressure_loss_weight" in overrides
        or not {"future_gap_loss_weight", "future_demand_loss_weight"} & overrides
    ):
        return float(value)
    return max(
        float(getattr(args, "future_gap_loss_weight", 0.0)),
        float(getattr(args, "future_demand_loss_weight", 0.0)),
    )


def _actor_future_pressure_weight_arg(args: argparse.Namespace) -> float:
    value = getattr(args, "actor_future_pressure_weight", None)
    overrides = set(getattr(args, "config_overrides", ()))
    if value is not None and (
        "actor_future_pressure_weight" in overrides
        or "actor_future_demand_weight" not in overrides
    ):
        return float(value)
    return float(getattr(args, "actor_future_demand_weight", 0.0))


def _write_history(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    tmp_path.replace(path)


def _write_summary(rows: list[dict[str, float | str]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_mamr_data_sources(
    path: Path,
    *,
    train_bundle: object,
    eval_bundle: object,
    train_city_states: Path | None,
    eval_city_states: Path | None,
) -> None:
    def bundle_info(bundle: object, city_states: Path | None) -> dict[str, object]:
        demand = getattr(bundle, "demand")
        metadata = dict(getattr(demand, "metadata", None) or {})
        rates = np.asarray(getattr(demand, "rates", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        raw_path = getattr(bundle, "raw_trips_path", None)
        return {
            "city_states": str(city_states) if city_states is not None else None,
            "raw_trips": str(raw_path) if raw_path is not None else None,
            "demand_source": metadata.get("source"),
            "uses_raw_trips": metadata.get("source") == "mamr_raw_trips",
            "aggregate_city_states": bool(metadata.get("aggregate_city_states", False)),
            "raw_time_mode": metadata.get("raw_time_mode"),
            "day_count": metadata.get("day_count"),
            "episode_count": _demand_episode_count(demand),
            "orders_per_episode": float(rates.sum()) if rates.size else 0.0,
        }

    path.write_text(
        json.dumps(
            {
                "train": bundle_info(train_bundle, train_city_states),
                "eval": bundle_info(eval_bundle, eval_city_states),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _resolve_chengdu_trajectory_split(
    args: argparse.Namespace,
    resolve_trajectory_paths_fn: Callable[[object], list[Path]],
) -> tuple[list[Path], list[Path], dict[str, object]]:
    train_arg = getattr(args, "train_trajectory", None)
    test_arg = getattr(args, "test_trajectory", None)
    if train_arg is not None or test_arg is not None:
        if train_arg is None or test_arg is None:
            raise ValueError("--train-trajectory and --test-trajectory must be provided together")
        train_paths = resolve_trajectory_paths_fn(train_arg)
        test_paths = resolve_trajectory_paths_fn(test_arg)
        split_mode = "explicit"
        all_paths = [*train_paths, *test_paths]
        undated_files: list[str] = []
    else:
        all_paths = resolve_trajectory_paths_fn(getattr(args, "trajectory"))
        train_paths, test_paths, undated_files = _split_chengdu_trajectory_paths_by_date(all_paths)
        split_mode = "default_date_range"

    train_set = {str(path.resolve()) for path in train_paths}
    test_set = {str(path.resolve()) for path in test_paths}
    overlap = sorted(train_set & test_set)
    if overlap:
        raise ValueError(f"train/test trajectory inputs overlap: {overlap[:3]}")
    if not train_paths:
        raise RuntimeError(
            "no Chengdu trajectory training files found. Provide --train-trajectory or use files dated 2016-11-08..2016-11-30."
        )
    if not test_paths:
        raise RuntimeError(
            "no Chengdu trajectory evaluation files found. Provide --test-trajectory or use files dated 2016-11-01..2016-11-07."
        )

    return train_paths, test_paths, {
        "split_mode": split_mode,
        "train_date_range": list(CHENGDU_TRAJECTORY_TRAIN_RANGE),
        "test_date_range": list(CHENGDU_TRAJECTORY_TEST_RANGE),
        "all_files": len(all_paths),
        "train_files": len(train_paths),
        "test_files": len(test_paths),
        "undated_files": undated_files[:20],
        "train_paths": [str(path) for path in train_paths],
        "test_paths": [str(path) for path in test_paths],
    }


def _split_chengdu_trajectory_paths_by_date(paths: list[Path]) -> tuple[list[Path], list[Path], list[str]]:
    train_paths: list[Path] = []
    test_paths: list[Path] = []
    undated_files: list[str] = []
    train_start, train_end = CHENGDU_TRAJECTORY_TRAIN_RANGE
    test_start, test_end = CHENGDU_TRAJECTORY_TEST_RANGE
    for path in paths:
        date = _extract_chengdu_yyyymmdd(path)
        if date is None:
            undated_files.append(str(path))
            continue
        if train_start <= date <= train_end:
            train_paths.append(path)
        elif test_start <= date <= test_end:
            test_paths.append(path)
    return train_paths, test_paths, undated_files


def _extract_chengdu_yyyymmdd(path: Path) -> int | None:
    text = str(path)
    for pattern in (r"(2016)[-_年]?11[-_月]?([0-3]\d)", r"(201611[0-3]\d)"):
        match = re.search(pattern, text)
        if not match:
            continue
        if len(match.groups()) == 1:
            return int(match.group(1))
        return int(f"{match.group(1)}11{match.group(2)}")
    return None


def _resolve_mamr_optional_path(root: Path, explicit: str | Path, candidates: tuple[str, ...]) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.exists() and not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return None


def _demand_episode_count(demand: DemandModel | None) -> int:
    count = getattr(demand, "episode_count", None)
    if isinstance(count, int) and count > 0:
        return count
    events_by_day = getattr(demand, "events_by_day", None)
    if events_by_day is not None:
        try:
            return max(len(events_by_day), 1)
        except TypeError:
            return 1
    return 1


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


def _load_road_network(args: argparse.Namespace, use_default_cache: bool = True) -> RoadNetwork | None:
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
    if use_default_cache and cache.exists():
        return RoadNetwork.from_graphml(cache, default_speed_kmph=args.road_speed_kmph)
    return None


def _load_road_cost_matrices(args: argparse.Namespace) -> RoadCostMatrices | None:
    if not getattr(args, "road_matrix", ""):
        return None
    return load_road_cost_matrices(args.road_matrix)


if __name__ == "__main__":
    main()
