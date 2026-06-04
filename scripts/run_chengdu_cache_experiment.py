from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

try:
    from scripts._common import assert_same_grid, load_pickle
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from _common import assert_same_grid, load_pickle

import numpy as np
import torch

from taxi_dispatch.baselines import diffusion_actions, park_actions, random_actions
from taxi_dispatch.chengdu import (
    DEFAULT_PEAK_HOTSPOT_WINDOWS_TEXT,
    _scale_day_steps,
    apply_peak_hotspot_boost_to_demand,
)
from taxi_dispatch.config import parse_args_with_config
from taxi_dispatch.data import EmpiricalTripDemand, EventTripDemand, TabularDemand
from taxi_dispatch.experiment import (
    _baseline_config,
    _actor_future_pressure_weight_arg,
    _future_pressure_loss_weight_arg,
    _write_history,
    aggregate_training_epochs,
    evaluate_fv_bicoord,
    evaluate_policy,
    train_fv_bicoord,
)
from taxi_dispatch.fv_bicoord import TRAINING_ARCHITECTURES
from taxi_dispatch.plotting import plot_cell_cancellations, plot_training_curves
from taxi_dispatch.road_network import RoadNetwork, load_road_cost_matrices


def scale_cached_demand(demand, demand_scale: float):
    scale = max(float(demand_scale), 0.0)
    if abs(scale - 1.0) < 1e-9:
        return demand
    metadata = dict(getattr(demand, "metadata", {}) or {})
    metadata["demand_scale"] = scale
    if isinstance(demand, EventTripDemand):
        events_by_day = tuple(
            _scale_day_steps([list(step_events) for step_events in day_steps], scale)
            for day_steps in demand.events_by_day
        )
        return EventTripDemand(
            rates=np.asarray(demand.rates, dtype=np.float32) * scale,
            od_probs=np.asarray(demand.od_probs, dtype=np.float32).copy(),
            mean_trip_minutes=np.asarray(demand.mean_trip_minutes, dtype=np.float32).copy(),
            events_by_day=events_by_day,
            stochastic=demand.stochastic,
            metadata=metadata,
        )
    if isinstance(demand, EmpiricalTripDemand):
        return EmpiricalTripDemand(
            rates=np.asarray(demand.rates, dtype=np.float32) * scale,
            od_probs=np.asarray(demand.od_probs, dtype=np.float32).copy(),
            mean_trip_minutes=np.asarray(demand.mean_trip_minutes, dtype=np.float32).copy(),
            stochastic=demand.stochastic,
            metadata=metadata,
        )
    if isinstance(demand, TabularDemand):
        return TabularDemand(
            rates=np.asarray(demand.rates, dtype=np.float32) * scale,
            od_probs=np.asarray(demand.od_probs, dtype=np.float32).copy(),
            stochastic=demand.stochastic,
        )
    if hasattr(demand, "rates"):
        try:
            scaled = replace(demand)
            scaled.rates = np.asarray(demand.rates, dtype=np.float32) * scale
            return scaled
        except TypeError:
            pass
    return demand


def write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_road_network(args):
    if args.road_network:
        return RoadNetwork.load_or_download(
            cache_path=args.road_network,
            place=args.osm_place,
            network_type=args.road_network_type,
            default_speed_kmph=args.default_speed_kmph,
            download=args.download_road_network,
        )
    if args.download_road_network:
        return RoadNetwork.load_or_download(
            cache_path=args.road_network_cache,
            place=args.osm_place,
            network_type=args.road_network_type,
            default_speed_kmph=args.default_speed_kmph,
            download=True,
        )
    if args.road_network_cache and Path(args.road_network_cache).exists():
        return RoadNetwork.from_graphml(args.road_network_cache, default_speed_kmph=args.default_speed_kmph)
    return None


def validate_road_matrices(distance: np.ndarray, time: np.ndarray, num_cells: int) -> None:
    expected = (int(num_cells), int(num_cells))
    if np.asarray(distance).shape != expected or np.asarray(time).shape != expected:
        raise ValueError(f"road matrices must both have shape {expected}")


def load_road_matrices(args, train_pack: dict) -> tuple[np.ndarray | None, np.ndarray | None, str, int, int]:
    if args.road_matrix:
        matrices = load_road_cost_matrices(args.road_matrix)
        validate_road_matrices(matrices.distance_km, matrices.time_minutes, train_pack["grid"].num_cells)
        return (
            matrices.distance_km,
            matrices.time_minutes,
            str(args.road_matrix),
            int(matrices.fallback_distance_count),
            int(matrices.fallback_time_count),
        )
    if "road_distance_matrix" in train_pack and "road_time_matrix" in train_pack:
        distance = train_pack["road_distance_matrix"]
        time = train_pack["road_time_matrix"]
        validate_road_matrices(distance, time, train_pack["grid"].num_cells)
        return (
            distance,
            time,
            str(train_pack.get("road_matrix_source", "cache")),
            int(train_pack.get("road_matrix_fallback_distance_count", 0)),
            int(train_pack.get("road_matrix_fallback_time_count", 0)),
        )
    return None, None, "", 0, 0


def region_value_weight_arg(args) -> float:
    if args.region_value_weight is not None:
        return float(args.region_value_weight)
    if args.future_value_weight is not None:
        return float(args.future_value_weight)
    return 0.25


def region_value_loss_weight_arg(args) -> float:
    if args.region_value_loss_weight is not None:
        return float(args.region_value_loss_weight)
    if args.future_value_loss_weight is not None:
        return float(args.future_value_loss_weight)
    return 0.0


def _default(config_defaults: dict[str, object], key: str, fallback: object) -> object:
    return config_defaults.get(key, fallback)


def build_parser(config_defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    defaults = dict(config_defaults or {})
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=_default(defaults, "config", None),
        help="Built-in config preset name from taxi_dispatch.config or an external JSON config file.",
    )
    parser.add_argument("--train-cache", type=Path, default=_default(defaults, "train_cache", None))
    parser.add_argument("--test-cache", type=Path, default=_default(defaults, "test_cache", None))
    parser.add_argument("--method", choices=["all", "park", "random", "diffusion", "fv_bicoord"], default=_default(defaults, "method", "all"))
    parser.add_argument("--taxis", type=int, default=_default(defaults, "taxis", 6000))
    parser.add_argument("--episodes", type=int, default=_default(defaults, "episodes", 50))
    parser.add_argument("--eval-episodes", type=int, default=_default(defaults, "eval_episodes", 9))
    parser.add_argument("--demand-scale", type=float, default=_default(defaults, "demand_scale", 1.8))
    parser.add_argument("--start-hour", type=float, default=_default(defaults, "start_hour", 6.0))
    parser.add_argument("--step-minutes", type=int, default=_default(defaults, "step_minutes", None))
    parser.add_argument("--peak-hotspot-multiplier", type=float, default=_default(defaults, "peak_hotspot_multiplier", 1.0))
    parser.add_argument(
        "--peak-hotspot-windows",
        default=_default(defaults, "peak_hotspot_windows", DEFAULT_PEAK_HOTSPOT_WINDOWS_TEXT),
    )
    parser.add_argument("--peak-hotspot-cells", default=_default(defaults, "peak_hotspot_cells", ""))
    parser.add_argument("--peak-hotspot-top-cells", type=int, default=_default(defaults, "peak_hotspot_top_cells", 10))
    parser.add_argument("--seed", type=int, default=_default(defaults, "seed", 0))
    parser.add_argument(
        "--same-step-reposition-service",
        action=argparse.BooleanOptionalAction,
        default=_default(defaults, "same_step_reposition_service", False),
    )
    parser.add_argument(
        "--reposition-before-assignment",
        action=argparse.BooleanOptionalAction,
        default=_default(defaults, "reposition_before_assignment", False),
    )
    parser.add_argument("--hidden-dim", type=int, default=_default(defaults, "hidden_dim", 128))
    parser.add_argument("--actor-lr", type=float, default=_default(defaults, "actor_lr", 1e-4))
    parser.add_argument("--critic-lr", type=float, default=_default(defaults, "critic_lr", 1e-3))
    parser.add_argument("--tau", type=float, default=_default(defaults, "tau", 0.005))
    parser.add_argument("--gamma", type=float, default=_default(defaults, "gamma", 0.9))
    parser.add_argument("--fv-temporal-window", type=int, default=_default(defaults, "fv_temporal_window", 3))
    parser.add_argument("--fv-road-time-weight", type=float, default=_default(defaults, "fv_road_time_weight", 0.2))
    parser.add_argument("--fv-graph-temperature", type=float, default=_default(defaults, "fv_graph_temperature", 10.0))
    parser.add_argument("--fv-attention-heads", type=int, default=_default(defaults, "fv_attention_heads", 4))
    parser.add_argument("--fv-symmetric-adjacency", action="store_true", default=_default(defaults, "fv_symmetric_adjacency", False))
    parser.add_argument(
        "--fv-training-architecture",
        choices=TRAINING_ARCHITECTURES,
        default=_default(defaults, "fv_training_architecture", "full"),
    )
    parser.add_argument(
        "--fv-matching-mode",
        choices=["value_guided", "min_cost", "greedy"],
        default=_default(defaults, "fv_matching_mode", "value_guided"),
        help="Lower-layer matching mode used by the FV/Bi-STAR policy.",
    )
    parser.add_argument(
        "--fv-use-local-global-heads",
        action=argparse.BooleanOptionalAction,
        default=_default(defaults, "fv_use_local_global_heads", True),
    )
    parser.add_argument(
        "--fv-use-future-pressure-head",
        action=argparse.BooleanOptionalAction,
        default=_default(defaults, "fv_use_future_pressure_head", True),
    )
    parser.add_argument(
        "--fv-use-supply-sufficiency-gate",
        action=argparse.BooleanOptionalAction,
        default=_default(defaults, "fv_use_supply_sufficiency_gate", False),
    )
    parser.add_argument(
        "--fv-source-surplus-threshold",
        type=float,
        default=_default(defaults, "fv_source_surplus_threshold", 0.0),
    )
    parser.add_argument(
        "--fv-target-shortage-threshold",
        type=float,
        default=_default(defaults, "fv_target_shortage_threshold", 0.0),
    )
    parser.add_argument(
        "--fv-global-shortage-threshold",
        type=float,
        default=_default(defaults, "fv_global_shortage_threshold", 0.0),
    )
    parser.add_argument(
        "--fv-use-dynamic-soft-expansion-gate",
        action=argparse.BooleanOptionalAction,
        default=_default(defaults, "fv_use_dynamic_soft_expansion_gate", False),
    )
    parser.add_argument(
        "--fv-dynamic-source-surplus-threshold",
        type=float,
        default=_default(defaults, "fv_dynamic_source_surplus_threshold", 0.0),
    )
    parser.add_argument(
        "--fv-dynamic-target-shortage-threshold",
        type=float,
        default=_default(defaults, "fv_dynamic_target_shortage_threshold", 0.0),
    )
    parser.add_argument(
        "--fv-dynamic-global-shortage-threshold",
        type=float,
        default=_default(defaults, "fv_dynamic_global_shortage_threshold", 0.0),
    )
    parser.add_argument(
        "--fv-pressure-gate-strength",
        type=float,
        default=_default(defaults, "fv_pressure_gate_strength", 0.2),
    )
    parser.add_argument(
        "--fv-pressure-temperature-minutes",
        type=float,
        default=_default(defaults, "fv_pressure_temperature_minutes", 10.0),
    )
    parser.add_argument(
        "--fv-pressure-max-neighbor-time-minutes",
        type=float,
        default=_default(defaults, "fv_pressure_max_neighbor_time_minutes", 30.0),
    )
    parser.add_argument(
        "--fv-pressure-clip",
        type=float,
        default=_default(defaults, "fv_pressure_clip", 10.0),
    )
    parser.add_argument(
        "--fv-use-response-protected-move-budget",
        action=argparse.BooleanOptionalAction,
        default=_default(defaults, "fv_use_response_protected_move_budget", False),
    )
    parser.add_argument(
        "--fv-response-budget-incoming-discount",
        type=float,
        default=_default(defaults, "fv_response_budget_incoming_discount", 0.3),
    )
    parser.add_argument(
        "--fv-response-budget-safety-buffer",
        type=float,
        default=_default(defaults, "fv_response_budget_safety_buffer", 1.0),
    )
    parser.add_argument(
        "--fv-response-budget-min-idle-to-move",
        type=float,
        default=_default(defaults, "fv_response_budget_min_idle_to_move", 1.0),
    )
    parser.add_argument(
        "--fv-response-budget-pressure-strength",
        type=float,
        default=_default(defaults, "fv_response_budget_pressure_strength", 0.2),
    )
    parser.add_argument(
        "--fv-local-residual-scale",
        type=float,
        default=_default(defaults, "fv_local_residual_scale", 0.1),
    )
    parser.add_argument(
        "--fv-global-bias-scale",
        type=float,
        default=_default(defaults, "fv_global_bias_scale", 0.1),
    )
    parser.add_argument("--region-value-loss-weight", type=float, default=_default(defaults, "region_value_loss_weight", None))
    parser.add_argument(
        "--future-value-loss-weight",
        type=float,
        default=_default(defaults, "future_value_loss_weight", None),
        help="Deprecated alias for --region-value-loss-weight.",
    )
    parser.add_argument("--future-pressure-loss-weight", type=float, default=_default(defaults, "future_pressure_loss_weight", None))
    parser.add_argument("--future-gap-loss-weight", type=float, default=_default(defaults, "future_gap_loss_weight", 0.0))
    parser.add_argument("--future-demand-loss-weight", type=float, default=_default(defaults, "future_demand_loss_weight", 0.0))
    parser.add_argument("--intensity-loss-weight", type=float, default=_default(defaults, "intensity_loss_weight", 0.0))
    parser.add_argument("--actor-future-pressure-weight", type=float, default=_default(defaults, "actor_future_pressure_weight", None))
    parser.add_argument("--actor-future-demand-weight", type=float, default=_default(defaults, "actor_future_demand_weight", 0.0))
    parser.add_argument("--actor-region-value-weight", type=float, default=_default(defaults, "actor_region_value_weight", 0.0))
    parser.add_argument("--future-demand-steps", type=int, default=_default(defaults, "future_demand_steps", 0))
    parser.add_argument("--region-value-weight", type=float, default=_default(defaults, "region_value_weight", None))
    parser.add_argument(
        "--future-value-weight",
        type=float,
        default=_default(defaults, "future_value_weight", None),
        help="Deprecated alias for --region-value-weight.",
    )
    parser.add_argument("--future-gap-weight", type=float, default=_default(defaults, "future_gap_weight", 0.25))
    parser.add_argument("--dispatch-intensity-weight", type=float, default=_default(defaults, "dispatch_intensity_weight", 0.25))
    parser.add_argument("--trip-time-weight", type=float, default=_default(defaults, "trip_time_weight", 0.05))
    parser.add_argument("--service-revenue-weight", type=float, default=_default(defaults, "service_revenue_weight", 0.05))
    parser.add_argument("--fare-base", type=float, default=_default(defaults, "fare_base", 14.0))
    parser.add_argument("--fare-per-km", type=float, default=_default(defaults, "fare_per_km", 2.6))
    parser.add_argument("--fare-per-minute", type=float, default=_default(defaults, "fare_per_minute", 0.5))
    parser.add_argument("--wait-time-penalty-weight", type=float, default=_default(defaults, "wait_time_penalty_weight", 0.1))
    parser.add_argument("--wait-time-tolerance-minutes", type=float, default=_default(defaults, "wait_time_tolerance_minutes", 5.0))
    parser.add_argument("--wait-time-penalty-exponent", type=float, default=_default(defaults, "wait_time_penalty_exponent", 1.0))
    parser.add_argument("--cancellation-penalty", type=float, default=_default(defaults, "cancellation_penalty", 3.0))
    parser.add_argument("--relocation-cost-weight", type=float, default=_default(defaults, "relocation_cost_weight", 0.5))
    parser.add_argument("--device", default=_default(defaults, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out", type=Path, default=_default(defaults, "out", None))
    parser.add_argument("--plots", action="store_true", default=_default(defaults, "plots", False))

    parser.add_argument("--road-network", type=Path, default=_default(defaults, "road_network", None))
    parser.add_argument(
        "--road-network-cache",
        type=Path,
        default=_default(defaults, "road_network_cache", Path("data/processed/chengdu_road.graphml")),
    )
    parser.add_argument("--road-matrix", type=Path, default=_default(defaults, "road_matrix", None))
    parser.add_argument("--download-road-network", action="store_true", default=_default(defaults, "download_road_network", False))
    parser.add_argument("--osm-place", default=_default(defaults, "osm_place", "Chengdu, Sichuan, China"))
    parser.add_argument("--road-network-type", default=_default(defaults, "road_network_type", "drive"))
    parser.add_argument("--default-speed-kmph", type=float, default=_default(defaults, "default_speed_kmph", 30.0))
    parser.add_argument("--road-speed-kmph", dest="default_speed_kmph", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--require-road-costs", action="store_true", default=_default(defaults, "require_road_costs", False))
    parser.add_argument("--fv-entropy-coef", type=float, default=_default(defaults, "fv_entropy_coef", 0.01))
    parser.add_argument(
        "--fv-park-mask-surplus-threshold",
        type=float,
        default=_default(defaults, "fv_park_mask_surplus_threshold", None),
        help="Disable stay in oversupplied cells above this scaled supply-demand gap when a neighbor has unmet demand.",
    )
    parser.add_argument(
        "--fv-park-mask-neighbor-need-threshold",
        type=float,
        default=_default(defaults, "fv_park_mask_neighbor_need_threshold", 0.0),
    )
    return parser


def main() -> None:
    args = parse_args_with_config(
        build_parser,
        required=(
            ("train_cache", "--train-cache"),
            ("test_cache", "--test-cache"),
            ("out", "--out"),
        ),
    )

    args.out.mkdir(parents=True, exist_ok=True)

    train_pack = load_pickle(args.train_cache)
    test_pack = load_pickle(args.test_cache)
    assert_same_grid(train_pack["grid"], test_pack["grid"])
    train_demand = scale_cached_demand(train_pack["demand"], args.demand_scale)
    test_demand = scale_cached_demand(test_pack["demand"], args.demand_scale)
    train_step_minutes = int(args.step_minutes or getattr(train_pack["config"], "step_minutes", 10))
    test_step_minutes = int(args.step_minutes or getattr(test_pack["config"], "step_minutes", train_step_minutes))
    train_demand, train_boost_cells, train_boost_windows = apply_peak_hotspot_boost_to_demand(
        train_demand,
        start_hour=args.start_hour,
        step_minutes=train_step_minutes,
        peak_hotspot_multiplier=args.peak_hotspot_multiplier,
        peak_hotspot_windows=args.peak_hotspot_windows,
        peak_hotspot_cells=args.peak_hotspot_cells,
        peak_hotspot_top_cells=args.peak_hotspot_top_cells,
    )
    test_demand, test_boost_cells, test_boost_windows = apply_peak_hotspot_boost_to_demand(
        test_demand,
        start_hour=args.start_hour,
        step_minutes=test_step_minutes,
        peak_hotspot_multiplier=args.peak_hotspot_multiplier,
        peak_hotspot_windows=args.peak_hotspot_windows,
        peak_hotspot_cells=args.peak_hotspot_cells,
        peak_hotspot_top_cells=args.peak_hotspot_top_cells,
    )
    road_network = load_road_network(args)
    (
        road_distance_matrix,
        road_time_matrix,
        road_matrix_source,
        road_matrix_fallback_distance_count,
        road_matrix_fallback_time_count,
    ) = load_road_matrices(args, train_pack)
    if road_time_matrix is None and road_network is not None:
        matrices = road_network.cell_cost_matrices(train_pack["grid"], fallback_speed_kmph=args.default_speed_kmph)
        road_distance_matrix = matrices.distance_km
        road_time_matrix = matrices.time_minutes
        validate_road_matrices(road_distance_matrix, road_time_matrix, train_pack["grid"].num_cells)
        road_matrix_source = "computed_from_road_network"
        road_matrix_fallback_distance_count = int(matrices.fallback_distance_count)
        road_matrix_fallback_time_count = int(matrices.fallback_time_count)
    if road_distance_matrix is not None and road_time_matrix is not None:
        routing_mode = "road_time_matrix"
    else:
        routing_mode = "road_network" if road_network is not None else "euclidean_fallback"
    if road_network is None and road_time_matrix is None:
        message = "no road network loaded; using Euclidean/grid-distance fallback routing."
        if args.require_road_costs:
            raise SystemExit(
                "ERROR: " + message + " Pass --road-matrix, --road-network, or --download-road-network."
            )
        print("WARNING: " + message)

    train_config = replace(
        train_pack["config"],
        fleet_size=args.taxis,
        seed=args.seed,
        demand_scale=args.demand_scale,
        travel_speed_kmph=args.default_speed_kmph,
        future_demand_steps=args.future_demand_steps,
        region_value_weight=region_value_weight_arg(args),
        future_value_weight=region_value_weight_arg(args),
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
        relocation_cost_weight=args.relocation_cost_weight,
        same_step_reposition_service=args.same_step_reposition_service,
        reposition_before_assignment=args.reposition_before_assignment,
        pickup_scope="origin",
        matching_mode=args.fv_matching_mode,
    )
    test_config = replace(
        test_pack["config"],
        fleet_size=args.taxis,
        seed=args.seed,
        demand_scale=args.demand_scale,
        travel_speed_kmph=args.default_speed_kmph,
        future_demand_steps=args.future_demand_steps,
        region_value_weight=region_value_weight_arg(args),
        future_value_weight=region_value_weight_arg(args),
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
        relocation_cost_weight=args.relocation_cost_weight,
        same_step_reposition_service=args.same_step_reposition_service,
        reposition_before_assignment=args.reposition_before_assignment,
        pickup_scope="origin",
        matching_mode=args.fv_matching_mode,
    )
    baseline_test_config = _baseline_config(test_config)
    park_test_config = replace(
        baseline_test_config,
        pickup_scope="origin",
        same_step_reposition_service=args.same_step_reposition_service,
        reposition_before_assignment=args.reposition_before_assignment,
    )
    random_test_config = replace(
        baseline_test_config,
        pickup_scope="origin",
        same_step_reposition_service=args.same_step_reposition_service,
        reposition_before_assignment=args.reposition_before_assignment,
    )
    diffusion_test_config = replace(
        baseline_test_config,
        pickup_scope="origin",
        same_step_reposition_service=args.same_step_reposition_service,
        reposition_before_assignment=args.reposition_before_assignment,
    )
    eval_initial_taxi_distribution = train_pack.get("initial_taxi_distribution")

    methods = ["park", "random", "diffusion", "fv_bicoord"] if args.method == "all" else [args.method]

    # 保存实验配置，方便复现
    (args.out / "run_config.json").write_text(
        json.dumps(
            {
                "config": str(args.config) if args.config else None,
                "config_source": getattr(args, "config_source", None),
                "config_preset": getattr(args, "config_preset", None),
                "config_overrides": list(getattr(args, "config_overrides", [])),
                "train_cache": str(args.train_cache),
                "test_cache": str(args.test_cache),
                "method": args.method,
                "taxis": args.taxis,
                "episodes": args.episodes,
                "eval_episodes": args.eval_episodes,
                "demand_scale": args.demand_scale,
                "start_hour": args.start_hour,
                "step_minutes": train_step_minutes,
                "peak_hotspot_multiplier": args.peak_hotspot_multiplier,
                "peak_hotspot_windows": args.peak_hotspot_windows,
                "peak_hotspot_cells": args.peak_hotspot_cells,
                "peak_hotspot_top_cells": args.peak_hotspot_top_cells,
                "train_peak_hotspot_cells": train_boost_cells,
                "test_peak_hotspot_cells": test_boost_cells,
                "train_peak_hotspot_windows": train_boost_windows,
                "test_peak_hotspot_windows": test_boost_windows,
                "train_peak_hotspot_periods": (getattr(train_demand, "metadata", {}) or {}).get(
                    "peak_hotspot_periods", ()
                ),
                "test_peak_hotspot_periods": (getattr(test_demand, "metadata", {}) or {}).get(
                    "peak_hotspot_periods", ()
                ),
                "seed": args.seed,
                "travel_speed_kmph": args.default_speed_kmph,
                "fv_temporal_window": args.fv_temporal_window,
                "fv_road_time_weight": args.fv_road_time_weight,
                "fv_graph_temperature": args.fv_graph_temperature,
                "fv_attention_heads": args.fv_attention_heads,
                "fv_symmetric_adjacency": args.fv_symmetric_adjacency,
                "fv_training_architecture": args.fv_training_architecture,
                "fv_matching_mode": args.fv_matching_mode,
                "fv_use_local_global_heads": args.fv_use_local_global_heads,
                "fv_use_future_pressure_head": args.fv_use_future_pressure_head,
                "fv_use_supply_sufficiency_gate": args.fv_use_supply_sufficiency_gate,
                "fv_source_surplus_threshold": args.fv_source_surplus_threshold,
                "fv_target_shortage_threshold": args.fv_target_shortage_threshold,
                "fv_global_shortage_threshold": args.fv_global_shortage_threshold,
                "fv_use_dynamic_soft_expansion_gate": args.fv_use_dynamic_soft_expansion_gate,
                "fv_dynamic_source_surplus_threshold": args.fv_dynamic_source_surplus_threshold,
                "fv_dynamic_target_shortage_threshold": args.fv_dynamic_target_shortage_threshold,
                "fv_dynamic_global_shortage_threshold": args.fv_dynamic_global_shortage_threshold,
                "fv_pressure_gate_strength": args.fv_pressure_gate_strength,
                "fv_pressure_temperature_minutes": args.fv_pressure_temperature_minutes,
                "fv_pressure_max_neighbor_time_minutes": args.fv_pressure_max_neighbor_time_minutes,
                "fv_pressure_clip": args.fv_pressure_clip,
                "fv_use_response_protected_move_budget": args.fv_use_response_protected_move_budget,
                "fv_response_budget_incoming_discount": args.fv_response_budget_incoming_discount,
                "fv_response_budget_safety_buffer": args.fv_response_budget_safety_buffer,
                "fv_response_budget_min_idle_to_move": args.fv_response_budget_min_idle_to_move,
                "fv_response_budget_pressure_strength": args.fv_response_budget_pressure_strength,
                "fv_local_residual_scale": args.fv_local_residual_scale,
                "fv_global_bias_scale": args.fv_global_bias_scale,
                "region_value_loss_weight": region_value_loss_weight_arg(args),
                "future_value_loss_weight": region_value_loss_weight_arg(args),
                "future_pressure_loss_weight": _future_pressure_loss_weight_arg(args),
                "future_gap_loss_weight": args.future_gap_loss_weight,
                "future_demand_loss_weight": args.future_demand_loss_weight,
                "intensity_loss_weight": args.intensity_loss_weight,
                "actor_future_pressure_weight": _actor_future_pressure_weight_arg(args),
                "actor_future_demand_weight": args.actor_future_demand_weight,
                "actor_region_value_weight": args.actor_region_value_weight,
                "future_demand_steps": args.future_demand_steps,
                "region_value_weight": region_value_weight_arg(args),
                "future_value_weight": region_value_weight_arg(args),
                "future_gap_weight": args.future_gap_weight,
                "dispatch_intensity_weight": args.dispatch_intensity_weight,
                "trip_time_weight": args.trip_time_weight,
                "service_revenue_weight": args.service_revenue_weight,
                "fare_base": args.fare_base,
                "fare_per_km": args.fare_per_km,
                "fare_per_minute": args.fare_per_minute,
                "wait_time_penalty_weight": args.wait_time_penalty_weight,
                "wait_time_tolerance_minutes": args.wait_time_tolerance_minutes,
                "wait_time_penalty_exponent": args.wait_time_penalty_exponent,
                "cancellation_penalty": args.cancellation_penalty,
                "relocation_cost_weight": args.relocation_cost_weight,
                "same_step_reposition_service": getattr(test_config, "same_step_reposition_service", True),
                "reposition_before_assignment": getattr(test_config, "reposition_before_assignment", False),
                "fv_entropy_coef": args.fv_entropy_coef,
                "fv_park_mask_surplus_threshold": args.fv_park_mask_surplus_threshold,
                "fv_park_mask_neighbor_need_threshold": args.fv_park_mask_neighbor_need_threshold,
                "baseline_matching_mode": getattr(baseline_test_config, "matching_mode", "greedy"),
                "baseline_pickup_scope": getattr(baseline_test_config, "pickup_scope", "origin"),
                "park_pickup_scope": getattr(park_test_config, "pickup_scope", "origin"),
                "random_pickup_scope": getattr(random_test_config, "pickup_scope", "origin"),
                "diffusion_pickup_scope": getattr(diffusion_test_config, "pickup_scope", "origin"),
                "park_same_step_reposition_service": getattr(
                    park_test_config,
                    "same_step_reposition_service",
                    True,
                ),
                "park_reposition_before_assignment": getattr(
                    park_test_config,
                    "reposition_before_assignment",
                    False,
                ),
                "random_same_step_reposition_service": getattr(
                    random_test_config,
                    "same_step_reposition_service",
                    True,
                ),
                "random_reposition_before_assignment": getattr(
                    random_test_config,
                    "reposition_before_assignment",
                    False,
                ),
                "diffusion_same_step_reposition_service": getattr(
                    diffusion_test_config,
                    "same_step_reposition_service",
                    True,
                ),
                "diffusion_reposition_before_assignment": getattr(
                    diffusion_test_config,
                    "reposition_before_assignment",
                    False,
                ),
                "fv_bicoord_matching_mode": getattr(test_config, "matching_mode", "value_guided"),
                "fv_bicoord_pickup_scope": getattr(test_config, "pickup_scope", "origin"),
                "fv_bicoord_same_step_reposition_service": getattr(
                    test_config,
                    "same_step_reposition_service",
                    True,
                ),
                "fv_bicoord_reposition_before_assignment": getattr(
                    test_config,
                    "reposition_before_assignment",
                    False,
                ),
                "eval_initial_distribution_source": (
                    "train_cache" if eval_initial_taxi_distribution is not None else "uniform_default"
                ),
                "road_network": str(args.road_network) if args.road_network else None,
                "road_network_cache": str(args.road_network_cache) if args.road_network_cache else None,
                "road_matrix": road_matrix_source or None,
                "routing_mode": routing_mode,
                "road_matrix_fallback_distance_count": int(road_matrix_fallback_distance_count),
                "road_matrix_fallback_time_count": int(road_matrix_fallback_time_count),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    trained_fv_agent = None
    if "fv_bicoord" in methods:
        trained_fv_agent, fv_history = train_fv_bicoord(
            config=train_config,
            episodes=args.episodes,
            hidden_dim=args.hidden_dim,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            tau=args.tau,
            gamma=args.gamma,
            device=args.device,
            seed=args.seed + 70000,
            out_dir=args.out,
            road_time_weight=args.fv_road_time_weight,
            temporal_window=args.fv_temporal_window,
            graph_temperature=args.fv_graph_temperature,
            attention_heads=args.fv_attention_heads,
            symmetric_adjacency=args.fv_symmetric_adjacency,
            region_value_loss_weight=region_value_loss_weight_arg(args),
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
            demand=train_demand,
            grid=train_pack["grid"],
            initial_taxi_distribution=train_pack.get("initial_taxi_distribution"),
            road_network=road_network,
            road_distance_matrix=road_distance_matrix,
            road_time_matrix=road_time_matrix,
            live_history_path=args.out / "fv_bicoord_training_history.csv",
            live_epoch_history_path=args.out / "fv_bicoord_training_epochs.csv",
        )

        hist_path = args.out / "fv_bicoord_training_history.csv"
        _write_history(fv_history, hist_path)

        epoch_history = aggregate_training_epochs(fv_history)
        epoch_hist_path = args.out / "fv_bicoord_training_epochs.csv"
        _write_history(epoch_history, epoch_hist_path)

        if args.plots:
            plot_training_curves(fv_history, args.out / "fv_bicoord_training_curves.png")
            plot_training_curves(epoch_history, args.out / "fv_bicoord_training_epoch_curves.png")

    rows = []
    for method in methods:
        if method == "park":
            metrics, env = evaluate_policy(
                park_test_config,
                lambda env, obs, state: park_actions(env),
                args.eval_episodes,
                args.seed + 10000,
                demand=test_demand,
                grid=test_pack["grid"],
                initial_taxi_distribution=eval_initial_taxi_distribution,
                road_network=road_network,
                road_distance_matrix=road_distance_matrix,
                road_time_matrix=road_time_matrix,
            )
        elif method == "random":
            rng = np.random.default_rng(args.seed + 20000)
            metrics, env = evaluate_policy(
                random_test_config,
                lambda env, obs, state: random_actions(env, rng),
                args.eval_episodes,
                args.seed + 30000,
                demand=test_demand,
                grid=test_pack["grid"],
                initial_taxi_distribution=eval_initial_taxi_distribution,
                road_network=road_network,
                road_distance_matrix=road_distance_matrix,
                road_time_matrix=road_time_matrix,
            )
        elif method == "diffusion":
            metrics, env = evaluate_policy(
                diffusion_test_config,
                lambda env, obs, state: diffusion_actions(env),
                args.eval_episodes,
                args.seed + 40000,
                demand=test_demand,
                grid=test_pack["grid"],
                initial_taxi_distribution=eval_initial_taxi_distribution,
                road_network=road_network,
                road_distance_matrix=road_distance_matrix,
                road_time_matrix=road_time_matrix,
            )
        elif method == "fv_bicoord":
            metrics, env = evaluate_fv_bicoord(
                test_config,
                trained_fv_agent,
                args.eval_episodes,
                args.seed + 80000,
                demand=test_demand,
                grid=test_pack["grid"],
                initial_taxi_distribution=eval_initial_taxi_distribution,
                road_network=road_network,
                road_distance_matrix=road_distance_matrix,
                road_time_matrix=road_time_matrix,
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
            raise ValueError(method)

        row = {"method": method, **asdict(metrics)}
        rows.append(row)

        print(
            f"{method:9s} "
            f"response={metrics.response_rate:.4f} "
            f"wait={metrics.response_time_seconds:.1f}s "
            f"occupied={metrics.occupied_rate:.4f} "
            f"Normalized GMV={metrics.normalized_gmv:.4f} "
            f"orders={metrics.orders:.1f} "
            f"cancellations={metrics.cancellations:.1f}"
        )

        env.save_cell_cancellations(args.out / f"{method}_cell_cancellations.csv")
        if args.plots:
            plot_cell_cancellations(env, args.out / f"{method}_cell_cancellations.png")

    write_summary(rows, args.out / "summary.csv")
    print(f"wrote results to {args.out}")


if __name__ == "__main__":
    main()
