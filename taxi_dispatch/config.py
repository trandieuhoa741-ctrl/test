from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Iterable, Sequence


CONFIG_PRESETS: dict[str, dict[str, object]] = {
    "run_chengdu_cache_experiment": {
        "train_cache": "data/processed_mamr/chengdu_train_20161108_20161130_N142_T108.pkl",
        "test_cache": "data/processed_mamr/chengdu_test_20161101_20161107_N142_T108.pkl",
        "method": "all",
        "taxis": 4000,
        "episodes": 50,
        "eval_episodes": 7,
        "demand_scale": 1.8,
        "seed": 0,
        "same_step_reposition_service": False,
        "reposition_before_assignment": False,
        "hidden_dim": 128,
        "actor_lr": 0.0001,
        "critic_lr": 0.001,
        "tau": 0.005,
        "gamma": 0.9,
        "fv_temporal_window": 3,
        "fv_road_time_weight": 0.2,
        "fv_graph_temperature": 10.0,
        "fv_attention_heads": 4,
        "fv_symmetric_adjacency": False,
        "fv_matching_mode": "value_guided",
        "fv_use_local_global_heads": True,
        "fv_local_residual_scale": 0.1,
        "fv_global_bias_scale": 0.1,
        "region_value_loss_weight": 0.0,
        "future_pressure_loss_weight": 0.0,
        "future_gap_loss_weight": 0.0,
        "future_demand_loss_weight": 0.0,
        "intensity_loss_weight": 0.0,
        "actor_future_pressure_weight": 0.0,
        "actor_future_demand_weight": 0.0,
        "actor_region_value_weight": 0.0,
        "future_demand_steps": 0,
        "region_value_weight": 0.25,
        "future_gap_weight": 0.25,
        "dispatch_intensity_weight": 0.25,
        "trip_time_weight": 0.05,
        "service_revenue_weight": 0.05,
        "fare_base": 14.0,
        "fare_per_km": 2.6,
        "fare_per_minute": 0.5,
        "wait_time_penalty_weight": 0.1,
        "wait_time_tolerance_minutes": 5.0,
        "wait_time_penalty_exponent": 1.0,
        "cancellation_penalty": 3.0,
        "relocation_cost_weight": 0.5,
        "device": "cuda",
        "out": "outputs/chengdu_cache_experiment",
        "plots": False,
        "road_matrix": "data/processed/chengdu_osm_road_matrix_N142.npz",
        "road_network": None,
        "road_network_cache": "data/processed/chengdu_road.graphml",
        "download_road_network": False,
        "osm_place": "Chengdu, Sichuan, China",
        "road_network_type": "drive",
        "default_speed_kmph": 30.0,
        "require_road_costs": False,
        "fv_entropy_coef": 0.01,
        "fv_park_mask_surplus_threshold": None,
        "fv_park_mask_neighbor_need_threshold": 0.0,
    },
    "chengdu_4000_mamr_fv_tuned": {
        "train_cache": "data/processed_mamr/chengdu_train_20161108_20161130_N142_T108.pkl",
        "test_cache": "data/processed_mamr/chengdu_test_20161101_20161107_N142_T108.pkl",
        "method": "all",
        "taxis": 4000,
        "episodes": 60,
        "eval_episodes": 7,
        "demand_scale": 1.8,
        "seed": 0,
        "same_step_reposition_service": False,
        "reposition_before_assignment": False,
        "hidden_dim": 256,
        "actor_lr": 0.0003,
        "critic_lr": 0.001,
        "tau": 0.005,
        "gamma": 0.9,
        "fv_temporal_window": 5,
        "fv_road_time_weight": 0.08,
        "fv_graph_temperature": 10.0,
        "fv_attention_heads": 4,
        "fv_symmetric_adjacency": False,
        "fv_matching_mode": "value_guided",
        "fv_use_local_global_heads": False,
        "fv_local_residual_scale": 0.1,
        "fv_global_bias_scale": 0.1,
        "region_value_loss_weight": 0.2,
        "future_pressure_loss_weight": 0.2,
        "future_gap_loss_weight": 0.2,
        "future_demand_loss_weight": 0.1,
        "intensity_loss_weight": 0.05,
        "actor_future_pressure_weight": 0.0,
        "actor_future_demand_weight": 0.0,
        "actor_region_value_weight": 0.0,
        "future_demand_steps": 0,
        "region_value_weight": 0.4,
        "future_gap_weight": 0.2,
        "dispatch_intensity_weight": 0.15,
        "trip_time_weight": 0.01,
        "service_revenue_weight": 0.1,
        "fare_base": 14.0,
        "fare_per_km": 2.6,
        "fare_per_minute": 0.5,
        "wait_time_penalty_weight": 0.03,
        "wait_time_tolerance_minutes": 5.0,
        "wait_time_penalty_exponent": 1.0,
        "cancellation_penalty": 5.0,
        "relocation_cost_weight": 0.3,
        "device": "cuda",
        "out": "outputs/chengdu_4000_mamr_fv_tuned_v1",
        "plots": False,
        "road_matrix": "data/processed/chengdu_osm_road_matrix_N142.npz",
        "road_network": None,
        "road_network_cache": "data/processed/chengdu_road.graphml",
        "download_road_network": False,
        "osm_place": "Chengdu, Sichuan, China",
        "road_network_type": "drive",
        "default_speed_kmph": 30.0,
        "require_road_costs": False,
        "fv_entropy_coef": 0.003,
        "fv_park_mask_surplus_threshold": 0.05,
        "fv_park_mask_neighbor_need_threshold": 0.0,
    },
    "chengdu_4000_response_v1": {
        "train_cache": "data/processed_mamr/chengdu_train_20161108_20161130_N142_T108.pkl",
        "test_cache": "data/processed_mamr/chengdu_test_20161101_20161107_N142_T108.pkl",
        "method": "all",
        "taxis": 4000,
        "episodes": 60,
        "eval_episodes": 7,
        "demand_scale": 1.8,
        "seed": 0,
        "same_step_reposition_service": False,
        "reposition_before_assignment": False,
        "hidden_dim": 256,
        "actor_lr": 0.00003,
        "critic_lr": 0.0005,
        "tau": 0.005,
        "gamma": 0.9,
        "fv_temporal_window": 5,
        "fv_road_time_weight": 0.10,
        "fv_graph_temperature": 2.4,
        "fv_attention_heads": 4,
        "fv_symmetric_adjacency": False,
        "fv_training_architecture": "rstr",
        "fv_matching_mode": "value_guided",
        "fv_use_local_global_heads": False,
        "fv_local_residual_scale": 0.1,
        "fv_global_bias_scale": 0.1,
        "region_value_loss_weight": 0.0,
        "future_pressure_loss_weight": 0.0,
        "future_gap_loss_weight": 0.0,
        "future_demand_loss_weight": 0.0,
        "intensity_loss_weight": 0.0,
        "actor_future_pressure_weight": 0.0,
        "actor_future_demand_weight": 0.0,
        "actor_region_value_weight": 0.0,
        "future_demand_steps": 0,
        "region_value_weight": 0.18,
        "future_gap_weight": 0.04,
        "dispatch_intensity_weight": 0.04,
        "trip_time_weight": 0.01,
        "service_revenue_weight": 0.04,
        "fare_base": 14.0,
        "fare_per_km": 2.6,
        "fare_per_minute": 0.5,
        "wait_time_penalty_weight": 0.025,
        "wait_time_tolerance_minutes": 5.0,
        "wait_time_penalty_exponent": 1.0,
        "cancellation_penalty": 4.5,
        "relocation_cost_weight": 0.45,
        "device": "cuda",
        "out": "outputs/chengdu_4000_response_v1",
        "plots": False,
        "road_matrix": "data/processed/chengdu_osm_road_matrix_N142.npz",
        "road_network": None,
        "road_network_cache": "data/processed/chengdu_road.graphml",
        "download_road_network": False,
        "osm_place": "Chengdu, Sichuan, China",
        "road_network_type": "drive",
        "default_speed_kmph": 30.0,
        "require_road_costs": False,
        "fv_entropy_coef": 0.02,
        "fv_park_mask_surplus_threshold": 0.10,
        "fv_park_mask_neighbor_need_threshold": 0.02,
    },
    "chengdu_origin_response_first": {
        "train_cache": "data/processed/chengdu_train_20161101_20161121_N142_T108.pkl",
        "test_cache": "data/processed/chengdu_test_20161122_20161130_N142_T108.pkl",
        "method": "all",
        "taxis": 6000,
        "episodes": 60,
        "eval_episodes": 9,
        "demand_scale": 1.8,
        "seed": 0,
        "same_step_reposition_service": True,
        "reposition_before_assignment": True,
        "hidden_dim": 256,
        "actor_lr": 0.0003,
        "critic_lr": 0.001,
        "tau": 0.005,
        "gamma": 0.9,
        "fv_temporal_window": 5,
        "fv_road_time_weight": 0.05,
        "fv_graph_temperature": 10.0,
        "fv_attention_heads": 4,
        "fv_symmetric_adjacency": False,
        "fv_matching_mode": "value_guided",
        "fv_use_local_global_heads": False,
        "fv_local_residual_scale": 0.1,
        "fv_global_bias_scale": 0.1,
        "region_value_loss_weight": 0.2,
        "future_pressure_loss_weight": 0.25,
        "future_gap_loss_weight": 0.25,
        "future_demand_loss_weight": 0.15,
        "intensity_loss_weight": 0.08,
        "actor_future_pressure_weight": 0.25,
        "actor_future_demand_weight": 0.25,
        "actor_region_value_weight": 0.08,
        "future_demand_steps": 0,
        "region_value_weight": 0.25,
        "future_gap_weight": 0.35,
        "dispatch_intensity_weight": 0.3,
        "trip_time_weight": 0.005,
        "service_revenue_weight": 0.03,
        "fare_base": 14.0,
        "fare_per_km": 2.6,
        "fare_per_minute": 0.5,
        "wait_time_penalty_weight": 0.05,
        "wait_time_tolerance_minutes": 5.0,
        "wait_time_penalty_exponent": 1.0,
        "cancellation_penalty": 8.0,
        "relocation_cost_weight": 0.12,
        "device": "cuda",
        "out": "outputs/chengdu_origin_response_first",
        "plots": False,
        "road_matrix": "data/processed/chengdu_osm_road_matrix_N142.npz",
        "road_network": None,
        "road_network_cache": "data/processed/chengdu_road.graphml",
        "download_road_network": False,
        "osm_place": "Chengdu, Sichuan, China",
        "road_network_type": "drive",
        "default_speed_kmph": 30.0,
        "require_road_costs": False,
        "fv_entropy_coef": 0.004,
        "fv_park_mask_surplus_threshold": 0.02,
        "fv_park_mask_neighbor_need_threshold": 0.0,
    },
}

CONFIG_PRESETS["chengdu_4000_response_light_full"] = {
    **CONFIG_PRESETS["chengdu_4000_response_v1"],
    "region_value_loss_weight": 0.0,
    "future_pressure_loss_weight": 0.0,
    "future_gap_loss_weight": 0.0,
    "future_demand_loss_weight": 0.0,
    "actor_future_pressure_weight": 0.0,
    "actor_future_demand_weight": 0.0,
    "actor_region_value_weight": 0.0,
    "out": "outputs/chengdu_4000_response_light_full",
}

CONFIG_PRESETS["chengdu_4000_response_no_region_value"] = {
    **CONFIG_PRESETS["chengdu_4000_response_v1"],
    "region_value_loss_weight": 0.0,
    "actor_region_value_weight": 0.0,
    "region_value_weight": 0.0,
    "future_value_weight": 0.0,
    "out": "outputs/chengdu_4000_response_no_region_value",
}

CONFIG_PRESETS["chengdu_4000_response_no_value_matching"] = {
    **CONFIG_PRESETS["chengdu_4000_response_v1"],
    "fv_matching_mode": "greedy",
    "region_value_weight": 0.0,
    "future_value_weight": 0.0,
    "out": "outputs/chengdu_4000_response_no_value_matching",
}

CONFIG_PRESETS["chengdu_4000_response_light_full_local_global"] = {
    **CONFIG_PRESETS["chengdu_4000_response_light_full"],
    "fv_use_local_global_heads": True,
    "fv_local_residual_scale": 0.1,
    "fv_global_bias_scale": 0.1,
    "out": "outputs/chengdu_4000_response_light_full_local_global",
}

CONFIG_PRESETS["response_region_hetero_global"] = {
    **CONFIG_PRESETS["chengdu_4000_response_v1"],
    "fv_training_architecture": "full",
    "fv_matching_mode": "value_guided",
    "fv_use_local_global_heads": True,
    "fv_use_future_pressure_head": False,
    "fv_use_supply_sufficiency_gate": False,
    "fv_source_surplus_threshold": 0.0,
    "fv_target_shortage_threshold": 0.0,
    "fv_global_shortage_threshold": 0.0,
    "fv_use_dynamic_soft_expansion_gate": False,
    "fv_dynamic_source_surplus_threshold": 0.0,
    "fv_dynamic_target_shortage_threshold": 0.0,
    "fv_dynamic_global_shortage_threshold": 0.0,
    "fv_pressure_gate_strength": 0.2,
    "fv_pressure_temperature_minutes": 10.0,
    "fv_pressure_max_neighbor_time_minutes": 30.0,
    "fv_pressure_clip": 10.0,
    "fv_use_response_protected_move_budget": False,
    "fv_response_budget_incoming_discount": 0.3,
    "fv_response_budget_safety_buffer": 1.0,
    "fv_response_budget_min_idle_to_move": 1.0,
    "fv_response_budget_pressure_strength": 0.2,
    "fv_local_residual_scale": 0.1,
    "fv_global_bias_scale": 0.1,
    "region_value_loss_weight": 0.0,
    "future_pressure_loss_weight": 0.0,
    "future_gap_loss_weight": 0.0,
    "future_demand_loss_weight": 0.0,
    "intensity_loss_weight": 0.0,
    "actor_future_pressure_weight": 0.0,
    "actor_future_demand_weight": 0.0,
    "actor_region_value_weight": 0.0,
    "region_value_weight": 0.18,
    "future_value_weight": 0.18,
    "future_gap_weight": 0.0,
    "dispatch_intensity_weight": 0.0,
    "trip_time_weight": 0.02,
    "fv_entropy_coef": 0.02,
    "out": "outputs/response_region_hetero_global",
}

CONFIG_PRESET_ALIASES = {
    "chengdu_cache_experiment": "run_chengdu_cache_experiment",
    "chengdu_4000_stable_v2": "chengdu_4000_response_v1",
    "chengdu_response_86": "chengdu_4000_response_v1",
    "chengdu_response_light_full": "chengdu_4000_response_light_full",
    "response_light_full": "chengdu_4000_response_light_full",
    "chengdu_response_light_full_local_global": "chengdu_4000_response_light_full_local_global",
    "response_light_full_local_global": "chengdu_4000_response_light_full_local_global",
    "chengdu_response_region_hetero_global": "response_region_hetero_global",
    "region_hetero_global": "response_region_hetero_global",
    "chengdu_response_no_region_value": "chengdu_4000_response_no_region_value",
    "response_no_region_value": "chengdu_4000_response_no_region_value",
    "no_region_value": "chengdu_4000_response_no_region_value",
    "chengdu_response_no_value_matching": "chengdu_4000_response_no_value_matching",
    "response_no_value_matching": "chengdu_4000_response_no_value_matching",
    "no_value_matching": "chengdu_4000_response_no_value_matching",
    "chengdu_response_first": "chengdu_origin_response_first",
    "response_86": "chengdu_4000_response_v1",
    "response_first": "chengdu_origin_response_first",
    "chengdu_response_rstr": "chengdu_4000_response_v1",
    "response_rstr": "chengdu_4000_response_v1",
    "rstr": "chengdu_4000_response_v1",
}


def config_preset_names() -> tuple[str, ...]:
    return tuple(sorted(CONFIG_PRESETS))


def add_config_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    if not any(action.dest == "config" for action in parser._actions):
        parser.add_argument(
            "--config",
            type=Path,
            default=None,
            help="Built-in config preset name or JSON config file.",
        )
    return parser


def parse_args_with_config(
    build_parser: Callable[[], argparse.ArgumentParser],
    argv: Sequence[str] | None = None,
    required: Iterable[str | tuple[str, str]] = (),
) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap_args, _remaining = bootstrap.parse_known_args(raw_argv)

    parser = add_config_argument(build_parser())
    config_preset = None
    config_source = None
    if bootstrap_args.config is not None:
        try:
            defaults = load_config_defaults(bootstrap_args.config, parser)
        except ValueError as exc:
            parser.error(str(exc))
        config_preset = _resolve_config_preset(bootstrap_args.config)
        config_source = "builtin_preset" if config_preset is not None else "json_file"
        defaults["config"] = bootstrap_args.config
        parser.set_defaults(**defaults)

    args = parser.parse_args(raw_argv)
    args.config_preset = config_preset
    args.config_source = config_source
    args.config_overrides = sorted(_cli_override_dests(parser, raw_argv))
    _validate_required_args(parser, args, required)
    return args


def load_config_defaults(config_path: str | Path, parser: argparse.ArgumentParser) -> dict[str, object]:
    raw = _load_config_object(config_path)
    if not isinstance(raw, dict):
        raise ValueError(f"config must contain a JSON/Python object: {config_path}")

    return _convert_config_defaults(raw, parser, str(config_path))


def _load_config_object(config_path: str | Path) -> dict[str, object]:
    path = Path(config_path)
    if path.is_file():
        return _load_json_config(path)

    preset_key = _resolve_config_preset(config_path)
    if preset_key is not None:
        return dict(CONFIG_PRESETS[preset_key])

    return _load_json_config(path)


def _load_json_config(config_path: Path) -> dict[str, object]:
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        names = ", ".join(config_preset_names())
        raise ValueError(f"config preset or file not found: {config_path}. Built-in presets: {names}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"config file must contain a JSON object: {config_path}")
    return raw


def _resolve_config_preset(config_path: str | Path) -> str | None:
    raw_name = str(config_path).strip()
    path = Path(raw_name.replace("\\", "/"))
    candidates = (
        raw_name,
        raw_name.replace("-", "_"),
        path.name,
        path.name.replace("-", "_"),
        path.stem,
        path.stem.replace("-", "_"),
    )
    for candidate in candidates:
        key = candidate.lower()
        key = CONFIG_PRESET_ALIASES.get(key, key)
        if key in CONFIG_PRESETS:
            return key
    return None


def _convert_config_defaults(
    raw: dict[str, object],
    parser: argparse.ArgumentParser,
    config_label: str,
) -> dict[str, object]:
    action_map = _config_action_map(parser)
    defaults: dict[str, object] = {}
    unknown_keys: list[str] = []
    for raw_key, value in raw.items():
        key = str(raw_key).lstrip("-").replace("-", "_")
        if key.startswith("_"):
            continue
        action = action_map.get(key)
        if action is None:
            unknown_keys.append(str(raw_key))
            continue
        defaults[action.dest] = _convert_config_value(action, value)
    if unknown_keys:
        raise ValueError(f"unknown config key(s) in {config_label}: {', '.join(sorted(unknown_keys))}")
    return defaults


def _config_action_map(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    action_map: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        action_map[action.dest] = action
        for option in action.option_strings:
            action_map[option.lstrip("-").replace("-", "_")] = action
    return action_map


def _cli_override_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    action_map = _config_action_map(parser)
    overrides: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        action = action_map.get(option.lstrip("-").replace("-", "_"))
        if action is None or action.dest in {"help", "config"}:
            continue
        overrides.add(action.dest)
    return overrides


def _convert_config_value(action: argparse.Action, value: object) -> object:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
        if not isinstance(value, bool):
            raise ValueError(f"config key '{action.dest}' must be true or false")
        return value

    if value is None:
        return None

    if action.nargs in ("+", "*") or isinstance(action.nargs, int):
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, Sequence):
            raw_items = list(value)
        else:
            raise ValueError(f"config key '{action.dest}' must be a list")
        items = [_convert_scalar(action, item) for item in raw_items]
        _validate_choices(action, items)
        return items

    converted = _convert_scalar(action, value)
    _validate_choices(action, converted)
    return converted


def _convert_scalar(action: argparse.Action, value: object) -> object:
    if action.type is None:
        return value
    try:
        return action.type(value)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise ValueError(f"invalid value for config key '{action.dest}': {value!r}") from exc


def _validate_choices(action: argparse.Action, value: object) -> None:
    if action.choices is None:
        return
    values = value if isinstance(value, list) else [value]
    invalid = [item for item in values if item not in action.choices]
    if invalid:
        allowed = ", ".join(str(item) for item in action.choices)
        raise ValueError(f"invalid choice for config key '{action.dest}': {invalid[0]!r}; choose from {allowed}")


def _validate_required_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    required: Iterable[str | tuple[str, str]],
) -> None:
    missing = []
    for item in required:
        if isinstance(item, tuple):
            dest, label = item
        else:
            dest, label = item, f"--{item.replace('_', '-')}"
        value = getattr(args, dest, None)
        if value is None or value == "":
            missing.append(label)
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)} (or set them in --config preset/file)")
