import unittest
import csv
import io
import json
import pickle
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from taxi_dispatch.baselines import diffusion_actions, park_actions, random_actions
from taxi_dispatch.chengdu import (
    RAW_COLUMN_ORDER,
    build_chengdu_demand,
    build_chengdu_env,
    build_chengdu_trajectory_env,
    build_chengdu_trajectory_event_demand,
    build_chengdu_trajectory_od_matrices,
    select_high_demand_trajectory_cells,
    apply_peak_hotspot_boost_to_demand,
    _scale_step_events,
)
from taxi_dispatch.config import load_config_defaults, parse_args_with_config
from taxi_dispatch.data import EmpiricalTripDemand, EventTripDemand, TripEvent
from taxi_dispatch.data import TabularDemand
from taxi_dispatch.env import EnvConfig, DispatchEnv, _OrderRequest
from taxi_dispatch.experiment import (
    _baseline_config,
    _fv_bicoord_config,
    _destination_value_targets_from_state_matrix,
    _actor_future_pressure_weight_arg,
    _future_pressure_target_from_env,
    _future_pressure_loss_weight_arg,
    _future_pressure_targets_from_next_state_matrix,
    _region_value_loss_weight_arg,
    _region_value_weight_arg,
    _apply_response_protected_move_budget_from_env,
    _dynamic_soft_expansion_action_mask_from_env,
    _spatiotemporal_pressure_from_shortage,
    _supply_sufficiency_action_mask_from_env,
    aggregate_training_epochs,
    build_parser,
    evaluate_fv_bicoord,
    evaluate_policy,
    run as run_experiment,
    train_fv_bicoord,
)
from taxi_dispatch.grid import DEFAULT_HEX_AVERAGE_SIDE_KM, DEFAULT_HEX_CENTER_SPACING_KM, HexGrid
from taxi_dispatch.mamr_data import build_mamr_data_compatible_env, load_mamr_preprocessed
from taxi_dispatch.mamr_env import MAMRDispatchEnv, MAMREnv
from taxi_dispatch.road_network import RoadNetwork, load_road_cost_matrices
from taxi_dispatch.routing import (
    assign_taxis_min_cost,
    assign_taxis_min_cost_from_costs,
    match_orders_greedy_sparse,
    match_orders_min_cost,
    match_orders_min_cost_sparse,
)
from taxi_dispatch.fv_bicoord import FVBiCoordAgent, FVBiCoordNetwork, build_road_time_adjacency


class DispatchTests(unittest.TestCase):
    def test_builtin_config_preset_loads_hyperparameters(self):
        from scripts.run_chengdu_cache_experiment import build_parser as build_cache_parser

        parser = build_cache_parser()

        defaults = load_config_defaults("chengdu_origin_response_first", parser)

        self.assertEqual(defaults["train_cache"], Path("data/processed/chengdu_train_20161101_20161121_N142_T108.pkl"))
        self.assertTrue(defaults["same_step_reposition_service"])
        self.assertTrue(defaults["reposition_before_assignment"])
        self.assertEqual(defaults["cancellation_penalty"], 8.0)

    def test_legacy_config_json_path_resolves_to_builtin_preset(self):
        from scripts.run_chengdu_cache_experiment import build_parser as build_cache_parser

        parser = build_cache_parser()

        defaults = load_config_defaults(Path("configs/chengdu_4000_mamr_fv_tuned.json"), parser)

        self.assertEqual(defaults["episodes"], 60)
        self.assertEqual(defaults["fv_road_time_weight"], 0.08)

    def test_parse_args_records_config_preset_and_cli_overrides(self):
        from scripts.run_chengdu_cache_experiment import build_parser as build_cache_parser

        args = parse_args_with_config(
            build_cache_parser,
            argv=[
                "--config",
                "chengdu_4000_response_v1",
                "--episodes",
                "3",
                "--out",
                "outputs/test_response",
            ],
            required=(
                ("train_cache", "--train-cache"),
                ("test_cache", "--test-cache"),
                ("out", "--out"),
            ),
        )

        self.assertEqual(args.config_source, "builtin_preset")
        self.assertEqual(args.config_preset, "chengdu_4000_response_v1")
        self.assertEqual(args.episodes, 3)
        self.assertEqual(args.fv_training_architecture, "rstr")
        self.assertEqual(args.future_demand_steps, 0)
        self.assertEqual(args.future_pressure_loss_weight, 0.0)
        self.assertEqual(args.future_demand_loss_weight, 0.0)
        self.assertEqual(args.actor_future_pressure_weight, 0.0)
        self.assertEqual(args.actor_future_demand_weight, 0.0)
        self.assertEqual(args.actor_region_value_weight, 0.0)
        self.assertIn("episodes", args.config_overrides)
        self.assertIn("out", args.config_overrides)
        self.assertNotIn("config", args.config_overrides)

    def test_region_value_ablation_presets_load_expected_weights(self):
        from scripts.run_chengdu_cache_experiment import build_parser as build_cache_parser

        parser = build_cache_parser()

        light = load_config_defaults("response_light_full", parser)
        light_local_global = load_config_defaults("response_light_full_local_global", parser)
        no_region = load_config_defaults("no_region_value", parser)
        region_hetero_global = load_config_defaults("response_region_hetero_global", parser)

        self.assertEqual(light["fv_training_architecture"], "rstr")
        self.assertEqual(light["region_value_loss_weight"], 0.0)
        self.assertEqual(light["actor_region_value_weight"], 0.0)
        self.assertEqual(light["future_pressure_loss_weight"], 0.0)
        self.assertEqual(light["actor_future_pressure_weight"], 0.0)
        self.assertFalse(light["fv_use_local_global_heads"])
        self.assertTrue(light_local_global["fv_use_local_global_heads"])
        self.assertEqual(light_local_global["fv_local_residual_scale"], 0.1)
        self.assertEqual(light_local_global["fv_global_bias_scale"], 0.1)
        self.assertEqual(light_local_global["future_pressure_loss_weight"], 0.0)
        self.assertEqual(no_region["region_value_loss_weight"], 0.0)
        self.assertEqual(no_region["actor_region_value_weight"], 0.0)
        self.assertEqual(no_region["region_value_weight"], 0.0)

        no_value_matching = load_config_defaults("no_value_matching", parser)
        self.assertEqual(no_value_matching["fv_matching_mode"], "greedy")
        self.assertEqual(no_value_matching["region_value_weight"], 0.0)

        self.assertEqual(region_hetero_global["fv_training_architecture"], "full")
        self.assertTrue(region_hetero_global["fv_use_local_global_heads"])
        self.assertFalse(region_hetero_global["fv_use_future_pressure_head"])
        self.assertFalse(region_hetero_global["fv_use_supply_sufficiency_gate"])
        self.assertEqual(region_hetero_global["fv_source_surplus_threshold"], 0.0)
        self.assertEqual(region_hetero_global["fv_target_shortage_threshold"], 0.0)
        self.assertEqual(region_hetero_global["fv_global_shortage_threshold"], 0.0)
        self.assertFalse(region_hetero_global["fv_use_dynamic_soft_expansion_gate"])
        self.assertEqual(region_hetero_global["fv_dynamic_source_surplus_threshold"], 0.0)
        self.assertEqual(region_hetero_global["fv_dynamic_target_shortage_threshold"], 0.0)
        self.assertEqual(region_hetero_global["fv_dynamic_global_shortage_threshold"], 0.0)
        self.assertEqual(region_hetero_global["fv_pressure_gate_strength"], 0.2)
        self.assertEqual(region_hetero_global["fv_pressure_temperature_minutes"], 10.0)
        self.assertEqual(region_hetero_global["fv_pressure_max_neighbor_time_minutes"], 30.0)
        self.assertEqual(region_hetero_global["fv_pressure_clip"], 10.0)
        self.assertFalse(region_hetero_global["fv_use_response_protected_move_budget"])
        self.assertEqual(region_hetero_global["fv_response_budget_incoming_discount"], 0.3)
        self.assertEqual(region_hetero_global["fv_response_budget_safety_buffer"], 1.0)
        self.assertEqual(region_hetero_global["fv_response_budget_min_idle_to_move"], 1.0)
        self.assertEqual(region_hetero_global["fv_response_budget_pressure_strength"], 0.2)
        self.assertEqual(region_hetero_global["fv_matching_mode"], "value_guided")
        self.assertEqual(region_hetero_global["region_value_loss_weight"], 0.0)
        self.assertEqual(region_hetero_global["future_pressure_loss_weight"], 0.0)
        self.assertEqual(region_hetero_global["actor_region_value_weight"], 0.0)
        self.assertEqual(region_hetero_global["future_gap_weight"], 0.0)
        self.assertEqual(region_hetero_global["dispatch_intensity_weight"], 0.0)
        self.assertGreater(region_hetero_global["region_value_weight"], 0.0)

    def test_fv_bicoord_config_accepts_non_value_guided_matching_mode(self):
        config = EnvConfig(
            pickup_scope="origin_and_neighbor",
            matching_mode="value_guided",
            region_value_weight=1.0,
            trip_time_weight=1.0,
        )

        fv_config = _fv_bicoord_config(config, matching_mode="min_cost")

        self.assertEqual(fv_config.pickup_scope, "origin")
        self.assertEqual(fv_config.matching_mode, "min_cost")
        self.assertEqual(fv_config.region_value_weight, 1.0)

    def test_legacy_future_head_overrides_control_pressure_defaults(self):
        from scripts.run_chengdu_cache_experiment import build_parser as build_cache_parser

        args = parse_args_with_config(
            build_cache_parser,
            argv=[
                "--config",
                "chengdu_4000_response_v1",
                "--future-gap-loss-weight",
                "0.0",
                "--future-demand-loss-weight",
                "0.0",
                "--actor-future-demand-weight",
                "0.0",
                "--out",
                "outputs/test_response",
            ],
            required=(
                ("train_cache", "--train-cache"),
                ("test_cache", "--test-cache"),
                ("out", "--out"),
            ),
        )

        self.assertEqual(_future_pressure_loss_weight_arg(args), 0.0)
        self.assertEqual(_actor_future_pressure_weight_arg(args), 0.0)

    def test_grid_has_stay_and_six_action_slots(self):
        grid = HexGrid.create(num_cells=19)
        self.assertEqual(grid.neighbors.shape, (19, 7))
        np.testing.assert_array_equal(grid.neighbors[:, 0], np.arange(19))
        self.assertTrue(np.any(grid.neighbors[:, 1:] == -1))

    def test_default_hex_grid_average_side_length_is_paper_value(self):
        grid = HexGrid.create(num_cells=7)

        self.assertAlmostEqual(grid.average_side_length_km, DEFAULT_HEX_AVERAGE_SIDE_KM)
        self.assertAlmostEqual(grid.cell_width_km, DEFAULT_HEX_CENTER_SPACING_KM)
        self.assertAlmostEqual(EnvConfig().cell_width_km, DEFAULT_HEX_CENTER_SPACING_KM)

    def test_min_cost_flow_respects_target_quotas(self):
        taxi_xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        target_xy = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        assignments = assign_taxis_min_cost([0, 1, 2], taxi_xy, {0: 1, 1: 1}, target_xy)
        self.assertEqual(len(assignments), 2)
        counts = {0: 0, 1: 0}
        for assignment in assignments:
            counts[assignment.target_cell] += 1
        self.assertEqual(counts, {0: 1, 1: 1})

    def test_order_matching_uses_pickup_radius_and_min_cost(self):
        order_xy = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        taxi_xy = np.asarray([[0.5, 0.0], [8.0, 0.0], [20.0, 0.0]], dtype=np.float32)
        matches = match_orders_min_cost(order_xy, [0, 1, 2], taxi_xy, max_pickup_distance_km=2.5)
        pairs = {(match.order_index, match.taxi_id) for match in matches}
        self.assertEqual(pairs, {(0, 0), (1, 1)})

    def test_sparse_order_matching_uses_only_candidate_edges(self):
        matches = match_orders_min_cost_sparse(
            n_orders=2,
            taxi_ids=[10, 11, 12],
            candidate_edges=[
                (0, 10, 1.5),
                (0, 11, 1.0),
                (1, 11, 1.0),
                (1, 12, 3.0),
            ],
            max_pickup_distance_km=2.5,
        )
        pairs = {(match.order_index, match.taxi_id) for match in matches}
        self.assertEqual(pairs, {(0, 10), (1, 11)})

    def test_sparse_order_matching_can_use_shaped_cost_without_losing_pickup_distance(self):
        matches = match_orders_min_cost_sparse(
            n_orders=1,
            taxi_ids=[10, 11],
            candidate_edges=[
                (0, 10, 0.5, 3.0),
                (0, 11, 1.0, -2.0),
            ],
            max_pickup_distance_km=2.5,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].taxi_id, 11)
        self.assertAlmostEqual(matches[0].pickup_distance, 1.0)

    def test_greedy_order_matching_does_not_globally_reassign(self):
        matches = match_orders_greedy_sparse(
            n_orders=2,
            taxi_ids=[10, 11],
            candidate_edges=[(0, 10, 1.0, 1.0), (0, 11, 2.0, 2.0), (1, 10, 1.5, 1.5)],
            max_pickup_distance_km=3.0,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].order_index, 0)
        self.assertEqual(matches[0].taxi_id, 10)

    def test_environment_skips_origin_taxi_when_pickup_exceeds_response_deadline(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                cell_width_km=2.5,
                fleet_size=2,
                horizon_steps=1,
                pickup_radius_km=100.0,
                future_value_weight=0.0,
                future_gap_weight=0.0,
                trip_time_weight=0.0,
                seed=11,
            ),
            grid=grid,
        )
        env.reset()
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=1,
                created_step=0,
                created_minute=0,
                pickup_xy=np.asarray([0.0, 0.0], dtype=np.float32),
            )
        ]
        env.current_minute = 0
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env.idle_by_cell[0].append(0)
        env.idle_by_cell[1].append(1)
        env.taxi_cell[0] = 0
        env.taxi_cell[1] = 1
        env.taxi_xy[0] = np.asarray([20.0, 0.0], dtype=np.float32)
        env.taxi_xy[1] = np.asarray([0.5, 0.0], dtype=np.float32)
        env._reset_step_reward_components()

        env._match_orders()

        self.assertEqual(env.served_orders, 1)
        self.assertEqual(env.taxi_cell[0], 0)
        self.assertEqual(env.taxi_cell[1], -1)
        self.assertEqual(env.waiting_orders, [])

    def test_region_value_guided_matching_cost_uses_trip_and_marginal_region_value(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=1,
                horizon_steps=1,
                trip_time_weight=0.1,
                region_value_weight=2.0,
                future_value_weight=0.0,
                future_gap_weight=0.5,
                dispatch_intensity_weight=0.0,
                seed=19,
            ),
            grid=grid,
        )
        env.taxi_cell[0] = 0
        env.region_value_guidance[1] = 2.0
        env.future_gap_guidance[1] = 1.0
        env.upper_guidance_active = True
        order = _OrderRequest(
            origin=0,
            destination=1,
            created_step=0,
            created_minute=0,
            pickup_xy=np.asarray([0.0, 0.0], dtype=np.float32),
            trip_minutes=10.0,
        )

        edge = env._candidate_match_edge(
            order_index=0,
            taxi_id=0,
            pickup_distance=0.5,
            pickup_time=4.0,
            order=order,
        )

        self.assertEqual(edge[:3], (0, 0, 0.5))
        self.assertAlmostEqual(edge[3], -3.5)

        env.taxi_cell[0] = 2
        env.region_value_guidance[2] = 1.5
        marginal_edge = env._candidate_match_edge(
            order_index=0,
            taxi_id=0,
            pickup_distance=0.5,
            pickup_time=4.0,
            order=order,
        )
        self.assertEqual(marginal_edge[:3], (0, 0, 0.5))
        self.assertAlmostEqual(marginal_edge[3], -0.5)

        env.upper_guidance_active = False
        plain_edge = env._candidate_match_edge(
            order_index=0,
            taxi_id=0,
            pickup_distance=0.5,
            pickup_time=4.0,
            order=order,
        )
        self.assertEqual(plain_edge[:3], (0, 0, 0.5))
        self.assertAlmostEqual(plain_edge[3], 4.0)

    def test_min_cost_matching_mode_ignores_upper_guidance_cost(self):
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=1,
                horizon_steps=1,
                matching_mode="min_cost",
                trip_time_weight=0.1,
                region_value_weight=2.0,
                future_gap_weight=0.5,
                seed=19,
            ),
            grid=HexGrid.create(7),
        )
        env.region_value_guidance[1] = 2.0
        env.future_gap_guidance[1] = 1.0
        env.upper_guidance_active = True
        order = _OrderRequest(
            origin=0,
            destination=1,
            created_step=0,
            created_minute=0,
            pickup_xy=np.asarray([0.0, 0.0], dtype=np.float32),
            trip_minutes=10.0,
        )

        edge = env._candidate_match_edge(0, 0, 0.5, 4.0, order)

        self.assertEqual(edge[:3], (0, 0, 0.5))
        self.assertAlmostEqual(edge[3], 4.0)

    def test_baseline_config_uses_greedy_matching(self):
        config = EnvConfig(
            future_demand_steps=3,
            region_value_weight=0.7,
            future_value_weight=0.7,
            future_gap_weight=0.5,
            dispatch_intensity_weight=0.5,
            trip_time_weight=0.2,
            matching_mode="value_guided",
        )

        baseline = _baseline_config(config)

        self.assertEqual(baseline.matching_mode, "greedy")
        self.assertEqual(baseline.future_demand_steps, 0)
        self.assertEqual(baseline.resolved_region_value_weight(), 0.0)
        self.assertEqual(baseline.future_gap_weight, 0.0)
        self.assertEqual(baseline.dispatch_intensity_weight, 0.0)
        self.assertEqual(baseline.trip_time_weight, 0.0)

    def test_matching_mode_accepts_greed_alias(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=1, horizon_steps=1, matching_mode="greed"), grid=HexGrid.create(7))
        self.assertEqual(env._matching_mode(), "greedy")

    def test_future_value_weight_is_legacy_region_value_alias(self):
        config = EnvConfig(region_value_weight=1.25, future_value_weight=9.0)
        self.assertEqual(config.resolved_region_value_weight(), 1.25)
        legacy_config = EnvConfig(future_value_weight=0.75)
        self.assertEqual(legacy_config.resolved_region_value_weight(), 0.75)

    def test_region_value_cli_aliases_prefer_canonical_names(self):
        args = build_parser().parse_args(
            [
                "--region-value-weight",
                "0.7",
                "--future-value-weight",
                "9.0",
                "--region-value-loss-weight",
                "0.2",
                "--future-value-loss-weight",
                "9.0",
            ]
        )
        self.assertEqual(_region_value_weight_arg(args), 0.7)
        self.assertEqual(_region_value_loss_weight_arg(args), 0.2)

    def test_upper_guidance_is_clipped_before_matching_cost(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, seed=23), grid=HexGrid.create(7))
        actions = park_actions(env)

        env._parse_policy_output(
            {
                "actions": actions,
                "region_value": np.full(env.grid_number, 5.0, dtype=np.float32),
                "future_pressure": np.full(env.grid_number, 5.0, dtype=np.float32),
            }
        )

        np.testing.assert_allclose(env.region_value_guidance, np.zeros(env.grid_number))
        np.testing.assert_allclose(env.future_gap_guidance, np.ones(env.grid_number))
        np.testing.assert_allclose(env.dispatch_intensity_guidance, np.ones(env.grid_number))

    def test_region_value_guidance_is_standardized_not_clipped(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, seed=23), grid=HexGrid.create(7))
        values = np.asarray([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32)

        env._parse_policy_output({"actions": park_actions(env), "region_value": values})

        self.assertLess(float(env.region_value_guidance[0]), 0.0)
        self.assertGreater(float(env.region_value_guidance[-1]), 0.0)
        self.assertAlmostEqual(float(env.region_value_guidance.mean()), 0.0, places=6)

    def test_random_actions_are_normalized_on_valid_actions(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, seed=23), grid=grid)

        actions = random_actions(env, np.random.default_rng(0))

        np.testing.assert_allclose(actions.sum(axis=1), np.ones(env.grid_number), atol=1e-6)
        self.assertTrue(np.all(actions[env.available_actions == 0] == 0.0))

    def test_random_actions_sample_each_idle_vehicle_from_valid_actions_including_stay(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=5, horizon_steps=1, seed=23), grid=grid)
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env.idle_by_cell[0] = list(range(5))
        rng = np.random.default_rng(1)
        expected_rng = np.random.default_rng(1)
        valid_actions = np.asarray(list(grid.valid_actions(0)), dtype=np.int64)
        expected_counts = np.bincount(expected_rng.choice(valid_actions, size=5), minlength=env.action_dim)

        actions = random_actions(env, rng)

        np.testing.assert_allclose(actions[0], expected_counts / 5.0, atol=1e-6)
        self.assertGreater(float(actions[0, 0]), 0.0)
        np.testing.assert_allclose(actions.sum(axis=1), np.ones(env.grid_number), atol=1e-6)
        self.assertTrue(np.all(actions[env.available_actions == 0] == 0.0))

    def test_regional_state_raw_returns_copy(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, seed=23), grid=HexGrid.create(7))

        raw = env.regional_state_raw()
        raw[0, env.idle_supply_feature_index] = 99.0

        self.assertNotEqual(env.regional_state_raw()[0, env.idle_supply_feature_index], 99.0)

    def test_diffusion_actions_handle_bad_state_values_and_normalize(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, seed=23), grid=grid)
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = np.nan
        env._state_raw[1, env.idle_supply_feature_index] = np.inf
        env._state_raw[2, env.incoming_supply_feature_index] = -np.inf

        actions = diffusion_actions(env)

        self.assertTrue(np.isfinite(actions).all())
        np.testing.assert_allclose(actions.sum(axis=1), np.ones(env.grid_number), atol=1e-6)
        self.assertTrue(np.all(actions[env.available_actions == 0] == 0.0))

    def test_diffusion_actions_uses_fractional_supply_ratio(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, seed=23), grid=grid)
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = 0.5

        actions = diffusion_actions(env)

        self.assertAlmostEqual(float(actions[0, 0]), 0.35, places=6)
        self.assertAlmostEqual(float(actions[0, 1:].sum()), 0.65, places=6)

    def test_diffusion_actions_ignore_incoming_supply(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, seed=23), grid=grid)
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.incoming_supply_feature_index] = 100.0

        actions = diffusion_actions(env)

        self.assertAlmostEqual(float(actions[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(actions[0, 1:].sum()), 0.0, places=6)

    def test_order_service_records_empty_pickup_distance_before_taxi_becomes_busy(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=1, horizon_steps=1, seed=23), grid=grid)
        env.reset()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env._set_taxi_idle_at_cell(0, 1)
        env.current_minute = 0
        env._reset_step_reward_components()
        order = _OrderRequest(
            origin=0,
            destination=2,
            created_step=0,
            created_minute=0,
            pickup_xy=grid.xy[0].copy(),
        )
        pickup_distance = env._cell_to_cell_distance(1, 0)

        env._dispatch_order_service(order, taxi_id=0, pickup_distance=pickup_distance)

        self.assertEqual(env.taxi_cell[0], -1)
        self.assertAlmostEqual(env._step_empty_distance[0], pickup_distance)

    def test_order_matching_uses_origin_taxi_before_neighbor_taxi(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(num_cells=7, fleet_size=2, horizon_steps=1, pickup_radius_km=10.0, seed=23),
            grid=grid,
        )
        env.reset()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env._set_taxi_idle_at_cell(0, 0)
        env._set_taxi_idle_at_cell(1, 1)
        env.taxi_xy[0] = grid.xy[0] + np.asarray([1.0, 0.0], dtype=np.float32)
        env.taxi_xy[1] = grid.xy[0] + np.asarray([0.1, 0.0], dtype=np.float32)
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=2,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0].copy(),
            )
        ]
        env.current_minute = 0
        env._reset_step_reward_components()

        env._match_orders()

        self.assertEqual(env.served_orders, 1)
        self.assertEqual(env.taxi_cell[0], -1)
        self.assertEqual(env.taxi_cell[1], 1)
        self.assertEqual(env.waiting_orders, [])

    def test_order_matching_uses_neighbor_taxi_only_after_origin_fails(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(num_cells=7, fleet_size=1, horizon_steps=1, pickup_radius_km=10.0, seed=23),
            grid=grid,
        )
        env.reset()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env._set_taxi_idle_at_cell(0, 1)
        env.taxi_xy[0] = grid.xy[0] + np.asarray([0.1, 0.0], dtype=np.float32)
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=2,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0].copy(),
            )
        ]
        env.current_minute = 0
        env._reset_step_reward_components()

        env._match_orders()

        self.assertEqual(env.served_orders, 1)
        self.assertEqual(env.taxi_cell[0], -1)
        self.assertEqual(env.waiting_orders, [])

    def test_origin_pickup_scope_ignores_neighbor_taxi(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=1,
                horizon_steps=1,
                pickup_radius_km=10.0,
                pickup_scope="origin",
                seed=23,
            ),
            grid=grid,
        )
        env.reset()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env._set_taxi_idle_at_cell(0, 1)
        env.taxi_xy[0] = grid.xy[0] + np.asarray([0.1, 0.0], dtype=np.float32)
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=2,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0].copy(),
            )
        ]
        env.current_minute = 0
        env._reset_step_reward_components()

        env._match_orders()

        self.assertEqual(env.served_orders, 0)
        self.assertEqual(env.taxi_cell[0], 1)
        self.assertEqual(len(env.waiting_orders), 1)

    def test_order_matching_uses_nearest_neighbor_taxi_after_origin_fails(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(num_cells=7, fleet_size=2, horizon_steps=1, pickup_radius_km=0.1, seed=23),
            grid=grid,
        )
        env.reset()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        neighbors = [int(cell) for cell in grid.neighbors[0, 1:] if cell >= 0]
        env._set_taxi_idle_at_cell(0, neighbors[0])
        env._set_taxi_idle_at_cell(1, neighbors[1])
        env.taxi_xy[0] = grid.xy[0] + np.asarray([1.0, 0.0], dtype=np.float32)
        env.taxi_xy[1] = grid.xy[0] + np.asarray([0.1, 0.0], dtype=np.float32)
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=2,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0].copy(),
            )
        ]
        env.current_minute = 0
        env._reset_step_reward_components()

        env._match_orders()

        self.assertEqual(env.served_orders, 1)
        self.assertEqual(env.taxi_cell[0], neighbors[0])
        self.assertEqual(env.taxi_cell[1], -1)
        self.assertEqual(env.waiting_orders, [])

    def test_value_guided_matching_can_prioritize_high_value_order(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=1,
                horizon_steps=1,
                matching_mode="value_guided",
                region_value_weight=10.0,
                future_gap_weight=0.0,
                dispatch_intensity_weight=0.0,
                trip_time_weight=0.0,
                seed=23,
            ),
            grid=grid,
        )
        env.reset()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env._set_taxi_idle_at_cell(0, 0)
        env.taxi_xy[0] = grid.xy[0].copy()
        env._reset_step_reward_components()
        env.region_value_guidance[:] = 0.0
        env.region_value_guidance[2] = 1.0
        env.upper_guidance_active = True
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=1,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0].copy(),
            ),
            _OrderRequest(
                origin=0,
                destination=2,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0].copy(),
            ),
        ]
        env.current_minute = 0

        env._match_orders()

        busy = [arrival for arrivals in env.busy_arrivals for arrival in arrivals]
        self.assertEqual(env.served_orders, 1)
        self.assertEqual(busy[0][1], 2)
        self.assertEqual([order.destination for order in env.waiting_orders], [1])

    def test_value_guided_matching_respects_pickup_radius(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=1,
                horizon_steps=1,
                max_wait_steps=2,
                pickup_radius_km=0.5,
                matching_mode="value_guided",
                region_value_weight=1.0,
                seed=24,
            ),
            grid=grid,
        )
        env.reset()
        neighbor = int(next(cell for cell in grid.neighbors[0, 1:] if cell >= 0))
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env._set_taxi_idle_at_cell(0, neighbor)
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=1,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0].copy(),
            )
        ]
        env.current_minute = 0
        env._reset_step_reward_components()
        env.region_value_guidance[1] = 1.0
        env.upper_guidance_active = True

        env._match_orders()

        self.assertEqual(env.served_orders, 0)
        self.assertEqual(env.taxi_cell[0], neighbor)
        self.assertEqual(len(env.waiting_orders), 1)

    def test_pickup_over_response_deadline_is_not_served_and_then_cancels(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                cell_width_km=2.5,
                fleet_size=1,
                horizon_steps=1,
                step_minutes=10,
                max_wait_steps=1,
                pickup_radius_km=100.0,
                seed=25,
            ),
            grid=grid,
        )
        env.reset()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env.idle_by_cell[0].append(0)
        env.taxi_cell[0] = 0
        env.taxi_xy[0] = np.asarray([6.0, 0.0], dtype=np.float32)
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=1,
                created_step=0,
                created_minute=0,
                pickup_xy=np.asarray([0.0, 0.0], dtype=np.float32),
            )
        ]
        env.total_orders = 1
        env.cell_orders[0] = 1
        env.current_minute = 0
        env._reset_step_reward_components()

        env._match_orders()

        self.assertEqual(env.served_orders, 0)
        self.assertEqual(env.taxi_cell[0], 0)
        self.assertEqual(len(env.waiting_orders), 1)

        env.current_minute = 10
        env._age_and_cancel_orders()

        self.assertEqual(env.cancelled_orders, 1)
        self.assertEqual(env.waiting_orders, [])
        self.assertEqual(env.metrics().response_rate, 0.0)

    def test_nearest_dispatch_failure_restores_idle_taxi(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=1,
                horizon_steps=1,
                step_minutes=10,
                max_wait_steps=1,
                pickup_radius_km=100.0,
                seed=26,
            ),
            grid=grid,
        )
        env.reset()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env.idle_by_cell[0].append(0)
        env.taxi_cell[0] = 0
        env.taxi_xy[0] = grid.xy[0].copy()
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=1,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0].copy(),
            )
        ]
        env.current_minute = 0
        env._reset_step_reward_components()
        calls = {"count": 0}

        def pickup_minutes(_taxi_id, _order, _pickup_distance):
            calls["count"] += 1
            return 0.0 if calls["count"] == 1 else 11.0

        env._road_pickup_minutes = pickup_minutes

        env._match_orders()

        self.assertEqual(env.served_orders, 0)
        self.assertEqual(env.cancelled_orders, 1)
        self.assertEqual(env.taxi_cell[0], 0)
        self.assertEqual(env.idle_by_cell[0], [0])
        self.assertEqual(env.waiting_orders, [])

    def test_value_guided_dispatch_failure_restores_idle_taxi(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=1,
                horizon_steps=1,
                step_minutes=10,
                max_wait_steps=1,
                pickup_radius_km=100.0,
                matching_mode="value_guided",
                seed=27,
            ),
            grid=grid,
        )
        env.reset()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env.idle_by_cell[0].append(0)
        env.taxi_cell[0] = 0
        env.taxi_xy[0] = grid.xy[0].copy()
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=1,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0].copy(),
            )
        ]
        env.current_minute = 0
        env._reset_step_reward_components()
        env.upper_guidance_active = True
        env._road_pickup_minutes = lambda _taxi_id, _order, _pickup_distance: 11.0

        env._match_orders()

        self.assertEqual(env.served_orders, 0)
        self.assertEqual(env.cancelled_orders, 1)
        self.assertEqual(env.taxi_cell[0], 0)
        self.assertEqual(env.idle_by_cell[0], [0])
        self.assertEqual(env.waiting_orders, [])

    def test_unmatched_order_waits_then_cancels(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, step_minutes=10, max_wait_steps=1, seed=23),
            grid=grid,
        )
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=2,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0].copy(),
            )
        ]
        env.current_minute = 0

        env._match_orders()

        self.assertEqual(len(env.waiting_orders), 1)
        env.current_minute = 10
        env._age_and_cancel_orders()
        self.assertEqual(env.cancelled_orders, 1)
        self.assertEqual(env.waiting_orders, [])

    def test_execution_feedback_enters_next_state(self):
        grid = HexGrid.create(7)
        od_counts = np.zeros((1, 7, 7), dtype=np.float32)
        duration_sum = np.zeros_like(od_counts)
        duration_count = np.zeros_like(od_counts)
        od_counts[0, 0, 1] = 1
        demand = EmpiricalTripDemand.from_counts(od_counts, duration_sum, duration_count)
        env = DispatchEnv(
            EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, step_minutes=2, max_wait_steps=0, seed=29),
            demand=demand,
            grid=grid,
        )

        _observations, state, _rewards, _actor_rewards, _done = env.advance(lambda cur_env, obs, st: park_actions(cur_env))
        state_matrix = state.reshape(env.grid_number, env.state_feature_dim)

        self.assertEqual(env.cancelled_orders, 1)
        self.assertEqual(env._state_raw[0, env.match_rate_feature_index], 0.0)
        self.assertEqual(env._state_raw[0, env.reject_rate_feature_index], 1.0)
        self.assertEqual(state_matrix[0, env.reject_rate_feature_index], 1.0)

    def test_future_shortage_target_is_not_exposed_as_current_state_feature(self):
        grid = HexGrid.create(7)
        rates = np.zeros((2, 7), dtype=np.float32)
        rates[1, 0] = 5.0
        od_probs = np.full((2, 7, 7), 1.0 / 7.0, dtype=np.float32)
        demand = TabularDemand(rates=rates, od_probs=od_probs)
        env = DispatchEnv(
            EnvConfig(num_cells=7, fleet_size=0, horizon_steps=2, future_demand_steps=1, seed=37),
            demand=demand,
            grid=grid,
        )

        env.reset()

        self.assertGreater(env._state_raw[0, env.future_demand_feature_index], 0.0)
        self.assertEqual(env._state_raw[0, env.observed_gap_feature_index], 0.0)
        self.assertEqual(env.observed_gap_target[0], 0.0)
        targets = env.auxiliary_targets()
        self.assertIn("observed_gap", targets)
        self.assertNotIn("future_gap", targets)

    def test_current_need_and_future_need_are_exposed_separately(self):
        grid = HexGrid.create(7)
        rates = np.zeros((2, 7), dtype=np.float32)
        rates[0, 1] = 2.0
        rates[1, 0] = 5.0
        od_probs = np.full((2, 7, 7), 1.0 / 7.0, dtype=np.float32)
        env = DispatchEnv(
            EnvConfig(num_cells=7, fleet_size=0, horizon_steps=2, future_demand_steps=1, seed=38),
            demand=TabularDemand(rates=rates, od_probs=od_probs),
            grid=grid,
        )

        env.reset()
        env.waiting_orders = [
            _OrderRequest(
                origin=1,
                destination=0,
                created_step=0,
                created_minute=0,
                pickup_xy=np.asarray([0.0, 0.0], dtype=np.float32),
            ),
            _OrderRequest(
                origin=1,
                destination=0,
                created_step=0,
                created_minute=0,
                pickup_xy=np.asarray([0.0, 0.0], dtype=np.float32),
            ),
        ]
        env._update_state_and_rewards()

        self.assertEqual(env._state_raw[0, env.current_need_feature_index], 0.0)
        self.assertEqual(env._state_raw[0, env.future_need_feature_index], 5.0)
        self.assertEqual(env._state_raw[1, env.current_need_feature_index], 2.0)
        self.assertEqual(env._state_raw[1, env.future_need_feature_index], 0.0)
        targets = env.auxiliary_targets()
        np.testing.assert_allclose(targets["observed_gap"], env._state_raw[:, env.current_need_feature_index])
        np.testing.assert_allclose(targets["future_need"], env._state_raw[:, env.future_need_feature_index])

    def test_regional_state_sanitizes_nonfinite_features(self):
        grid = HexGrid.create(7)
        rates = np.zeros((2, 7), dtype=np.float32)
        od_probs = np.full((2, 7, 7), 1.0 / 7.0, dtype=np.float32)
        demand = TabularDemand(rates=rates, od_probs=od_probs)
        env = DispatchEnv(
            EnvConfig(num_cells=7, fleet_size=0, horizon_steps=2, future_demand_steps=1, seed=39),
            demand=demand,
            grid=grid,
        )
        env.demand.rates[1, 0] = np.nan
        env.demand.rates[1, 1] = np.inf
        env._feedback_match_rate[0] = np.nan
        env._feedback_reject_rate[1] = np.inf
        env._feedback_empty_distance[2] = np.inf
        env._feedback_execution_bias[3] = -np.inf
        env.feature_scales[0] = 0.0

        env._update_state_and_rewards()

        self.assertTrue(np.isfinite(env.state).all())
        self.assertTrue(np.isfinite(env.observations).all())
        self.assertTrue(np.isfinite(env.raw_reward).all())
        self.assertTrue(np.isfinite(env.actor_rewards).all())

    def test_future_demand_lookahead_pads_episode_tail(self):
        grid = HexGrid.create(7)
        rates = np.zeros((2, 7), dtype=np.float32)
        rates[1, 0] = 2.0
        od_probs = np.full((2, 7, 7), 1.0 / 7.0, dtype=np.float32)
        demand = TabularDemand(rates=rates, od_probs=od_probs)
        env = DispatchEnv(
            EnvConfig(num_cells=7, fleet_size=0, horizon_steps=2, future_demand_steps=3, seed=37),
            demand=demand,
            grid=grid,
        )

        env.step_index = 1

        np.testing.assert_array_equal(env._future_demand_by_cell(), rates[1] * 3.0)

    def test_advance_returns_realized_step_actor_rewards(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=5, horizon_steps=1, seed=41), grid=HexGrid.create(7))
        captured: list[np.ndarray] = []

        def policy(cur_env, obs, state):
            captured.append(cur_env.actor_rewards.copy())
            return park_actions(cur_env)

        _obs, _state, _critic_rewards, actor_rewards, _done = env.advance(policy)

        self.assertEqual(len(captured), 1)
        np.testing.assert_allclose(actor_rewards, env.actor_rewards)

    def test_advance_actor_rewards_are_post_execution_refresh(self):
        grid = HexGrid.create(7)
        rates = np.zeros((2, 7), dtype=np.float32)
        rates[0, 1] = 1.0
        od_probs = np.full((2, 7, 7), 1.0 / 7.0, dtype=np.float32)
        od_probs[:, 1, :] = 0.0
        od_probs[:, 1, 0] = 1.0
        initial = np.zeros(7, dtype=np.float32)
        initial[0] = 1.0
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=1,
                horizon_steps=2,
                max_wait_steps=2,
                pickup_radius_km=100.0,
                relocation_cost_weight=0.0,
                future_imbalance_weight=0.0,
                seed=42,
            ),
            demand=TabularDemand(rates=rates, od_probs=od_probs),
            grid=grid,
            initial_taxi_distribution=initial,
        )
        env.reset()
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])
        captured: list[np.ndarray] = []

        def policy(cur_env, obs, state):
            captured.append(cur_env.actor_rewards.copy())
            return park_actions(cur_env)

        _obs, _state, _critic_rewards, actor_rewards, _done = env.advance(policy)

        np.testing.assert_allclose(actor_rewards, env.actor_rewards)
        self.assertGreater(float(actor_rewards[0, action_to_1]), 0.0)
        self.assertNotAlmostEqual(float(actor_rewards[0, action_to_1]), float(captured[0][0, action_to_1]))

    def test_advance_matches_current_orders_before_repositioning_idle_taxis(self):
        grid = HexGrid.create(7)
        rates = np.zeros((1, 7), dtype=np.float32)
        rates[0, 0] = 1.0
        od_probs = np.full((1, 7, 7), 1.0 / 7.0, dtype=np.float32)
        od_probs[0, 0, :] = 0.0
        od_probs[0, 0, 1] = 1.0
        initial = np.zeros(7, dtype=np.float32)
        initial[0] = 1.0
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=1,
                horizon_steps=1,
                pickup_scope="origin",
                same_step_reposition_service=False,
                pickup_radius_km=100.0,
                seed=44,
            ),
            demand=TabularDemand(rates=rates, od_probs=od_probs),
            grid=grid,
            initial_taxi_distribution=initial,
        )
        env.reset()
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])

        def policy(cur_env, obs, state):
            actions = park_actions(cur_env)
            actions[0] = 0.0
            actions[0, action_to_1] = 1.0
            return actions

        env.advance(policy)

        self.assertEqual(env.served_orders, 1)
        self.assertEqual(env.cancelled_orders, 0)
        self.assertEqual(env.repositioned, 0)
        self.assertEqual(env.waiting_orders, [])
        idle_ids = [taxi_id for cell in env.idle_by_cell for taxi_id in cell]
        busy_ids = [taxi_id for arrivals in env.busy_arrivals for taxi_id, _cell, _duration in arrivals]
        reposition_ids = [taxi_id for arrivals in env.reposition_arrivals for taxi_id, _cell in arrivals]
        self.assertEqual(reposition_ids, [])
        self.assertCountEqual(idle_ids + busy_ids + reposition_ids, [0])

    def test_reposition_policy_observes_post_assignment_idle_supply(self):
        grid = HexGrid.create(7)
        rates = np.zeros((1, 7), dtype=np.float32)
        rates[0, 0] = 1.0
        od_probs = np.full((1, 7, 7), 1.0 / 7.0, dtype=np.float32)
        od_probs[0, 0, :] = 0.0
        od_probs[0, 0, 1] = 1.0
        initial = np.zeros(7, dtype=np.float32)
        initial[0] = 1.0
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=1,
                horizon_steps=1,
                pickup_scope="origin",
                same_step_reposition_service=False,
                pickup_radius_km=100.0,
                seed=45,
            ),
            demand=TabularDemand(rates=rates, od_probs=od_probs),
            grid=grid,
            initial_taxi_distribution=initial,
        )
        env.reset()
        pre_idle_supply: list[float] = []
        post_idle_supply: list[float] = []
        post_actor_rewards: list[np.ndarray] = []

        def guidance_policy(cur_env, obs, state):
            pre_idle_supply.append(float(cur_env._state_raw[0, cur_env.idle_supply_feature_index]))
            return park_actions(cur_env)

        def reposition_policy(cur_env, obs, state):
            post_idle_supply.append(float(cur_env._state_raw[0, cur_env.idle_supply_feature_index]))
            post_actor_rewards.append(cur_env.actor_rewards.copy())
            return park_actions(cur_env)

        _obs, _state, _critic_rewards, actor_rewards, _done = env.advance(
            guidance_policy,
            reposition_policy=reposition_policy,
        )

        self.assertEqual(pre_idle_supply, [1.0])
        self.assertEqual(post_idle_supply, [0.0])
        self.assertEqual(env.served_orders, 1)
        self.assertEqual(env.repositioned, 0)
        np.testing.assert_allclose(actor_rewards, post_actor_rewards[0])

    def test_reposition_before_assignment_can_rescue_origin_only_order(self):
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5)
        rates = np.zeros((1, 2), dtype=np.float32)
        rates[0, 1] = 1.0
        od_probs = np.zeros((1, 2, 2), dtype=np.float32)
        od_probs[0, 1, 0] = 1.0
        initial = np.asarray([1.0, 0.0], dtype=np.float32)
        env = DispatchEnv(
            EnvConfig(
                num_cells=2,
                cell_width_km=2.5,
                fleet_size=1,
                horizon_steps=1,
                step_minutes=10,
                max_wait_steps=1,
                pickup_scope="origin",
                same_step_reposition_service=True,
                reposition_before_assignment=True,
                pickup_radius_km=100.0,
                travel_speed_kmph=30.0,
                seed=47,
            ),
            demand=TabularDemand(rates=rates, od_probs=od_probs),
            grid=grid,
            initial_taxi_distribution=initial,
        )
        env.reset()
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])

        def policy(cur_env, obs, state):
            actions = park_actions(cur_env)
            actions[0] = 0.0
            actions[0, action_to_1] = 1.0
            return actions

        env.advance(policy)

        self.assertEqual(env.served_orders, 1)
        self.assertEqual(env.cancelled_orders, 0)
        self.assertEqual(env.repositioned, 1)
        self.assertEqual(env.waiting_orders, [])
        self.assertGreater(env.response_time_seconds, 5.0 * 60.0)

    def test_step_warns_as_deprecated_legacy_api(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, seed=42), grid=HexGrid.create(7))
        env.reset()

        with self.assertWarnsRegex(FutureWarning, "advance"):
            env.step(park_actions(env))

    def test_actor_rewards_use_current_target_need_not_future_oracle_need(self):
        grid = HexGrid.create(7)
        rates = np.zeros((2, 7), dtype=np.float32)
        rates[1, 1] = 5.0
        demand = TabularDemand(rates=rates, od_probs=np.full((2, 7, 7), 1.0 / 7.0, dtype=np.float32))
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=0,
                horizon_steps=2,
                future_demand_steps=1,
                relocation_cost_weight=0.0,
                future_imbalance_weight=0.0,
                seed=43,
            ),
            demand=demand,
            grid=grid,
        )
        env.reset()
        env.config.reward_action_move_cost = 0.0
        env._update_state_and_rewards()
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])

        self.assertGreater(env._state_raw[1, env.future_demand_feature_index], 0.0)
        self.assertAlmostEqual(float(env.actor_rewards[0, action_to_1]), 0.0)

    def test_stay_reward_is_zero_without_current_need(self):
        grid = HexGrid.create(7)
        rates = np.zeros((1, 7), dtype=np.float32)
        od_probs = np.full((1, 7, 7), 1.0 / 7.0, dtype=np.float32)
        initial = np.zeros(7, dtype=np.float32)
        initial[0] = 1.0
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=20,
                horizon_steps=1,
                relocation_cost_weight=0.0,
                seed=48,
            ),
            demand=TabularDemand(rates=rates, od_probs=od_probs),
            grid=grid,
            initial_taxi_distribution=initial,
        )

        env.reset()

        self.assertEqual(float(env._state_raw[0, env.demand_feature_index]), 0.0)
        self.assertGreater(float(env._state_raw[0, env.idle_supply_feature_index]), 0.0)
        self.assertAlmostEqual(float(env.actor_rewards[0, 0]), 0.0)

    def test_rewards_do_not_include_future_imbalance_penalty(self):
        grid = HexGrid.create(7)
        rates = np.zeros((2, 7), dtype=np.float32)
        rates[1, 0] = 5.0
        demand = TabularDemand(rates=rates, od_probs=np.full((2, 7, 7), 1.0 / 7.0, dtype=np.float32))
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=0,
                horizon_steps=2,
                future_demand_steps=1,
                relocation_cost_weight=0.0,
                future_imbalance_weight=1.0,
                seed=44,
            ),
            demand=demand,
            grid=grid,
        )
        env.reset()
        env.config.reward_action_move_cost = 0.0
        env._update_state_and_rewards()
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])

        self.assertGreater(env._state_raw[0, env.future_demand_feature_index], 0.0)
        self.assertAlmostEqual(float(env.raw_reward[0]), 0.0)
        self.assertAlmostEqual(float(env.actor_rewards[0, action_to_1]), 0.0)

    def test_critic_reward_uses_normalized_match_profit_and_relocation_cost(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=0,
                horizon_steps=1,
                relocation_cost_weight=10.0,
                idle_time_penalty_weight=10.0,
                execution_mismatch_penalty=10.0,
                seed=45,
            ),
            grid=grid,
        )
        env.reset()
        env._step_service_revenue[0] = 100.0
        env._step_wait_minutes[0] = 10.0
        env._step_wait_penalty_minutes[0] = 10.0
        env._step_cancelled_orders[0] = 2.0
        env._step_matched_orders[0] = 4.0
        env._step_order_requests[0] = 5.0
        env._step_relocation_cost[0] = 0.25
        env._feedback_idle_time[0] = 999.0
        env._feedback_execution_bias[0] = 999.0

        env._update_state_and_rewards()

        match_profit_norm = 100.0 / (5.0 * 25.0)
        repo_cost = 0.25 / env.config.step_minutes
        local_reward = match_profit_norm - 0.05 * repo_cost
        global_reward = local_reward / env.grid_number
        expected = 0.75 * local_reward + 0.25 * global_reward
        self.assertAlmostEqual(float(env.raw_reward[0]), expected)
        self.assertAlmostEqual(float(env.raw_reward.sum()), local_reward)
        self.assertAlmostEqual(env.reward_debug["match_profit_norm_mean"], match_profit_norm / env.grid_number)
        self.assertIn("repo_balance_improve_mean", env.reward_debug)

    def test_match_reward_decreases_when_travel_cost_increases(self):
        grid = HexGrid.create(7)
        base_env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, seed=45), grid=grid)
        costly_env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, seed=45), grid=grid)
        for env in (base_env, costly_env):
            env.reset()
            env._step_service_revenue[0] = 100.0
            env._step_service_minutes[0] = 10.0
            env._step_matched_orders[0] = 1.0
            env._step_order_requests[0] = 1.0
        costly_env.config.reward_match_time_cost_weight = 2.0

        base_env._update_state_and_rewards()
        costly_env._update_state_and_rewards()

        self.assertGreater(float(base_env.raw_reward[0]), float(costly_env.raw_reward[0]))
        self.assertGreater(base_env.reward_debug["match_profit_norm_mean"], costly_env.reward_debug["match_profit_norm_mean"])

    def test_repo_balance_reward_tracks_neighborhood_improvement_and_regression(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, seed=45), grid=HexGrid.create(7))
        before_supply = np.zeros(env.grid_number, dtype=np.float32)
        before_demand = np.zeros(env.grid_number, dtype=np.float32)
        after_supply = np.zeros(env.grid_number, dtype=np.float32)
        worse_supply = np.zeros(env.grid_number, dtype=np.float32)
        before_supply[0] = 4.0
        before_demand[0] = 1.0
        before_demand[1] = 3.0
        after_supply[0] = 2.0
        after_supply[1] = 2.0
        worse_supply[0] = 5.0

        env._repo_reward_supply_before = before_supply
        env._repo_reward_demand_before = before_demand
        _before, _after, improve = env._repo_balance_reward_components(after_supply, before_demand)
        env._repo_reward_supply_before = before_supply
        env._repo_reward_demand_before = before_demand
        _before, _after, regression = env._repo_balance_reward_components(worse_supply, before_demand)

        self.assertGreater(float(improve[0]), 0.0)
        self.assertLess(float(regression[0]), 0.0)
        zero_balance = env._balance_degree(
            np.zeros(env.grid_number, dtype=np.float32),
            np.zeros(env.grid_number, dtype=np.float32),
        )
        self.assertTrue(np.isfinite(zero_balance).all())

    def test_repo_reward_uses_decision_demand_snapshot_not_final_waiting_demand(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=2, seed=45), grid=HexGrid.create(7))
        env.reset()
        decision_demand = np.zeros(env.grid_number, dtype=np.float32)
        decision_demand[1] = 3.0
        final_waiting_demand = np.zeros(env.grid_number, dtype=np.float32)
        final_waiting_demand[0] = 99.0
        future_demand = np.zeros(env.grid_number, dtype=np.float32)
        env._repo_reward_demand_before = decision_demand.copy()

        repo_demand = env._repo_reward_demand_for_update(final_waiting_demand, future_demand)

        np.testing.assert_allclose(repo_demand, decision_demand)

    def test_capture_reposition_reward_before_ignores_post_assignment_state_raw_demand(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=2, seed=45), grid=HexGrid.create(7))
        env.reset()
        env.config.reward_repo_future_demand_weight = 0.0
        env._step_decision_demand[:] = 0.0
        env._step_decision_demand[1] = 3.0
        env._state_raw[:, env.demand_feature_index] = 0.0
        env._state_raw[0, env.demand_feature_index] = 99.0
        env._state_raw[:, env.future_demand_feature_index] = 0.0
        env._state_raw[2, env.future_demand_feature_index] = 99.0

        env._capture_reposition_reward_before()

        expected = np.zeros(env.grid_number, dtype=np.float32)
        expected[1] = 3.0
        np.testing.assert_allclose(env._repo_reward_demand_before, expected)

    def test_repo_reward_can_include_future_demand_with_decision_demand(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=2, seed=45), grid=HexGrid.create(7))
        env.reset()
        env.config.reward_repo_future_demand_weight = 0.25
        env._step_decision_demand[:] = 0.0
        env._step_decision_demand[1] = 2.0
        final_waiting_demand = np.zeros(env.grid_number, dtype=np.float32)
        final_waiting_demand[0] = 99.0
        future_demand = np.zeros(env.grid_number, dtype=np.float32)
        future_demand[2] = 4.0

        repo_demand = env._repo_reward_demand_for_update(final_waiting_demand, future_demand)

        expected = np.zeros(env.grid_number, dtype=np.float32)
        expected[1] = 2.0
        expected[2] = 1.0
        np.testing.assert_allclose(repo_demand, expected)

    def test_relocation_cost_is_recorded_as_empty_minutes(self):
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5)
        env = DispatchEnv(
            EnvConfig(
                num_cells=2,
                cell_width_km=2.5,
                fleet_size=1,
                horizon_steps=1,
                travel_speed_kmph=30.0,
                seed=45,
            ),
            grid=grid,
        )
        env.reset()
        env._reset_step_reward_components()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env.idle_by_cell[0] = [0]
        env.taxi_cell[0] = 0
        env.taxi_xy[0] = grid.xy[0].copy()
        env.current_minute = 0
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])
        actions = park_actions(env)
        actions[0] = 0.0
        actions[0, action_to_1] = 1.0

        env._apply_repositioning(actions)

        expected_distance = env._cell_to_cell_distance(0, 1)
        expected_minutes = float(max(1, int(np.ceil(env._distance_to_travel_minutes(expected_distance)))))
        self.assertAlmostEqual(float(env._step_empty_distance[0]), expected_distance)
        self.assertAlmostEqual(float(env._step_relocation_cost[0]), expected_minutes)

    def test_reward_outputs_are_finite_clipped_shaped_and_debugged(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, seed=45), grid=HexGrid.create(7))
        env.reset()
        env._step_service_revenue[:] = 1.0e9
        env._step_matched_orders[:] = 1.0
        env._step_order_requests[:] = 1.0

        env._update_state_and_rewards()

        self.assertEqual(env.raw_reward.shape, (env.grid_number,))
        self.assertEqual(env.actor_rewards.shape, (env.grid_number, env.action_dim))
        self.assertTrue(np.isfinite(env.raw_reward).all())
        self.assertTrue(np.isfinite(env.actor_rewards).all())
        self.assertLessEqual(float(env.raw_reward.max()), 5.0)
        self.assertGreaterEqual(float(env.raw_reward.min()), -5.0)
        self.assertLessEqual(float(env.actor_rewards.max()), 5.0)
        self.assertGreaterEqual(float(env.actor_rewards.min()), -5.0)
        for key in (
            "match_profit_mean",
            "match_profit_norm_mean",
            "repo_balance_before_mean",
            "repo_balance_after_mean",
            "repo_balance_improve_mean",
            "repo_cost_mean",
            "local_reward_mean",
            "global_reward",
            "critic_reward_mean",
            "actor_reward_mean",
        ):
            self.assertIn(key, env.reward_debug)

    def test_service_revenue_uses_base_distance_and_time_fare(self):
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5)
        road_distance = np.asarray([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32)
        road_time = np.asarray([[0.0, 8.0], [8.0, 0.0]], dtype=np.float32)
        env = DispatchEnv(
            EnvConfig(
                num_cells=2,
                cell_width_km=2.5,
                fleet_size=1,
                horizon_steps=1,
                pickup_radius_km=100.0,
                fare_base=8.0,
                fare_per_km=1.9,
                fare_per_minute=0.5,
                seed=46,
            ),
            grid=grid,
            road_distance_matrix=road_distance,
            road_time_matrix=road_time,
        )
        env.reset()
        order = _OrderRequest(
            origin=0,
            destination=1,
            created_step=0,
            created_minute=0,
            pickup_xy=grid.xy[0],
            dropoff_xy=grid.xy[1],
        )

        self.assertTrue(env._dispatch_order_service(order, taxi_id=0, pickup_distance=0.0))

        expected_fare = 8.0 + 1.9 * 2.0 + 0.5 * 8.0
        self.assertAlmostEqual(env.gmv, expected_fare)
        self.assertAlmostEqual(float(env._step_service_revenue[0]), expected_fare, places=6)

    def test_wait_penalty_uses_tolerance_threshold(self):
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=0,
                horizon_steps=1,
                wait_time_tolerance_minutes=5.0,
                wait_time_penalty_exponent=2.0,
                seed=45,
            ),
            grid=HexGrid.create(7),
        )

        self.assertAlmostEqual(env._quality_wait_penalty_minutes(4.0), 0.0)
        self.assertAlmostEqual(env._quality_wait_penalty_minutes(10.0), 2.5)

    def test_actor_rewards_include_normalized_order_matching_reward(self):
        grid = HexGrid.create(7)
        initial = np.zeros(7, dtype=np.float32)
        initial[0] = 1.0
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=1,
                horizon_steps=1,
                relocation_cost_weight=0.0,
                seed=46,
            ),
            grid=grid,
            initial_taxi_distribution=initial,
        )
        env.reset()
        env._step_service_revenue[0] = 1000.0
        env._step_matched_orders[0] = 1.0
        env._step_order_requests[0] = 1.0

        env._update_state_and_rewards()

        local_reward = 1.0
        global_reward = local_reward / env.grid_number
        expected_stay = 0.8 * local_reward + 0.2 * global_reward
        self.assertAlmostEqual(float(env.actor_rewards[0, 0]), expected_stay, places=6)
        self.assertLess(float(env.actor_rewards[0, 0]), 2.0)

    def test_actor_move_reward_uses_normalized_action_cost_without_local_reward(self):
        grid = HexGrid.create(7)
        initial = np.zeros(7, dtype=np.float32)
        initial[0] = 1.0
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=5,
                horizon_steps=1,
                relocation_cost_weight=0.0,
                seed=48,
            ),
            grid=grid,
            initial_taxi_distribution=initial,
        )
        env.reset()
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=1,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0],
            )
        ]

        env._update_state_and_rewards(demand_cutoff_minute=env._step_end_minute())
        move_action = next(action for action, dst in enumerate(grid.neighbors[0]) if action > 0 and dst >= 0)

        expected_cost = float(env._normalized_action_move_costs()[0, move_action])
        self.assertAlmostEqual(float(env.actor_rewards[0, move_action]), -expected_cost)

    def test_actor_rewards_do_not_reward_unserved_current_shortage(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=0,
                horizon_steps=1,
                cancellation_penalty=4.5,
                seed=49,
            ),
            grid=grid,
        )
        env.reset()
        env.config.reward_action_move_cost = 0.0
        env.waiting_orders = [
            _OrderRequest(
                origin=0,
                destination=1,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0],
            ),
            _OrderRequest(
                origin=0,
                destination=1,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0],
            ),
        ]

        env._update_state_and_rewards(demand_cutoff_minute=env._step_end_minute())

        self.assertAlmostEqual(float(env.actor_rewards[0, 0]), 0.0)

    def test_actor_move_reward_uses_dst_src_and_global_reward_mix(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=0,
                horizon_steps=1,
                relocation_cost_weight=0.0,
                seed=50,
            ),
            grid=grid,
        )
        env.reset()
        env.config.reward_action_move_cost = 0.0
        env._step_service_revenue[1] = 100.0
        env._step_matched_orders[1] = 1.0
        env._step_order_requests[1] = 1.0
        env._update_state_and_rewards()
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])

        dst_local_reward = 1.0
        src_local_reward = 0.0
        global_reward = dst_local_reward / env.grid_number
        expected = 0.6 * dst_local_reward + 0.2 * src_local_reward + 0.2 * global_reward
        self.assertAlmostEqual(float(env.actor_rewards[0, action_to_1]), expected, places=6)

    def test_min_cost_flow_can_use_road_cost_matrix(self):
        costs = np.asarray([[9.0, 1.0], [1.0, 9.0]], dtype=np.float32)
        assignments = assign_taxis_min_cost_from_costs([10, 11], {0: 1, 1: 1}, costs)
        pairs = {(assignment.taxi_id, assignment.target_cell) for assignment in assignments}
        self.assertEqual(pairs, {(10, 1), (11, 0)})

    def test_road_network_shortest_paths_and_env_integration(self):
        with TemporaryDirectory() as tmp:
            nodes_path = Path(tmp) / "nodes.csv"
            edges_path = Path(tmp) / "edges.csv"
            nodes_path.write_text(
                "node,lon,lat\n"
                "a,104.0000,30.6500\n"
                "b,104.0100,30.6500\n"
                "c,104.0200,30.6500\n",
                encoding="utf-8",
            )
            edges_path.write_text(
                "u,v,length_m,speed_kmph\n"
                "a,b,1000,60\n"
                "b,c,1000,60\n"
                "a,c,5000,60\n"
                "c,b,1000,60\n"
                "b,a,1000,60\n",
                encoding="utf-8",
            )
            road = RoadNetwork.from_csv(nodes_path, edges_path)
            self.assertAlmostEqual(road.shortest_path_distance_km("a", "c"), 2.0)
            road._node_kdtree = None
            self.assertEqual(
                road.nearest_nodes(np.asarray([104.001, 104.019]), np.asarray([30.65, 30.65])),
                ["a", "c"],
            )

            grid = HexGrid.from_axial_coords(HexGrid.create(7).coords, cell_width_km=2.5, projection_origin=(104.0, 30.65))
            od_counts = np.zeros((1, grid.num_cells, grid.num_cells), dtype=np.float32)
            duration_sum = np.zeros_like(od_counts)
            duration_count = np.zeros_like(od_counts)
            od_counts[0, 0, 1] = 1
            duration_sum[0, 0, 1] = 5
            duration_count[0, 0, 1] = 1
            demand = EmpiricalTripDemand.from_counts(od_counts, duration_sum, duration_count)
            env = DispatchEnv(
                EnvConfig(num_cells=grid.num_cells, fleet_size=1, horizon_steps=1, seed=3),
                demand=demand,
                grid=grid,
                initial_taxi_distribution=np.asarray([1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                road_network=road,
            )
            metrics, _env = evaluate_policy(
                env.config,
                lambda cur_env, obs, st: park_actions(cur_env),
                episodes=1,
                seed=3,
                demand=demand,
                grid=grid,
                initial_taxi_distribution=env.initial_taxi_distribution,
                road_network=road,
            )
            self.assertEqual(metrics.orders, 1)
            self.assertEqual(metrics.cancellations, 0)

    def test_road_matrix_reports_fallback_counts(self):
        with TemporaryDirectory() as tmp:
            grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5, projection_origin=(104.0, 30.65))
            nodes_path = Path(tmp) / "nodes.csv"
            edges_path = Path(tmp) / "edges.csv"
            nodes_path.write_text(
                "node,lon,lat\n"
                f"a,{float(grid.lonlat[0, 0]):.8f},{float(grid.lonlat[0, 1]):.8f}\n"
                f"b,{float(grid.lonlat[1, 0]):.8f},{float(grid.lonlat[1, 1]):.8f}\n",
                encoding="utf-8",
            )
            edges_path.write_text("u,v,length_m,speed_kmph\n", encoding="utf-8")
            road = RoadNetwork.from_csv(nodes_path, edges_path)

            matrices = road.cell_cost_matrices(grid)

            self.assertEqual(matrices.fallback_distance_count, 2)
            self.assertEqual(matrices.fallback_time_count, 2)

    def test_precomputed_road_time_matrix_overrides_empirical_trip_minutes(self):
        with TemporaryDirectory() as tmp:
            grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5, projection_origin=(104.0, 30.65))
            nodes_path = Path(tmp) / "nodes.csv"
            edges_path = Path(tmp) / "edges.csv"
            nodes_path.write_text(
                "node,lon,lat\n"
                f"a,{float(grid.lonlat[0, 0]):.8f},{float(grid.lonlat[0, 1]):.8f}\n"
                f"b,{float(grid.lonlat[1, 0]):.8f},{float(grid.lonlat[1, 1]):.8f}\n",
                encoding="utf-8",
            )
            edges_path.write_text(
                "u,v,length_m,speed_kmph\n"
                "a,b,1200,60\n"
                "b,a,1200,60\n",
                encoding="utf-8",
            )
            road = RoadNetwork.from_csv(nodes_path, edges_path)
            matrices = road.cell_cost_matrices(grid)

            matrix_path = Path(tmp) / "road_cells.npz"
            matrices.save_npz(matrix_path)
            loaded = load_road_cost_matrices(matrix_path)

            env = DispatchEnv(
                EnvConfig(num_cells=2, cell_width_km=2.5, fleet_size=1, horizon_steps=1, step_minutes=10, seed=5),
                grid=grid,
                road_distance_matrix=loaded.distance_km,
                road_time_matrix=loaded.time_minutes,
            )
            order = _OrderRequest(
                origin=0,
                destination=1,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[0],
                dropoff_xy=grid.xy[1],
                trip_minutes=99.0,
            )

            self.assertAlmostEqual(float(loaded.distance_km[0, 1]), 1.2, places=3)
            self.assertAlmostEqual(float(loaded.time_minutes[0, 1]), 1.2, places=3)
            self.assertAlmostEqual(env._road_trip_distance(order), 1.5, places=3)
            self.assertAlmostEqual(env._road_trip_minutes(order), 1.2, places=3)

    def test_road_matrix_same_cell_pickup_uses_vehicle_order_geometry(self):
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5)
        road_distance = np.asarray([[0.0, 2.5], [2.5, 0.0]], dtype=np.float32)
        road_time = np.asarray([[0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
        env = DispatchEnv(
            EnvConfig(
                num_cells=2,
                cell_width_km=2.5,
                fleet_size=1,
                horizon_steps=1,
                pickup_radius_km=100.0,
                travel_speed_kmph=30.0,
                seed=5,
            ),
            grid=grid,
            road_distance_matrix=road_distance,
            road_time_matrix=road_time,
        )
        env.reset()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env.idle_by_cell[0].append(0)
        env.taxi_cell[0] = 0
        env.taxi_xy[0] = grid.xy[0] + np.asarray([1.0, 0.0], dtype=np.float32)
        order = _OrderRequest(
            origin=0,
            destination=1,
            created_step=0,
            created_minute=0,
            pickup_xy=grid.xy[0].copy(),
            dropoff_xy=grid.xy[1].copy(),
        )
        env.current_minute = 0
        env._reset_step_reward_components()

        taxi_ids, candidate_edges = env._pickup_candidate_edges([order], pickup_scope="origin")

        self.assertEqual(taxi_ids, [0])
        self.assertEqual(len(candidate_edges), 1)
        self.assertAlmostEqual(candidate_edges[0][2], 1.0)
        self.assertAlmostEqual(candidate_edges[0][3], 2.0)

        env.waiting_orders = [order]
        env._match_orders()

        self.assertEqual(env.served_orders, 1)
        self.assertAlmostEqual(env.response_time_seconds, 120.0)
        self.assertAlmostEqual(env._step_empty_distance[0], 1.0)

    def test_road_matrix_same_cell_pickup_has_positive_floor_for_coincident_points(self):
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5)
        road_distance = np.asarray([[0.0, 2.5], [2.5, 0.0]], dtype=np.float32)
        road_time = np.asarray([[0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
        env = DispatchEnv(
            EnvConfig(
                num_cells=2,
                cell_width_km=2.5,
                fleet_size=1,
                horizon_steps=1,
                pickup_radius_km=100.0,
                travel_speed_kmph=30.0,
                seed=5,
            ),
            grid=grid,
            road_distance_matrix=road_distance,
            road_time_matrix=road_time,
        )
        env.reset()
        env.taxi_cell[0] = 0
        env.taxi_xy[0] = grid.xy[0].copy()
        order = _OrderRequest(
            origin=0,
            destination=1,
            created_step=0,
            created_minute=0,
            pickup_xy=grid.xy[0].copy(),
            dropoff_xy=grid.xy[1].copy(),
        )

        pickup_distance = env._pickup_distance_for_taxi(0, order)
        pickup_minutes = env._road_pickup_minutes(0, order, pickup_distance)

        self.assertGreater(pickup_distance, 0.0)
        self.assertGreater(pickup_minutes, 0.0)

    def test_road_matrix_same_cell_trip_uses_order_trip_minutes(self):
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5)
        road_distance = np.asarray([[0.0, 2.5], [2.5, 0.0]], dtype=np.float32)
        road_time = np.asarray([[0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
        env = DispatchEnv(
            EnvConfig(num_cells=2, cell_width_km=2.5, fleet_size=1, horizon_steps=1, seed=5),
            grid=grid,
            road_distance_matrix=road_distance,
            road_time_matrix=road_time,
        )
        order = _OrderRequest(
            origin=0,
            destination=0,
            created_step=0,
            created_minute=0,
            pickup_xy=grid.xy[0].copy(),
            dropoff_xy=grid.xy[0] + np.asarray([0.2, 0.0], dtype=np.float32),
            trip_minutes=8.0,
        )

        self.assertAlmostEqual(env._road_trip_minutes(order), 8.0)

    def test_road_matrix_same_cell_sample_trip_minutes_uses_empirical_mean(self):
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5)
        od_counts = np.zeros((1, 2, 2), dtype=np.float32)
        trip_minutes_sum = np.zeros_like(od_counts)
        trip_minutes_count = np.zeros_like(od_counts)
        od_counts[0, 0, 0] = 2.0
        trip_minutes_sum[0, 0, 0] = 14.0
        trip_minutes_count[0, 0, 0] = 2.0
        demand = EmpiricalTripDemand.from_counts(od_counts, trip_minutes_sum, trip_minutes_count)
        road_distance = np.asarray([[0.0, 2.5], [2.5, 0.0]], dtype=np.float32)
        road_time = np.asarray([[0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
        env = DispatchEnv(
            EnvConfig(num_cells=2, cell_width_km=2.5, fleet_size=1, horizon_steps=1, seed=5),
            demand=demand,
            grid=grid,
            road_distance_matrix=road_distance,
            road_time_matrix=road_time,
        )

        values = env._sample_trip_minutes(0, np.asarray([0, 1], dtype=np.int64))

        self.assertAlmostEqual(float(values[0]), 7.0)
        self.assertAlmostEqual(float(values[1]), 5.0)

    def test_chengdu_train_trajectory_od_matrix_filters_test_dates(self):
        bounds = (103.96, 30.62, 104.04, 30.68)
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5, projection_origin=(104.0, 30.65))

        def write_trajectory_tar(path: Path, date: str, minutes: int) -> None:
            rows = [
                ("driver_id", "order_id", "time", "lon", "lat"),
                ("d1", "o1", f"{date} 08:00:00", f"{float(grid.lonlat[0, 0]):.8f}", f"{float(grid.lonlat[0, 1]):.8f}"),
                ("d1", "o1", f"{date} 08:{minutes:02d}:00", f"{float(grid.lonlat[1, 0]):.8f}", f"{float(grid.lonlat[1, 1]):.8f}"),
            ]
            payload = ("\n".join(",".join(row) for row in rows) + "\n").encode("utf-8")
            with tarfile.open(path, "w:gz") as tar:
                info = tarfile.TarInfo(path.with_suffix(".csv").name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            write_trajectory_tar(raw_dir / "2016_1101.tar.gz", "2016-11-01", 40)
            write_trajectory_tar(raw_dir / "2016_1108.tar.gz", "2016-11-08", 10)
            train_cache = root / "train.pkl"
            with train_cache.open("wb") as f:
                pickle.dump({"config": EnvConfig(num_cells=2), "grid": grid, "stats": {"bounds": bounds}}, f)

            matrix_path = root / "train_od.npz"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/precompute_train_trajectory_od_matrices.py",
                    "--raw-dir",
                    str(raw_dir),
                    "--train-cache",
                    str(train_cache),
                    "--out",
                    str(matrix_path),
                    "--source",
                    "raw",
                    "--bounds",
                    ",".join(str(x) for x in bounds),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            matrices = load_road_cost_matrices(matrix_path)
            self.assertAlmostEqual(float(matrices.time_minutes[0, 1]), 10.0)
            self.assertGreater(float(matrices.distance_km[0, 1]), 0.0)
            stats = json.loads(matrix_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(stats["train_date_range"], [20161108, 20161130])
            self.assertEqual(stats["train_files"], 1)
            self.assertEqual(stats["skipped_test_or_other_files"], 1)
            self.assertEqual(stats["rows_used"], 1)

    def test_chengdu_trajectory_od_matrix_uses_selected_cells_and_fallbacks(self):
        bounds = (103.96, 30.62, 104.04, 30.68)
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0), (0, 1)], cell_width_km=2.5, projection_origin=(104.0, 30.65))
        assignment_grid = HexGrid.create_geographic_fixed(bounds, cell_width_km=2.5, padding_km=2.5)
        with TemporaryDirectory() as tmp:
            tar_path = Path(tmp) / "2016_1101.tar.gz"
            rows = [
                ("driver_id", "order_id", "time", "lon", "lat"),
                ("d1", "o1", "2016-11-01 08:00:00", f"{float(grid.lonlat[0, 0]):.8f}", f"{float(grid.lonlat[0, 1]):.8f}"),
                ("d1", "o1", "2016-11-01 08:12:00", f"{float(grid.lonlat[1, 0]):.8f}", f"{float(grid.lonlat[1, 1]):.8f}"),
            ]
            payload = ("\n".join(",".join(row) for row in rows) + "\n").encode("utf-8")
            with tarfile.open(tar_path, "w:gz") as tar:
                info = tarfile.TarInfo("2016_1101.csv")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            matrices, stats = build_chengdu_trajectory_od_matrices(
                [tar_path],
                grid=grid,
                assignment_grid=assignment_grid,
                bounds=bounds,
            )

            self.assertAlmostEqual(float(matrices.time_minutes[0, 1]), 12.0)
            self.assertEqual(stats["rows_used"], 1)
            self.assertEqual(stats["observed_od_pairs"], 1)
            self.assertEqual(matrices.fallback_time_count, 5)
            self.assertGreater(float(matrices.time_minutes[1, 2]), 0.0)

    def test_cache_experiment_evaluation_uses_train_initial_distribution(self):
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0), (2, 0)], cell_width_km=2.5)
        rates = np.zeros((1, 3), dtype=np.float32)
        rates[0, 0] = 1.0
        od_probs = np.zeros((1, 3, 3), dtype=np.float32)
        od_probs[0, :, 1] = 1.0
        mean_trip_minutes = np.full((1, 3, 3), 5.0, dtype=np.float32)
        event = TripEvent(origin=0, destination=1, minute_offset=0, trip_minutes=5.0, pickup_xy=tuple(grid.xy[0]))
        demand = EventTripDemand(
            rates=rates,
            od_probs=od_probs,
            mean_trip_minutes=mean_trip_minutes,
            events_by_day=(((event,),),),
        )
        config = EnvConfig(
            num_cells=3,
            cell_width_km=2.5,
            fleet_size=10,
            horizon_steps=1,
            step_minutes=1,
            pickup_radius_km=1.5,
            seed=5,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_cache = root / "train.pkl"
            test_cache = root / "test.pkl"
            with train_cache.open("wb") as f:
                pickle.dump(
                    {
                        "config": config,
                        "grid": grid,
                        "demand": demand,
                        "initial_taxi_distribution": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    },
                    f,
                )
            with test_cache.open("wb") as f:
                pickle.dump(
                    {
                        "config": config,
                        "grid": grid,
                        "demand": demand,
                        "initial_taxi_distribution": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                    },
                    f,
                )

            out_dir = root / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_chengdu_cache_experiment.py",
                    "--train-cache",
                    str(train_cache),
                    "--test-cache",
                    str(test_cache),
                    "--method",
                    "park",
                    "--taxis",
                    "10",
                    "--eval-episodes",
                    "1",
                    "--demand-scale",
                    "1.0",
                    "--seed",
                    "0",
                    "--out",
                    str(out_dir),
                    "--road-network-cache",
                    str(root / "missing.graphml"),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            with (out_dir / "summary.csv").open("r", newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["method"], "park")
            self.assertEqual(float(row["served_orders"]), 1.0)
            run_config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(run_config["eval_initial_distribution_source"], "train_cache")
            self.assertFalse(run_config["same_step_reposition_service"])
            self.assertFalse(run_config["reposition_before_assignment"])
            self.assertEqual(run_config["park_pickup_scope"], "origin")
            self.assertEqual(run_config["random_pickup_scope"], "origin")
            self.assertEqual(run_config["diffusion_pickup_scope"], "origin")
            self.assertFalse(run_config["diffusion_same_step_reposition_service"])
            self.assertFalse(run_config["diffusion_reposition_before_assignment"])
            self.assertEqual(run_config["fv_bicoord_matching_mode"], "value_guided")
            self.assertEqual(run_config["fv_bicoord_pickup_scope"], "origin")
            self.assertFalse(run_config["fv_bicoord_same_step_reposition_service"])
            self.assertFalse(run_config["fv_bicoord_reposition_before_assignment"])

    def test_road_pickup_fallback_uses_configured_travel_speed(self):
        env = DispatchEnv(
            EnvConfig(num_cells=2, fleet_size=1, horizon_steps=1, cell_width_km=2.5, step_minutes=10, seed=5),
            grid=HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5),
        )

        self.assertAlmostEqual(env._distance_to_travel_minutes(1.25), 2.5)
        self.assertAlmostEqual(env._road_pickup_minutes(0, env.waiting_orders[0] if env.waiting_orders else _OrderRequest(
            origin=0,
            destination=1,
            created_step=0,
            created_minute=0,
            pickup_xy=env.grid.xy[0],
        ), 1.25), 2.5)

    def test_environment_advance_shapes(self):
        env = DispatchEnv(EnvConfig(num_cells=19, fleet_size=80, horizon_steps=4, seed=123))
        observations, state = env.reset()
        self.assertEqual(observations.shape, (19, env.action_dim * env.state_feature_dim))
        self.assertEqual(state.shape, (19 * env.state_feature_dim,))

        next_observations, next_state, rewards, actor_rewards, done = env.advance(
            lambda cur_env, obs, st: park_actions(cur_env)
        )
        self.assertEqual(next_observations.shape, observations.shape)
        self.assertEqual(next_state.shape, state.shape)
        self.assertEqual(rewards.shape, (19,))
        self.assertEqual(actor_rewards.shape, (19, 7))
        self.assertFalse(done)

    def test_environment_exposes_mamr_style_action_mask(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1, seed=31))
        self.assertEqual(env.available_actions.shape, (7, 7))
        np.testing.assert_array_equal(env.available_actions[:, 0], np.ones(7, dtype=np.float32))
        self.assertEqual(float(env.available_actions[0].sum()), 7.0)
        self.assertTrue(np.any(env.available_actions[1:, 1:] == 0.0))

    def test_mamr_env_alias_keeps_dispatch_environment_contract(self):
        env = MAMRDispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1, seed=31))
        self.assertIsInstance(env, DispatchEnv)
        self.assertIs(MAMREnv, MAMRDispatchEnv)
        observations, state = env.reset()
        self.assertEqual(observations.shape, (7, env.action_dim * env.state_feature_dim))
        self.assertEqual(state.shape, (7 * env.state_feature_dim,))

    def test_mamr_preprocessed_files_convert_to_dispatch_environment(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hex_bin_attributes.csv").write_text(
                "hex_id,north_east_neighbor,north_neighbor,north_west_neighbor,"
                "south_east_neighbor,south_neighbor,south_west_neighbor,"
                "east,north_east,north_west,south_east,south_west,west\n"
                '10,20,,,,,,"[0.01, 0.0]","[0.0, 0.01]","[-0.01, 0.01]",'
                '"[0.0, -0.01]","[-0.01, -0.01]","[-0.01, 0.0]"\n'
                '20,,30,,,,10,"[0.03, 0.0]","[0.02, 0.01]","[0.0, 0.01]",'
                '"[0.02, -0.01]","[0.0, -0.01]","[0.01, 0.0]"\n'
                '30,,,,,,20,"[0.05, 0.0]","[0.04, 0.01]","[0.02, 0.01]",'
                '"[0.04, -0.01]","[0.02, -0.01]","[0.03, 0.0]"\n',
                encoding="utf-8",
            )
            (root / "hex_distances.csv").write_text(
                "pickup_bin,dropoff_bin,straight_line_distance\n"
                "10,20,1.0\n"
                "20,10,1.0\n"
                "20,30,1.5\n"
                "30,20,1.5\n"
                "10,30,2.5\n"
                "30,10,2.5\n",
                encoding="utf-8",
            )
            (root / "driver_distribution.csv").write_text(
                "hex_id,driver_count\n10,5\n20,3\n30,2\n",
                encoding="utf-8",
            )
            city_states = {
                0: {
                    "time_unit_duration": 10,
                    "ride_count_matrix": np.asarray(
                        [[0, 2, 0], [0, 0, 1], [0, 0, 0]],
                        dtype=np.float32,
                    ),
                    "travel_time_matrix": np.asarray(
                        [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
                        dtype=np.float32,
                    ),
                }
            }
            with (root / "city_states.dill").open("wb") as f:
                pickle.dump(city_states, f)

            bundle = load_mamr_preprocessed(root, fleet_size=10)
            self.assertEqual(bundle.hex_ids, (10, 20, 30))
            self.assertEqual(bundle.grid.num_cells, 3)
            self.assertEqual(bundle.grid.neighbors[0, 1], 1)
            np.testing.assert_allclose(bundle.initial_taxi_distribution, [0.5, 0.3, 0.2])
            self.assertAlmostEqual(float(bundle.hex_distance_matrix[0, 2]), 2.5)
            self.assertEqual(bundle.config.step_minutes, 10)
            self.assertEqual(bundle.config.horizon_steps, 1)

            configured_bundle = load_mamr_preprocessed(
                root,
                fleet_size=10,
                distance_unit="km",
                base_config=EnvConfig(
                    num_cells=99,
                    fleet_size=999,
                    horizon_steps=99,
                    step_minutes=5,
                    future_demand_steps=0,
                    region_value_weight=0.0,
                    future_value_weight=0.0,
                    future_gap_weight=0.0,
                    dispatch_intensity_weight=0.0,
                    trip_time_weight=0.0,
                    seed=123,
                ),
            )
            self.assertEqual(configured_bundle.config.num_cells, 3)
            self.assertEqual(configured_bundle.config.horizon_steps, 1)
            self.assertEqual(configured_bundle.config.future_demand_steps, 0)
            self.assertEqual(configured_bundle.config.region_value_weight, 0.0)
            self.assertEqual(configured_bundle.config.future_value_weight, 0.0)
            self.assertEqual(configured_bundle.config.future_gap_weight, 0.0)
            self.assertEqual(configured_bundle.config.dispatch_intensity_weight, 0.0)
            self.assertEqual(configured_bundle.config.trip_time_weight, 0.0)

            bundle.demand.reset_episode(np.random.default_rng(0))
            events = bundle.demand.sample_requests(0, np.random.default_rng(1))
            self.assertEqual(len(events), 3)
            self.assertEqual([event.minute_offset for event in events[:2]], [0, 9])
            self.assertEqual(events[0].origin, 0)
            self.assertEqual(events[0].destination, 1)

            env, env_bundle = build_mamr_data_compatible_env(root, fleet_size=10, distance_unit="km")
            self.assertIsInstance(env, MAMRDispatchEnv)
            self.assertEqual(env_bundle.hex_ids, bundle.hex_ids)
            metrics, _ = evaluate_policy(
                env.config,
                lambda cur_env, obs, st: park_actions(cur_env),
                episodes=1,
                seed=3,
                demand=env.demand,
                grid=env.grid,
                initial_taxi_distribution=env.initial_taxi_distribution,
                hex_distance_matrix=env.hex_distance_matrix,
            )
            self.assertEqual(metrics.orders, 3)

    def test_mamr_loader_accepts_original_nested_layout(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "city_states").mkdir(parents=True)
            (root / "data" / "hex_bins").mkdir(parents=True)
            (root / "envs").mkdir(parents=True)
            (root / "data" / "hex_bins" / "hex_bin_attributes.csv").write_text(
                "hex_id,north_east_neighbor,north_neighbor,north_west_neighbor,"
                "south_east_neighbor,south_neighbor,south_west_neighbor,"
                "east,north_east,north_west,south_east,south_west,west\n"
                '0,1,,,,,,"[0.01, 0.0]","[0.0, 0.01]","[-0.01, 0.01]",'
                '"[0.0, -0.01]","[-0.01, -0.01]","[-0.01, 0.0]"\n'
                '1,,,,,,0,"[0.03, 0.0]","[0.02, 0.01]","[0.0, 0.01]",'
                '"[0.02, -0.01]","[0.0, -0.01]","[0.01, 0.0]"\n',
                encoding="utf-8",
            )
            (root / "data" / "hex_bins" / "hex_distances.csv").write_text(
                "pickup_bin,dropoff_bin,straight_line_distance\n"
                "0,1,1.0\n"
                "1,0,1.0\n",
                encoding="utf-8",
            )
            (root / "envs" / "driver_distribution.csv").write_text(
                "hex_id,driver_count\n0,7\n1,3\n",
                encoding="utf-8",
            )
            city_states = {
                0: {
                    "time_unit_duration": 5,
                    "ride_count_matrix": np.asarray([[0, 1], [0, 0]], dtype=np.float32),
                    "travel_time_matrix": np.asarray([[0, 1], [1, 0]], dtype=np.float32),
                }
            }
            with (root / "data" / "city_states" / "city_states.dill").open("wb") as f:
                pickle.dump(city_states, f)

            bundle = load_mamr_preprocessed(root, fleet_size=10, distance_unit="km")
            self.assertEqual(bundle.config.step_minutes, 5)
            self.assertEqual(bundle.grid.neighbors[0, 1], 1)
            np.testing.assert_allclose(bundle.initial_taxi_distribution, [0.7, 0.3])

    def test_mamr_loader_accepts_drivers_d30_and_raw_test_days(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hex_bin_attributes.csv").write_text(
                "hex_id,north_east_neighbor,north_neighbor,north_west_neighbor,"
                "south_east_neighbor,south_neighbor,south_west_neighbor,"
                "east,north_east,north_west,south_east,south_west,west\n"
                '0,1,,,,,,"[0.01, 0.0]","[0.0, 0.01]","[-0.01, 0.01]",'
                '"[0.0, -0.01]","[-0.01, -0.01]","[-0.01, 0.0]"\n'
                '1,,,,,,0,"[0.03, 0.0]","[0.02, 0.01]","[0.0, 0.01]",'
                '"[0.02, -0.01]","[0.0, -0.01]","[0.01, 0.0]"\n',
                encoding="utf-8",
            )
            (root / "hex_distances.csv").write_text(
                "pickup_bin,dropoff_bin,straight_line_distance\n0,1,1.0\n1,0,1.0\n",
                encoding="utf-8",
            )
            (root / "drivers_d30.csv").write_text("counts_9\n7\n3\n", encoding="utf-8")
            city_states = {
                step: {
                    "time_unit_duration": 10,
                    "ride_count_matrix": np.zeros((2, 2), dtype=np.float32),
                    "travel_time_matrix": np.ones((2, 2), dtype=np.float32),
                }
                for step in range(2)
            }
            with (root / "city_states_test.dill").open("wb") as f:
                pickle.dump(city_states, f)
            (root / "raw_test.csv").write_text(
                "PU_time,pickup_bin,dropoff_bin,weight,duration_seconds\n"
                "180,0,1,1,600\n"
                "86520,0,1,2,900\n",
                encoding="utf-8",
            )

            bundle = load_mamr_preprocessed(
                root,
                city_states_path=root / "city_states_test.dill",
                raw_trips_path=root / "raw_test.csv",
                fleet_size=0,
                distance_unit="km",
            )
            np.testing.assert_allclose(bundle.initial_taxi_distribution, [0.7, 0.3])
            self.assertEqual(bundle.demand.episode_count, 2)
            self.assertEqual(bundle.demand.metadata["day_count"], 2)
            self.assertEqual(bundle.demand.metadata["raw_time_mode"], "episode_relative")

            metrics, _ = evaluate_policy(
                bundle.config,
                lambda cur_env, obs, st: park_actions(cur_env),
                episodes=2,
                seed=3,
                demand=bundle.demand,
                grid=bundle.grid,
                initial_taxi_distribution=bundle.initial_taxi_distribution,
                hex_distance_matrix=bundle.hex_distance_matrix,
                env_cls=MAMRDispatchEnv,
            )
            self.assertEqual(metrics.orders, 1.5)

    def test_mamr_raw_loader_aligns_unix_wall_clock_timestamps_to_city_state_start(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hex_bin_attributes.csv").write_text(
                "hex_id,north_east_neighbor,north_neighbor,north_west_neighbor,"
                "south_east_neighbor,south_neighbor,south_west_neighbor,"
                "east,north_east,north_west,south_east,south_west,west\n"
                '0,1,,,,,,"[0.01, 0.0]","[0.0, 0.01]","[-0.01, 0.01]",'
                '"[0.0, -0.01]","[-0.01, -0.01]","[-0.01, 0.0]"\n'
                '1,,,,,,0,"[0.03, 0.0]","[0.02, 0.01]","[0.0, 0.01]",'
                '"[0.02, -0.01]","[0.0, -0.01]","[0.01, 0.0]"\n',
                encoding="utf-8",
            )
            (root / "hex_distances.csv").write_text(
                "pickup_bin,dropoff_bin,straight_line_distance\n0,1,1.0\n1,0,1.0\n",
                encoding="utf-8",
            )
            city_states = {
                step: {
                    "time": datetime(2016, 11, 1, 8, step * 10),
                    "time_unit_duration": 10,
                    "ride_count_matrix": np.zeros((2, 2), dtype=np.float32),
                    "travel_time_matrix": np.ones((2, 2), dtype=np.float32),
                }
                for step in range(2)
            }
            with (root / "city_states_test.dill").open("wb") as f:
                pickle.dump(city_states, f)
            first_day = int(datetime(2016, 11, 22, 8, 3, tzinfo=timezone.utc).timestamp())
            second_day = int(datetime(2016, 11, 23, 8, 2, tzinfo=timezone.utc).timestamp())
            (root / "raw_test.csv").write_text(
                "PU_time,pickup_bin,dropoff_bin,weight,duration_seconds\n"
                f"{first_day},0,1,1,600\n"
                f"{second_day},0,1,2,900\n",
                encoding="utf-8",
            )

            bundle = load_mamr_preprocessed(
                root,
                city_states_path=root / "city_states_test.dill",
                raw_trips_path=root / "raw_test.csv",
                fleet_size=0,
                distance_unit="km",
            )

            self.assertEqual(bundle.demand.episode_count, 2)
            self.assertEqual(bundle.demand.metadata["raw_time_mode"], "unix_wall_clock")
            bundle.demand.reset_episode(np.random.default_rng(0), episode_index=0)
            self.assertEqual([event.minute_offset for event in bundle.demand.sample_requests(0, np.random.default_rng(1))], [3])
            bundle.demand.reset_episode(np.random.default_rng(0), episode_index=1)
            self.assertEqual([event.minute_offset for event in bundle.demand.sample_requests(0, np.random.default_rng(1))], [2, 2])

    def test_response_time_is_averaged_over_served_orders(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=1, horizon_steps=1, seed=17))
        env.total_orders = 4
        env.served_orders = 2
        env.response_time_seconds = 120.0
        self.assertEqual(env.metrics().response_time_seconds, 60.0)

    def test_evaluation_metrics_are_order_weighted_across_episodes(self):
        class FakeEvalEnv:
            def __init__(self, config, **_kwargs):
                self.config = config
                self.episode_index = 0
                self.total_orders = 0
                self.served_orders = 0
                self.cancelled_orders = 0
                self.response_time_seconds = 0.0
                self.occupied_minutes = 0.0
                self.gmv = 0.0
                self.repositioned = 0

            def reset(self, _seed=None, episode_index=None):
                self.episode_index = int(episode_index or 0)
                return np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.float32)

            def advance(self, _policy):
                if self.episode_index == 0:
                    self.total_orders = 1
                    self.served_orders = 1
                    self.cancelled_orders = 0
                    self.response_time_seconds = 10.0
                else:
                    self.total_orders = 9
                    self.served_orders = 0
                    self.cancelled_orders = 9
                    self.response_time_seconds = 0.0
                return (
                    np.zeros((1, 1), dtype=np.float32),
                    np.zeros(1, dtype=np.float32),
                    np.zeros(1, dtype=np.float32),
                    np.zeros((1, 1), dtype=np.float32),
                    True,
                )

        metrics, _ = evaluate_policy(
            EnvConfig(num_cells=1, fleet_size=1, horizon_steps=1, step_minutes=10),
            lambda _env, _obs, _state: np.zeros((1, 1), dtype=np.float32),
            episodes=2,
            seed=1,
            env_cls=FakeEvalEnv,
        )
        self.assertAlmostEqual(metrics.response_rate, 0.1)
        self.assertAlmostEqual(metrics.cancellation_rate, 0.9)
        self.assertAlmostEqual(metrics.response_time_seconds, 10.0)
        self.assertAlmostEqual(metrics.orders, 5.0)

    def test_mamr_experiment_auto_loads_raw_train_and_test(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "mamr"
            (root / "city_states").mkdir(parents=True)
            (root / "hex_bins").mkdir(parents=True)
            (root / "hex_bins" / "hex_bin_attributes.csv").write_text(
                "hex_id,north_east_neighbor,north_neighbor,north_west_neighbor,"
                "south_east_neighbor,south_neighbor,south_west_neighbor,"
                "east,north_east,north_west,south_east,south_west,west\n"
                '0,1,,,,,,"[0.01, 0.0]","[0.0, 0.01]","[-0.01, 0.01]",'
                '"[0.0, -0.01]","[-0.01, -0.01]","[-0.01, 0.0]"\n'
                '1,,,,,,0,"[0.03, 0.0]","[0.02, 0.01]","[0.0, 0.01]",'
                '"[0.02, -0.01]","[0.0, -0.01]","[0.01, 0.0]"\n',
                encoding="utf-8",
            )
            (root / "hex_bins" / "hex_distances.csv").write_text(
                "pickup_bin,dropoff_bin,straight_line_distance\n0,1,1.0\n1,0,1.0\n",
                encoding="utf-8",
            )
            (root / "driver_distribution.csv").write_text("hex_id,driver_count\n0,1\n1,1\n", encoding="utf-8")
            train_states = {
                step: {
                    "time_unit_duration": 10,
                    "ride_count_matrix": np.asarray([[0, 5], [0, 0]], dtype=np.float32),
                    "travel_time_matrix": np.ones((2, 2), dtype=np.float32),
                }
                for step in range(2)
            }
            test_states = {
                step: {
                    "time_unit_duration": 10,
                    "ride_count_matrix": np.zeros((2, 2), dtype=np.float32),
                    "travel_time_matrix": np.ones((2, 2), dtype=np.float32),
                }
                for step in range(2)
            }
            with (root / "city_states" / "city_states_train.dill").open("wb") as f:
                pickle.dump(train_states, f)
            with (root / "city_states" / "city_states_test.dill").open("wb") as f:
                pickle.dump(test_states, f)
            (root / "raw_train.csv").write_text(
                "PU_time,pickup_bin,dropoff_bin,weight,duration_seconds\n"
                "60,0,1,1,600\n",
                encoding="utf-8",
            )
            (root / "raw_test.csv").write_text(
                "PU_time,pickup_bin,dropoff_bin,weight,duration_seconds\n"
                "180,0,1,1,600\n"
                "86520,0,1,2,900\n",
                encoding="utf-8",
            )
            out_dir = Path(tmp) / "out"
            args = build_parser().parse_args(
                [
                    "--real-demand",
                    "mamr",
                    "--method",
                    "park",
                    "--mamr-data",
                    str(root),
                    "--taxis",
                    "0",
                    "--horizon-steps",
                    "2",
                    "--step-minutes",
                    "10",
                    "--demand-scale",
                    "1.0",
                    "--out",
                    str(out_dir),
                    "--no-plots",
                ]
            )
            run_experiment(args)
            with (out_dir / "summary.csv").open("r", newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["method"], "park")
            self.assertEqual(float(row["orders"]), 1.5)
            sources = json.loads((out_dir / "mamr_data_sources.json").read_text(encoding="utf-8"))
            self.assertTrue(sources["train"]["uses_raw_trips"])
            self.assertTrue(sources["eval"]["uses_raw_trips"])
            self.assertEqual(Path(sources["train"]["raw_trips"]).name, "raw_train.csv")
            self.assertEqual(Path(sources["eval"]["raw_trips"]).name, "raw_test.csv")

            out_dir_raw = Path(tmp) / "out_raw"
            args_with_raw = build_parser().parse_args(
                [
                    "--real-demand",
                    "mamr",
                    "--method",
                    "park",
                    "--mamr-data",
                    str(root),
                    "--mamr-test-raw",
                    str(root / "raw_test.csv"),
                    "--taxis",
                    "0",
                    "--horizon-steps",
                    "2",
                    "--step-minutes",
                    "10",
                    "--demand-scale",
                    "1.0",
                    "--out",
                    str(out_dir_raw),
                    "--no-plots",
                ]
            )
            run_experiment(args_with_raw)
            with (out_dir_raw / "summary.csv").open("r", newline="", encoding="utf-8") as f:
                row_with_raw = next(csv.DictReader(f))
            self.assertEqual(float(row_with_raw["orders"]), 1.5)

    def test_repositioning_uses_largest_remainder_for_small_fleets(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=2, horizon_steps=1, seed=17))
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        for taxi_id in range(env.config.fleet_size):
            env._set_taxi_idle_at_cell(taxi_id, 0)

        actions = park_actions(env)
        actions[0] = 0.0
        actions[0, 1] = 0.5
        actions[0, 2] = 0.5
        env.current_minute = 0
        env._apply_repositioning(actions)

        self.assertEqual(env.repositioned, 2)
        self.assertEqual(len(env.idle_by_cell[0]), 0)

    def test_repositioned_taxi_can_match_after_physical_arrival_minute(self):
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5)
        env = DispatchEnv(
            EnvConfig(
                num_cells=2,
                cell_width_km=2.5,
                fleet_size=1,
                horizon_steps=2,
                step_minutes=10,
                max_wait_steps=2,
                pickup_radius_km=100.0,
                travel_speed_kmph=30.0,
                seed=18,
            ),
            grid=grid,
        )
        env.reset()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env._set_taxi_idle_at_cell(0, 0)
        env.taxi_xy[0] = grid.xy[0].copy()
        env.waiting_orders = [
            _OrderRequest(
                origin=1,
                destination=0,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[1].copy(),
            )
        ]
        env.current_minute = 0
        env._reset_step_reward_components()
        actions = park_actions(env)
        actions[0] = 0.0
        actions[0, int(np.flatnonzero(grid.neighbors[0] == 1)[0])] = 1.0

        env._apply_repositioning(actions)
        self.assertEqual(env.taxi_cell[0], -1)
        self.assertEqual(env.idle_by_cell[1], [])
        self.assertEqual(env.reposition_arrivals[5], [(0, 1)])

        env._run_assignment_window()

        self.assertEqual(env.served_orders, 1)
        self.assertEqual(env.cancelled_orders, 0)
        self.assertEqual(env.waiting_orders, [])

    def test_next_step_reposition_service_cannot_match_current_step_order(self):
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5)
        env = DispatchEnv(
            EnvConfig(
                num_cells=2,
                cell_width_km=2.5,
                fleet_size=1,
                horizon_steps=2,
                step_minutes=10,
                max_wait_steps=2,
                pickup_radius_km=100.0,
                travel_speed_kmph=30.0,
                same_step_reposition_service=False,
                seed=18,
            ),
            grid=grid,
        )
        env.reset()
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env._set_taxi_idle_at_cell(0, 0)
        env.taxi_xy[0] = grid.xy[0].copy()
        env.waiting_orders = [
            _OrderRequest(
                origin=1,
                destination=0,
                created_step=0,
                created_minute=0,
                pickup_xy=grid.xy[1].copy(),
            )
        ]
        env.total_orders = 1
        env.cell_orders[1] = 1
        env.current_minute = 0
        env._reset_step_reward_components()
        actions = park_actions(env)
        actions[0] = 0.0
        actions[0, int(np.flatnonzero(grid.neighbors[0] == 1)[0])] = 1.0

        env._apply_repositioning(actions)
        self.assertEqual(env.reposition_arrivals[10], [(0, 1)])

        env._run_assignment_window()

        self.assertEqual(env.served_orders, 0)
        self.assertEqual(env.cancelled_orders, 1)
        self.assertEqual(env.waiting_orders, [])
        self.assertEqual(env.taxi_cell[0], -1)

    def test_fv_bicoord_agent_masks_invalid_actions(self):
        grid = HexGrid.create(7)
        road_time = np.full((7, 7), 10.0, dtype=np.float32)
        np.fill_diagonal(road_time, 0.0)
        adjacency = build_road_time_adjacency(grid, road_time_matrix=road_time, step_minutes=10)
        action_costs = np.zeros((7, 7), dtype=np.float32)
        for cell in range(7):
            for action, dst in enumerate(grid.neighbors[cell]):
                action_costs[cell, action] = np.inf if dst < 0 else float(road_time[cell, int(dst)])
        agent = FVBiCoordAgent(
            agent_n=7,
            feature_dim=5,
            hidden_dim=8,
            action_dim=7,
            adjacency=adjacency,
            action_costs=action_costs,
            temporal_window=2,
            device="cpu",
        )
        sequence = np.zeros((2, 7, 5), dtype=np.float32)
        available_actions = (grid.neighbors >= 0).astype(np.float32)
        decision = agent.take_decision(sequence, available_actions)
        actions = decision["actions"]

        self.assertEqual(actions.shape, (7, 7))
        self.assertTrue(np.all(actions[available_actions == 0.0] == 0.0))
        np.testing.assert_allclose(actions.sum(axis=1), np.ones(7, dtype=np.float32), atol=1e-5)
        self.assertEqual(decision["actions"].shape, (7, 7))
        self.assertEqual(decision["region_value"].shape, (7,))
        self.assertEqual(decision["future_pressure"].shape, (7,))
        self.assertEqual(decision["future_gap"].shape, (7,))
        self.assertEqual(decision["future_demand"].shape, (7,))
        self.assertEqual(decision["dispatch_intensity"].shape, (7,))
        np.testing.assert_allclose(decision["future_gap"], decision["future_pressure"])
        np.testing.assert_allclose(decision["future_demand"], decision["future_pressure"])
        np.testing.assert_allclose(decision["dispatch_intensity"], np.clip(decision["future_pressure"], 0.0, 1.0))

        clean_agent = FVBiCoordAgent(
            agent_n=7,
            feature_dim=5,
            hidden_dim=8,
            action_dim=7,
            adjacency=adjacency,
            action_costs=action_costs,
            temporal_window=2,
            use_future_pressure_head=False,
            device="cpu",
        )
        clean_decision = clean_agent.take_decision(sequence, available_actions)
        self.assertEqual(set(clean_decision), {"actions", "region_value"})

    def test_raw_supply_sufficiency_gate_blocks_moves_from_non_surplus_source(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = 0.0
        env._state_raw[0, env.incoming_supply_feature_index] = 0.0
        env._state_raw[0, env.current_need_feature_index] = 1.0
        env._state_raw[1, env.current_need_feature_index] = 2.0

        mask = _supply_sufficiency_action_mask_from_env(env, np.ones_like(env.available_actions, dtype=np.float32))

        self.assertEqual(float(mask[0, 0]), 1.0)
        self.assertEqual(float(mask[0, 1:].sum()), 0.0)

    def test_raw_supply_sufficiency_gate_blocks_moves_to_non_shortage_target(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = 3.0
        env._state_raw[1, env.idle_supply_feature_index] = 3.0
        env._state_raw[2, env.current_need_feature_index] = 1.0
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])

        mask = _supply_sufficiency_action_mask_from_env(env, np.ones_like(env.available_actions, dtype=np.float32))

        self.assertEqual(float(mask[0, action_to_1]), 0.0)

    def test_raw_supply_sufficiency_gate_allows_moves_to_shortage_target(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = 3.0
        env._state_raw[1, env.current_need_feature_index] = 2.0
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])

        mask = _supply_sufficiency_action_mask_from_env(env, np.ones_like(env.available_actions, dtype=np.float32))

        self.assertEqual(float(mask[0, action_to_1]), 1.0)

    def test_raw_supply_sufficiency_gate_allows_only_stay_when_global_shortage_is_small(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        env._state_raw[:, :] = 0.0
        env._state_raw[:, env.idle_supply_feature_index] = 2.0
        env._state_raw[:, env.current_need_feature_index] = 1.0

        mask = _supply_sufficiency_action_mask_from_env(
            env,
            np.ones_like(env.available_actions, dtype=np.float32),
            global_shortage_threshold=0.0,
        )

        np.testing.assert_allclose(mask[:, 0], np.ones(env.grid_number, dtype=np.float32))
        self.assertEqual(float(mask[:, 1:].sum()), 0.0)

    def test_spatiotemporal_pressure_uses_road_time_kernel(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        road_time = np.full((7, 7), 999.0, dtype=np.float32)
        np.fill_diagonal(road_time, 0.0)
        road_time[0, 1] = 5.0
        road_time[0, 2] = 60.0
        env.road_time_matrix = road_time
        shortage = np.zeros(7, dtype=np.float32)
        shortage[1] = 8.0
        shortage[2] = 20.0

        pressure = _spatiotemporal_pressure_from_shortage(
            env,
            shortage,
            pressure_temperature_minutes=10.0,
            max_neighbor_time_minutes=30.0,
            pressure_clip=50.0,
        )

        self.assertAlmostEqual(float(pressure[0]), 8.0, delta=1.0e-4)
        self.assertAlmostEqual(float(pressure[2]), 0.0, delta=1.0e-6)

    def test_dynamic_soft_expansion_gate_strength_zero_matches_hard_gate(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = 0.0
        env._state_raw[0, env.current_need_feature_index] = 1.0
        env._state_raw[1, env.current_need_feature_index] = 2.0
        available = np.ones_like(env.available_actions, dtype=np.float32)

        hard_mask = _supply_sufficiency_action_mask_from_env(env, available)
        dynamic_mask = _dynamic_soft_expansion_action_mask_from_env(
            env,
            available,
            pressure_strength=0.0,
        )

        np.testing.assert_allclose(dynamic_mask, hard_mask)

    def test_dynamic_soft_expansion_gate_near_pressure_allows_non_surplus_source(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        road_time = np.full((7, 7), 999.0, dtype=np.float32)
        np.fill_diagonal(road_time, 0.0)
        road_time[0, 1] = 2.0
        env.road_time_matrix = road_time
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.current_need_feature_index] = 1.0
        env._state_raw[1, env.current_need_feature_index] = 100.0
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])

        mask = _dynamic_soft_expansion_action_mask_from_env(
            env,
            np.ones_like(env.available_actions, dtype=np.float32),
            pressure_strength=1.0,
            pressure_temperature_minutes=10.0,
            max_neighbor_time_minutes=30.0,
            pressure_clip=200.0,
        )

        self.assertEqual(float(mask[0, action_to_1]), 1.0)

    def test_dynamic_soft_expansion_gate_far_pressure_does_not_open_move(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        road_time = np.full((7, 7), 999.0, dtype=np.float32)
        np.fill_diagonal(road_time, 0.0)
        road_time[0, 1] = 60.0
        env.road_time_matrix = road_time
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.current_need_feature_index] = 1.0
        env._state_raw[1, env.current_need_feature_index] = 100.0

        mask = _dynamic_soft_expansion_action_mask_from_env(
            env,
            np.ones_like(env.available_actions, dtype=np.float32),
            pressure_strength=1.0,
            pressure_temperature_minutes=10.0,
            max_neighbor_time_minutes=30.0,
            pressure_clip=200.0,
        )

        self.assertEqual(float(mask[0, 0]), 1.0)
        self.assertEqual(float(mask[0, 1:].sum()), 0.0)

    def test_dynamic_soft_expansion_gate_blocks_weak_pressure_target(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        road_time = np.full((7, 7), 999.0, dtype=np.float32)
        np.fill_diagonal(road_time, 0.0)
        road_time[1, 2] = 60.0
        env.road_time_matrix = road_time
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = 3.0
        env._state_raw[1, env.idle_supply_feature_index] = 3.0
        env._state_raw[2, env.current_need_feature_index] = 10.0
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])

        mask = _dynamic_soft_expansion_action_mask_from_env(
            env,
            np.ones_like(env.available_actions, dtype=np.float32),
            pressure_strength=1.0,
            pressure_temperature_minutes=10.0,
            max_neighbor_time_minutes=30.0,
            pressure_clip=200.0,
        )

        self.assertEqual(float(mask[0, action_to_1]), 0.0)

    def test_dynamic_soft_expansion_gate_allows_only_stay_when_global_shortage_is_small(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        env._state_raw[:, :] = 0.0
        env._state_raw[:, env.idle_supply_feature_index] = 2.0
        env._state_raw[:, env.current_need_feature_index] = 1.0

        mask = _dynamic_soft_expansion_action_mask_from_env(
            env,
            np.ones_like(env.available_actions, dtype=np.float32),
            global_shortage_threshold=0.0,
            pressure_strength=1.0,
        )

        np.testing.assert_allclose(mask[:, 0], np.ones(env.grid_number, dtype=np.float32))
        self.assertEqual(float(mask[:, 1:].sum()), 0.0)

    def test_response_protected_move_budget_protects_current_need(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = 10.0
        env._state_raw[0, env.current_need_feature_index] = 8.0
        actions = np.zeros_like(env.available_actions, dtype=np.float32)
        actions[:, 0] = 1.0
        actions[0, 0] = 0.2
        actions[0, 1] = 0.4
        actions[0, 2] = 0.4

        projected = _apply_response_protected_move_budget_from_env(
            env,
            actions,
            incoming_discount=0.0,
            safety_buffer=1.0,
        )

        self.assertLessEqual(float(projected[0, 1:].sum()), 0.1 + 1.0e-6)
        self.assertGreaterEqual(float(projected[0, 0]), 0.9 - 1.0e-6)
        self.assertTrue(np.isfinite(projected).all())
        np.testing.assert_allclose(projected.sum(axis=1), np.ones(env.grid_number, dtype=np.float32), atol=1.0e-6)

    def test_response_protected_move_budget_forces_stay_without_surplus(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = 5.0
        env._state_raw[0, env.current_need_feature_index] = 5.0
        actions = np.zeros_like(env.available_actions, dtype=np.float32)
        actions[:, 0] = 1.0
        actions[0, 0] = 0.2
        actions[0, 1] = 0.4
        actions[0, 2] = 0.4

        projected = _apply_response_protected_move_budget_from_env(
            env,
            actions,
            incoming_discount=0.0,
            safety_buffer=1.0,
        )

        self.assertLess(float(projected[0, 1:].sum()), 1.0e-6)
        self.assertGreater(float(projected[0, 0]), 1.0 - 1.0e-6)

    def test_response_protected_move_budget_preserves_actor_direction_preference(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = 10.0
        env._state_raw[1, env.current_need_feature_index] = 10.0
        env._state_raw[2, env.current_need_feature_index] = 10.0
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])
        action_to_2 = int(np.flatnonzero(env.grid.neighbors[0] == 2)[0])
        actions = np.zeros_like(env.available_actions, dtype=np.float32)
        actions[:, 0] = 1.0
        actions[0, 0] = 0.1
        actions[0, action_to_1] = 0.6
        actions[0, action_to_2] = 0.3

        projected = _apply_response_protected_move_budget_from_env(
            env,
            actions,
            incoming_discount=0.0,
            safety_buffer=0.0,
        )

        self.assertAlmostEqual(float(projected[0, action_to_1] / projected[0, action_to_2]), 2.0, places=5)

    def test_response_protected_move_budget_uses_expected_flux_not_one_vehicle(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = 10.0
        env._state_raw[0, env.current_need_feature_index] = 10.0
        env._state_raw[1, env.current_need_feature_index] = 1.0
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])
        actions = np.zeros_like(env.available_actions, dtype=np.float32)
        actions[:, 0] = 1.0
        actions[0, 0] = 0.5
        actions[0, action_to_1] = 0.5

        projected = _apply_response_protected_move_budget_from_env(
            env,
            {"actions": actions},
            incoming_discount=0.0,
        )
        debug = projected["response_budget_debug"]

        self.assertAlmostEqual(debug["expected_flux_max"], 5.0, places=6)
        self.assertAlmostEqual(debug["expected_flux_mean"], 5.0, places=6)
        self.assertLess(debug["ratio_improvement_mean"], 0.0)
        self.assertLess(debug["scaled_ratio_improvement_mean"], 0.0)
        self.assertLess(float(projected["actions"][0, action_to_1]), 1.0e-6)

    def test_response_protected_move_budget_scales_improvement_by_pair_demand(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=120, horizon_steps=1))
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = 120.0
        env._state_raw[0, env.current_need_feature_index] = 100.0
        env._state_raw[1, env.idle_supply_feature_index] = 80.0
        env._state_raw[1, env.current_need_feature_index] = 100.0
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])
        actions = np.zeros_like(env.available_actions, dtype=np.float32)
        actions[:, 0] = 1.0
        actions[0, 0] = 0.999
        actions[0, action_to_1] = 0.001

        projected = _apply_response_protected_move_budget_from_env(
            env,
            {"actions": actions},
            incoming_discount=0.0,
        )
        debug = projected["response_budget_debug"]

        self.assertGreater(debug["ratio_improvement_mean"], 0.0)
        self.assertGreater(debug["scaled_ratio_improvement_mean"], debug["ratio_improvement_mean"])
        self.assertGreater(debug["mean_ratio_gate"], 0.9)
        self.assertGreater(debug["move_action_rate_after_gate"], 0.0)
        self.assertTrue(np.isfinite(projected["actions"]).all())
        np.testing.assert_allclose(projected["actions"].sum(axis=1), np.ones(env.grid_number, dtype=np.float32), atol=1.0e-6)
        np.testing.assert_allclose(env.available_actions[:, 0], np.ones(env.grid_number, dtype=np.float32))
        self.assertTrue(np.all(projected["actions"][:, 0] >= 0.0))

    def test_response_protected_move_budget_keeps_improving_move_available(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1))
        env._state_raw[:, :] = 0.0
        env._state_raw[0, env.idle_supply_feature_index] = 10.0
        env._state_raw[1, env.current_need_feature_index] = 10.0
        action_to_1 = int(np.flatnonzero(env.grid.neighbors[0] == 1)[0])
        actions = np.zeros_like(env.available_actions, dtype=np.float32)
        actions[:, 0] = 1.0
        actions[0, 0] = 0.5
        actions[0, action_to_1] = 0.5

        projected = _apply_response_protected_move_budget_from_env(
            env,
            {"actions": actions},
            incoming_discount=0.0,
        )

        self.assertGreater(float(projected["actions"][0, action_to_1]), 0.49)
        self.assertIn("response_budget_debug", projected)
        self.assertGreater(projected["response_budget_debug"]["mean_ratio_gate"], 0.0)
        self.assertGreater(projected["response_budget_debug"]["move_action_rate_after_gate"], 0.0)

    def test_supply_sufficiency_gate_no_longer_pre_actor_masks_reposition_policy(self):
        class RecordingAgent:
            temporal_window = 1

            def __init__(self) -> None:
                self.masks: list[np.ndarray] = []

            def available_actions_for_sequence(self, _sequence, available_actions):
                mask = np.asarray(available_actions, dtype=np.float32).copy()
                self.masks.append(mask)
                return mask

            def take_decision(self, _sequence, available_actions):
                actions = np.asarray(available_actions, dtype=np.float32).copy()
                row_sums = actions.sum(axis=1, keepdims=True)
                actions = np.divide(actions, np.maximum(row_sums, 1.0), out=np.zeros_like(actions), where=row_sums > 0.0)
                return {"actions": actions, "region_value": np.zeros(actions.shape[0], dtype=np.float32)}

        class GateProbeEnv:
            grid_number = 7
            state_feature_dim = len(DispatchEnv.state_feature_names)
            idle_supply_feature_index = DispatchEnv.idle_supply_feature_index
            incoming_supply_feature_index = DispatchEnv.incoming_supply_feature_index
            current_need_feature_index = DispatchEnv.current_need_feature_index

            def __init__(self, config, **_kwargs) -> None:
                self.config = config
                self.grid = HexGrid.create(7)
                self.available_actions = np.ones((7, 7), dtype=np.float32)
                self._state_raw = np.zeros((7, self.state_feature_dim), dtype=np.float32)
                self._state_raw[0, self.current_need_feature_index] = 1.0
                self._state_raw[1, self.current_need_feature_index] = 2.0
                self.total_orders = 0
                self.served_orders = 0
                self.cancelled_orders = 0
                self.response_time_seconds = 0.0
                self.occupied_minutes = 0.0
                self.gmv = 0.0
                self.repositioned = 0

            def reset(self, _seed, episode_index=None):
                state = self._state_raw.reshape(-1).copy()
                return state.copy(), state

            def regional_state_raw(self):
                return self._state_raw.copy()

            def advance(self, guidance_policy, reposition_policy=None):
                state = self._state_raw.reshape(-1).copy()
                guidance_policy(self, state.copy(), state)
                if reposition_policy is not None:
                    reposition_policy(self, state.copy(), state)
                return state.copy(), state, np.zeros(7, dtype=np.float32), np.zeros((7, 7), dtype=np.float32), True

        agent = RecordingAgent()

        evaluate_fv_bicoord(
            EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1),
            agent,
            episodes=1,
            seed=0,
            env_cls=GateProbeEnv,
            use_supply_sufficiency_gate=True,
        )

        self.assertEqual(len(agent.masks), 2)
        guidance_mask, reposition_mask = agent.masks
        np.testing.assert_allclose(guidance_mask, np.ones((7, 7), dtype=np.float32))
        np.testing.assert_allclose(reposition_mask, np.ones((7, 7), dtype=np.float32))

    def test_response_protected_move_budget_applies_only_to_reposition_policy(self):
        class RecordingAgent:
            temporal_window = 1

            def __init__(self) -> None:
                self.masks: list[np.ndarray] = []

            def available_actions_for_sequence(self, _sequence, available_actions):
                mask = np.asarray(available_actions, dtype=np.float32).copy()
                self.masks.append(mask)
                return mask

            def take_decision(self, _sequence, available_actions):
                actions = np.asarray(available_actions, dtype=np.float32).copy()
                row_sums = actions.sum(axis=1, keepdims=True)
                actions = np.divide(actions, np.maximum(row_sums, 1.0), out=np.zeros_like(actions), where=row_sums > 0.0)
                return {"actions": actions, "region_value": np.zeros(actions.shape[0], dtype=np.float32)}

        class GateProbeEnv:
            grid_number = 7
            state_feature_dim = len(DispatchEnv.state_feature_names)
            idle_supply_feature_index = DispatchEnv.idle_supply_feature_index
            incoming_supply_feature_index = DispatchEnv.incoming_supply_feature_index
            current_need_feature_index = DispatchEnv.current_need_feature_index

            def __init__(self, config, **_kwargs) -> None:
                self.config = config
                self.grid = HexGrid.create(7)
                self.available_actions = np.ones((7, 7), dtype=np.float32)
                self.decisions: list[dict[str, np.ndarray]] = []
                self._state_raw = np.zeros((7, self.state_feature_dim), dtype=np.float32)
                self._state_raw[0, self.idle_supply_feature_index] = 5.0
                self._state_raw[0, self.current_need_feature_index] = 5.0
                self.total_orders = 0
                self.served_orders = 0
                self.cancelled_orders = 0
                self.response_time_seconds = 0.0
                self.occupied_minutes = 0.0
                self.gmv = 0.0
                self.repositioned = 0

            def reset(self, _seed, episode_index=None):
                state = self._state_raw.reshape(-1).copy()
                return state.copy(), state

            def regional_state_raw(self):
                return self._state_raw.copy()

            def advance(self, guidance_policy, reposition_policy=None):
                state = self._state_raw.reshape(-1).copy()
                guidance_decision = guidance_policy(self, state.copy(), state)
                self.decisions.append(guidance_decision)
                if reposition_policy is not None:
                    reposition_decision = reposition_policy(self, state.copy(), state)
                    self.decisions.append(reposition_decision)
                return state.copy(), state, np.zeros(7, dtype=np.float32), np.zeros((7, 7), dtype=np.float32), True

        agent = RecordingAgent()

        _metrics, env = evaluate_fv_bicoord(
            EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1),
            agent,
            episodes=1,
            seed=0,
            env_cls=GateProbeEnv,
            use_dynamic_soft_expansion_gate=True,
            use_response_protected_move_budget=True,
            pressure_gate_strength=0.0,
        )

        self.assertEqual(len(agent.masks), 2)
        guidance_mask, reposition_mask = agent.masks
        np.testing.assert_allclose(guidance_mask, np.ones((7, 7), dtype=np.float32))
        np.testing.assert_allclose(reposition_mask, np.ones((7, 7), dtype=np.float32))
        self.assertGreater(float(env.decisions[0]["actions"][0, 1:].sum()), 0.0)
        self.assertLess(float(env.decisions[1]["actions"][0, 1:].sum()), 1.0e-6)
        self.assertGreater(float(env.decisions[1]["actions"][0, 0]), 1.0 - 1.0e-6)

    def test_dynamic_soft_expansion_gate_no_longer_pre_actor_hard_masks(self):
        class RecordingAgent:
            temporal_window = 1

            def __init__(self) -> None:
                self.masks: list[np.ndarray] = []

            def available_actions_for_sequence(self, _sequence, available_actions):
                mask = np.asarray(available_actions, dtype=np.float32).copy()
                self.masks.append(mask)
                return mask

            def take_decision(self, _sequence, available_actions):
                actions = np.asarray(available_actions, dtype=np.float32).copy()
                row_sums = actions.sum(axis=1, keepdims=True)
                actions = np.divide(actions, np.maximum(row_sums, 1.0), out=np.zeros_like(actions), where=row_sums > 0.0)
                return {"actions": actions, "region_value": np.zeros(actions.shape[0], dtype=np.float32)}

        class GateProbeEnv:
            grid_number = 7
            state_feature_dim = len(DispatchEnv.state_feature_names)
            idle_supply_feature_index = DispatchEnv.idle_supply_feature_index
            incoming_supply_feature_index = DispatchEnv.incoming_supply_feature_index
            current_need_feature_index = DispatchEnv.current_need_feature_index

            def __init__(self, config, **_kwargs) -> None:
                self.config = config
                self.grid = HexGrid.create(7)
                self.available_actions = np.ones((7, 7), dtype=np.float32)
                self._state_raw = np.zeros((7, self.state_feature_dim), dtype=np.float32)
                self._state_raw[0, self.current_need_feature_index] = 1.0
                self._state_raw[1, self.current_need_feature_index] = 100.0
                road_time = np.full((7, 7), 999.0, dtype=np.float32)
                np.fill_diagonal(road_time, 0.0)
                road_time[0, 1] = 2.0
                self.road_time_matrix = road_time
                self.total_orders = 0
                self.served_orders = 0
                self.cancelled_orders = 0
                self.response_time_seconds = 0.0
                self.occupied_minutes = 0.0
                self.gmv = 0.0
                self.repositioned = 0

            def reset(self, _seed, episode_index=None):
                state = self._state_raw.reshape(-1).copy()
                return state.copy(), state

            def regional_state_raw(self):
                return self._state_raw.copy()

            def advance(self, guidance_policy, reposition_policy=None):
                state = self._state_raw.reshape(-1).copy()
                guidance_policy(self, state.copy(), state)
                if reposition_policy is not None:
                    reposition_policy(self, state.copy(), state)
                return state.copy(), state, np.zeros(7, dtype=np.float32), np.zeros((7, 7), dtype=np.float32), True

        agent = RecordingAgent()

        evaluate_fv_bicoord(
            EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1),
            agent,
            episodes=1,
            seed=0,
            env_cls=GateProbeEnv,
            use_supply_sufficiency_gate=True,
            use_dynamic_soft_expansion_gate=True,
            use_response_protected_move_budget=True,
            pressure_gate_strength=1.0,
            pressure_temperature_minutes=10.0,
            pressure_max_neighbor_time_minutes=30.0,
            pressure_clip=200.0,
        )

        action_to_1 = int(np.flatnonzero(HexGrid.create(7).neighbors[0] == 1)[0])
        self.assertEqual(len(agent.masks), 2)
        _guidance_mask, reposition_mask = agent.masks
        np.testing.assert_allclose(reposition_mask, np.ones((7, 7), dtype=np.float32))
        self.assertEqual(float(reposition_mask[0, action_to_1]), 1.0)

    def test_supply_sufficiency_gate_blocks_moves_from_non_surplus_source(self):
        action_destinations = np.asarray(
            [
                [0, 1, 2, 0],
                [1, 0, 2, 1],
                [2, 0, 1, 2],
            ],
            dtype=np.int64,
        )
        agent = FVBiCoordAgent(
            agent_n=3,
            feature_dim=6,
            hidden_dim=8,
            action_dim=4,
            adjacency=np.eye(3, dtype=np.float32),
            action_costs=np.zeros((3, 4), dtype=np.float32),
            action_destinations=action_destinations,
            temporal_window=2,
            use_supply_sufficiency_gate=True,
            source_surplus_threshold=0.0,
            target_shortage_threshold=0.0,
            device="cpu",
        )
        sequence = np.zeros((2, 3, 6), dtype=np.float32)
        sequence[-1, 0, agent.idle_supply_feature_index] = 0.0
        sequence[-1, 0, agent.current_need_feature_index] = 1.0
        sequence[-1, 1, agent.current_need_feature_index] = 2.0

        mask = agent.available_actions_for_sequence(sequence, np.ones((3, 4), dtype=np.float32))

        self.assertEqual(float(mask[0, 0]), 1.0)
        self.assertEqual(float(mask[0, 1:].sum()), 0.0)

    def test_supply_sufficiency_gate_blocks_moves_to_non_shortage_target(self):
        action_destinations = np.asarray(
            [
                [0, 1, 2, 0],
                [1, 0, 2, 1],
                [2, 0, 1, 2],
            ],
            dtype=np.int64,
        )
        agent = FVBiCoordAgent(
            agent_n=3,
            feature_dim=6,
            hidden_dim=8,
            action_dim=4,
            adjacency=np.eye(3, dtype=np.float32),
            action_costs=np.zeros((3, 4), dtype=np.float32),
            action_destinations=action_destinations,
            temporal_window=2,
            use_supply_sufficiency_gate=True,
            source_surplus_threshold=0.0,
            target_shortage_threshold=0.0,
            device="cpu",
        )
        sequence = np.zeros((2, 3, 6), dtype=np.float32)
        sequence[-1, 0, agent.idle_supply_feature_index] = 3.0
        sequence[-1, 1, agent.idle_supply_feature_index] = 3.0
        sequence[-1, 2, agent.current_need_feature_index] = 1.0

        mask = agent.available_actions_for_sequence(sequence, np.ones((3, 4), dtype=np.float32))

        self.assertEqual(int(action_destinations[0, 1]), 1)
        self.assertEqual(float(mask[0, 1]), 0.0)

    def test_supply_sufficiency_gate_allows_moves_to_shortage_target(self):
        action_destinations = np.asarray(
            [
                [0, 1, 2, 0],
                [1, 0, 2, 1],
                [2, 0, 1, 2],
            ],
            dtype=np.int64,
        )
        agent = FVBiCoordAgent(
            agent_n=3,
            feature_dim=6,
            hidden_dim=8,
            action_dim=4,
            adjacency=np.eye(3, dtype=np.float32),
            action_costs=np.zeros((3, 4), dtype=np.float32),
            action_destinations=action_destinations,
            temporal_window=2,
            use_supply_sufficiency_gate=True,
            source_surplus_threshold=0.0,
            target_shortage_threshold=0.0,
            device="cpu",
        )
        sequence = np.zeros((2, 3, 6), dtype=np.float32)
        sequence[-1, 0, agent.idle_supply_feature_index] = 3.0
        sequence[-1, 1, agent.current_need_feature_index] = 2.0

        mask = agent.available_actions_for_sequence(sequence, np.ones((3, 4), dtype=np.float32))

        self.assertEqual(int(action_destinations[0, 1]), 1)
        self.assertEqual(float(mask[0, 1]), 1.0)

    def test_supply_sufficiency_gate_allows_only_stay_when_global_shortage_is_small(self):
        action_destinations = np.asarray(
            [
                [0, 1, 2, 0],
                [1, 0, 2, 1],
                [2, 0, 1, 2],
            ],
            dtype=np.int64,
        )
        agent = FVBiCoordAgent(
            agent_n=3,
            feature_dim=6,
            hidden_dim=8,
            action_dim=4,
            adjacency=np.eye(3, dtype=np.float32),
            action_costs=np.zeros((3, 4), dtype=np.float32),
            action_destinations=action_destinations,
            temporal_window=2,
            use_supply_sufficiency_gate=True,
            global_shortage_threshold=0.0,
            device="cpu",
        )
        sequence = np.zeros((2, 3, 6), dtype=np.float32)
        sequence[-1, :, agent.idle_supply_feature_index] = 2.0
        sequence[-1, :, agent.current_need_feature_index] = 1.0

        mask = agent.available_actions_for_sequence(sequence, np.ones((3, 4), dtype=np.float32))

        np.testing.assert_allclose(mask[:, 0], np.ones(3, dtype=np.float32))
        self.assertEqual(float(mask[:, 1:].sum()), 0.0)

    def test_fv_bicoord_can_mask_surplus_stay_action(self):
        grid = HexGrid.create(7)
        adjacency = build_road_time_adjacency(grid, step_minutes=10)
        action_costs = np.zeros((7, 7), dtype=np.float32)
        agent = FVBiCoordAgent(
            agent_n=7,
            feature_dim=6,
            hidden_dim=8,
            action_dim=7,
            adjacency=adjacency,
            action_costs=action_costs,
            action_destinations=grid.neighbors,
            temporal_window=2,
            park_mask_surplus_threshold=0.25,
            park_mask_neighbor_need_threshold=0.1,
            device="cpu",
        )
        sequence = np.zeros((2, 7, 6), dtype=np.float32)
        available_actions = (grid.neighbors >= 0).astype(np.float32)
        neighbor = int(grid.neighbors[0, 1])
        sequence[-1, 0, agent.idle_supply_feature_index] = 1.0
        sequence[-1, 0, agent.supply_demand_gap_feature_index] = 0.5
        sequence[-1, neighbor, agent.current_need_feature_index] = 0.5

        masked = agent.available_actions_for_sequence(sequence, available_actions)

        self.assertEqual(float(masked[0, 0]), 0.0)
        self.assertGreater(float(masked[0].sum()), 0.0)

        sequence[-1, 0, agent.supply_demand_gap_feature_index] = -0.5
        unmasked = agent.available_actions_for_sequence(sequence, available_actions)
        self.assertEqual(float(unmasked[0, 0]), 1.0)

    def test_fv_bicoord_network_aligns_static_action_mask_across_batch(self):
        net = FVBiCoordNetwork(feature_dim=3, hidden_dim=8, action_dim=4, attention_heads=2)
        sequence = torch.zeros((2, 2, 3, 3), dtype=torch.float32)
        adjacency = torch.eye(3, dtype=torch.float32)
        available_actions = torch.asarray(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=torch.float32,
        )

        probs, _values = net(sequence, adjacency, available_actions=available_actions)

        self.assertEqual(probs.shape, (2, 3, 4))
        self.assertTrue(torch.all(probs[:, available_actions == 0.0] == 0.0))

        with self.assertRaisesRegex(ValueError, "available_actions"):
            net(sequence, adjacency, available_actions=torch.ones((2, 4), dtype=torch.float32))

    def test_fv_bicoord_network_sanitizes_nonfinite_sequence_inputs(self):
        net = FVBiCoordNetwork(feature_dim=3, hidden_dim=8, action_dim=4, attention_heads=2)
        sequence = torch.zeros((2, 2, 3, 3), dtype=torch.float32)
        sequence[0, 0, 0, 0] = float("nan")
        sequence[1, 1, 2, 1] = float("inf")

        probs, values, heads = net(sequence, torch.eye(3, dtype=torch.float32), return_heads=True)

        self.assertTrue(torch.isfinite(probs).all())
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(all(torch.isfinite(value).all() for value in heads.values()))

    def test_local_global_heads_preserve_shapes_and_action_mask(self):
        net = FVBiCoordNetwork(
            feature_dim=3,
            hidden_dim=8,
            action_dim=4,
            attention_heads=2,
            num_cells=3,
            use_local_global_heads=True,
        )
        sequence = torch.zeros((2, 2, 3, 3), dtype=torch.float32)
        adjacency = torch.eye(3, dtype=torch.float32)
        available_actions = torch.asarray(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=torch.float32,
        )

        probs, values, heads = net(
            sequence,
            adjacency,
            available_actions=available_actions,
            return_heads=True,
        )

        self.assertEqual(probs.shape, (2, 3, 4))
        self.assertEqual(values.shape, (2, 3))
        self.assertEqual(heads["region_value"].shape, (2, 3))
        self.assertTrue(torch.all(probs[:, available_actions == 0.0] == 0.0))

    def test_global_value_bias_outputs_region_conditioned_vector(self):
        net = FVBiCoordNetwork(
            feature_dim=3,
            hidden_dim=8,
            action_dim=4,
            attention_heads=2,
            num_cells=3,
            use_local_global_heads=True,
            global_bias_scale=1.0,
        )
        with torch.no_grad():
            net.critic.weight.zero_()
            net.critic.bias.zero_()
            net.global_value_bias.weight.zero_()
            net.global_value_bias.bias.copy_(torch.asarray([0.0, 1.0, 2.0]))

        sequence = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
        adjacency = torch.eye(3, dtype=torch.float32)

        _probs, values, heads = net(sequence, adjacency, return_heads=True)
        region_value = heads["region_value"][0].detach()

        self.assertEqual(tuple(net.global_value_bias.bias.shape), (3,))
        torch.testing.assert_close(heads["region_value"], values)
        self.assertLess(float(region_value[0]), float(region_value[1]))
        self.assertLess(float(region_value[1]), float(region_value[2]))

    def test_zero_initialized_local_global_heads_match_shared_head_outputs(self):
        torch.manual_seed(19)
        base = FVBiCoordNetwork(feature_dim=3, hidden_dim=8, action_dim=4, attention_heads=2)
        local_global = FVBiCoordNetwork(
            feature_dim=3,
            hidden_dim=8,
            action_dim=4,
            attention_heads=2,
            num_cells=3,
            use_local_global_heads=True,
        )
        local_global.load_state_dict(base.state_dict(), strict=False)
        sequence = torch.randn((2, 2, 3, 3), dtype=torch.float32)
        adjacency = torch.eye(3, dtype=torch.float32)

        base_probs, base_values, base_heads = base(sequence, adjacency, return_heads=True)
        lg_probs, lg_values, lg_heads = local_global(sequence, adjacency, return_heads=True)

        torch.testing.assert_close(lg_probs, base_probs, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(lg_values, base_values, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(base_heads["region_value"], base_values, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(lg_heads["region_value"], base_heads["region_value"], atol=1e-6, rtol=1e-6)

    def test_auxiliary_heads_can_detach_shared_encoder_gradients(self):
        torch.manual_seed(9)
        net = FVBiCoordNetwork(feature_dim=3, hidden_dim=8, action_dim=4, attention_heads=2)
        sequence = torch.randn((1, 2, 3, 3), dtype=torch.float32)
        adjacency = torch.eye(3, dtype=torch.float32)

        _probs, _values, heads = net(
            sequence,
            adjacency,
            return_heads=True,
            detach_auxiliary_heads=True,
        )
        loss = heads["future_pressure"].sum()
        loss.backward()

        encoder_params = [
            *net.temporal.parameters(),
            *net.graph_attn1.parameters(),
            *net.graph_attn2.parameters(),
        ]
        self.assertTrue(all(param.grad is None for param in encoder_params))
        self.assertIsNotNone(net.future_pressure.weight.grad)

    def test_local_global_value_residuals_receive_critic_gradients(self):
        torch.manual_seed(10)
        net = FVBiCoordNetwork(
            feature_dim=3,
            hidden_dim=8,
            action_dim=4,
            attention_heads=2,
            num_cells=3,
            use_local_global_heads=True,
        )
        sequence = torch.randn((1, 2, 3, 3), dtype=torch.float32)
        adjacency = torch.eye(3, dtype=torch.float32)

        _probs, values, heads = net(
            sequence,
            adjacency,
            return_heads=True,
        )
        loss = values.sum()
        loss.backward()

        torch.testing.assert_close(heads["region_value"], values)
        self.assertIsNotNone(net.critic.weight.grad)
        self.assertIsNotNone(net.local_region_value_weight.grad)
        self.assertIsNotNone(net.global_value_bias.weight.grad)
        self.assertIsNone(net.region_value.weight.grad)

    def test_road_time_adjacency_preserves_direction_by_default(self):
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5)
        road_time = np.asarray([[0.0, 1.0], [9.0, 0.0]], dtype=np.float32)

        directed = build_road_time_adjacency(grid, road_time_matrix=road_time, temperature=10.0)
        symmetric = build_road_time_adjacency(grid, road_time_matrix=road_time, temperature=10.0, symmetric=True)

        self.assertNotAlmostEqual(float(directed[0, 1]), float(directed[1, 0]))
        self.assertGreater(float(symmetric[1, 0]), float(directed[1, 0]))

    def test_auxiliary_head_loss_uses_explicit_future_pressure_target(self):
        adjacency = np.eye(2, dtype=np.float32)
        action_costs = np.zeros((2, 7), dtype=np.float32)
        agent = FVBiCoordAgent(
            agent_n=2,
            feature_dim=6,
            hidden_dim=8,
            action_dim=7,
            adjacency=adjacency,
            action_costs=action_costs,
            future_value_loss_weight=0.0,
            future_gap_loss_weight=1.0,
            intensity_loss_weight=0.0,
            device="cpu",
        )
        heads = {
            "region_value": torch.zeros((1, 2), dtype=torch.float32),
            "future_pressure": torch.zeros((1, 2), dtype=torch.float32),
        }
        next_sequences = torch.zeros((1, 2, 2, 6), dtype=torch.float32)
        next_sequences[:, -1, :, agent.future_gap_feature_index] = 9.0
        targets = {"future_pressure": torch.zeros((1, 2), dtype=torch.float32)}

        loss = agent._auxiliary_head_loss(heads, next_sequences=next_sequences, targets=targets)

        self.assertAlmostEqual(float(loss), 0.0)
        missing_target_loss = agent._auxiliary_head_loss(heads, next_sequences=next_sequences, targets=None)
        self.assertAlmostEqual(float(missing_target_loss), 0.0)

    def test_region_value_is_critic_derived_matching_value(self):
        torch.manual_seed(12)
        net = FVBiCoordNetwork(feature_dim=4, hidden_dim=8, action_dim=3, attention_heads=2)
        sequence = torch.zeros((1, 2, 2, 4), dtype=torch.float32)
        adjacency = torch.eye(2, dtype=torch.float32)

        _probs, values, heads = net(sequence, adjacency, return_heads=True)

        self.assertIn("critic_value", heads)
        self.assertIn("future_pressure", heads)
        self.assertFalse(hasattr(net, "future_gap"))
        self.assertFalse(hasattr(net, "future_demand"))
        self.assertFalse(hasattr(net, "dispatch_intensity"))
        self.assertTrue(torch.equal(heads["critic_value"], values))
        self.assertTrue(torch.equal(heads["region_value"], values))
        self.assertTrue(hasattr(net, "region_value"))

    def test_auxiliary_targets_merge_future_pressure_sources(self):
        agent = FVBiCoordAgent(
            agent_n=2,
            feature_dim=6,
            hidden_dim=8,
            action_dim=7,
            adjacency=np.eye(2, dtype=np.float32),
            action_costs=np.zeros((2, 7), dtype=np.float32),
            device="cpu",
        )
        transitions = [
            {
                "auxiliary_targets": {
                    "region_value": np.asarray([1.0, 2.0], dtype=np.float32),
                    "future_gap": np.asarray([0.5, 0.0], dtype=np.float32),
                    "future_demand": np.asarray([0.25, 0.75], dtype=np.float32),
                    "dispatch_intensity": np.asarray([0.5, 0.0], dtype=np.float32),
                }
            },
            {"auxiliary_targets": {"region_value": np.asarray([3.0, 4.0], dtype=np.float32)}},
        ]

        targets = agent._transition_auxiliary_targets(transitions)

        self.assertIsNotNone(targets)
        np.testing.assert_allclose(targets["future_pressure"].cpu().numpy(), [[0.5, 0.75], [0.0, 0.0]])
        np.testing.assert_allclose(targets["future_pressure_mask"].cpu().numpy(), [[1.0, 1.0], [0.0, 0.0]])
        self.assertNotIn("future_gap", targets)
        self.assertNotIn("future_demand", targets)
        np.testing.assert_allclose(targets["region_value_mask"].cpu().numpy(), [[1.0, 1.0], [1.0, 1.0]])

    def test_fv_update_uses_next_available_actions_for_target_net(self):
        agent = FVBiCoordAgent(
            agent_n=2,
            feature_dim=4,
            hidden_dim=8,
            action_dim=3,
            adjacency=np.eye(2, dtype=np.float32),
            action_costs=np.zeros((2, 3), dtype=np.float32),
            temporal_window=1,
            attention_heads=1,
            device="cpu",
        )
        current_mask = np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)
        next_mask = np.asarray([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        captured: dict[str, np.ndarray] = {}
        original_forward = agent.target_net.forward

        def capture_forward(
            sequence,
            adjacency,
            available_actions=None,
            action_costs=None,
            road_time_weight=0.0,
            return_heads=False,
        ):
            captured["available_actions"] = available_actions.detach().cpu().numpy().copy()
            return original_forward(
                sequence,
                adjacency,
                available_actions=available_actions,
                action_costs=action_costs,
                road_time_weight=road_time_weight,
                return_heads=return_heads,
            )

        agent.target_net.forward = capture_forward
        transitions = [
            {
                "sequence": np.zeros((1, 2, 4), dtype=np.float32),
                "next_sequence": np.ones((1, 2, 4), dtype=np.float32),
                "critic_rewards": np.zeros(2, dtype=np.float32),
                "actor_rewards": np.zeros((2, 3), dtype=np.float32),
                "available_actions": current_mask,
                "next_available_actions": next_mask,
                "done": False,
            }
        ]

        agent.update(transitions)

        np.testing.assert_allclose(captured["available_actions"], next_mask[None, :, :])

    def test_actor_advantages_are_normalized_and_clipped(self):
        advantages = torch.asarray([[[100.0, -100.0, 50.0]]], dtype=torch.float32)
        available_actions = torch.asarray([[[1.0, 1.0, 0.0]]], dtype=torch.float32)

        normalized = FVBiCoordAgent._normalize_actor_advantages(advantages, available_actions)

        self.assertLessEqual(float(torch.abs(normalized).max()), 3.0)
        self.assertAlmostEqual(float(normalized[available_actions > 0.0].mean()), 0.0, places=6)

        single_valid = torch.asarray([[[50.0, -50.0]]], dtype=torch.float32)
        single_mask = torch.asarray([[[1.0, 0.0]]], dtype=torch.float32)
        clipped = FVBiCoordAgent._normalize_actor_advantages(single_valid, single_mask)
        self.assertAlmostEqual(float(clipped[0, 0, 0]), 3.0)

    def test_future_gap_fallback_uses_supply_demand_gap_feature(self):
        agent = FVBiCoordAgent(
            agent_n=2,
            feature_dim=6,
            hidden_dim=8,
            action_dim=7,
            adjacency=np.eye(2, dtype=np.float32),
            action_costs=np.zeros((2, 7), dtype=np.float32),
            future_value_loss_weight=0.0,
            future_gap_loss_weight=1.0,
            intensity_loss_weight=0.0,
            device="cpu",
        )
        heads = {
            "region_value": torch.zeros((1, 2), dtype=torch.float32),
            "future_pressure": torch.zeros((1, 2), dtype=torch.float32),
        }
        next_sequences = torch.zeros((1, 2, 2, 6), dtype=torch.float32)
        next_sequences[:, -1, :, agent.supply_demand_gap_feature_index] = torch.asarray([-2.0, 1.0])

        loss = agent._auxiliary_head_loss(heads, next_sequences=next_sequences, targets=None)

        self.assertGreater(float(loss), 0.0)

    def test_actor_loss_aggregates_actions_before_cells(self):
        actor_loss_terms = torch.ones((1, 2, 4), dtype=torch.float32)
        available_actions = torch.asarray([[[1, 1, 0, 0], [1, 1, 1, 1]]], dtype=torch.float32)

        loss = FVBiCoordAgent._actor_loss_from_terms(actor_loss_terms, available_actions)

        self.assertAlmostEqual(float(loss), 3.0)

    def test_actor_loss_can_weight_cells_by_idle_supply(self):
        actor_loss_terms = torch.asarray([[[10.0, 10.0], [1.0, 1.0]]], dtype=torch.float32)
        available_actions = torch.ones((1, 2, 2), dtype=torch.float32)
        cell_weights = torch.asarray([[0.0, 1.0]], dtype=torch.float32)

        loss = FVBiCoordAgent._actor_loss_from_terms(actor_loss_terms, available_actions, cell_weights)

        self.assertAlmostEqual(float(loss), 2.0)

    def test_entropy_regularization_records_policy_entropy(self):
        grid = HexGrid.create(7)
        agent = FVBiCoordAgent(
            agent_n=7,
            feature_dim=5,
            hidden_dim=8,
            action_dim=7,
            adjacency=build_road_time_adjacency(grid, step_minutes=10),
            action_costs=np.zeros((7, 7), dtype=np.float32),
            future_value_loss_weight=0.0,
            future_gap_loss_weight=0.0,
            intensity_loss_weight=0.0,
            entropy_coef=0.2,
            gamma=1.0,
            temporal_window=2,
            device="cpu",
        )
        transition = {
            "sequence": np.zeros((2, 7, 5), dtype=np.float32),
            "next_sequence": np.zeros((2, 7, 5), dtype=np.float32),
            "critic_rewards": np.ones(7, dtype=np.float32),
            "actor_rewards": np.ones((7, 7), dtype=np.float32),
            "available_actions": (grid.neighbors >= 0).astype(np.float32),
            "done": True,
        }

        actor_loss, critic_loss = agent.update([transition])

        self.assertTrue(np.isfinite(actor_loss))
        self.assertTrue(np.isfinite(critic_loss))
        self.assertGreater(agent.last_policy_entropy, 0.0)
        self.assertLess(agent.last_entropy_loss, 0.0)
        self.assertTrue(np.isfinite(agent.last_auxiliary_loss))
        self.assertTrue(np.isfinite(agent.last_total_grad_norm))
        self.assertTrue(np.isfinite(agent.last_encoder_grad_norm))

    def test_actor_bootstrap_uses_action_destination_values(self):
        agent = FVBiCoordAgent(
            agent_n=2,
            feature_dim=3,
            hidden_dim=4,
            action_dim=3,
            adjacency=np.eye(2, dtype=np.float32),
            action_costs=np.zeros((2, 3), dtype=np.float32),
            action_destinations=np.asarray([[0, 1, -1], [1, 0, -1]], dtype=np.int64),
            device="cpu",
        )
        next_values = torch.asarray([[10.0, 20.0]], dtype=torch.float32)

        next_action_values = agent._next_values_for_actions(next_values)

        np.testing.assert_allclose(next_action_values.detach().numpy()[0], [[10.0, 20.0, 10.0], [20.0, 10.0, 20.0]])

    def test_destination_value_target_uses_regional_need_pressure(self):
        env = DispatchEnv(EnvConfig(num_cells=2, fleet_size=0, horizon_steps=1, seed=3), grid=HexGrid.create(2))
        state_matrix = np.zeros((2, env.state_feature_dim), dtype=np.float32)
        state_matrix[:, env.observed_gap_feature_index] = [0.25, 0.1]
        state_matrix[:, env.future_need_feature_index] = [0.2, 0.5]

        targets = _destination_value_targets_from_state_matrix(env, state_matrix)

        np.testing.assert_allclose(targets["region_value"], [0.25, 0.5])

    def test_destination_value_target_uses_arrival_step_pressure_when_eta_available(self):
        grid = HexGrid.from_axial_coords([(0, 0), (1, 0)], cell_width_km=2.5)
        rates = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 4.0],
                [0.0, 8.0],
            ],
            dtype=np.float32,
        )
        demand = TabularDemand(
            rates=rates,
            od_probs=np.full((3, 2, 2), 0.5, dtype=np.float32),
        )
        env = DispatchEnv(
            EnvConfig(num_cells=2, fleet_size=2, horizon_steps=3, step_minutes=10, seed=3),
            demand=demand,
            grid=grid,
            road_time_matrix=np.asarray([[0.0, 10.0], [10.0, 0.0]], dtype=np.float32),
        )
        state_matrix = np.zeros((2, env.state_feature_dim), dtype=np.float32)
        state_matrix[:, env.observed_gap_feature_index] = [0.25, 0.1]
        state_matrix[:, env.future_need_feature_index] = [0.2, 0.5]
        state_matrix[0, env.idle_supply_feature_index] = 1.0

        targets = _destination_value_targets_from_state_matrix(env, state_matrix)

        np.testing.assert_allclose(targets["region_value"], [0.25, 4.0])

    def test_future_pressure_auxiliary_target_uses_next_observed_gap(self):
        env = DispatchEnv(EnvConfig(num_cells=2, fleet_size=0, horizon_steps=1, seed=3), grid=HexGrid.create(2))
        next_state_matrix = np.zeros((2, env.state_feature_dim), dtype=np.float32)
        next_state_matrix[:, env.observed_gap_feature_index] = [0.25, 0.5]

        targets = _future_pressure_targets_from_next_state_matrix(env, next_state_matrix)

        self.assertNotIn("region_value", targets)
        np.testing.assert_allclose(targets["future_pressure"], [0.25, 0.5])

    def test_future_pressure_target_uses_configured_lookahead(self):
        grid = HexGrid.create(2)
        rates = np.asarray([[0.0, 0.0], [3.0, 1.0]], dtype=np.float32)
        env = DispatchEnv(
            EnvConfig(num_cells=2, fleet_size=4, horizon_steps=2, future_demand_steps=1, seed=3),
            demand=TabularDemand(rates=rates, od_probs=np.full((2, 2, 2), 0.5, dtype=np.float32)),
            grid=grid,
        )

        env.reset()
        target = _future_pressure_target_from_env(env)

        np.testing.assert_allclose(target, rates[1] / env.feature_scale)

    def test_terminal_transitions_do_not_bootstrap_from_next_value(self):
        torch.manual_seed(123)
        grid = HexGrid.create(7)
        adjacency = build_road_time_adjacency(grid)
        action_costs = np.zeros((7, 7), dtype=np.float32)
        agent_a = FVBiCoordAgent(
            agent_n=7,
            feature_dim=5,
            hidden_dim=8,
            action_dim=7,
            adjacency=adjacency,
            action_costs=action_costs,
            future_value_loss_weight=0.0,
            future_gap_loss_weight=0.0,
            intensity_loss_weight=0.0,
            gamma=1.0,
            temporal_window=2,
            device="cpu",
        )
        agent_b = FVBiCoordAgent(
            agent_n=7,
            feature_dim=5,
            hidden_dim=8,
            action_dim=7,
            adjacency=adjacency,
            action_costs=action_costs,
            future_value_loss_weight=0.0,
            future_gap_loss_weight=0.0,
            intensity_loss_weight=0.0,
            gamma=1.0,
            temporal_window=2,
            device="cpu",
        )
        agent_b.net.load_state_dict(agent_a.net.state_dict())
        agent_b.target_net.load_state_dict(agent_a.target_net.state_dict())
        transition = {
            "sequence": np.zeros((2, 7, 5), dtype=np.float32),
            "critic_rewards": np.ones(7, dtype=np.float32),
            "actor_rewards": np.ones((7, 7), dtype=np.float32),
            "available_actions": (grid.neighbors >= 0).astype(np.float32),
            "done": True,
        }
        transition_a = {**transition, "next_sequence": np.zeros((2, 7, 5), dtype=np.float32)}
        transition_b = {**transition, "next_sequence": np.ones((2, 7, 5), dtype=np.float32) * 100.0}

        actor_loss_a, critic_loss_a = agent_a.update([transition_a])
        actor_loss_b, critic_loss_b = agent_b.update([transition_b])

        self.assertAlmostEqual(actor_loss_a, actor_loss_b, places=5)
        self.assertAlmostEqual(critic_loss_a, critic_loss_b, places=5)

    def test_state_separates_idle_and_incoming_supply_for_current_window(self):
        grid = HexGrid.create(7)
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=0,
                horizon_steps=3,
                step_minutes=10,
                taxi_to_idle_weight=0.5,
                future_demand_steps=0,
                seed=23,
            ),
            grid=grid,
        )
        env.idle_by_cell[1].append(10)
        env.busy_arrivals[5].append((20, 2, 7))
        env.busy_arrivals[15].append((21, 3, 7))

        env.step_index = 0
        env._update_state_and_rewards()
        self.assertEqual(env._state_raw[1, env.idle_supply_feature_index], 1.0)
        self.assertEqual(env._state_raw[2, env.incoming_supply_feature_index], 1.0)
        self.assertEqual(env._state_raw[3, env.incoming_supply_feature_index], 0.0)
        self.assertEqual(env._state_raw[2, env.supply_demand_gap_feature_index], 0.5)

        env.step_index = 1
        env._update_state_and_rewards()
        self.assertEqual(env._state_raw[2, env.incoming_supply_feature_index], 0.0)
        self.assertEqual(env._state_raw[3, env.incoming_supply_feature_index], 1.0)

    def test_tabular_demand_from_csv(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "od.csv"
            path.write_text("step,origin,destination,count\n0,1,2,3\n1,2,1,2\n", encoding="utf-8")
            demand = TabularDemand.from_csv(path, num_cells=4, horizon_steps=3)
            rng = np.random.default_rng(0)
            np.testing.assert_array_equal(demand.generate(0, rng), np.asarray([0, 3, 0, 0]))
            sampled = demand.sample_destinations(origin=1, step=0, n=5, rng=rng)
            np.testing.assert_array_equal(sampled, np.full(5, 2))

    def test_deterministic_demand_rounding_preserves_fractional_total(self):
        rng = np.random.default_rng(0)
        rates = np.asarray([[0.4, 0.4, 0.4]], dtype=np.float32)
        od_probs = np.full((1, 3, 3), 1.0 / 3.0, dtype=np.float32)
        tabular = TabularDemand(rates=rates, od_probs=od_probs)

        np.testing.assert_array_equal(tabular.generate(0, rng), np.asarray([1, 0, 0]))

        od_counts = np.zeros((1, 3, 3), dtype=np.float32)
        trip_minutes_sum = np.zeros_like(od_counts)
        trip_minutes_count = np.zeros_like(od_counts)
        od_counts[0, :, 0] = 0.4
        empirical = EmpiricalTripDemand.from_counts(od_counts, trip_minutes_sum, trip_minutes_count)

        np.testing.assert_array_equal(empirical.generate(0, rng), np.asarray([1, 0, 0]))

    def test_empirical_trip_demand_uses_duration_means(self):
        od_counts = np.zeros((2, 3, 3), dtype=np.float32)
        duration_sum = np.zeros_like(od_counts)
        duration_count = np.zeros_like(od_counts)
        od_counts[0, 1, 2] = 3
        duration_sum[0, 1, 2] = 36
        duration_count[0, 1, 2] = 3
        demand = EmpiricalTripDemand.from_counts(od_counts, duration_sum, duration_count)
        rng = np.random.default_rng(0)
        np.testing.assert_array_equal(demand.generate(0, rng), np.asarray([0, 3, 0]))
        np.testing.assert_allclose(demand.sample_trip_minutes(1, 0, np.asarray([2, 2]), rng), [12, 12])

    def test_trip_minute_fallback_uses_empirical_global_mean(self):
        rates = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        od_probs = np.full((1, 3, 3), 1.0 / 3.0, dtype=np.float32)
        mean_trip_minutes = np.zeros((1, 3, 3), dtype=np.float32)
        mean_trip_minutes[0, 0, 1] = 8.0
        demand = EventTripDemand(
            rates=rates,
            od_probs=od_probs,
            mean_trip_minutes=mean_trip_minutes,
            events_by_day=(((),),),
        )

        minutes = demand.sample_trip_minutes(0, 0, np.asarray([2]), np.random.default_rng(0))

        np.testing.assert_allclose(minutes, [8.0])

    def test_chengdu_raw_loader_deduplicates_and_aggregates(self):
        bounds = (104.0, 30.6, 104.1, 30.7)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "orders.csv"
            rows = [
                RAW_COLUMN_ORDER,
                ("a", "2016-11-01 08:03:00", "2016-11-01 08:13:00", "104.020", "30.620", "104.080", "30.680"),
                ("a", "2016-11-01 08:03:00", "2016-11-01 08:13:00", "104.020", "30.620", "104.080", "30.680"),
                ("b", "2016-11-01 08:07:00", "2016-11-01 08:27:00", "104.030", "30.630", "104.070", "30.670"),
                ("c", "2016-11-01 12:00:00", "2016-11-01 12:10:00", "104.030", "30.630", "104.070", "30.670"),
            ]
            path.write_text("\n".join(",".join(row) for row in rows) + "\n", encoding="utf-8")
            grid = HexGrid.create_geographic(7, bounds)
            demand, stats = build_chengdu_demand(
                [path],
                grid=grid,
                horizon_steps=2,
                step_minutes=10,
                start_hour=8,
                bounds=bounds,
            )
            self.assertEqual(stats.rows_used, 2)
            self.assertEqual(stats.rows_duplicate, 1)
            self.assertEqual(int(demand.rates.sum()), 2)

    def test_chengdu_env_runs_paper_style_advance(self):
        bounds = (104.0, 30.6, 104.1, 30.7)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "orders.csv"
            rows = [
                RAW_COLUMN_ORDER,
                ("a", "2016-11-01 08:03:00", "2016-11-01 08:13:00", "104.020", "30.620", "104.080", "30.680"),
                ("b", "2016-11-01 08:07:00", "2016-11-01 08:27:00", "104.030", "30.630", "104.070", "30.670"),
            ]
            path.write_text("\n".join(",".join(row) for row in rows) + "\n", encoding="utf-8")
            env, stats = build_chengdu_env(
                raw_paths=[path],
                config=EnvConfig(num_cells=7, fleet_size=10, horizon_steps=2, step_minutes=10, seed=5),
                start_hour=8,
                bounds=bounds,
            )
            self.assertEqual(stats.rows_used, 2)
            observations, state = env.reset()
            self.assertEqual(observations.shape, (7, env.action_dim * env.state_feature_dim))
            next_observations, next_state, rewards, actor_rewards, done = env.advance(lambda cur_env, obs, st: park_actions(cur_env))
            self.assertEqual(next_observations.shape, observations.shape)
            self.assertEqual(next_state.shape, state.shape)
            self.assertEqual(rewards.shape, (7,))
            self.assertEqual(actor_rewards.shape, (7, 7))
            self.assertFalse(done)

    def test_chengdu_training_entry_rejects_single_source_eval(self):
        result = subprocess.run(
            [sys.executable, "scripts/train_chengdu_fv_bicoord.py"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no longer runs Chengdu Bi-STAR directly from one raw source", result.stderr + result.stdout)

    def test_chengdu_trajectory_env_uses_fixed_paper_grid(self):
        bounds = (104.0, 30.6, 104.1, 30.7)
        with TemporaryDirectory() as tmp:
            tar_path = Path(tmp) / "2016_1101.tar.gz"
            rows = [
                ("司机ID", "订单ID", "GPS时间", "轨迹点经度", "轨迹点纬度"),
                ("d1", "o1", "2016-11-01 08:03:00", "104.020", "30.620"),
                ("d1", "o1", "2016-11-01 08:08:00", "104.021", "30.621"),
                ("d1", "o1", "2016-11-01 08:13:00", "104.022", "30.622"),
                ("d2", "o2", "2016-11-01 08:07:00", "104.030", "30.630"),
                ("d2", "o2", "2016-11-01 08:12:00", "104.031", "30.631"),
                ("d2", "o2", "2016-11-01 08:17:00", "104.032", "30.632"),
            ]
            payload = ("\n".join(",".join(row) for row in rows) + "\n").encode("utf-8")
            with tarfile.open(tar_path, "w:gz") as tar:
                info = tarfile.TarInfo("2016_1101.csv")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            env, stats = build_chengdu_trajectory_env(
                trajectory_paths=[tar_path],
                config=EnvConfig(num_cells=7, cell_width_km=2.5, fleet_size=10, horizon_steps=2, step_minutes=10, seed=5),
                start_hour=8,
                bounds=bounds,
                high_demand_min_orders_per_minute=0.0,
            )
            self.assertEqual(stats.source, "chengdu_trajectory_orders")
            self.assertEqual(stats.rows_used, 2)
            self.assertEqual(stats.trajectory_points, 6)
            self.assertEqual(env.grid.num_cells, 7)
            self.assertAlmostEqual(env.grid.cell_width_km, 2.5)
            observations, state = env.reset()
            next_observations, next_state, rewards, actor_rewards, done = env.advance(
                lambda cur_env, obs, st: park_actions(cur_env)
            )
            self.assertEqual(next_observations.shape, observations.shape)
            self.assertEqual(next_state.shape, state.shape)
            self.assertEqual(rewards.shape, (7,))
            self.assertEqual(actor_rewards.shape, (7, 7))
            self.assertFalse(done)

    def test_chengdu_trajectory_event_demand_preserves_minute_offsets(self):
        bounds = (104.0, 30.6, 104.1, 30.7)
        with TemporaryDirectory() as tmp:
            tar_path = Path(tmp) / "2016_1101.tar.gz"
            rows = [
                ("driver_id", "order_id", "time", "lon", "lat"),
                ("d1", "o1", "2016-11-01 08:03:00", "104.020", "30.620"),
                ("d1", "o1", "2016-11-01 08:08:00", "104.021", "30.621"),
                ("d1", "o1", "2016-11-01 08:13:00", "104.022", "30.622"),
                ("d2", "o2", "2016-11-01 08:07:00", "104.030", "30.630"),
                ("d2", "o2", "2016-11-01 08:12:00", "104.031", "30.631"),
                ("d2", "o2", "2016-11-01 08:17:00", "104.032", "30.632"),
            ]
            payload = ("\n".join(",".join(row) for row in rows) + "\n").encode("utf-8")
            with tarfile.open(tar_path, "w:gz") as tar:
                info = tarfile.TarInfo("2016_1101.csv")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            assignment_grid = HexGrid.create_geographic_fixed(bounds, cell_width_km=2.5, padding_km=2.5)
            selected_coords = select_high_demand_trajectory_cells(
                [tar_path],
                grid=assignment_grid,
                num_cells=7,
                horizon_steps=2,
                step_minutes=10,
                start_hour=8,
                bounds=bounds,
                min_orders_per_minute=0.0,
            )
            grid = HexGrid.from_axial_coords(
                selected_coords,
                cell_width_km=assignment_grid.cell_width_km,
                projection_origin=assignment_grid.projection_origin,
            )
            demand, stats, _initial = build_chengdu_trajectory_event_demand(
                [tar_path],
                raw_file_count=1,
                grid=grid,
                assignment_grid=assignment_grid,
                horizon_steps=2,
                step_minutes=10,
                start_hour=8,
                bounds=bounds,
                high_demand_min_orders_per_minute=0.0,
            )

            self.assertIsInstance(demand, EventTripDemand)
            self.assertEqual(stats.source, "chengdu_trajectory_event_orders")
            self.assertEqual(stats.rows_used, 2)
            demand.reset_episode(np.random.default_rng(0))
            events = demand.sample_requests(0, np.random.default_rng(1))
            self.assertEqual([event.minute_offset for event in events], [3, 7])

    def test_chengdu_trajectory_experiment_trains_aggregate_and_evaluates_events(self):
        bounds = (104.0, 30.6, 104.1, 30.7)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()

            def write_trajectory_tar(path: Path, date: str) -> None:
                rows = [
                    ("driver_id", "order_id", "time", "lon", "lat"),
                    ("d1", "o1", f"{date} 08:03:00", "104.020", "30.620"),
                    ("d1", "o1", f"{date} 08:08:00", "104.021", "30.621"),
                    ("d1", "o1", f"{date} 08:13:00", "104.022", "30.622"),
                    ("d2", "o2", f"{date} 08:07:00", "104.030", "30.630"),
                    ("d2", "o2", f"{date} 08:12:00", "104.031", "30.631"),
                    ("d2", "o2", f"{date} 08:17:00", "104.032", "30.632"),
                ]
                payload = ("\n".join(",".join(row) for row in rows) + "\n").encode("utf-8")
                with tarfile.open(path, "w:gz") as tar:
                    info = tarfile.TarInfo(path.with_suffix("").with_suffix(".csv").name)
                    info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))

            write_trajectory_tar(raw_dir / "2016_1101.tar.gz", "2016-11-01")
            write_trajectory_tar(raw_dir / "2016_1108.tar.gz", "2016-11-08")

            out_dir = root / "out"
            args = build_parser().parse_args(
                [
                    "--real-demand",
                    "chengdu-trajectory",
                    "--trajectory",
                    str(raw_dir),
                    "--method",
                    "park",
                    "--cells",
                    "7",
                    "--taxis",
                    "10",
                    "--horizon-steps",
                    "2",
                    "--step-minutes",
                    "10",
                    "--start-hour",
                    "8",
                    "--bounds",
                    ",".join(str(value) for value in bounds),
                    "--min-orders-per-minute",
                    "0",
                    "--demand-scale",
                    "1.0",
                    "--road-network-cache",
                    str(root / "missing.graphml"),
                    "--out",
                    str(out_dir),
                    "--no-plots",
                ]
            )

            run_experiment(args)

            train_stats = json.loads((out_dir / "chengdu_demand_stats.json").read_text(encoding="utf-8"))
            eval_stats = json.loads((out_dir / "chengdu_eval_demand_stats.json").read_text(encoding="utf-8"))
            split_stats = json.loads((out_dir / "chengdu_trajectory_split_stats.json").read_text(encoding="utf-8"))
            with (out_dir / "summary.csv").open("r", newline="", encoding="utf-8") as f:
                rows_out = list(csv.DictReader(f))

            self.assertEqual(train_stats["source"], "chengdu_trajectory_orders")
            self.assertEqual(eval_stats["source"], "chengdu_trajectory_event_orders")
            self.assertEqual(split_stats["split_mode"], "default_date_range")
            self.assertEqual(split_stats["train_files"], 1)
            self.assertEqual(split_stats["test_files"], 1)
            self.assertEqual(eval_stats["days"], 1)
            self.assertEqual(float(rows_out[0]["orders"]), 2.0)

    def test_event_trip_demand_stochastic_resamples_event_stream(self):
        rates = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        od_probs = np.full((1, 3, 3), 1.0 / 3.0, dtype=np.float32)
        mean_trip_minutes = np.zeros((1, 3, 3), dtype=np.float32)
        event = TripEvent(origin=0, destination=1, minute_offset=0, trip_minutes=5.0)
        demand = EventTripDemand(
            rates=rates,
            od_probs=od_probs,
            mean_trip_minutes=mean_trip_minutes,
            events_by_day=(((event,),),),
            stochastic=True,
        )

        rng = np.random.default_rng(0)
        request_counts = [len(demand.sample_requests(0, rng)) for _ in range(4)]

        self.assertEqual(request_counts, [1, 0, 0, 3])
        np.testing.assert_array_equal(demand.generate(0, np.random.default_rng(0)), np.asarray([1, 0, 0]))
        self.assertFalse(EventTripDemand.__dataclass_fields__["_day_index"].init)

    def test_event_stream_demand_scale_changes_real_event_count(self):
        events = [
            TripEvent(origin=0, destination=1, minute_offset=0, trip_minutes=5.0),
            TripEvent(origin=0, destination=1, minute_offset=5, trip_minutes=5.0),
        ]

        downscaled = _scale_step_events(events, 0.5)
        upscaled = _scale_step_events(events, 1.5)

        self.assertEqual(len(downscaled), 1)
        self.assertEqual(len(upscaled), 3)

    def test_peak_hotspot_boost_only_changes_peak_hotspot_orders(self):
        rates = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        od_probs = np.full((2, 2, 2), 0.5, dtype=np.float32)
        mean_trip_minutes = np.zeros((2, 2, 2), dtype=np.float32)
        events = (
            (
                TripEvent(origin=0, destination=1, minute_offset=0, trip_minutes=5.0),
                TripEvent(origin=1, destination=0, minute_offset=1, trip_minutes=5.0),
            ),
            (
                TripEvent(origin=0, destination=1, minute_offset=0, trip_minutes=5.0),
                TripEvent(origin=1, destination=0, minute_offset=1, trip_minutes=5.0),
            ),
        )
        demand = EventTripDemand(
            rates=rates,
            od_probs=od_probs,
            mean_trip_minutes=mean_trip_minutes,
            events_by_day=(events,),
        )

        boosted, cells, windows = apply_peak_hotspot_boost_to_demand(
            demand,
            start_hour=8.0,
            step_minutes=10,
            peak_hotspot_multiplier=3.0,
            peak_hotspot_windows="8-8.2",
            peak_hotspot_cells="0",
        )

        self.assertEqual(cells, (0,))
        self.assertEqual(windows, ((8.0, 8.2),))
        np.testing.assert_allclose(boosted.rates, [[3.0, 1.0], [1.0, 1.0]])
        boosted.reset_episode(np.random.default_rng(0))
        self.assertEqual(len(boosted.sample_requests(0, np.random.default_rng(1))), 4)
        self.assertEqual(len(boosted.sample_requests(1, np.random.default_rng(1))), 2)
        self.assertEqual(boosted.metadata["peak_hotspot_periods"][0]["period"], "morning_peak")

    def test_peak_hotspot_boost_selects_separate_cells_for_morning_midday_evening(self):
        rates = np.zeros((12, 3), dtype=np.float32)
        rates[0, 0] = 5.0
        rates[5, 1] = 5.0
        rates[10, 2] = 5.0
        od_probs = np.full((12, 3, 3), 1.0 / 3.0, dtype=np.float32)
        mean_trip_minutes = np.zeros((12, 3, 3), dtype=np.float32)
        demand = EventTripDemand(
            rates=rates,
            od_probs=od_probs,
            mean_trip_minutes=mean_trip_minutes,
            events_by_day=tuple(tuple(() for _ in range(12)) for _ in range(1)),
        )

        boosted, cells, windows = apply_peak_hotspot_boost_to_demand(
            demand,
            start_hour=7.0,
            step_minutes=60,
            peak_hotspot_multiplier=2.0,
            peak_hotspot_windows="morning_peak=7-8,midday_peak=12-13,evening_peak=17-18",
            peak_hotspot_top_cells=1,
        )

        self.assertEqual(cells, (0, 1, 2))
        self.assertEqual(windows, ((7.0, 8.0), (12.0, 13.0), (17.0, 18.0)))
        np.testing.assert_allclose(boosted.rates[[0, 5, 10], [0, 1, 2]], [10.0, 10.0, 10.0])
        periods = boosted.metadata["peak_hotspot_periods"]
        self.assertEqual([period["period"] for period in periods], ["morning_peak", "midday_peak", "evening_peak"])
        self.assertEqual([period["cells"] for period in periods], [(0,), (1,), (2,)])

    def test_chengdu_preprocess_defaults_to_aggregate_od_with_three_peak_windows(self):
        from scripts.preprocess_chengdu_train_test import build_parser as build_preprocess_parser

        parser = build_preprocess_parser()
        args = parser.parse_args([])
        event_args = parser.parse_args(["--event-stream"])

        self.assertTrue(args.aggregate_od)
        self.assertIn("midday_peak=12-14", args.peak_hotspot_windows)
        self.assertFalse(event_args.aggregate_od)

    def test_trajectory_demand_scale_and_peak_boost_keep_real_initial_distribution(self):
        bounds = (104.0, 30.6, 104.1, 30.7)
        with TemporaryDirectory() as tmp:
            tar_path = Path(tmp) / "2016_1101.tar.gz"
            rows = [
                ("driver_id", "order_id", "time", "lon", "lat"),
                ("d1", "o1", "2016-11-01 08:03:00", "104.020", "30.620"),
                ("d1", "o1", "2016-11-01 08:08:00", "104.021", "30.621"),
                ("d1", "o1", "2016-11-01 08:13:00", "104.022", "30.622"),
                ("d2", "o2", "2016-11-01 08:07:00", "104.030", "30.630"),
                ("d2", "o2", "2016-11-01 08:12:00", "104.031", "30.631"),
                ("d2", "o2", "2016-11-01 08:17:00", "104.032", "30.632"),
            ]
            payload = ("\n".join(",".join(row) for row in rows) + "\n").encode("utf-8")
            with tarfile.open(tar_path, "w:gz") as tar:
                info = tarfile.TarInfo("2016_1101.csv")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            assignment_grid = HexGrid.create_geographic_fixed(bounds, cell_width_km=2.5, padding_km=2.5)
            selected_coords = select_high_demand_trajectory_cells(
                [tar_path],
                grid=assignment_grid,
                num_cells=7,
                horizon_steps=2,
                step_minutes=10,
                start_hour=8,
                bounds=bounds,
                min_orders_per_minute=0.0,
            )
            grid = HexGrid.from_axial_coords(
                selected_coords,
                cell_width_km=assignment_grid.cell_width_km,
                projection_origin=assignment_grid.projection_origin,
            )
            demand, stats, initial = build_chengdu_trajectory_event_demand(
                [tar_path],
                raw_file_count=1,
                grid=grid,
                assignment_grid=assignment_grid,
                horizon_steps=2,
                step_minutes=10,
                start_hour=8,
                bounds=bounds,
                high_demand_min_orders_per_minute=0.0,
                demand_scale=2.0,
                peak_hotspot_multiplier=3.0,
                peak_hotspot_windows="8-8.2",
                peak_hotspot_top_cells=1,
            )

            self.assertIsNotNone(initial)
            self.assertAlmostEqual(float(initial.sum()), 2.0)
            self.assertGreater(float(demand.rates.sum()), 4.0)
            self.assertEqual(stats.peak_hotspot_multiplier, 3.0)
            self.assertEqual(len(stats.peak_hotspot_cells), 1)

    def test_event_orders_cancel_at_step_boundary(self):
        grid = HexGrid.create(7)
        rates = np.zeros((2, 7), dtype=np.float32)
        rates[0, 0] = 1
        od_probs = np.full((2, 7, 7), 1.0 / 7.0, dtype=np.float32)
        mean_trip_minutes = np.zeros((2, 7, 7), dtype=np.float32)
        event = TripEvent(origin=0, destination=1, minute_offset=9, trip_minutes=5.0, pickup_xy=(0.0, 0.0))
        demand = EventTripDemand(
            rates=rates,
            od_probs=od_probs,
            mean_trip_minutes=mean_trip_minutes,
            events_by_day=(((event,), ()),),
        )
        env = DispatchEnv(
            EnvConfig(num_cells=7, fleet_size=0, horizon_steps=2, step_minutes=10, max_wait_steps=1, seed=7),
            demand=demand,
            grid=grid,
        )

        env.advance(lambda cur_env, obs, st: park_actions(cur_env))
        self.assertEqual(env.total_orders, 1)
        self.assertEqual(env.cancelled_orders, 1)
        self.assertEqual(env.waiting_orders, [])

        env.advance(lambda cur_env, obs, st: park_actions(cur_env))
        self.assertEqual(env.cancelled_orders, 1)

    def test_order_from_previous_step_cannot_be_served_after_boundary(self):
        grid = HexGrid.create(7)
        rates = np.zeros((2, 7), dtype=np.float32)
        rates[0, 0] = 1
        od_probs = np.full((2, 7, 7), 1.0 / 7.0, dtype=np.float32)
        mean_trip_minutes = np.zeros((2, 7, 7), dtype=np.float32)
        event = TripEvent(origin=0, destination=1, minute_offset=9, trip_minutes=5.0, pickup_xy=tuple(grid.xy[0]))
        demand = EventTripDemand(
            rates=rates,
            od_probs=od_probs,
            mean_trip_minutes=mean_trip_minutes,
            events_by_day=(((event,), ()),),
        )
        env = DispatchEnv(
            EnvConfig(
                num_cells=7,
                fleet_size=1,
                horizon_steps=2,
                step_minutes=10,
                max_wait_steps=2,
                pickup_radius_km=10.0,
                seed=7,
            ),
            demand=demand,
            grid=grid,
        )
        env.idle_by_cell = [[] for _ in range(env.grid_number)]
        env.taxi_cell[0] = -1
        env.reposition_arrivals[10].append((0, 0))

        env.advance(lambda cur_env, obs, st: park_actions(cur_env))
        self.assertEqual(env.cancelled_orders, 1)
        self.assertEqual(env.served_orders, 0)
        self.assertEqual(env.waiting_orders, [])

        env.advance(lambda cur_env, obs, st: park_actions(cur_env))
        self.assertEqual(env.cancelled_orders, 1)
        self.assertEqual(env.served_orders, 0)

    def test_policy_state_includes_current_step_event_orders(self):
        grid = HexGrid.create(7)
        rates = np.zeros((1, 7), dtype=np.float32)
        rates[0, 0] = 2
        od_probs = np.full((1, 7, 7), 1.0 / 7.0, dtype=np.float32)
        mean_trip_minutes = np.zeros((1, 7, 7), dtype=np.float32)
        events = (
            TripEvent(origin=0, destination=1, minute_offset=0, trip_minutes=5.0, pickup_xy=(0.0, 0.0)),
            TripEvent(origin=0, destination=1, minute_offset=9, trip_minutes=5.0, pickup_xy=(0.0, 0.0)),
        )
        demand = EventTripDemand(
            rates=rates,
            od_probs=od_probs,
            mean_trip_minutes=mean_trip_minutes,
            events_by_day=((events,),),
        )
        env = DispatchEnv(
            EnvConfig(num_cells=7, fleet_size=0, horizon_steps=1, step_minutes=10, max_wait_steps=1, seed=7),
            demand=demand,
            grid=grid,
        )
        seen_demand = []

        def policy(cur_env, obs, st):
            seen_demand.append(float(cur_env._state_raw[:, 0].sum()))
            return park_actions(cur_env)

        env.advance(policy)

        self.assertEqual(env.total_orders, 2)
        self.assertEqual(seen_demand, [2.0])

    def test_experiment_policy_sees_orders_generated_by_advance(self):
        grid = HexGrid.create(7)
        od_counts = np.zeros((1, 7, 7), dtype=np.float32)
        trip_minutes_sum = np.zeros_like(od_counts)
        trip_minutes_count = np.zeros_like(od_counts)
        od_counts[0, 0, 1] = 3
        trip_minutes_sum[0, 0, 1] = 30
        trip_minutes_count[0, 0, 1] = 3
        demand = EmpiricalTripDemand.from_counts(od_counts, trip_minutes_sum, trip_minutes_count)
        config = EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1, seed=13)
        seen_demand = []

        def policy(cur_env, observations, state):
            seen_demand.append(float(state.reshape(cur_env.grid_number, cur_env.state_feature_dim)[:, 0].sum()))
            return park_actions(cur_env)

        metrics, _env = evaluate_policy(config, policy, episodes=1, seed=13, demand=demand, grid=grid)
        self.assertEqual(metrics.orders, 3)
        self.assertEqual(len(seen_demand), 1)
        self.assertGreater(seen_demand[0], 0.0)

    def test_fv_final_checkpoint_uses_post_update_policy_state(self):
        grid = HexGrid.create(7)
        rates = np.zeros((1, 7), dtype=np.float32)
        rates[0, 0] = 1.0
        od_probs = np.full((1, 7, 7), 1.0 / 7.0, dtype=np.float32)
        demand = TabularDemand(rates=rates, od_probs=od_probs)
        original_update = FVBiCoordAgent.update

        def mutating_update(agent, transitions):
            with torch.no_grad():
                for network in (agent.net, agent.target_net):
                    for parameter in network.parameters():
                        parameter.fill_(123.0)
            agent.last_policy_entropy = 0.0
            agent.last_entropy_loss = 0.0
            return 0.0, 0.0

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            try:
                FVBiCoordAgent.update = mutating_update
                agent, history = train_fv_bicoord(
                    config=EnvConfig(num_cells=7, fleet_size=10, horizon_steps=1, seed=13),
                    episodes=1,
                    hidden_dim=8,
                    actor_lr=1e-4,
                    critic_lr=1e-3,
                    tau=0.005,
                    gamma=0.9,
                    device="cpu",
                    seed=13,
                    out_dir=out_dir,
                    road_time_weight=0.0,
                    temporal_window=1,
                    graph_temperature=10.0,
                    attention_heads=1,
                    symmetric_adjacency=False,
                    region_value_loss_weight=0.1,
                    future_gap_loss_weight=0.1,
                    future_demand_loss_weight=0.1,
                    intensity_loss_weight=0.02,
                    demand=demand,
                    grid=grid,
                )
            finally:
                FVBiCoordAgent.update = original_update

            self.assertTrue(
                all(torch.allclose(parameter, torch.full_like(parameter, 123.0)) for parameter in agent.net.parameters())
            )
            final_saved = torch.load(out_dir / "fv_bicoord_model" / "fv_bicoord_actor_critic.pt", map_location="cpu")
            self.assertTrue(all(torch.allclose(value, torch.full_like(value, 123.0)) for value in final_saved.values()))
            final_metadata = json.loads((out_dir / "fv_bicoord_final_checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(final_metadata["checkpoint_type"], "final")

    def test_fv_training_reset_uses_episode_index(self):
        class RecordingDemand:
            def __init__(self):
                self.rates = np.zeros((1, 7), dtype=np.float32)
                self.od_probs = np.full((1, 7, 7), 1.0 / 7.0, dtype=np.float32)
                self.episode_indices: list[int | None] = []
                self.episode_count = 2

            def reset_episode(self, _rng, episode_index=None):
                self.episode_indices.append(episode_index)

            def generate(self, _step, _rng):
                return np.zeros(7, dtype=np.int64)

            def sample_destinations(self, _origin, _step, n, _rng):
                return np.zeros(n, dtype=np.int64)

        demand = RecordingDemand()
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            live_history_path = out_dir / "live_history.csv"
            live_epoch_history_path = out_dir / "live_epochs.csv"
            _agent, history = train_fv_bicoord(
                config=EnvConfig(num_cells=7, fleet_size=3, horizon_steps=1, seed=21),
                episodes=2,
                hidden_dim=8,
                actor_lr=1e-4,
                critic_lr=1e-3,
                tau=0.005,
                gamma=0.9,
                device="cpu",
                seed=21,
                out_dir=out_dir,
                road_time_weight=0.0,
                temporal_window=1,
                graph_temperature=10.0,
                attention_heads=1,
                symmetric_adjacency=False,
                region_value_loss_weight=0.1,
                future_gap_loss_weight=0.1,
                future_demand_loss_weight=0.1,
                intensity_loss_weight=0.02,
                demand=demand,
                grid=HexGrid.create(7),
                live_history_path=live_history_path,
                live_epoch_history_path=live_epoch_history_path,
            )
            with live_history_path.open("r", newline="", encoding="utf-8") as f:
                live_rows = list(csv.DictReader(f))
            with live_epoch_history_path.open("r", newline="", encoding="utf-8") as f:
                live_epoch_rows = list(csv.DictReader(f))

        self.assertEqual(demand.episode_indices[-2:], [0, 1])
        self.assertEqual([int(row["epoch"]) for row in history], [1, 1])
        self.assertEqual([int(row["epoch_episode"]) for row in history], [1, 2])
        epoch_history = aggregate_training_epochs(history)
        self.assertEqual(len(epoch_history), 1)
        self.assertEqual(int(epoch_history[0]["episodes"]), 2)
        self.assertEqual([int(float(row["episode"])) for row in live_rows], [1, 2])
        self.assertEqual(len(live_epoch_rows), 1)
        self.assertEqual(int(float(live_epoch_rows[0]["episodes"])), 2)

    def test_training_epoch_aggregation_is_order_weighted(self):
        history = [
            {
                "episode": 1.0,
                "epoch": 1.0,
                "epoch_size": 2.0,
                "reward": 1.0,
                "response_rate": 1.0,
                "response_time_seconds": 10.0,
                "cancellation_rate": 0.0,
                "occupied_rate": 0.1,
                "normalized_gmv": 1.0,
                "orders": 1.0,
                "served_orders": 1.0,
                "cancellations": 0.0,
                "actor_loss": 0.1,
                "critic_loss": 0.2,
                "policy_entropy": 0.3,
                "entropy_loss": -0.01,
            },
            {
                "episode": 2.0,
                "epoch": 1.0,
                "epoch_size": 2.0,
                "reward": 3.0,
                "response_rate": 0.0,
                "response_time_seconds": 100.0,
                "cancellation_rate": 1.0,
                "occupied_rate": 0.3,
                "normalized_gmv": 3.0,
                "orders": 9.0,
                "served_orders": 0.0,
                "cancellations": 9.0,
                "actor_loss": 0.3,
                "critic_loss": 0.4,
                "policy_entropy": 0.5,
                "entropy_loss": -0.03,
            },
        ]

        epoch_history = aggregate_training_epochs(history)

        self.assertEqual(len(epoch_history), 1)
        self.assertAlmostEqual(epoch_history[0]["response_rate"], 0.1)
        self.assertAlmostEqual(epoch_history[0]["response_time_seconds"], 10.0)
        self.assertAlmostEqual(epoch_history[0]["reward"], 2.0)

    def test_experiment_runs_chengdu_fv_bicoord_training_entry(self):
        bounds = (104.0, 30.6, 104.1, 30.7)
        with TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "orders.csv"
            rows = [
                RAW_COLUMN_ORDER,
                ("a", "2016-11-01 08:03:00", "2016-11-01 08:13:00", "104.020", "30.620", "104.080", "30.680"),
                ("b", "2016-11-01 08:07:00", "2016-11-01 08:27:00", "104.030", "30.630", "104.070", "30.670"),
            ]
            raw_path.write_text("\n".join(",".join(row) for row in rows) + "\n", encoding="utf-8")
            out_dir = Path(tmp) / "out"
            args = build_parser().parse_args(
                [
                    "--real-demand",
                    "chengdu",
                    "--raw",
                    str(raw_path),
                    "--method",
                    "fv_bicoord",
                    "--episodes",
                    "1",
                    "--eval-episodes",
                    "1",
                    "--cells",
                    "7",
                    "--taxis",
                    "10",
                    "--horizon-steps",
                    "1",
                    "--step-minutes",
                    "10",
                    "--start-hour",
                    "8",
                    "--bounds",
                    ",".join(str(x) for x in bounds),
                    "--hidden-dim",
                    "8",
                    "--device",
                    "cpu",
                    "--out",
                    str(out_dir),
                    "--no-plots",
                ]
            )

            run_experiment(args)
            self.assertTrue((out_dir / "fv_bicoord_training_history.csv").exists())
            self.assertTrue((out_dir / "fv_bicoord_training_epochs.csv").exists())
            self.assertTrue((out_dir / "summary.csv").exists())
            self.assertTrue((out_dir / "chengdu_grid.csv").exists())
            self.assertTrue((out_dir / "chengdu_demand_stats.json").exists())

    def test_synthetic_experiment_ignores_default_chengdu_road_cache(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            args = build_parser().parse_args(
                [
                    "--method",
                    "fv_bicoord",
                    "--episodes",
                    "1",
                    "--eval-episodes",
                    "1",
                    "--cells",
                    "7",
                    "--taxis",
                    "30",
                    "--horizon-steps",
                    "2",
                    "--hidden-dim",
                    "8",
                    "--device",
                    "cpu",
                    "--out",
                    str(out_dir),
                    "--no-plots",
                ]
            )

            run_experiment(args)
            self.assertTrue((out_dir / "fv_bicoord_training_history.csv").exists())
            self.assertTrue((out_dir / "fv_bicoord_training_epochs.csv").exists())
            self.assertTrue((out_dir / "summary.csv").exists())

    def test_synthetic_experiment_runs_fv_bicoord_method(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            args = build_parser().parse_args(
                [
                    "--method",
                    "fv_bicoord",
                    "--episodes",
                    "1",
                    "--eval-episodes",
                    "1",
                    "--cells",
                    "7",
                    "--taxis",
                    "30",
                    "--horizon-steps",
                    "2",
                    "--hidden-dim",
                    "8",
                    "--fv-temporal-window",
                    "2",
                    "--device",
                    "cpu",
                    "--out",
                    str(out_dir),
                    "--no-plots",
                ]
            )

            run_experiment(args)
            self.assertTrue((out_dir / "fv_bicoord_training_history.csv").exists())
            self.assertTrue((out_dir / "summary.csv").exists())
            with (out_dir / "summary.csv").open("r", newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["method"], "fv_bicoord")

    def test_agent_update_smoke(self):
        env = DispatchEnv(EnvConfig(num_cells=7, fleet_size=30, horizon_steps=2, seed=7))
        adjacency = build_road_time_adjacency(env.grid, step_minutes=env.config.step_minutes)
        agent = FVBiCoordAgent(
            agent_n=env.grid_number,
            feature_dim=env.state_feature_dim,
            hidden_dim=16,
            action_dim=env.action_dim,
            adjacency=adjacency,
            action_costs=env.action_costs,
            temporal_window=2,
            device="cpu",
        )
        _observations, initial_state = env.reset()
        sequence = None
        decision = None

        def policy(cur_env, obs, st):
            nonlocal sequence, decision
            state_matrix = np.asarray(st, dtype=np.float32).reshape(cur_env.grid_number, cur_env.state_feature_dim)
            sequence = np.stack([state_matrix, state_matrix], axis=0).astype(np.float32)
            decision = agent.take_decision(sequence, cur_env.available_actions)
            return decision

        next_observations, next_state, critic_rewards, actor_rewards, _ = env.advance(policy)
        self.assertIsNotNone(sequence)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["actions"].shape, (env.grid_number, env.action_dim))

        next_state_matrix = np.asarray(next_state, dtype=np.float32).reshape(env.grid_number, env.state_feature_dim)
        next_sequence = np.stack(
            [
                np.asarray(initial_state, dtype=np.float32).reshape(env.grid_number, env.state_feature_dim),
                next_state_matrix,
            ],
            axis=0,
        ).astype(np.float32)
        actor_loss, critic_loss = agent.update(
            [
                {
                    "sequence": sequence,
                    "next_sequence": next_sequence,
                    "critic_rewards": critic_rewards,
                    "actor_rewards": actor_rewards,
                    "available_actions": env.available_actions.copy(),
                }
            ]
        )
        self.assertTrue(np.isfinite(actor_loss))
        self.assertTrue(np.isfinite(critic_loss))
        self.assertEqual(next_observations.shape, (env.grid_number, env.observation_dim))


if __name__ == "__main__":
    unittest.main()
