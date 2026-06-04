from __future__ import annotations

import numpy as np

from .env import DispatchEnv


def park_actions(env: DispatchEnv) -> np.ndarray:
    actions = np.zeros((env.grid_number, env.action_dim), dtype=np.float32)
    actions[:, 0] = 1.0
    return actions


def random_actions(env: DispatchEnv, rng: np.random.Generator) -> np.ndarray:
    actions = np.zeros((env.grid_number, env.action_dim), dtype=np.float32)
    for cell in range(env.grid_number):
        idle_count = len(env.idle_by_cell[cell])
        valid = np.asarray(list(env.grid.valid_actions(cell)), dtype=np.int64)
        if idle_count <= 0 or valid.size == 0:
            actions[cell, 0] = 1.0
            continue
        sampled = rng.choice(valid, size=idle_count)
        counts = np.bincount(sampled, minlength=env.action_dim).astype(np.float32)
        actions[cell] = counts / float(idle_count)
    return _valid_action_probs(env, actions)


def diffusion_actions(env: DispatchEnv) -> np.ndarray:
    """Supply-only neighboring diffusion baseline from the paper."""

    raw = env.regional_state_raw()
    idle_supply = np.nan_to_num(raw[:, env.idle_supply_feature_index], nan=0.0, posinf=0.0, neginf=0.0)
    supply = np.maximum(idle_supply, 0.0)
    actions = np.zeros((env.grid_number, env.action_dim), dtype=np.float32)

    for cell in range(env.grid_number):
        local = env.grid.local_cells(cell)
        local_mean = float(np.mean(supply[local])) if local else float(supply[cell])
        excess = max(float(supply[cell] - local_mean), 0.0)
        if excess <= 0 or supply[cell] <= 0:
            actions[cell, 0] = 1.0
            continue

        deficits: list[tuple[int, float]] = []
        for action in range(1, env.action_dim):
            dst = int(env.grid.neighbors[cell, action])
            if dst < 0:
                continue
            deficit = max(local_mean - float(supply[dst]), 0.0)
            if deficit > 0:
                deficits.append((action, deficit))

        if not deficits:
            actions[cell, 0] = 1.0
            continue

        move_fraction = min(excess / max(float(supply[cell]), 1e-6), 0.65)
        total_deficit = sum(value for _, value in deficits)
        if not np.isfinite(total_deficit) or total_deficit <= 0:
            actions[cell, 0] = 1.0
            continue
        actions[cell, 0] = 1.0 - move_fraction
        for action, deficit in deficits:
            actions[cell, action] = move_fraction * deficit / total_deficit

    return _valid_action_probs(env, actions)


def _valid_action_probs(env: DispatchEnv, actions: np.ndarray) -> np.ndarray:
    probs = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    probs = np.maximum(probs, 0.0)
    probs *= env.available_actions
    row_sums = probs.sum(axis=1, keepdims=True)
    invalid_rows = row_sums[:, 0] <= 0
    if np.any(invalid_rows):
        probs[invalid_rows] = 0.0
        probs[invalid_rows, 0] = 1.0
        row_sums = probs.sum(axis=1, keepdims=True)
    return (probs / row_sums).astype(np.float32)
