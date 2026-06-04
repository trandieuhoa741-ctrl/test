from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .grid import HexGrid


TRAINING_ARCHITECTURES = ("full", "rstr")


class SharedGraphAttentionLayer(torch.nn.Module):
    """Shared multi-head attention over the road-time regional graph."""

    def __init__(self, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        heads = max(int(attention_heads), 1)
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.hidden_dim = int(hidden_dim)
        self.attention_heads = heads
        self.head_dim = hidden_dim // heads
        self.query = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, hidden_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.spatial_bias_alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, cells, hidden = x.shape
        graph = adjacency.to(device=x.device, dtype=x.dtype)
        if graph.ndim == 2:
            graph = graph.unsqueeze(0).expand(batch, -1, -1)
        mask = graph > 0.0

        q = self.query(x).view(batch, cells, self.attention_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch, cells, self.attention_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch, cells, self.attention_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / float(self.head_dim) ** 0.5
        logits = logits + self.spatial_bias_alpha * torch.log(graph.clamp_min(1e-6)).unsqueeze(1)
        logits = logits.masked_fill(~mask.unsqueeze(1), -1.0e10)
        attention = torch.softmax(logits, dim=-1)
        attended = torch.matmul(attention, v).transpose(1, 2).reshape(batch, cells, hidden)
        return self.norm(x + F.relu(self.out(attended)))


class FVBiCoordNetwork(torch.nn.Module):
    """Shared spatio-temporal graph attention network for Bi-STAR."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        action_dim: int,
        attention_heads: int = 4,
        num_cells: int | None = None,
        use_local_global_heads: bool = False,
        use_future_pressure_head: bool = True,
        local_residual_scale: float = 0.1,
        global_bias_scale: float = 0.1,
    ) -> None:
        super().__init__()
        attention_heads = _compatible_attention_heads(hidden_dim, attention_heads)
        if use_local_global_heads and num_cells is None:
            raise ValueError("num_cells is required when use_local_global_heads=True")
        self.num_cells = None if num_cells is None else int(num_cells)
        self.use_local_global_heads = bool(use_local_global_heads)
        self.use_future_pressure_head = bool(use_future_pressure_head)
        self.local_residual_scale = float(local_residual_scale)
        self.global_bias_scale = float(global_bias_scale)
        self.temporal = torch.nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.graph_attn1 = SharedGraphAttentionLayer(hidden_dim, attention_heads)
        self.graph_attn2 = SharedGraphAttentionLayer(hidden_dim, attention_heads)
        self.actor = torch.nn.Linear(hidden_dim, action_dim)
        self.critic = torch.nn.Linear(hidden_dim, 1)
        # Kept for old checkpoints/tests; matching value is critic-derived via _critic_value_logits.
        self.region_value = torch.nn.Linear(hidden_dim, 1)
        self.future_pressure = torch.nn.Linear(hidden_dim, 1) if self.use_future_pressure_head else None
        if self.use_local_global_heads:
            assert self.num_cells is not None
            self.local_actor_weight = torch.nn.Parameter(torch.zeros(self.num_cells, hidden_dim, action_dim))
            self.local_actor_bias = torch.nn.Parameter(torch.zeros(self.num_cells, action_dim))
            self.local_region_value_weight = torch.nn.Parameter(torch.zeros(self.num_cells, hidden_dim, 1))
            self.local_region_value_bias = torch.nn.Parameter(torch.zeros(self.num_cells, 1))
            self.global_actor_bias = torch.nn.Linear(hidden_dim, action_dim)
            self.global_value_bias = torch.nn.Linear(hidden_dim, self.num_cells)
            self._reset_local_global_heads()
        else:
            self.local_actor_weight = None
            self.local_actor_bias = None
            self.local_region_value_weight = None
            self.local_region_value_bias = None
            self.global_actor_bias = None
            self.global_value_bias = None

    def _reset_local_global_heads(self) -> None:
        if not self.use_local_global_heads:
            return
        torch.nn.init.zeros_(self.local_actor_weight)
        torch.nn.init.zeros_(self.local_actor_bias)
        torch.nn.init.zeros_(self.local_region_value_weight)
        torch.nn.init.zeros_(self.local_region_value_bias)
        torch.nn.init.zeros_(self.global_actor_bias.weight)
        torch.nn.init.zeros_(self.global_actor_bias.bias)
        torch.nn.init.zeros_(self.global_value_bias.weight)
        torch.nn.init.zeros_(self.global_value_bias.bias)

    def forward(
        self,
        sequence: torch.Tensor,
        adjacency: torch.Tensor,
        available_actions: torch.Tensor | None = None,
        action_costs: torch.Tensor | None = None,
        road_time_weight: float = 0.0,
        return_heads: bool = False,
        detach_auxiliary_heads: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if sequence.ndim != 4:
            raise ValueError("sequence must have shape (batch, time, cells, features)")
        sequence = torch.nan_to_num(sequence, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        batch, time_steps, cells, features = sequence.shape
        temporal_input = sequence.permute(0, 2, 1, 3).reshape(batch * cells, time_steps, features)
        _, hidden = self.temporal(temporal_input)
        x = hidden[-1].reshape(batch, cells, -1)

        x = self.graph_attn1(x, adjacency)
        x = self.graph_attn2(x, adjacency)

        logits = self._actor_logits(x)
        if action_costs is not None and road_time_weight > 0.0:
            costs = action_costs.to(device=logits.device, dtype=logits.dtype)
            costs = self._align_batched_matrix(costs, logits, "action_costs")
            costs = torch.nan_to_num(costs, nan=0.0, posinf=1.0e6, neginf=0.0)
            logits = logits - float(road_time_weight) * costs
        if available_actions is not None:
            mask = self._align_available_actions(available_actions, logits)
            logits = logits.masked_fill(~mask, -1.0e10)
        action_probs = F.softmax(logits, dim=-1)
        action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
        values = self._critic_value_logits(x).squeeze(-1)
        values = torch.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        if not return_heads:
            return action_probs, values
        auxiliary_x = x.detach() if detach_auxiliary_heads else x
        heads = {
            "critic_value": values,
            "region_value": values,
        }
        if self.future_pressure is not None:
            heads["future_pressure"] = F.softplus(self.future_pressure(auxiliary_x)).squeeze(-1)
        heads = {key: torch.nan_to_num(value, nan=0.0, posinf=1.0e6, neginf=-1.0e6) for key, value in heads.items()}
        return action_probs, values, heads

    def _actor_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.actor(x)
        if not self.use_local_global_heads:
            return logits
        self._validate_local_global_cell_count(x)
        local_logits = torch.einsum("bch,cha->bca", x, self.local_actor_weight) + self.local_actor_bias.unsqueeze(0)
        global_state = x.mean(dim=1)
        global_logits = self.global_actor_bias(global_state).unsqueeze(1)
        return (
            logits
            + float(self.local_residual_scale) * local_logits
            + float(self.global_bias_scale) * global_logits
        )

    def _critic_value_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.critic(x)
        if not self.use_local_global_heads:
            return logits
        self._validate_local_global_cell_count(x)
        local_value = (
            torch.einsum("bch,chv->bcv", x, self.local_region_value_weight)
            + self.local_region_value_bias.unsqueeze(0)
        )
        global_state = x.mean(dim=1)
        global_value = self.global_value_bias(global_state).unsqueeze(-1)
        return (
            logits
            + float(self.local_residual_scale) * local_value
            + float(self.global_bias_scale) * global_value
        )

    def _region_value_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self._critic_value_logits(x)

    def _validate_local_global_cell_count(self, x: torch.Tensor) -> None:
        if self.num_cells is not None and x.shape[1] != self.num_cells:
            raise ValueError(f"sequence cells ({x.shape[1]}) must match num_cells ({self.num_cells})")

    def local_head_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        if not self.use_local_global_heads:
            return ()
        return (
            self.local_actor_weight,
            self.local_actor_bias,
            self.local_region_value_weight,
            self.local_region_value_bias,
        )

    def local_actor_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        if not self.use_local_global_heads:
            return ()
        return (self.local_actor_weight, self.local_actor_bias)

    def local_value_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        if not self.use_local_global_heads:
            return ()
        return (self.local_region_value_weight, self.local_region_value_bias)

    def global_head_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        if not self.use_local_global_heads:
            return ()
        return (
            *self.global_actor_bias.parameters(),
            *self.global_value_bias.parameters(),
        )

    def global_actor_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        if not self.use_local_global_heads:
            return ()
        return tuple(self.global_actor_bias.parameters())

    def global_value_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        if not self.use_local_global_heads:
            return ()
        return tuple(self.global_value_bias.parameters())

    @staticmethod
    def _align_batched_matrix(matrix: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
        expected_tail = reference.shape[1:]
        if matrix.ndim == 2:
            if tuple(matrix.shape) != tuple(expected_tail):
                raise ValueError(f"{name} must have shape {tuple(expected_tail)} or {tuple(reference.shape)}")
            return matrix.unsqueeze(0).expand(reference.shape[0], -1, -1)
        if matrix.ndim == 3:
            if tuple(matrix.shape[1:]) != tuple(expected_tail):
                raise ValueError(f"{name} must have trailing shape {tuple(expected_tail)}")
            if matrix.shape[0] == reference.shape[0]:
                return matrix
            if matrix.shape[0] == 1:
                return matrix.expand(reference.shape[0], -1, -1)
        raise ValueError(f"{name} must have shape {tuple(expected_tail)} or {tuple(reference.shape)}")

    @staticmethod
    def _align_available_actions(available_actions: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        mask = available_actions.to(device=logits.device, dtype=torch.bool)
        mask = FVBiCoordNetwork._align_batched_matrix(mask, logits, "available_actions")
        has_valid_action = mask.any(dim=-1, keepdim=True)
        stay_fallback = torch.zeros_like(mask)
        stay_fallback[..., 0] = True
        return torch.where(has_valid_action, mask, stay_fallback)


class FVBiCoordAgent:
    """Region-value-guided bi-level coordination actor-critic.

    The shared spatio-temporal graph backbone feeds region-heterogeneous
    residual heads plus global coordination biases. The lower environment uses
    destination-oriented region value for value-guided matching.
    """

    def __init__(
        self,
        agent_n: int,
        feature_dim: int,
        hidden_dim: int,
        action_dim: int,
        adjacency: np.ndarray,
        action_costs: np.ndarray,
        action_destinations: np.ndarray | None = None,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        tau: float = 0.005,
        gamma: float = 0.9,
        road_time_weight: float = 0.2,
        temporal_window: int = 3,
        attention_heads: int = 4,
        future_value_loss_weight: float | None = None,
        future_gap_loss_weight: float = 0.0,
        future_demand_loss_weight: float = 0.0,
        intensity_loss_weight: float = 0.0,
        region_value_loss_weight: float | None = None,
        actor_future_demand_weight: float = 0.0,
        actor_region_value_weight: float = 0.0,
        future_pressure_loss_weight: float | None = None,
        actor_future_pressure_weight: float | None = None,
        entropy_coef: float = 0.01,
        park_mask_surplus_threshold: float | None = None,
        park_mask_neighbor_need_threshold: float = 0.0,
        future_gap_feature_index: int = 5,
        supply_demand_gap_feature_index: int = 4,
        idle_supply_feature_index: int = 1,
        current_need_feature_index: int = 5,
        incoming_supply_feature_index: int | None = None,
        use_local_global_heads: bool = True,
        use_future_pressure_head: bool = True,
        use_supply_sufficiency_gate: bool = False,
        source_surplus_threshold: float = 0.0,
        target_shortage_threshold: float = 0.0,
        global_shortage_threshold: float = 0.0,
        local_residual_scale: float = 0.1,
        global_bias_scale: float = 0.1,
        training_architecture: str = "full",
        device: str | torch.device | None = None,
    ) -> None:
        self.agent_n = int(agent_n)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.road_time_weight = float(road_time_weight)
        self.temporal_window = max(int(temporal_window), 1)
        self.attention_heads = _compatible_attention_heads(self.hidden_dim, attention_heads)
        if region_value_loss_weight is None:
            region_value_loss_weight = 0.0 if future_value_loss_weight is None else float(future_value_loss_weight)
        self.region_value_loss_weight = float(region_value_loss_weight)
        self.future_value_loss_weight = self.region_value_loss_weight
        if future_pressure_loss_weight is None:
            future_pressure_loss_weight = max(float(future_gap_loss_weight), float(future_demand_loss_weight))
        self.future_pressure_loss_weight = float(future_pressure_loss_weight)
        self.future_gap_loss_weight = float(future_gap_loss_weight)
        self.future_demand_loss_weight = float(future_demand_loss_weight)
        self.intensity_loss_weight = float(intensity_loss_weight)
        if actor_future_pressure_weight is None:
            actor_future_pressure_weight = actor_future_demand_weight
        self.actor_future_pressure_weight = max(float(actor_future_pressure_weight), 0.0)
        self.actor_future_demand_weight = self.actor_future_pressure_weight
        self.use_future_pressure_head = bool(use_future_pressure_head)
        if not self.use_future_pressure_head and (
            self.future_pressure_loss_weight > 0.0 or self.actor_future_pressure_weight > 0.0
        ):
            raise ValueError("future pressure weights require use_future_pressure_head=True")
        self.actor_region_value_weight = max(float(actor_region_value_weight), 0.0)
        self.entropy_coef = max(float(entropy_coef), 0.0)
        self.park_mask_surplus_threshold = (
            None if park_mask_surplus_threshold is None else float(park_mask_surplus_threshold)
        )
        self.park_mask_neighbor_need_threshold = max(float(park_mask_neighbor_need_threshold), 0.0)
        self.future_gap_feature_index = int(future_gap_feature_index)
        self.supply_demand_gap_feature_index = int(supply_demand_gap_feature_index)
        self.idle_supply_feature_index = int(idle_supply_feature_index)
        self.current_need_feature_index = int(current_need_feature_index)
        self.incoming_supply_feature_index = (
            None if incoming_supply_feature_index is None else int(incoming_supply_feature_index)
        )
        self.use_local_global_heads = bool(use_local_global_heads)
        self.local_residual_scale = float(local_residual_scale)
        self.global_bias_scale = float(global_bias_scale)
        self.use_supply_sufficiency_gate = bool(use_supply_sufficiency_gate)
        self.source_surplus_threshold = float(source_surplus_threshold)
        self.target_shortage_threshold = float(target_shortage_threshold)
        self.global_shortage_threshold = float(global_shortage_threshold)
        self.training_architecture = _normalize_training_architecture(training_architecture)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_policy_entropy = 0.0
        self.last_entropy_loss = 0.0
        self.last_auxiliary_loss = 0.0
        self.last_total_loss = 0.0
        self.last_total_grad_norm = 0.0
        self.last_encoder_grad_norm = 0.0
        self.last_actor_head_grad_norm = 0.0
        self.last_critic_head_grad_norm = 0.0
        self.last_auxiliary_head_grad_norm = 0.0
        self.last_local_head_grad_norm = 0.0
        self.last_global_head_grad_norm = 0.0

        self.net = FVBiCoordNetwork(
            feature_dim,
            hidden_dim,
            action_dim,
            attention_heads=self.attention_heads,
            num_cells=self.agent_n,
            use_local_global_heads=self.use_local_global_heads,
            use_future_pressure_head=self.use_future_pressure_head,
            local_residual_scale=self.local_residual_scale,
            global_bias_scale=self.global_bias_scale,
        ).to(self.device)
        self.target_net = FVBiCoordNetwork(
            feature_dim,
            hidden_dim,
            action_dim,
            attention_heads=self.attention_heads,
            num_cells=self.agent_n,
            use_local_global_heads=self.use_local_global_heads,
            use_future_pressure_head=self.use_future_pressure_head,
            local_residual_scale=self.local_residual_scale,
            global_bias_scale=self.global_bias_scale,
        ).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())

        self.adjacency = torch.as_tensor(adjacency, dtype=torch.float32, device=self.device)
        self.action_costs = torch.as_tensor(_normalize_action_costs(action_costs), dtype=torch.float32, device=self.device)
        action_destinations_np = _normalize_action_destinations(action_destinations, self.agent_n, self.action_dim)
        self.action_destinations_np = action_destinations_np
        self.action_destinations = torch.as_tensor(
            action_destinations_np,
            dtype=torch.long,
            device=self.device,
        )
        optimizer_groups: list[dict[str, object]] = [
            {"params": self.net.temporal.parameters(), "lr": actor_lr},
            {"params": self.net.graph_attn1.parameters(), "lr": actor_lr},
            {"params": self.net.graph_attn2.parameters(), "lr": actor_lr},
            {"params": self.net.actor.parameters(), "lr": actor_lr},
            {"params": self.net.critic.parameters(), "lr": critic_lr},
        ]
        local_actor_params = list(self.net.local_actor_parameters())
        if local_actor_params:
            optimizer_groups.append({"params": local_actor_params, "lr": actor_lr})
        global_actor_params = list(self.net.global_actor_parameters())
        if global_actor_params:
            optimizer_groups.append({"params": global_actor_params, "lr": actor_lr})
        local_value_params = list(self.net.local_value_parameters())
        if local_value_params:
            optimizer_groups.append({"params": local_value_params, "lr": critic_lr})
        global_value_params = list(self.net.global_value_parameters())
        if global_value_params:
            optimizer_groups.append({"params": global_value_params, "lr": critic_lr})
        if self.training_architecture != "rstr" and self.net.future_pressure is not None:
            optimizer_groups.append({"params": self.net.future_pressure.parameters(), "lr": critic_lr})
        self.optimizer = torch.optim.Adam(optimizer_groups)

    @torch.no_grad()
    def take_decision(
        self,
        sequence: np.ndarray | list[list[list[float]]],
        available_actions: np.ndarray | list[list[float]] | None = None,
    ) -> dict[str, np.ndarray]:
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.shape != (self.temporal_window, self.agent_n, self.feature_dim):
            raise ValueError(f"sequence must have shape {(self.temporal_window, self.agent_n, self.feature_dim)}")
        masks = self.available_actions_for_sequence(seq, available_actions)
        seq_tensor = torch.as_tensor(seq[None, ...], dtype=torch.float32, device=self.device)
        mask_tensor = None if masks is None else torch.as_tensor(masks[None, ...], dtype=torch.float32, device=self.device)
        if self.training_architecture == "rstr":
            probs, values = self.net(
                seq_tensor,
                self.adjacency,
                available_actions=mask_tensor,
                action_costs=self.action_costs,
                road_time_weight=self.road_time_weight,
            )
            return {
                "actions": probs[0].cpu().numpy().astype(np.float32),
                "region_value": values[0].cpu().numpy().astype(np.float32),
            }
        probs, _values, heads = self.net(
            seq_tensor,
            self.adjacency,
            available_actions=mask_tensor,
            action_costs=self.action_costs,
            road_time_weight=self.road_time_weight,
            return_heads=True,
            detach_auxiliary_heads=True,
        )
        decision = {
            "actions": probs[0].cpu().numpy().astype(np.float32),
            "region_value": heads["region_value"][0].cpu().numpy().astype(np.float32),
        }
        if "future_pressure" in heads:
            future_pressure = heads["future_pressure"][0].cpu().numpy().astype(np.float32)
            dispatch_intensity = np.clip(future_pressure, 0.0, 1.0).astype(np.float32)
            decision.update(
                {
                    "future_pressure": future_pressure,
                    "future_gap": future_pressure,
                    "future_demand": future_pressure,
                    "dispatch_intensity": dispatch_intensity,
                }
            )
        return decision

    def available_actions_for_sequence(
        self,
        sequence: np.ndarray | list[list[list[float]]],
        available_actions: np.ndarray | list[list[float]] | None,
    ) -> np.ndarray | None:
        if available_actions is None:
            return None
        seq = np.asarray(sequence, dtype=np.float32)
        masks = np.asarray(available_actions, dtype=np.float32).copy()
        if masks.shape != (self.agent_n, self.action_dim):
            raise ValueError(f"available_actions must have shape {(self.agent_n, self.action_dim)}")
        if seq.shape != (self.temporal_window, self.agent_n, self.feature_dim):
            raise ValueError(f"sequence must have shape {(self.temporal_window, self.agent_n, self.feature_dim)}")
        if self.use_supply_sufficiency_gate:
            masks = self._mask_supply_sufficiency_actions(seq, masks)
        if self.park_mask_surplus_threshold is None:
            return masks
        return self._mask_surplus_park_actions(seq, masks)

    def _mask_supply_sufficiency_actions(self, sequence: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if sequence.shape != (self.temporal_window, self.agent_n, self.feature_dim):
            raise ValueError(f"sequence must have shape {(self.temporal_window, self.agent_n, self.feature_dim)}")
        if not (0 <= self.idle_supply_feature_index < self.feature_dim):
            return masks
        if not (0 <= self.current_need_feature_index < self.feature_dim):
            return masks

        gated = np.asarray(masks, dtype=np.float32).copy()
        latest = np.asarray(sequence[-1], dtype=np.float32)
        idle = np.nan_to_num(
            latest[:, self.idle_supply_feature_index],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if self.incoming_supply_feature_index is None or not (
            0 <= self.incoming_supply_feature_index < self.feature_dim
        ):
            incoming = np.zeros_like(idle)
        else:
            incoming = np.nan_to_num(
                latest[:, self.incoming_supply_feature_index],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        need = np.nan_to_num(
            latest[:, self.current_need_feature_index],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        supply = np.maximum(idle + incoming, 0.0)
        need = np.maximum(need, 0.0)
        shortage = np.maximum(need - supply, 0.0)
        surplus = np.maximum(supply - need, 0.0)

        if float(np.sum(shortage)) <= self.global_shortage_threshold:
            gated[:, :] = 0.0
            gated[:, 0] = 1.0
            return gated

        for src in range(self.agent_n):
            gated[src, 0] = 1.0
            if surplus[src] <= self.source_surplus_threshold:
                gated[src, 1:] = 0.0
                continue

            for action in range(1, self.action_dim):
                if gated[src, action] <= 0.0:
                    continue
                dst = int(self.action_destinations_np[src, action])
                if dst < 0 or dst >= self.agent_n:
                    gated[src, action] = 0.0
                    continue
                if shortage[dst] <= self.target_shortage_threshold:
                    gated[src, action] = 0.0

            if not np.any(gated[src] > 0.0):
                gated[src, 0] = 1.0

        return gated

    def update(self, transitions: list[dict[str, Any]]) -> tuple[float, float]:
        if not transitions:
            return 0.0, 0.0

        sequences = torch.as_tensor(
            np.asarray([item["sequence"] for item in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        next_sequences = torch.as_tensor(
            np.asarray([item["next_sequence"] for item in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        critic_rewards = torch.as_tensor(
            np.asarray([item["critic_rewards"] for item in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        actor_rewards = torch.as_tensor(
            np.asarray([item["actor_rewards"] for item in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        available_actions = torch.as_tensor(
            np.asarray([item["available_actions"] for item in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        next_available_actions = torch.as_tensor(
            np.asarray(
                [item.get("next_available_actions", item["available_actions"]) for item in transitions],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            np.asarray([bool(item.get("done", False)) for item in transitions], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        actor_cell_weights = self._transition_actor_cell_weights(transitions)
        auxiliary_targets = None
        auxiliary_sequences = sequences

        if self.training_architecture == "rstr":
            probs, values = self.net(
                sequences,
                self.adjacency,
                available_actions=available_actions,
                action_costs=self.action_costs,
                road_time_weight=self.road_time_weight,
            )
            heads: dict[str, torch.Tensor] = {}
        else:
            auxiliary_targets = self._transition_auxiliary_targets(transitions)
            auxiliary_sequences = self._transition_auxiliary_sequences(transitions, sequences)
            probs, values, heads = self.net(
                sequences,
                self.adjacency,
                available_actions=available_actions,
                action_costs=self.action_costs,
                road_time_weight=self.road_time_weight,
                return_heads=True,
                detach_auxiliary_heads=True,
            )
            actor_rewards = self._shape_actor_rewards(actor_rewards, values, heads)
        with torch.no_grad():
            _next_probs, next_values = self.target_net(
                next_sequences,
                self.adjacency,
                available_actions=next_available_actions,
                action_costs=self.action_costs,
                road_time_weight=self.road_time_weight,
            )
            nonterminal = (1.0 - dones).view(-1, 1)
            critic_targets = critic_rewards + self.gamma * nonterminal * next_values
            next_action_values = self._next_values_for_actions(next_values)
            actor_advantages = (
                actor_rewards
                + self.gamma * nonterminal.unsqueeze(-1) * next_action_values
                - values.unsqueeze(-1)
            )
            actor_advantages = self._normalize_actor_advantages(actor_advantages, available_actions)

        critic_loss = F.mse_loss(values, critic_targets)
        log_probs = torch.log(probs.clamp_min(1e-8))
        actor_loss_terms = -probs.detach() * log_probs * actor_advantages.detach()
        actor_loss = self._actor_loss_from_terms(actor_loss_terms, available_actions, actor_cell_weights)
        policy_entropy = self._policy_entropy(probs, log_probs, available_actions, actor_cell_weights)
        entropy_loss = -float(self.entropy_coef) * policy_entropy
        auxiliary_loss = torch.zeros((), dtype=values.dtype, device=values.device)
        if self.training_architecture != "rstr":
            auxiliary_heads = heads
            if auxiliary_sequences is not sequences:
                _aux_probs, _aux_values, auxiliary_heads = self.net(
                    auxiliary_sequences,
                    self.adjacency,
                    return_heads=True,
                    detach_auxiliary_heads=True,
                )
            auxiliary_loss = self._auxiliary_head_loss(
                auxiliary_heads,
                next_sequences=next_sequences,
                targets=auxiliary_targets,
            )
        loss = actor_loss + critic_loss + auxiliary_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.last_auxiliary_loss = float(auxiliary_loss.detach().cpu())
        self.last_total_loss = float(loss.detach().cpu())
        self.last_encoder_grad_norm = self._grad_norm(
            (
                *self.net.temporal.parameters(),
                *self.net.graph_attn1.parameters(),
                *self.net.graph_attn2.parameters(),
            )
        )
        self.last_actor_head_grad_norm = self._grad_norm(self.net.actor.parameters())
        self.last_critic_head_grad_norm = self._grad_norm(self.net.critic.parameters())
        if self.training_architecture == "rstr":
            self.last_auxiliary_head_grad_norm = 0.0
        else:
            auxiliary_params = [] if self.net.future_pressure is None else list(self.net.future_pressure.parameters())
            self.last_auxiliary_head_grad_norm = self._grad_norm(auxiliary_params)
        self.last_local_head_grad_norm = self._grad_norm(self.net.local_head_parameters())
        self.last_global_head_grad_norm = self._grad_norm(self.net.global_head_parameters())
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.last_total_grad_norm = float(total_grad_norm.detach().cpu())
        self.optimizer.step()
        self._soft_update()
        self.last_policy_entropy = float(policy_entropy.detach().cpu())
        self.last_entropy_loss = float(entropy_loss.detach().cpu())
        return float(actor_loss.detach().cpu()), float(critic_loss.detach().cpu())

    def _mask_surplus_park_actions(self, sequence: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if not (0 <= self.idle_supply_feature_index < self.feature_dim):
            return masks
        if not (0 <= self.supply_demand_gap_feature_index < self.feature_dim):
            return masks
        current = np.asarray(sequence[-1], dtype=np.float32)
        idle_supply = np.nan_to_num(current[:, self.idle_supply_feature_index], nan=0.0, posinf=1.0e6, neginf=0.0)
        surplus_gap = np.nan_to_num(
            current[:, self.supply_demand_gap_feature_index],
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
        )
        if 0 <= self.current_need_feature_index < self.feature_dim:
            current_need = np.nan_to_num(
                current[:, self.current_need_feature_index],
                nan=0.0,
                posinf=1.0e6,
                neginf=0.0,
            )
            current_need = np.maximum(current_need, 0.0)
        else:
            current_need = np.maximum(-surplus_gap, 0.0)

        neighbor_need = np.zeros(self.agent_n, dtype=np.float32)
        valid_moves = masks[:, 1:] > 0.0
        for cell in range(self.agent_n):
            best = 0.0
            for action in range(1, self.action_dim):
                if not valid_moves[cell, action - 1]:
                    continue
                destination = int(self.action_destinations_np[cell, action])
                if 0 <= destination < self.agent_n:
                    best = max(best, float(current_need[destination]))
            neighbor_need[cell] = best

        has_nonstay_action = valid_moves.any(axis=1)
        disable_stay = (
            (masks[:, 0] > 0.0)
            & has_nonstay_action
            & (idle_supply > 0.0)
            & (surplus_gap >= float(self.park_mask_surplus_threshold))
            & (neighbor_need >= float(self.park_mask_neighbor_need_threshold))
            & (neighbor_need > current_need + 1.0e-6)
        )
        masks[disable_stay, 0] = 0.0
        return masks

    def _auxiliary_head_loss(
        self,
        heads: dict[str, torch.Tensor],
        next_sequences: torch.Tensor | None = None,
        targets: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        region_value = heads["region_value"]
        pressure = heads.get("future_pressure")
        loss = torch.zeros((), dtype=region_value.dtype, device=region_value.device)
        pressure_target = None if targets is None else targets.get("future_pressure")
        pressure_mask = None if targets is None else targets.get("future_pressure_mask")
        if pressure is not None and pressure_target is None:
            pressure_target = self._fallback_future_pressure_target(next_sequences, pressure)

        if self.future_pressure_loss_weight > 0.0 and pressure is not None and pressure_target is not None:
            pressure_target = pressure_target.to(device=pressure.device, dtype=pressure.dtype)
            loss = loss + float(self.future_pressure_loss_weight) * self._masked_smooth_l1_loss(
                pressure,
                pressure_target,
                pressure_mask,
            )
        return loss

    def _shape_actor_rewards(
        self,
        actor_rewards: torch.Tensor,
        values: torch.Tensor,
        heads: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        shaped = actor_rewards
        if self.actor_future_pressure_weight > 0.0 and "future_pressure" in heads:
            future_pressure = heads["future_pressure"].detach().clamp_min(0.0)
            shaped = shaped + float(self.actor_future_pressure_weight) * self._cell_values_for_actions(future_pressure)
        if self.actor_region_value_weight > 0.0 and "region_value" in heads:
            region_values = self._standardize_cell_values(heads["region_value"].detach())
            shaped = shaped + float(self.actor_region_value_weight) * self._cell_values_for_actions(region_values)
        return shaped

    @staticmethod
    def _normalize_actor_advantages(
        actor_advantages: torch.Tensor,
        available_actions: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = available_actions > 0.0
        valid_advantages = actor_advantages[valid_mask]
        if valid_advantages.numel() > 1:
            adv_mean = valid_advantages.mean()
            adv_std = valid_advantages.std().clamp_min(1.0e-6)
            actor_advantages = (actor_advantages - adv_mean) / adv_std
        return torch.clamp(actor_advantages, -3.0, 3.0)

    def _fallback_future_pressure_target(
        self,
        next_sequences: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if next_sequences is None or next_sequences.ndim != 4:
            return None
        if not (0 <= self.supply_demand_gap_feature_index < next_sequences.shape[-1]):
            return None
        supply_demand_gap = next_sequences[:, -1, :, self.supply_demand_gap_feature_index]
        return (-supply_demand_gap).clamp_min(0.0).to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _masked_smooth_l1_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = F.smooth_l1_loss(prediction, target, reduction="none")
        return FVBiCoordAgent._masked_mean(values, mask)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return values.mean()
        weights = mask.to(device=values.device, dtype=values.dtype)
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    @staticmethod
    def _grad_norm(parameters: Any) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            value = float(parameter.grad.detach().norm(2).cpu())
            total += value * value
        return float(total ** 0.5)

    @staticmethod
    def _actor_loss_from_terms(
        actor_loss_terms: torch.Tensor,
        available_actions: torch.Tensor,
        cell_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_f = available_actions.float()
        actor_loss_per_cell = (actor_loss_terms * valid_f).sum(dim=-1)
        cell_mask = (valid_f.sum(dim=-1) > 0).float()
        weights = FVBiCoordAgent._normalized_cell_weights(cell_mask, cell_weights)
        return (actor_loss_per_cell * weights).sum() / weights.sum().clamp_min(1.0e-6)

    @staticmethod
    def _policy_entropy(
        probs: torch.Tensor,
        log_probs: torch.Tensor,
        available_actions: torch.Tensor,
        cell_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        entropy_per_cell = -(probs * log_probs * available_actions.float()).sum(dim=-1)
        cell_mask = (available_actions.float().sum(dim=-1) > 1.0).float()
        weights = FVBiCoordAgent._normalized_cell_weights(cell_mask, cell_weights)
        return (entropy_per_cell * weights).sum() / weights.sum().clamp_min(1.0e-6)

    @staticmethod
    def _normalized_cell_weights(
        cell_mask: torch.Tensor,
        cell_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cell_weights is None:
            return cell_mask
        weights = cell_weights.to(device=cell_mask.device, dtype=cell_mask.dtype)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0e6, neginf=0.0).clamp_min(0.0)
        weights = weights * cell_mask
        if weights.sum().detach().cpu().item() <= 0.0:
            return cell_mask
        return weights

    def _next_values_for_actions(self, next_values: torch.Tensor) -> torch.Tensor:
        return self._cell_values_for_actions(next_values)

    def _cell_values_for_actions(self, cell_values: torch.Tensor) -> torch.Tensor:
        destinations = self.action_destinations.to(device=cell_values.device)
        flat_destinations = destinations.reshape(1, -1).expand(cell_values.shape[0], -1)
        gathered = cell_values.gather(1, flat_destinations)
        return gathered.reshape(cell_values.shape[0], self.agent_n, self.action_dim)

    @staticmethod
    def _standardize_cell_values(values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, keepdim=True).clamp_min(1.0e-6)
        return ((values - mean) / std).clamp(-3.0, 3.0)

    def _transition_auxiliary_targets(self, transitions: list[dict[str, Any]]) -> dict[str, torch.Tensor] | None:
        target_rows = [item.get("auxiliary_targets") for item in transitions]
        if not target_rows or all(row is None for row in target_rows):
            return None
        targets: dict[str, torch.Tensor] = {}

        region_target = self._stack_optional_target_rows(target_rows, ("region_value",))
        if region_target is not None:
            targets["region_value"], targets["region_value_mask"] = region_target

        if self.use_future_pressure_head:
            pressure_target = self._stack_optional_target_rows(
                target_rows,
                ("future_pressure", "future_gap", "future_demand", "dispatch_intensity"),
                combine="max",
            )
            if pressure_target is not None:
                targets["future_pressure"], targets["future_pressure_mask"] = pressure_target
        return targets or None

    def _stack_optional_target_rows(
        self,
        target_rows: list[object],
        keys: tuple[str, ...],
        *,
        combine: str = "first",
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        template = None
        for row in target_rows:
            if not isinstance(row, dict):
                continue
            for key in keys:
                if key in row:
                    template = np.asarray(row[key], dtype=np.float32)
                    break
            if template is not None:
                break
        if template is None:
            return None

        values = []
        masks = []
        for row in target_rows:
            row_values = []
            if isinstance(row, dict):
                for key in keys:
                    if key in row:
                        row_values.append(np.asarray(row[key], dtype=np.float32))
                        if combine == "first":
                            break
            if row_values:
                if combine == "max":
                    values.append(np.maximum.reduce(row_values).astype(np.float32))
                else:
                    values.append(row_values[0].astype(np.float32))
                masks.append(np.ones_like(template, dtype=np.float32))
            else:
                values.append(np.zeros_like(template, dtype=np.float32))
                masks.append(np.zeros_like(template, dtype=np.float32))
        return (
            torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.asarray(masks, dtype=np.float32), dtype=torch.float32, device=self.device),
        )

    def _transition_auxiliary_sequences(
        self,
        transitions: list[dict[str, Any]],
        default_sequences: torch.Tensor,
    ) -> torch.Tensor:
        if not any("auxiliary_sequence" in item for item in transitions):
            return default_sequences
        sequences = np.asarray(
            [item.get("auxiliary_sequence", item["sequence"]) for item in transitions],
            dtype=np.float32,
        )
        return torch.as_tensor(sequences, dtype=torch.float32, device=self.device)

    def _transition_actor_cell_weights(self, transitions: list[dict[str, Any]]) -> torch.Tensor | None:
        if not any("actor_cell_weights" in item for item in transitions):
            return None
        values = []
        for item in transitions:
            row = item.get("actor_cell_weights")
            if row is None:
                values.append(np.ones(self.agent_n, dtype=np.float32))
            else:
                weights = np.asarray(row, dtype=np.float32)
                if weights.shape != (self.agent_n,):
                    raise ValueError(f"actor_cell_weights must have shape {(self.agent_n,)}")
                values.append(weights)
        return torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32, device=self.device)

    def save(self, directory: str | Path, metadata: dict[str, Any] | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "agent_n": self.agent_n,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "tau": self.tau,
            "gamma": self.gamma,
            "road_time_weight": self.road_time_weight,
            "temporal_window": self.temporal_window,
            "attention_heads": self.attention_heads,
            "region_value_loss_weight": self.region_value_loss_weight,
            "future_value_loss_weight": self.future_value_loss_weight,
            "future_pressure_loss_weight": self.future_pressure_loss_weight,
            "future_gap_loss_weight": self.future_gap_loss_weight,
            "future_demand_loss_weight": self.future_demand_loss_weight,
            "intensity_loss_weight": self.intensity_loss_weight,
            "actor_future_pressure_weight": self.actor_future_pressure_weight,
            "actor_future_demand_weight": self.actor_future_demand_weight,
            "actor_region_value_weight": self.actor_region_value_weight,
            "entropy_coef": self.entropy_coef,
            "park_mask_surplus_threshold": self.park_mask_surplus_threshold,
            "park_mask_neighbor_need_threshold": self.park_mask_neighbor_need_threshold,
            "use_local_global_heads": self.use_local_global_heads,
            "use_future_pressure_head": self.use_future_pressure_head,
            "use_supply_sufficiency_gate": self.use_supply_sufficiency_gate,
            "source_surplus_threshold": self.source_surplus_threshold,
            "target_shortage_threshold": self.target_shortage_threshold,
            "global_shortage_threshold": self.global_shortage_threshold,
            "incoming_supply_feature_index": self.incoming_supply_feature_index,
            "local_residual_scale": self.local_residual_scale,
            "global_bias_scale": self.global_bias_scale,
            "training_architecture": self.training_architecture,
        }
        if metadata is not None:
            meta["checkpoint_metadata"] = dict(metadata)
        torch.save(meta, directory / "meta.pt")
        torch.save(self.net.state_dict(), directory / "fv_bicoord_actor_critic.pt")
        torch.save(self.target_net.state_dict(), directory / "fv_bicoord_target.pt")
        torch.save(self.optimizer.state_dict(), directory / "optimizer.pt")

    def _soft_update(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
                target_param.mul_(1.0 - self.tau)
                target_param.add_(param, alpha=self.tau)


def build_road_time_adjacency(
    grid: HexGrid,
    road_time_matrix: np.ndarray | None = None,
    hex_distance_matrix: np.ndarray | None = None,
    step_minutes: int = 10,
    temperature: float = 10.0,
    symmetric: bool = False,
) -> np.ndarray:
    weights = np.eye(grid.num_cells, dtype=np.float32)
    temp = max(float(temperature), 1e-3)
    for origin in range(grid.num_cells):
        for destination in grid.neighbors[origin, 1:]:
            if destination < 0:
                continue
            dst = int(destination)
            minutes = _edge_minutes(
                grid=grid,
                origin=origin,
                destination=dst,
                road_time_matrix=road_time_matrix,
                hex_distance_matrix=hex_distance_matrix,
                step_minutes=step_minutes,
            )
            weight = float(np.exp(-minutes / temp))
            weights[origin, dst] = max(weights[origin, dst], weight)
            if symmetric:
                weights[dst, origin] = max(weights[dst, origin], weight)
    row_sum = np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    return (weights / row_sum).astype(np.float32)


def _edge_minutes(
    grid: HexGrid,
    origin: int,
    destination: int,
    road_time_matrix: np.ndarray | None,
    hex_distance_matrix: np.ndarray | None,
    step_minutes: int,
) -> float:
    if road_time_matrix is not None:
        minutes = float(np.asarray(road_time_matrix, dtype=np.float32)[origin, destination])
        if np.isfinite(minutes) and minutes >= 0.0:
            return max(minutes, 0.0)
    if hex_distance_matrix is not None:
        distance = float(np.asarray(hex_distance_matrix, dtype=np.float32)[origin, destination])
        if np.isfinite(distance) and distance > 0.0:
            return distance / max(float(grid.cell_width_km), 0.1) * float(step_minutes)
    return grid.distance(origin, destination) / max(float(grid.cell_width_km), 0.1) * float(step_minutes)


def _normalize_action_costs(action_costs: np.ndarray) -> np.ndarray:
    costs = np.asarray(action_costs, dtype=np.float32).copy()
    finite = costs[np.isfinite(costs) & (costs > 0.0)]
    scale = float(np.percentile(finite, 75)) if finite.size else 1.0
    costs[~np.isfinite(costs)] = scale
    costs = np.maximum(costs / max(scale, 1e-6), 0.0)
    costs[:, 0] = 0.0
    return costs.astype(np.float32)


def _normalize_action_destinations(
    action_destinations: np.ndarray | None,
    agent_n: int,
    action_dim: int,
) -> np.ndarray:
    if action_destinations is None:
        return np.repeat(np.arange(agent_n, dtype=np.int64)[:, None], action_dim, axis=1)
    destinations = np.asarray(action_destinations, dtype=np.int64)
    if destinations.shape != (agent_n, action_dim):
        raise ValueError(f"action_destinations must have shape {(agent_n, action_dim)}")
    fallback = np.repeat(np.arange(agent_n, dtype=np.int64)[:, None], action_dim, axis=1)
    return np.where(destinations >= 0, destinations, fallback).astype(np.int64)


def _compatible_attention_heads(hidden_dim: int, requested_heads: int) -> int:
    heads = min(max(int(requested_heads), 1), max(int(hidden_dim), 1))
    while heads > 1 and int(hidden_dim) % heads != 0:
        heads -= 1
    return heads


def _normalize_training_architecture(value: str) -> str:
    architecture = str(value).strip().lower().replace("-", "_")
    aliases = {
        "full": "full",
        "bicoord": "full",
        "fv_bicoord": "full",
        "rstr": "rstr",
        "two_layer": "rstr",
        "two_layer_rstr": "rstr",
    }
    if architecture not in aliases:
        allowed = ", ".join(TRAINING_ARCHITECTURES)
        raise ValueError(f"training_architecture must be one of: {allowed}")
    return aliases[architecture]
