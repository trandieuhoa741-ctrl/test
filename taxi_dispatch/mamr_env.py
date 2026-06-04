from __future__ import annotations

from .env import EnvConfig, DispatchEnv


class MAMRDispatchEnv(DispatchEnv):
    """Regional dispatch simulator configured from MAMR-compatible artifacts.

    This class is not the original MAMR simulator. It keeps this repository's
    matching, repositioning, and metric logic while using MAMR-style city-state,
    hex-grid, distance, raw-trip, and driver-distribution files.
    """

    data_source = "mamr"

    @classmethod
    def from_bundle(cls, bundle: object) -> "MAMRDispatchEnv":
        return cls(
            bundle.config,
            demand=bundle.demand,
            grid=bundle.grid,
            initial_taxi_distribution=bundle.initial_taxi_distribution,
            hex_distance_matrix=bundle.hex_distance_matrix,
        )


MAMREnv = MAMRDispatchEnv


__all__ = ["EnvConfig", "MAMRDispatchEnv", "MAMREnv"]
