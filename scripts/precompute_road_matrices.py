from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts._common import (
        assert_same_grid,
        load_pickle,
        road_matrix_stats,
        save_pickle,
        write_matrix_long_csv,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from _common import (
        assert_same_grid,
        load_pickle,
        road_matrix_stats,
        save_pickle,
        write_matrix_long_csv,
    )

from taxi_dispatch.road_network import RoadNetwork


def load_road_network(args: argparse.Namespace) -> RoadNetwork:
    if args.road_nodes or args.road_edges:
        if not args.road_nodes or not args.road_edges:
            raise ValueError("--road-nodes and --road-edges must be provided together")
        return RoadNetwork.from_csv(args.road_nodes, args.road_edges, default_speed_kmph=args.road_speed_kmph)
    if args.road_network:
        return RoadNetwork.from_graphml(args.road_network, default_speed_kmph=args.road_speed_kmph)
    return RoadNetwork.load_or_download(
        cache_path=args.road_network_cache,
        place=args.road_place,
        network_type=args.road_network_type,
        default_speed_kmph=args.road_speed_kmph,
        download=args.download_road_network,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute selected-cell road distance/time matrices for leak-free Chengdu experiments."
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output .npz path.")
    parser.add_argument("--stats-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--inject-caches", action="store_true", help="Store matrices inside both cache pickle files.")
    parser.add_argument("--road-network", default="", help="Existing GraphML road graph.")
    parser.add_argument("--road-nodes", default="", help="Road node CSV with node,lon,lat columns.")
    parser.add_argument("--road-edges", default="", help="Road edge CSV with u,v,length_m columns.")
    parser.add_argument("--download-road-network", action="store_true")
    parser.add_argument("--road-network-cache", default="data/processed/chengdu_road.graphml")
    parser.add_argument("--road-place", default="Chengdu, Sichuan, China")
    parser.add_argument("--road-network-type", default="drive")
    parser.add_argument("--road-speed-kmph", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_pack = load_pickle(args.train_cache)
    test_pack = load_pickle(args.test_cache)
    grid = train_pack["grid"]
    assert_same_grid(grid, test_pack["grid"])

    road = load_road_network(args)
    matrices = road.cell_cost_matrices(grid, fallback_speed_kmph=args.road_speed_kmph)
    matrices.save_npz(args.out)

    stats_path = args.stats_out or args.out.with_suffix(".json")
    stats = {
        "train_cache": str(args.train_cache),
        "test_cache": str(args.test_cache),
        "road_matrix": str(args.out),
        "road_network": str(args.road_network or args.road_network_cache),
        **road_matrix_stats(matrices),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.csv_out:
        write_matrix_long_csv(
            args.csv_out,
            matrices,
            distance_field="road_distance_km",
            time_field="road_time_min",
        )

    if args.inject_caches:
        for path, pack in ((args.train_cache, train_pack), (args.test_cache, test_pack)):
            pack["road_distance_matrix"] = matrices.distance_km
            pack["road_time_matrix"] = matrices.time_minutes
            pack["road_matrix_path"] = str(args.out)
            save_pickle(path, pack)

    print("saved road matrix:", args.out)
    print("saved stats:", stats_path)
    if args.csv_out:
        print("saved csv:", args.csv_out)
    if args.inject_caches:
        print("injected matrices into:", args.train_cache, args.test_cache)


if __name__ == "__main__":
    main()
