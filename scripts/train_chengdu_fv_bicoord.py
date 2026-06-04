from __future__ import annotations

import sys

try:
    from scripts._common import ensure_project_root_on_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from _common import ensure_project_root_on_path

ensure_project_root_on_path()

from run_chengdu_cache_experiment import main as run_cache_experiment


def main() -> None:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        run_cache_experiment()
        return
    if "--train-cache" not in argv or "--test-cache" not in argv:
        raise SystemExit(
            "This entry no longer runs Chengdu Bi-STAR directly from one raw source, "
            "because that made train/eval use the same demand. First build split caches with:\n"
            "  python scripts/preprocess_chengdu_train_test.py --raw-dir data/2016年成都滴滴轨迹数据\n"
            "Then run:\n"
            "  python scripts/train_chengdu_fv_bicoord.py --train-cache data/processed/chengdu_train_20161108_20161130_N142_T108.pkl "
            "--test-cache data/processed/chengdu_test_20161101_20161107_N142_T108.pkl --out outputs/fv_bicoord_chengdu"
        )
    if "--method" not in argv:
        sys.argv.extend(["--method", "fv_bicoord"])
    run_cache_experiment()


if __name__ == "__main__":
    main()
