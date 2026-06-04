from __future__ import annotations

import csv
from pathlib import Path


CASES = (
    ("full_seed0", "Full", "complete model"),
    (
        "no_future_prediction_seed0",
        "w/o Future Prediction",
        "remove future pressure / future demand / future gap",
    ),
    ("no_region_value_seed0", "w/o Region Value", "remove region value head and actor region bias"),
    (
        "no_value_guided_matching_seed0",
        "w/o Value-Guided Matching",
        "use greedy lower matching without value guidance",
    ),
    ("no_region_value_no_future_seed0", "w/o Region + Future", "remove both region value and future prediction"),
)


def _read_metrics(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return row


def main() -> None:
    root = Path("outputs/target_d1p5_peak2p0")
    values: dict[str, dict[str, str]] = {}
    missing: list[Path] = []
    for case, _method, _description in CASES:
        summary = root / case / "summary.csv"
        if not summary.exists():
            missing.append(summary)
            continue
        values[case] = _read_metrics(summary)

    if missing:
        print("Missing summaries:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)

    print("method\tdescription\tresponse_rate\tNormalized GMV\tresponse_time_seconds\tcancellation_rate")
    for case, method, description in CASES:
        row = values[case]
        print(
            f"{method}\t{description}\t"
            f"{float(row['response_rate']):.4f}\t"
            f"{float(row['normalized_gmv']):.4f}\t"
            f"{float(row['response_time_seconds']):.1f}\t"
            f"{float(row['cancellation_rate']):.4f}"
        )


if __name__ == "__main__":
    main()
