from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TQGSACap result json files.")
    parser.add_argument(
        "--results-dir",
        default="scripts/yoloworld_lightning/results",
        help="Directory containing <experiment>.json files.",
    )
    parser.add_argument(
        "--out",
        default="scripts/yoloworld_lightning/results/summary.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    metric_keys = set()
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        metrics = payload.get("metrics", {})
        metric_keys.update(metrics.keys())
        rows.append({
            "experiment": payload.get("experiment", path.stem),
            **metrics,
        })

    metric_order = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rougel", "cider", "harmonic_mean"]
    fieldnames = ["experiment"] + [key for key in metric_order if key in metric_keys]
    fieldnames += sorted(metric_keys - set(fieldnames))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
