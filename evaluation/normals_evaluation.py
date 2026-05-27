"""Surface-normals evaluation: mean/median angular error plus threshold percentages on Structured3D."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation._base import (
    build_test_dataset, device, iter_samples, load_pred,
    print_eval_header, write_metrics,
)
from src.metrics.normals_metrics import MetricTracker
from src.utils.geometry_utils import unit_normals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Panorama surface-normals evaluation.")
    p.add_argument("--data_path", type=str, required=True,
                   help="Root directory of the dataset (parent of the dataset folder).")
    p.add_argument("--dataset", type=str, default="Structured3D", choices=["Structured3D"],
                   help="Currently only Structured3D ships GT normals with the release.")
    p.add_argument("--pred_path", type=str, required=True,
                   help="Per-checkpoint results directory written by inference.py.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    test_ds = build_test_dataset(args.dataset, args.data_path)
    pred_path = Path(args.pred_path)
    print_eval_header(
        task="normals",
        dataset=args.dataset,
        pred_path=pred_path,
        extras={"samples": len(test_ds)},
    )
    metrics = MetricTracker(["mean", "mse", "delta_5", "delta_22.5"])

    n = 0
    for batch in iter_samples(test_ds, f"Evaluating normals on {args.dataset}"):
        try:
            pred = load_pred(pred_path, "normals", batch["id"])
        except Exception:
            print(f"Could not load prediction for {batch['id']}, skipping.")
            continue
        gt, mask = batch["normals"].to(device), batch["mask"].to(device)
        metrics.update(unit_normals(pred), unit_normals(gt), mask)
        n += 1

    write_metrics(pred_path / "normals", metrics.calculate_final(n))


if __name__ == "__main__":
    main()
