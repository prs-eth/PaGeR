"""Depth evaluation: LS-aligned AbsRel / RMSE / delta-1 over valid ERP pixels."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation._base import (
    build_test_dataset, device, iter_samples, load_pred,
    print_eval_header, write_metrics,
)
from src.metrics.depth_metrics import MetricTracker, align_pred_gt

# Datasets shipping depth GT under licenses that allow redistributing the eval pipeline.
DEPTH_EVAL_DATASETS = ("Matterport3D360", "Stanford2D3DS", "ZuriPano")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Panorama depth evaluation.")
    p.add_argument("--data_path", type=str, required=True,
                   help="Root directory of the dataset (parent of the dataset folder).")
    p.add_argument("--dataset", type=str, required=True, choices=DEPTH_EVAL_DATASETS,
                   help="Which evaluation dataset to score against.")
    p.add_argument("--pred_path", type=str, required=True,
                   help="Per-checkpoint results directory written by inference.py.")
    p.add_argument("--alignment_type", type=str, default="metric",
                   choices=["metric", "scale", "scale_and_shift"],
                   help="LS alignment between prediction and GT before scoring.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    test_ds = build_test_dataset(args.dataset, args.data_path)
    pred_path = Path(args.pred_path)
    print_eval_header(
        task="depth",
        dataset=args.dataset,
        pred_path=pred_path,
        extras={"alignment": args.alignment_type, "samples": len(test_ds)},
    )
    metrics = MetricTracker(["abs_relative_difference", "rmse_linear", "delta1_acc"])

    n = 0
    for batch in iter_samples(test_ds, f"Evaluating depth on {args.dataset}"):
        try:
            pred = load_pred(pred_path, "depth", batch["id"],
                             target_hw=tuple(batch["depth"].shape[-2:]))
        except Exception:
            print(f"Could not load prediction for {batch['id']}, skipping.")
            continue
        gt, mask = batch["depth"].to(device), batch["mask"].to(device)
        aligned = align_pred_gt(pred, gt, mask, args.alignment_type)
        metrics.update(aligned, gt, mask, batch["id"])
        n += 1

    final, _ = metrics.calculate_final(n)
    write_metrics(
        pred_path / "depth", final,
        filename=f"evaluation_metrics_{args.alignment_type}.txt",
    )


if __name__ == "__main__":
    main()
