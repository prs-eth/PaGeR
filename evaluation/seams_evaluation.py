"""Cubemap-seam evaluation: pixel-coverage, edge-incidence, and edge-magnitude metrics.

See the PaGeR paper appendix (Table 4a) for the metric definitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation._base import (
    build_test_dataset, iter_samples, load_pred,
    print_eval_header, write_metrics,
)
from src.metrics.seams_metrics import MetricTracker

# Paper reports seam metrics on Replica360_4K (dense synthetic depth).
SEAMS_EVAL_DATASETS = ("Replica360_4K",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cubemap-seam evaluation.")
    p.add_argument("--data_path", type=str, required=True,
                   help="Root directory of the dataset (parent of the dataset folder).")
    p.add_argument("--dataset", type=str, required=True, choices=SEAMS_EVAL_DATASETS,
                   help="Dataset to evaluate against.")
    p.add_argument("--pred_path", type=str, required=True,
                   help="Per-checkpoint results directory written by inference.py.")
    p.add_argument("--pixel_jump_thresh", type=float, default=0.010,
                   help="Per-pixel |Δdepth| threshold on min-max-normalised depth above "
                        "which a seam pixel counts as visibly broken. Drives "
                        "seam_defect_density directly, and feeds the per-edge pixel count "
                        "used by seam_prevalence.")
    p.add_argument("--edge_mean_jump_thresh", type=float, default=0.008,
                   help="Per-edge mean-|Δdepth| threshold above which a whole cube edge "
                        "counts as broken. Drives seam_severity only.")
    p.add_argument("--min_broken_pixel_frac", type=float, default=0.05,
                   help="Per-edge minimum fraction of pixels exceeding --pixel_jump_thresh "
                        "for the edge to count toward seam_prevalence. Drives "
                        "seam_prevalence only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    test_ds = build_test_dataset(args.dataset, args.data_path)
    pred_path = Path(args.pred_path)
    print_eval_header(
        task="seams",
        dataset=args.dataset,
        pred_path=pred_path,
        extras={
            "pixel_jump_thresh": args.pixel_jump_thresh,
            "edge_mean_jump_thresh": args.edge_mean_jump_thresh,
            "min_broken_pixel_frac": args.min_broken_pixel_frac,
            "samples": len(test_ds),
        },
    )
    metrics = MetricTracker(
        ["seam_defect_density", "seam_prevalence", "seam_severity"],
        pixel_jump_thresh=args.pixel_jump_thresh,
        edge_mean_jump_thresh=args.edge_mean_jump_thresh,
        min_broken_pixel_frac=args.min_broken_pixel_frac,
    )

    n = 0
    for batch in iter_samples(test_ds, f"Evaluating seams on {args.dataset}"):
        try:
            pred = load_pred(pred_path, "depth", batch["id"])
        except Exception:
            print(f"Could not load prediction for {batch['id']}, skipping.")
            continue
        metrics.update(pred)
        n += 1

    write_metrics(pred_path / "seams", metrics.calculate_final(n))


if __name__ == "__main__":
    main()
