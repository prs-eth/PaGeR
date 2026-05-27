# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The PaGeR Authors.
"""Build GLB point clouds from cached PaGeR depth / normals predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from dataloaders.Matterport3D360_dataloader import Matterport3D360
from dataloaders.Replica360_4K_dataloader import Replica360_4K
from dataloaders.Stanford2D3DS_dataloader import Stanford2D3DS
from dataloaders.Structured3D_dataloader import Structured3D
from dataloaders.ZuriPano_dataloader import ZuriPano
from src.utils.geometry_utils import compute_edge_mask, erp_to_point_cloud_glb


_RNG = np.random.default_rng(0)


def subsample_erp_mask(mask: np.ndarray, max_points: int) -> np.ndarray:
    """cos(latitude)-weighted random subset to undo the ERP pole oversampling."""
    H, W = mask.shape
    v = (np.arange(H, dtype=np.float32) + 0.5) / H
    phi = (0.5 - v) * np.pi
    cos_phi = np.cos(phi)
    prob = (cos_phi[:, None] * np.ones(W, dtype=np.float32)) * mask
    total = prob.sum()
    if total > 0:
        prob *= min(1.0, max_points / total)
    return _RNG.random(mask.shape).astype(np.float32) < prob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GLB point clouds from PaGeR predictions.")
    parser.add_argument("--data_path", type=str, default="data",
                        help="Root directory of the dataset (parent of the dataset folder).")
    parser.add_argument("--dataset", type=str, default="ZuriPano",
                        choices=["ZuriPano", "Matterport3D360", "Stanford2D3DS", "Structured3D", "Replica360_4K"],
                        help="Which dataset's predictions to export.")
    parser.add_argument("--color_modality", type=str, default="rgb",
                        choices=["rgb", "normals"],
                        help="Source of per-point colour.")
    parser.add_argument("--depth_path", type=str, required=True,
                        help="Per-checkpoint results directory written by inference.py "
                             "(i.e. <results_path>/<checkpoint_name>/).")
    parser.add_argument("--normals_path", type=str, default=None,
                        help="Optional override for normals predictions. Defaults to --depth_path.")
    parser.add_argument("--max_points", type=int, default=1_000_000,
                        help="Approximate maximum number of points per cloud after subsampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_cls = globals()[args.dataset]
    test_ds = dataset_cls(data_path=Path(args.data_path))
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, num_workers=1, pin_memory=True,
        persistent_workers=True, prefetch_factor=1, shuffle=False,
    )

    depth_path = Path(args.depth_path)
    normals_path = Path(args.normals_path or args.depth_path)
    color_modality_dir = depth_path / (
        "depth" if args.color_modality == "rgb" else "normals"
    )
    save_dir = color_modality_dir / "point_clouds"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("PaGeR point-cloud export")
    print("=" * 64)
    print(f"  dataset      : {args.dataset} (test split, {len(test_ds)} samples)")
    print(f"  depth preds  : {depth_path}")
    if args.color_modality == "normals":
        print(f"  normals preds: {normals_path}")
    print(f"  colour       : {args.color_modality}")
    print(f"  max points   : {args.max_points:,}")
    print(f"  output       : {save_dir}")
    print("=" * 64)

    max_depth_thresh = 0.99 * test_ds.MAX_DEPTH
    progress = tqdm(test_loader, desc="Generating Point Cloud")
    for batch in progress:
        rgb = batch["rgb"][0].permute(1, 2, 0).float().numpy()
        sample_id = batch["id"][0]
        mask = (
            batch["mask"].squeeze().numpy()
            if "mask" in batch
            else np.ones((test_ds.HEIGHT, test_ds.WIDTH), dtype=bool)
        )
        file_name = sample_id + ".npz"

        depth_file = depth_path / "depth" / "preds" / file_name
        try:
            depth = np.load(depth_file)["arr_0"].squeeze(0)
        except Exception:
            print(f"Could not load depth prediction for {file_name}, skipping.")
            continue

        if args.color_modality == "rgb":
            color = rgb
        else:
            normals_file = normals_path / "normals" / "preds" / file_name
            try:
                # (3, H, W) unit normals in [-1, 1] → (H, W, 3) RGB in [0, 1].
                normals_np = np.load(normals_file)["arr_0"].transpose(1, 2, 0)
                color = np.clip((normals_np + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
            except Exception:
                print(f"Could not load normals prediction for {file_name}, skipping.")
                continue

        edge_mask = compute_edge_mask(depth, rel_thresh=0.002)
        # Drop pixels saturated at the far clamp (sky / out-of-range).
        sky_keep = depth < max_depth_thresh
        keep = mask & edge_mask & sky_keep
        keep = subsample_erp_mask(keep, args.max_points)
        erp_to_point_cloud_glb(
            color, depth, keep, export_path=save_dir / f"{sample_id}.glb",
        )

    print(f"Done. Point clouds written to {save_dir}")


if __name__ == "__main__":
    main()
