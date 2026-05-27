"""Shared plumbing for the per-modality evaluation CLIs (dataset, pred loading, progress, metrics output)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Quiet the DA3 backbone's chatty INFO logs at import time.
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

import numpy as np
import torch
from tqdm.auto import tqdm

from dataloaders.Matterport3D360_dataloader import Matterport3D360
from dataloaders.Replica360_4K_dataloader import Replica360_4K
from dataloaders.Stanford2D3DS_dataloader import Stanford2D3DS
from dataloaders.Structured3D_dataloader import Structured3D
from dataloaders.ZuriPano_dataloader import ZuriPano

DATASETS = {
    "ZuriPano": ZuriPano,
    "Matterport3D360": Matterport3D360,
    "Stanford2D3DS": Stanford2D3DS,
    "Structured3D": Structured3D,
    "Replica360_4K": Replica360_4K,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_test_dataset(name: str, data_path):
    return DATASETS[name](data_path=Path(data_path))


def load_pred(pred_path: Path, modality: str, sample_id: str,
              target_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Load <pred_path>/<modality>/preds/<sample_id>.npz, optionally bilinear-resizing to target_hw."""
    fpath = pred_path / modality / "preds" / f"{sample_id}.npz"
    arr = np.load(fpath)["arr_0"]
    t = torch.from_numpy(arr).float().to(device)
    if target_hw is not None and tuple(t.shape[-2:]) != tuple(target_hw):
        leading = t.shape[:-2]
        t = torch.nn.functional.interpolate(
            t.reshape(-1, 1, *t.shape[-2:]),
            size=target_hw, mode="bilinear", align_corners=False,
        ).reshape(*leading, *target_hw)
    return t


def iter_samples(test_ds, desc: str):
    return tqdm(test_ds, desc=desc)


def write_metrics(out_dir: Path, final_metrics: dict[str, float],
                  filename: str = "evaluation_metrics.txt") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "w") as f:
        for k, v in final_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print()
    print("Metrics")
    print("-" * 40)
    for k, v in final_metrics.items():
        print(f"  {k:<32s} {v:.4f}")
    print("-" * 40)
    print(f"Saved metrics to {out_path}")
    return out_path


def print_eval_header(*, task: str, dataset: str, pred_path: Path,
                      extras: dict | None = None) -> None:
    print("=" * 64)
    print(f"PaGeR evaluation — {task}")
    print("=" * 64)
    width = max([len(k) for k in (extras or {})] + [len("pred_path")])
    print(f"  {'dataset':<{width}s} : {dataset}")
    print(f"  {'pred_path':<{width}s} : {pred_path}")
    if extras:
        for k, v in extras.items():
            print(f"  {k:<{width}s} : {v}")
    print("=" * 64)
