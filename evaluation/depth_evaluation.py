import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from dataloaders.PanoInfinigen_dataloader import PanoInfinigen
from dataloaders.Matterport3D360_dataloader import Matterport3D360
from dataloaders.Stanford2D3DS_dataloader import Stanford2D3DS
from dataloaders.Structured3D_dataloader import Structured3D
from dataloaders.ScannetPP_dataloader import ScannetPP
from src.metrics.depth_metrics import MetricTracker, align_pred_gt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Depth evaluation script for panorama depth estimation.")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Directory containing the dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["PanoInfinigen", "Matterport3D360", "Stanford2D3DS", "Structured3D", "Structured3D_ScannetPP"],
        help="Data source to use. 'PanoInfinigen' for the synthetic dataset."
    )

    parser.add_argument(
        "--pred_path",
        type=str,
        help="Directory containing the results."
    )

    parser.add_argument(
        "--alignment_type",
        type=str,
        choices=["metric", "scale", "scale_and_shift"],
        help="Type of alignment to apply between prediction and ground truth."
    )


    parser.add_argument(
        "--save_error_maps",
        action="store_true",
        help="Whether to save error maps during evaluation."
    )

    parser.add_argument(
        "--error_maps_saving_frequency",
        type=int,
        help="Frequency of saving error maps (in terms of batches)."
    )

    return parser.parse_args()


def save_error_map(pred, gt, mask, save_path):
    error_map = (pred - gt) * mask
    error_map_np = error_map.squeeze().cpu().numpy()

    max_abs = np.nanmax(np.abs(error_map_np))
    vmin, vmax = -max_abs, max_abs

    cmap = plt.get_cmap('seismic')
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(error_map_np, cmap=cmap, norm=norm, origin='upper')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.5)
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_label('Red-too far, blue-too close', rotation=270, labelpad=15)
    cbar.ax.set_yticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def main():
    args = parse_args()

    dataset_cls = globals()[args.dataset]
    test_ds = dataset_cls(data_path=Path(args.data_path), split="test")
    tracked_metrics = ["abs_relative_difference", "rmse_linear", "delta1_acc", "delta2_acc"]
    pred_path = Path(args.pred_path) / args.dataset
    metrics = MetricTracker(tracked_metrics)
    eval_folder_name = "depth_evaluation"

    if args.alignment_type == 'metric':
        evaluation_dir_name = "metric"
    else:
        evaluation_dir_name = f"{args.alignment_type}"
    evaluation_dir = pred_path / eval_folder_name / evaluation_dir_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    error_maps_dir = evaluation_dir / "error_maps"
    error_maps_dir.mkdir(parents=True, exist_ok=True)

    num_samples = 0
    progress_bar = tqdm(test_ds, desc="Evaluating")
    for i, batch in enumerate(progress_bar):
        file_name = batch["id"] + ".npz"
        pred_file_path = pred_path / "eval" / file_name
        try:
            pred = np.load(pred_file_path)['arr_0']
            pred = torch.from_numpy(pred).float().to(device)
            num_samples += 1
        except:
            print(f"Could not load prediction for {file_name}, skipping.")
            continue
        gt = batch["depth"].mean(dim=0, keepdim=True).to(device)
        mask = batch["mask"].to(device)
        aligned_pred = align_pred_gt(pred, gt, mask, args.alignment_type)
        if args.save_error_maps and (i % args.error_maps_saving_frequency == 0):
            save_error_map(aligned_pred, gt, mask, error_maps_dir / f"{batch['id']}.png")
        metrics.update(aligned_pred, gt, mask, batch["id"])

    final_metrics_dict, error_list = metrics.calculate_final(num_samples)
    
    metrics_file_path = evaluation_dir / "evaluation_metrics.txt"
    with open(metrics_file_path, "w") as f:
        for metric, value in final_metrics_dict.items():
            f.write(f"{metric}: {value:.4f}\n")
    print(f"Saved evaluation metrics to {metrics_file_path}")

if __name__ == "__main__":
    main()
