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
from dataloaders.PanoInfinigen_dataloader import PanoInfinigen
from dataloaders.Structured3D_dataloader import Structured3D
from dataloaders.Replica360_4K_dataloader import Replica360_4K
from src.metrics.edge_metrics import MetricTracker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Edge evaluation script for panorama depth estimation.")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Directory containing the dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["PanoInfinigen", "Structured3D", "Replica360_4K"],
        help="Data source to use. 'PanoInfinigen' for the synthetic dataset."
    )

    parser.add_argument(
        "--pred_path",
        type=str,
        help="Directory containing the results."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_cls = globals()[args.dataset]
    test_ds = dataset_cls(data_path=Path(args.data_path), split="test")
    tracked_metrics = ["edge_dbe_completeness", "edge_dbe_accuracy", "edge_precision", "edge_recall"]
    pred_path = Path(args.pred_path) / args.dataset
    metrics = MetricTracker(tracked_metrics, test_ds.MAX_DEPTH)
    eval_folder_name = "edge_evaluation"

    evaluation_dir = Path(pred_path) / eval_folder_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    error_maps_dir = evaluation_dir / "error_maps"
    error_maps_dir.mkdir(parents=True, exist_ok=True)

    num_samples = 0
    progress_bar = tqdm(test_ds, desc="Evaluating")
    for batch in progress_bar:
        file_name = batch["id"] + ".npz"
        pred_file_path = pred_path / "eval" / file_name
        try:
            pred = np.load(pred_file_path)['arr_0']
            pred = torch.from_numpy(pred).float().squeeze(0).to(device)
            num_samples += 1
        except:
            print(f"Could not load prediction for {file_name}, skipping.")
            continue
        
        gt = batch["depth"].mean(dim=0, keepdim=True).squeeze(0).to(device)
        mask = batch["mask"].squeeze(0).to(device)
        metrics.update(pred, gt, mask, batch["id"])

    final_metrics_dict, error_list = metrics.calculate_final(num_samples)
    
    metrics_file_path = evaluation_dir / "evaluation_metrics.txt"
    with open(metrics_file_path, "w") as f:
        for metric, value in final_metrics_dict.items():
            f.write(f"{metric}: {value:.4f}\n")
    print(f"Saved evaluation metrics to {metrics_file_path}")

if __name__ == "__main__":
    main()