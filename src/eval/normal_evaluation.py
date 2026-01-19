import argparse
from dataloaders.PanoInfinigen_dataloader import PanoInfinigen
from dataloaders.Structured3D_dataloader import Structured3D
from pathlib import Path
from tqdm.auto import tqdm
import torch
from normal_metrics import MetricTracker
from utils.geometry_utils import unit_normals
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for panorama depth estimation using diffusion models.")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Directory containing the dataset."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["PanoInfinigen", "Structured3D"],
        help="Data source to use. 'PanoInfinigen' for the synthetic dataset."
    )

    parser.add_argument(
        "--pred_path",
        type=str,
        help="Directory containing the results."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode."
    )

    return parser.parse_args()



def main():
    args = parse_args()

    dataset_cls = globals()[args.dataset]
    test_ds = dataset_cls(data_path=Path(args.data_path), split="test", debug=args.debug)
    tracked_metrics = ["mean", "median", "mse", "delta_5", "delta_7.5", "delta_11.25", "delta_22.5", "delta_30"]
    pred_path = Path(args.pred_path) / args.dataset
    metrics = MetricTracker(tracked_metrics)
    eval_folder_name = "normal_evaluation"

    evaluation_dir = Path(pred_path) / eval_folder_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    num_samples = 0
    progress_bar = tqdm(test_ds, desc="Evaluating")
    for batch in progress_bar:
        file_name = batch["id"] + ".npz"
        pred_file_path = pred_path / "eval" / file_name
        try:
            pred = np.load(pred_file_path)['arr_0']
            pred = torch.from_numpy(pred).float().to(device)
            num_samples += 1
        except:
            print(f"Could not load prediction for {file_name}, skipping.")
            continue
        gt = batch["normal"].to(device)
        mask = batch["mask"].to(device)
        pred = unit_normals(pred)
        gt = unit_normals(gt)
        metrics.update(pred, gt, mask)

    final_metrics_dict = metrics.calculate_final(num_samples)
    
    metrics_file_path = evaluation_dir / "evaluation_metrics.txt"
    with open(metrics_file_path, "w") as f:
        for metric, value in final_metrics_dict.items():
            f.write(f"{metric}: {value:.4f}\n")
    print(f"Saved evaluation metrics to {metrics_file_path}")

if __name__ == "__main__":
    main()