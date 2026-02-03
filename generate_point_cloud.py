import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm.auto import tqdm

from dataloaders.PanoInTheWild_dataloader import PanoInTheWild
from dataloaders.PanoInfinigen_dataloader import PanoInfinigen
from dataloaders.Matterport3D360_dataloader import Matterport3D360
from dataloaders.Structured3D_dataloader import Structured3D
from dataloaders.Stanford2D3DS_dataloader import Stanford2D3DS
from dataloaders.Replica360_4K_dataloader import Replica360_4K
from src.utils.geometry_utils import compute_edge_mask, erp_to_pointcloud, erp_to_point_cloud_glb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def parse_args():
    parser = argparse.ArgumentParser(description="The generation script for point clouds from panorama depth and normals predictions.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Directory containing the dataset."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="PanoInfinigen",
        choices=["PanoInfinigen", "Matterport3D360", "PanoInTheWild", "Structured3D", "Stanford2D3DS", "Replica360_4K"],
        help="Data source to use. 'PanoInfinigen' for the synthetic dataset."
    )
    
    parser.add_argument(
        "--color_modality",
        type=str,
        default="rgb",
        choices=["rgb", "normals"],
        help="What data modality to use for coloring the point cloud."
    )

    parser.add_argument(
        "--depth_path",
        type=str,
        default="data",
        help="Directory containing the results."
    )

    parser.add_argument(
        "--normals_path",
        type=str,
        default="data",
        help="Directory containing the results."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    dataset_cls = globals()[args.dataset]
    test_ds = dataset_cls(data_path=Path(args.data_path), split="test")
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=1, num_workers=1, 
                                        pin_memory=True, persistent_workers=True, prefetch_factor=1, shuffle=False)
    depth_path = Path(args.depth_path) / args.dataset
    normals_path = Path(args.normals_path) / args.dataset
    save_dir = depth_path / f"{args.color_modality}_point_clouds"
    save_dir.mkdir(parents=True, exist_ok=True)
    progress_bar = tqdm(test_dataloader, desc="Generating Point Cloud")
    for i, batch in enumerate(progress_bar):
        rgb = batch["rgb"][0].permute(1, 2, 0).float().numpy()
        id = batch["id"][0]

        if 'mask' in batch:
            mask = batch['mask'].squeeze().numpy()
        else:
            mask = np.ones(rgb.shape[:2], dtype=bool)
        file_name = id + ".npz"

        depth_file_path = depth_path / "eval" / file_name
        try:
            depth = np.load(depth_file_path)['arr_0'].squeeze(0)
        except:
            print(f"Could not load depth prediction for {file_name}, skipping.")
            continue
        if args.color_modality == "rgb":
            color = rgb
        else:
            normals_file_path = normals_path / "eval" / file_name
            try:
                color = np.load(normals_file_path)['arr_0'].transpose(1,2,0)
            except:
                print(f"Could not load normals prediction for {file_name}, skipping.")
                continue

        edge_filtered_mask = compute_edge_mask(
            depth,
            abs_thresh=0.002,
            rel_thresh=0.002,
        )
        mask = mask & edge_filtered_mask
        pts, cols = erp_to_pointcloud(
            color, depth, mask)

        erp_to_point_cloud_glb(
            color, depth, mask,
            export_path=save_dir / f"{id}.glb"
        )


