import sys
import torch
import numpy as np
import logging
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from diffusers.utils import check_min_version
from huggingface_hub import hf_hub_download

from dataloaders.PanoInTheWild_dataloader import PanoInTheWild
from dataloaders.PanoInfinigen_dataloader import PanoInfinigen
from dataloaders.Matterport3D360_dataloader import Matterport3D360
from dataloaders.Structured3D_dataloader import Structured3D
from dataloaders.Stanford2D3DS_dataloader import Stanford2D3DS
from dataloaders.ScannetPP_dataloader import ScannetPP
from dataloaders.Replica360_4K_dataloader import Replica360_4K
from src.pager import Pager
from src.utils.utils import args_to_omegaconf, convert_paths_to_pathlib, prepare_image_for_logging, log_images_mosaic

check_min_version("0.27.0.dev0")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for panorama depth estimation using diffusion models.")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the YAML config file."
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="UNet checkpoint to load, either local path or HF repo ID."
    )
    parser.add_argument(
        "--enable_xformers", 
        action="store_true", 
        default=None,
        help="Whether or not to use xformers."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Directory containing the dataset."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["PanoInfinigen", "PanoInTheWild", "Matterport3D360", "Stanford2D3DS", 
                 "Structured3D", "Structured3D_ScannetPP", "Replica360_4K"],
        help="Data source to use. 'PanoInfinigen' for the synthetic dataset."
    )

    parser.add_argument(
        "--scenes",
        default=None,
        choices=["indoor", "outdoor", "both"], 
        help="Which scenes to use for training. 'indoor' for indoor scenes, 'outdoor' for outdoor scenes, " \
        "'both' for both indoor and outdoor scenes.",
    )

    parser.add_argument(
        "--img_report_frequency",
        type=int,
        default=None,
        help="How often to report image (in steps)."
    )

    parser.add_argument(
        "--generate_eval",
        action="store_true",
        default=None,
        help="Whether to generate evaluation .npz files."
    )

    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Directory to save results."
    )

    parser.add_argument(
        "--pred_only",
        action="store_true",
        default=None,
        help="Whether to only output the final pred without the concatenated input."
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    try:
        checkpoint_config_path = hf_hub_download(
            repo_id=args.checkpoint_path,
            filename="config.yaml"
        )
    except Exception as e:
            checkpoint_config_path = Path(args.checkpoint_path) / "config.yaml"
            
    cfg = OmegaConf.load(args.config)
    checkpoint_cfg = OmegaConf.load(checkpoint_config_path)
    cfg.model = checkpoint_cfg.model

    cfg = args_to_omegaconf(args, cfg)
    cfg = convert_paths_to_pathlib(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger("simple")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(message)s") 
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False 
    cmap = plt.get_cmap("Spectral")

    preds_path = Path(args.results_path) / f"{cfg.model.checkpoint_path.name}" / cfg.data.dataset
    preds_path.mkdir(parents=True, exist_ok=True)
    print(f"preds will be saved to {preds_path}")
    preds_img_dir = preds_path / "example_images"
    preds_img_dir.mkdir(parents=True, exist_ok=True)
    preds_eval_dir = preds_path / "eval"
    preds_eval_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_config = {}
    logger.info(f"Loading UNet weights from checkpoint at {cfg.model.checkpoint_path}")
    checkpoint_config[checkpoint_cfg.model.modality] = {"path": cfg.model.checkpoint_path, 
                                                        "mode": "trained", "config": checkpoint_cfg.model}

    pager = Pager(model_configs=checkpoint_config, pretrained_path = cfg.model.pretrained_path, 
                  train_modality=cfg.model.modality, device=device)
    pager.unet[cfg.model.modality].to(device, dtype=pager.weight_dtype)
    pager.unet[cfg.model.modality].eval()

    dataset_cls = globals()[cfg.data.dataset]
    test_ds = dataset_cls(data_path=Path(cfg.data.data_path), split="test", scenes=cfg.data.scenes, 
                          log_depth=cfg.model.log_scale)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=1, num_workers=1, pin_memory=True, 
                                                  persistent_workers=True, prefetch_factor=1, shuffle=False)

    pager.prepare_cubemap_PE(test_ds.HEIGHT, test_ds.WIDTH)
    min_depth = test_ds.LOG_MIN_DEPTH if cfg.model.log_scale else test_ds.MIN_DEPTH
    depth_range = test_ds.LOG_DEPTH_RANGE if cfg.model.log_scale else test_ds.DEPTH_RANGE
    progress_bar = tqdm(test_dataloader, desc=f"Test", total=len(test_dataloader))
    for i, batch in enumerate(progress_bar):
        with torch.inference_mode():
            batch["rgb_cubemap"] = batch["rgb_cubemap"].to(device)
            if 'mask' in batch:
                mask = batch["mask"].squeeze(0).to(device)
            else:
                mask = None
            pred_cubemap = pager(batch, cfg.model.modality)
            if cfg.model.modality == "depth": 
                pred, pred_image = pager.process_depth_output(pred_cubemap, orig_size=(dataset_cls.HEIGHT, dataset_cls.WIDTH), 
                                                              min_depth=min_depth, depth_range=depth_range, log_scale=cfg.model.log_scale, mask=mask)
                pred, pred_image = pred.cpu().numpy(), pred_image.cpu().numpy()
                pred_image = np.clip(pred_image, pred_image.min(), np.quantile(pred_image, 0.99))
                pred_image = prepare_image_for_logging(pred_image)
                pred_image = cmap(pred_image[0,...]/255.0)
                pred_image = (pred_image[..., :3] * 255).astype(np.uint8)
            elif cfg.model.modality == "normals":
                pred = pager.process_normals_output(pred_cubemap, orig_size=(dataset_cls.HEIGHT, dataset_cls.WIDTH))
                pred = pred.cpu().numpy()
                pred_image = pred.copy()
                pred_image = prepare_image_for_logging(pred_image).transpose(1,2,0)

            sample_id = batch["id"][0]

            if args.generate_eval:
                npy_path = preds_eval_dir / f"{sample_id}"
                np.savez(npy_path, pred)

            if i % cfg.logging.img_report_frequency == 0:
                if args.pred_only:
                    result = pred_image
                else:
                    pred_image = pred_image.transpose(2, 0, 1)
                    rgb = prepare_image_for_logging(batch["rgb"][0].cpu().numpy())
                    result = log_images_mosaic([rgb, pred_image])
                Image.fromarray(result).save(preds_img_dir / f"{sample_id}.jpg")


if __name__ == "__main__":
    main()
