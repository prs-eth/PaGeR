# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The PaGeR Authors.
"""Batch inference for PaGeR.

Runs a checkpoint on every panorama in the chosen dataset, writing preview
JPEGs and (with ``--generate_eval``) raw .npz arrays for the evaluation scripts.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Quiet the DA3 backbone's chatty INFO logs during model construction.
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

from dataloaders.Matterport3D360_dataloader import Matterport3D360
from dataloaders.Replica360_4K_dataloader import Replica360_4K
from dataloaders.Stanford2D3DS_dataloader import Stanford2D3DS
from dataloaders.Structured3D_dataloader import Structured3D
from dataloaders.ZuriPano_dataloader import ZuriPano
from src.pager import Pager
from src.utils.scene_classifier import get_classifier
from src.utils.utils import (
    args_to_omegaconf,
    convert_paths_to_pathlib,
    prepare_depth_for_logging,
    prepare_normals_for_logging,
)


# Both scale heads share the ``scale`` output key, so exactly one must run per forward pass.
SCALE_HEADS = {"indoor": "scale_indoor", "outdoor": "scale_outdoor"}

MODEL_TO_REPO = {
    "pager":              "prs-eth/PaGeR",
    "pager-metric-depth": "prs-eth/PaGeR-metric-depth",
    "pager-normals":      "prs-eth/PaGeR-normals",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaGeR batch inference.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the inference YAML config.")
    parser.add_argument("--checkpoint", type=str, default="pager",
                        help="Which checkpoint to run. Accepts a short alias for "
                             "the released checkpoints (pager, pager-metric-depth, "
                             "pager-normals), a HuggingFace Hub repo id, or a local "
                             "directory holding model.safetensors + config.yaml. "
                             "Default: pager (the unified prs-eth/PaGeR checkpoint).")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Root directory of the dataset (overrides cfg.data.data_path).")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["ZuriPano", "Matterport3D360", "Stanford2D3DS", "Structured3D", "Replica360_4K"],
                        help="Which evaluation dataset to run on.")
    parser.add_argument("--results_path", type=str, default="results",
                        help="Where to write previews and raw .npz predictions. "
                             "Defaults to ``./results`` (already in .gitignore).")
    parser.add_argument("--scene_mode", type=str, default="auto",
                        choices=["auto", "indoor", "outdoor"],
                        help="Which scale head to use on dual-scale checkpoints. "
                             "'auto' (default) routes each panorama via a CLIP "
                             "ViT-B/32 classifier on the 4 equatorial cubemap faces.")
    parser.add_argument("--generate_eval", action="store_true", default=None,
                        help="Write per-sample .npz files consumable by evaluation/*.")
    parser.add_argument("--img_report_frequency", type=int, default=None,
                        help="Save a JPEG preview every Nth sample (1 = every sample).")
    parser.add_argument("--sky_mask_threshold", type=float, default=0.3,
                        help="Sigmoid threshold for binarising the sky mask in depth/normals fill.")
    parser.add_argument("--sky_mask_softness", type=float, default=0.15,
                        help="Half-width of the smoothstep around --sky_mask_threshold. "
                             "0 reproduces the old hard binary mask; larger values give softer edges.")
    parser.add_argument("--mask_npz_with_sky", action="store_true", default=False,
                        help="If set, the saved raw .npz arrays are also sky-masked (PNG "
                             "previews are sky-masked regardless of this flag).")
    parser.add_argument("--sky_mask_open_kernel", type=int, default=5,
                        help="Kernel size for the morphological opening applied to the sky "
                             "probability map. 0 / 1 disables. Default 5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = MODEL_TO_REPO.get(args.checkpoint, args.checkpoint)

    try:
        checkpoint_config_path = hf_hub_download(
            repo_id=checkpoint_path, filename="config.yaml"
        )
    except Exception:
        checkpoint_config_path = Path(checkpoint_path) / "config.yaml"

    cfg = OmegaConf.load(args.config)
    model_cfg = OmegaConf.load(checkpoint_config_path)

    cfg = args_to_omegaconf(args, cfg)
    cfg = convert_paths_to_pathlib(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger("simple")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    cmap = plt.get_cmap("Spectral")

    preds_eval_dir = Path(args.results_path)
    preds_eval_dir.mkdir(parents=True, exist_ok=True)

    for modality in model_cfg.modalities:
        if modality in ("sky", "scale", "scale_indoor", "scale_outdoor"):
            continue
        (preds_eval_dir / modality / "preds").mkdir(parents=True, exist_ok=True)
        (preds_eval_dir / modality / "previews").mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("PaGeR inference")
    print("=" * 64)
    print(f"  checkpoint : {args.checkpoint}  ({checkpoint_path})")
    print(f"  modalities : {list(model_cfg.modalities)}")
    print(f"  dataset    : {cfg.data.dataset}")
    print(f"  device     : {device}")
    print(f"  results    : {preds_eval_dir}")

    print("Loading model ...", flush=True)
    pager = Pager(Path(checkpoint_path), cfg=model_cfg, device=device)
    pager.get_intrinsics_extrinsics(image_size=model_cfg.face_size, fov=getattr(model_cfg, "cube_fov", 90.0))
    pager.model.to(device, dtype=pager.weight_dtype)
    pager.model.eval()
    print("Model ready.")

    has_dual_scale = all(m in model_cfg.modalities for m in SCALE_HEADS.values())
    scene_classifier = None
    if has_dual_scale and args.scene_mode == "auto":
        scene_classifier = get_classifier(device=device)
        print("Scene routing : auto (CLIP ViT-B/32 on cubemap faces)")
    elif has_dual_scale:
        print(f"Scene routing : forced '{args.scene_mode}' → {SCALE_HEADS[args.scene_mode]}")

    # CLIP classifier expects raw RGB in [0, 1]; the cubemap is ImageNet-normalised.
    _imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _skip_heads_for(rgb_cubemap_normed):
        if not has_dual_scale:
            return None
        if args.scene_mode == "auto":
            cube_01 = rgb_cubemap_normed.to(device) * _imagenet_std + _imagenet_mean
            label, _ = scene_classifier.classify(cube_01)
            keep = SCALE_HEADS[label]
        else:
            keep = SCALE_HEADS[args.scene_mode]
        return {h for h in SCALE_HEADS.values() if h != keep}

    dataset_cls = globals()[cfg.data.dataset]
    test_ds = dataset_cls(
        data_path=Path(cfg.data.data_path),
        face_size=model_cfg.face_size,
        cube_fov=getattr(model_cfg, "cube_fov", 90.0),
    )
    pager.set_depth_range(test_ds.MIN_DEPTH, test_ds.MAX_DEPTH)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, num_workers=1, pin_memory=True,
        persistent_workers=True, prefetch_factor=1, shuffle=False,
    )

    print(f"Dataset       : {cfg.data.dataset} (test split, {len(test_ds)} samples)")
    print(f"Depth range   : [{test_ds.MIN_DEPTH}, {test_ds.MAX_DEPTH}] m")
    print("=" * 64)

    img_freq = max(1, int(getattr(cfg.logging, "img_report_frequency", 1)))
    progress = tqdm(test_loader, desc="Inference", total=len(test_loader))
    eval_modalities = [m for m in model_cfg.modalities
                       if m not in ("sky", "scale", "scale_indoor", "scale_outdoor")]
    for i, batch in enumerate(progress):
        with torch.inference_mode():
            sample_id = batch["id"][0]
            if args.generate_eval and all(
                (preds_eval_dir / m / "preds" / f"{sample_id}.npz").exists()
                for m in eval_modalities
            ):
                continue
            rgb_cubemap = batch["rgb_cubemap"].to(device)
            skip_heads = _skip_heads_for(rgb_cubemap[0])
            pred_dict = pager(rgb_cubemap, dtype=torch.float16, skip_heads=skip_heads)

            sky_pred = pred_dict["sky"][0] if "sky" in pred_dict else None
            log_scale = pred_dict.get("scale", None)
            npz_sky_mask = sky_pred if args.mask_npz_with_sky else None
            save_png = i % img_freq == 0

            for modality in model_cfg.modalities:
                if modality in ("sky", "scale", "scale_indoor", "scale_outdoor"):
                    continue

                log_fn = (
                    prepare_depth_for_logging
                    if modality == "depth"
                    else prepare_normals_for_logging
                )
                extra = {
                    "sky_mask_threshold": args.sky_mask_threshold,
                    "sky_mask_softness": args.sky_mask_softness,
                    "sky_mask_open_kernel": args.sky_mask_open_kernel,
                }
                if modality == "depth":
                    extra["log_scale"] = log_scale

                raw_image, result_image = log_fn(
                    pager, pred_dict[modality][0], npz_sky_mask,
                    (test_ds.HEIGHT, test_ds.WIDTH), cmap, **extra,
                )

                if args.generate_eval:
                    np.savez(
                        preds_eval_dir / modality / "preds" / f"{sample_id}",
                        raw_image,
                    )

                if save_png:
                    if npz_sky_mask is None and sky_pred is not None:
                        # Previews are always sky-masked; re-render since the .npz pass was unmasked.
                        _, result_image = log_fn(
                            pager, pred_dict[modality][0], sky_pred,
                            (test_ds.HEIGHT, test_ds.WIDTH), cmap, **extra,
                        )
                    Image.fromarray(np.transpose(result_image, (1, 2, 0))).save(
                        preds_eval_dir / modality / "previews" / f"{sample_id}.jpg",
                    )

    print(f"Done. Predictions written to {preds_eval_dir}")


if __name__ == "__main__":
    main()
