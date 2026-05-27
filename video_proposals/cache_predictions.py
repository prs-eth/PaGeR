"""Run PaGeR once on a curated example set and cache outputs for the video generators."""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.pager import Pager
from src.utils.geometry_utils import erp_to_cubemap, erp_to_pointcloud, compute_edge_mask
from src.utils.utils import prepare_depth_for_logging, prepare_normals_for_logging

EXAMPLES = [
    ("livingroom_synth",        "Indoor"),
    ("church_meeting_room",     "Indoor"),
    ("eth_campus_plaza",        "Outdoor"),
    ("zurich_street_corner",    "Outdoor"),
    ("medieval_kitchen",        "Indoor"),
]
ERP_W, ERP_H = 2048, 1024     # downsized from the demo's 4032×2016 to keep cache small
CACHE_DIR = ROOT / "video_proposals" / "cache"
CHECKPOINT_DIR = ROOT / "checkpoints" / "PaGeR"

SCENE_TO_SCALE_HEAD = {"Indoor": "scale_indoor", "Outdoor": "scale_outdoor"}


def main():
    device = torch.device("cuda")
    cfg = OmegaConf.load(CHECKPOINT_DIR / "config.yaml")
    face_size = int(cfg.face_size); cube_fov = float(getattr(cfg, "cube_fov", 90.0))
    pager = Pager(CHECKPOINT_DIR, cfg=cfg, device=device)
    pager.get_intrinsics_extrinsics(image_size=face_size, fov=cube_fov)
    pager.model.to(device, dtype=pager.weight_dtype); pager.model.eval()
    cmap = plt.get_cmap("Spectral")

    for stem, scene_hint in EXAMPLES:
        out = CACHE_DIR / stem
        out.mkdir(parents=True, exist_ok=True)
        if (out / "rgb.png").exists() and (out / "depth_raw.npy").exists():
            print(f"  [skip] {stem}"); continue

        # locate the source image
        src = next((ROOT / "assets" / "examples").glob(f"{stem}.*"))
        img_pil = Image.open(src).convert("RGB").resize((ERP_W, ERP_H), Image.LANCZOS)
        input_rgb = np.array(img_pil)

        img = torch.from_numpy(input_rgb).permute(2, 0, 1).to(torch.float32) / 255.0
        img = img * 2.0 - 1.0
        rgb_cubemap = erp_to_cubemap(img, face_w=face_size, fov=cube_fov).unsqueeze(0).to(device)

        # cache the cubemap faces so videos can show them
        faces_01 = ((rgb_cubemap[0].cpu().numpy() + 1.0) * 0.5).clip(0, 1)  # (6,3,H,W)
        np.save(out / "cubemap.npy", faces_01.astype(np.float32))

        # use the hand-tagged scene hint (open_clip not installed in this env)
        scene = scene_hint
        active = SCENE_TO_SCALE_HEAD[scene]
        skip = {h for h in SCENE_TO_SCALE_HEAD.values() if h != active}

        with torch.inference_mode():
            pred = pager(rgb_cubemap, dtype=torch.float16, skip_heads=skip)
        sky = pred["sky"][0]; log_scale = pred.get("scale", None)
        raw_depth, viz_depth = prepare_depth_for_logging(
            pager, pred["depth"][0], sky, (ERP_H, ERP_W), cmap, log_scale=log_scale)
        _, viz_normals = prepare_normals_for_logging(
            pager, pred["normals"][0], sky, (ERP_H, ERP_W))

        raw_depth = np.squeeze(raw_depth).astype(np.float32)
        viz_depth = np.transpose(viz_depth, (1, 2, 0))     # (H,W,3) uint8
        viz_normals = np.transpose(viz_normals, (1, 2, 0)) # (H,W,3) uint8

        # sky mask as a (H,W) float in [0,1]
        sky_prob = torch.sigmoid(sky).float().cpu().numpy()
        # sky head is in cubemap space — re-stitch through the same pipeline shape: rebuild a (H,W) sky.
        # Simpler proxy: derive from raw_depth being clipped at MAX_DEPTH (which is where sky is filled).
        # Both work for video purposes; here we'll use a depth-based mask which is already ERP-aligned.
        sky_mask = (raw_depth >= pager.MAX_DEPTH * 0.999).astype(np.float32)

        # point cloud (decimated)
        edge_mask = compute_edge_mask(raw_depth, rel_thresh=0.002)
        s = 2
        d_ds = raw_depth[::s, ::s]
        rgb_ds = input_rgb[::s, ::s].astype(np.float32) / 255.0
        norm_ds = viz_normals[::s, ::s].astype(np.float32) / 255.0
        edge_ds = edge_mask[::s, ::s]
        xyz = erp_to_pointcloud(torch.from_numpy(d_ds)).permute(1, 2, 0).numpy()
        keep = (d_ds > 0) & np.asarray(edge_ds, dtype=bool) & (d_ds < pager.MAX_DEPTH * 0.999)
        points = xyz[keep].astype(np.float32)
        rgb_pts = np.clip(rgb_ds[keep], 0, 1).astype(np.float32)
        norm_pts = np.clip(norm_ds[keep], 0, 1).astype(np.float32)

        # save everything
        img_pil.save(out / "rgb.png")
        Image.fromarray(viz_depth).save(out / "depth_viz.png")
        Image.fromarray(viz_normals).save(out / "normals_viz.png")
        Image.fromarray((sky_mask * 255).astype(np.uint8)).save(out / "sky_mask.png")
        np.save(out / "depth_raw.npy", raw_depth)
        np.save(out / "points_xyz.npy", points)
        np.save(out / "points_rgb.npy", rgb_pts)
        np.save(out / "points_normals.npy", norm_pts)
        # also save the classifier verdict
        (out / "scene.txt").write_text(f"{scene}\n")
        print(f"  [ok] {stem}: scene={scene}, depth={raw_depth.min():.2f}-{raw_depth.max():.2f}m, "
              f"pts={len(points):,}")


if __name__ == "__main__":
    main()
