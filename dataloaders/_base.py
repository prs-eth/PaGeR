"""Shared base for PaGeR evaluation datasets.

Subclasses set class attrs (HEIGHT/WIDTH/SUBDIR/MIN_DEPTH/MAX_DEPTH/POLE_CROP_*) and
implement _scan() and _load_depth(); optionally _load_normals() / _scene_mask().
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.utils.geometry_utils import erp_to_cubemap

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class PanoDataset(Dataset):
    HEIGHT: int = 1024
    WIDTH: int = 2048
    SUBDIR: str = ""
    MIN_DEPTH: float = 1e-2
    MAX_DEPTH: float = 75.0
    POLE_CROP_FRAC: float = 0.0
    POLE_CROP_RGB: bool = False

    def __init__(self, data_path, face_size: int = 504, cube_fov: float = 90.0):
        self.data_path = Path(data_path) / self.SUBDIR
        self.face_size = int(face_size)
        self.cube_fov = float(cube_fov)
        self.samples = self._scan()

    def _scan(self) -> list[dict]:
        raise NotImplementedError

    def _load_depth(self, entry: dict) -> np.ndarray:
        raise NotImplementedError

    def _load_normals(self, entry: dict):
        return None

    def _scene_mask(self, entry: dict, depth: np.ndarray):
        return None

    def _resize_nn(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[:2] == (self.HEIGHT, self.WIDTH):
            return arr
        return cv2.resize(arr, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)

    def _pole_band(self) -> int:
        return int(self.POLE_CROP_FRAC * self.HEIGHT)

    def _process_rgb(self, rgb_pil: Image.Image):
        rgb_np = np.array(rgb_pil).astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
        if self.POLE_CROP_RGB and self.POLE_CROP_FRAC > 0:
            crop = self._pole_band()
            rgb_tensor[:, :crop] = 0
            rgb_tensor[:, -crop:] = 0
        rgb_cubemap = erp_to_cubemap(
            NORMALIZE(rgb_tensor.clone()), face_w=self.face_size, fov=self.cube_fov,
        )
        return rgb_tensor.float(), rgb_cubemap.float()

    def _depth_mask_tensors(self, depth_np: np.ndarray, scene_mask=None):
        depth_np = np.clip(depth_np, self.MIN_DEPTH, self.MAX_DEPTH).astype(np.float32)
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
        mask_tensor = (depth_tensor < 0.99 * self.MAX_DEPTH) & \
                      (depth_tensor > 1.01 * self.MIN_DEPTH)
        if scene_mask is not None:
            mask_tensor = mask_tensor & torch.from_numpy(scene_mask).unsqueeze(0)
        if self.POLE_CROP_FRAC > 0:
            crop = self._pole_band()
            mask_tensor[:, :crop] = 0
            mask_tensor[:, -crop:] = 0
        return depth_tensor, mask_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        rgb_pil = Image.open(entry["rgb"]).convert("RGB").resize(
            (self.WIDTH, self.HEIGHT), Image.LANCZOS,
        )
        rgb_tensor, rgb_cubemap = self._process_rgb(rgb_pil)

        depth_np = self._resize_nn(self._load_depth(entry).astype(np.float32))
        scene_mask = self._scene_mask(entry, depth_np)
        if scene_mask is not None:
            scene_mask = self._resize_nn(scene_mask.astype(np.uint8)).astype(bool)
        depth_tensor, mask_tensor = self._depth_mask_tensors(depth_np, scene_mask)

        out = {
            "id": entry["id"],
            "rgb": rgb_tensor,
            "rgb_cubemap": rgb_cubemap,
            "depth": depth_tensor,
            "mask": mask_tensor,
        }
        normals_np = self._load_normals(entry)
        if normals_np is not None:
            normals_np = self._resize_nn(normals_np.astype(np.float32))
            out["normals"] = torch.from_numpy(normals_np).permute(2, 0, 1).float()
        return out
