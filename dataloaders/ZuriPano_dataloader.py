"""ZüriPano: real-world outdoor LiDAR panoramas, fetched from HuggingFace prs-eth/ZuriPano."""

from __future__ import annotations

import numpy as np
from datasets import load_dataset
from PIL import Image

from dataloaders._base import PanoDataset

# 16-bit PNG over [0, 200] m (~3 mm precision).
DEPTH_SCALE_M_PER_UNIT = 200.0 / 65535.0


class ZuriPano(PanoDataset):
    HEIGHT, WIDTH = 2048, 4096
    HF_REPO = "prs-eth/ZuriPano"
    HF_SPLIT = "test"

    def __init__(self, data_path=None, face_size: int = 504, cube_fov: float = 90.0):
        # data_path is unused (HF-backed); kept for API parity.
        self.face_size = int(face_size)
        self.cube_fov = float(cube_fov)
        self.hf = load_dataset(self.HF_REPO, split=self.HF_SPLIT)
        self.samples = list(range(len(self.hf)))

    def _scan(self):
        return self.samples

    def __getitem__(self, idx):
        sample = self.hf[idx]

        rgb_pil = sample["rgb"].convert("RGB").resize(
            (self.WIDTH, self.HEIGHT), Image.LANCZOS,
        )
        rgb_tensor, rgb_cubemap = self._process_rgb(rgb_pil)

        depth_np = np.asarray(sample["depth"], dtype=np.float32) * DEPTH_SCALE_M_PER_UNIT
        depth_np = self._resize_nn(depth_np)
        scene_mask = np.asarray(sample["mask"], dtype=np.uint8) > 0
        scene_mask = self._resize_nn(scene_mask.astype(np.uint8)).astype(bool)
        depth_tensor, mask_tensor = self._depth_mask_tensors(depth_np, scene_mask)

        return {
            "id": sample["id"].strip().replace(" ", "_"),
            "rgb": rgb_tensor,
            "rgb_cubemap": rgb_cubemap,
            "depth": depth_tensor,
            "mask": mask_tensor,
        }
