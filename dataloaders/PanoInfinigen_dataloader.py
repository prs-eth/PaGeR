import io
import torch
import lmdb
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
from src.utils.geometry_utils import roll_augment, roll_normals, erp_to_cubemap

class PanoInfinigen(Dataset):
    HEIGHT, WIDTH = 2160, 3840

    def __init__(self,
                 data_path: Path,
                 training: bool = True,
                 split: str = "train",
                 scenes: str = "both",
                 log_depth: bool = False,
                 data_augmentation: bool = False,
                 debug: bool = False,
                 readahead: bool = True):
        super().__init__()
        self.data_path = data_path / "PanoInfinigen"
        self.training = training 
        self.split, self.scenes = split, scenes
        self.log_depth = log_depth
        self.data_augmentation = data_augmentation
        self.debug = debug
        self.readahead = readahead

        tiny_val = False
        if self.split == "tiny_val":
            self.split = "val"
            tiny_val = True

        indoor_root  = self.data_path / "indoor"  / self.split
        outdoor_root = self.data_path / "outdoor" / self.split
        if scenes == "both":
            roots = [p for p in [indoor_root, outdoor_root] if p.exists()]
        elif scenes == "indoor":
            roots = [indoor_root]
        elif scenes == "outdoor":
            roots = [outdoor_root]
        else:
            raise ValueError(f"Unknown scenes option: {scenes}")

        if not roots:
            raise ValueError(f"No LMDB roots found for split={self.split}, scenes={scenes} in {self.data_path}")
        self.roots: List[Path] = roots

        view_index: List[Tuple[Path, int, str, int]] = []
        for root in self.roots:
            idx_path = root / "index.npy"
            if not idx_path.exists():
                continue
            arr = np.load(idx_path, allow_pickle=True).tolist()  
            if self.debug:
                arr = arr[: min(23, len(arr))]
            if tiny_val:
                if self.debug:
                    arr = arr[::5]
                else:
                    arr = arr[::50]
            for shard_idx, scene_id, num_views in arr:
                n = int(num_views)
                for v in range(n):
                    view_index.append((root, int(shard_idx), str(scene_id), v))
        if not view_index:
            raise ValueError("Empty per-view index — check your index.npy files.")
        self._index = view_index

        self._env_cache = {}
        self.set_depth_ranges()

    def __len__(self):
        return len(self._index)

    def set_depth_ranges(self):
        self.MIN_DEPTH = 1e-2
        self.MAX_DEPTH = 15.0 if self.scenes == "indoor" else 75.0

        self.LOG_MIN_DEPTH = np.log(self.MIN_DEPTH)
        self.DEPTH_RANGE = self.MAX_DEPTH - self.MIN_DEPTH
        self.LOG_DEPTH_RANGE = np.log(self.MAX_DEPTH) - np.log(self.MIN_DEPTH)
    
    def process_rgb(self, rgb_img_pil: Image.Image, shift_ratio=0.0):
        rgb_np = np.array(rgb_img_pil) / 255.0
        rgb_np = rgb_np * 2 - 1
        rgb_np = roll_augment(rgb_np, shift_ratio * self.WIDTH)
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
        rgb_cubemap_tensor = erp_to_cubemap(rgb_tensor)
        return rgb_tensor.type(torch.float32), rgb_cubemap_tensor.type(torch.float32)

    def process_depth(self, depth_np: np.ndarray, shift_ratio=0.0):
        depth_np = np.clip(depth_np, a_min=self.MIN_DEPTH, a_max=self.MAX_DEPTH).astype(np.float32)
        depth_np = roll_augment(depth_np, shift_ratio * self.WIDTH)

        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
        mask_tensor = depth_tensor < (0.99 * self.MAX_DEPTH)
        if self.training:
            if self.log_depth:
                depth_tensor = (torch.log(depth_tensor) - self.LOG_MIN_DEPTH) / self.LOG_DEPTH_RANGE
            else:
                min_depth = torch.quantile(depth_tensor[mask_tensor], 0.02)
                max_depth = torch.quantile(depth_tensor[mask_tensor], 0.98)
                depth_tensor = (depth_tensor - min_depth) / (max_depth - min_depth)
                depth_tensor = torch.clamp(depth_tensor, 0, 1)
            depth_tensor = (depth_tensor * 2) - 1

        depth_cubemap_tensor = erp_to_cubemap(depth_tensor)
        mask_cubemap_tensor = erp_to_cubemap(mask_tensor.float()) > 0.99

        depth_tensor = depth_tensor.repeat(3, 1, 1)
        depth_cubemap_tensor = depth_cubemap_tensor.repeat(1, 3, 1, 1)
        return depth_tensor, depth_cubemap_tensor, mask_tensor, mask_cubemap_tensor

    def process_normals(self, normals_np: np.ndarray, shift_ratio=0.0):
        normals_np = normals_np.astype(np.float32)
        normals_np = roll_augment(normals_np, shift_ratio * self.WIDTH)
        normals_np = roll_normals(normals_np, shift_ratio * self.WIDTH)
        normals_tensor = torch.from_numpy(normals_np).permute(2, 0, 1)
        normals_cubemap_tensor = erp_to_cubemap(normals_tensor)
        return normals_tensor, normals_cubemap_tensor
    
    def build_sample(self, rgb_img_pil, depth_np, normals_np):
        if self.data_augmentation:
            shift_ratio = np.random.uniform(0, 1)
        else:
            shift_ratio = 0
        rgb_tensor, rgb_cubemap_tensor = self.process_rgb(rgb_img_pil, shift_ratio)
        depth_tensor, depth_cubemap_tensor, mask_tensor, mask_cubemap_tensor = self.process_depth(depth_np, shift_ratio)
        normals_tensor, normals_cubemap_tensor = self.process_normals(normals_np, shift_ratio)
        return {
            "rgb": rgb_tensor,
            "rgb_cubemap": rgb_cubemap_tensor,
            "depth": depth_tensor,
            "mask": mask_tensor,
            "depth_cubemap": depth_cubemap_tensor,
            "mask_cubemap": mask_cubemap_tensor,
            "normals": normals_tensor,
            "normals_cubemap": normals_cubemap_tensor,
        }

    def _get_env(self, root: Path, shard_idx: int):
        key = (str(root), int(shard_idx))
        env = self._env_cache.get(key)
        if env is None:
            shard_dir = Path(root) / f"shard_{shard_idx:05d}.lmdb"
            env = lmdb.open(
                str(shard_dir),
                readonly=True, lock=False, subdir=True,
                readahead=self.readahead, max_readers=4096, meminit=False
            )
            self._env_cache[key] = env
        return env

    def _load_view(self, env, scene_id: str, view_idx: int):
        key_base = f"{scene_id}/{view_idx:04d}"
        with env.begin() as txn:
            rgb_bytes    = txn.get(f"{key_base}#rgb".encode())
            depth_bytes  = txn.get(f"{key_base}#depth".encode())
            normals_bytes = txn.get(f"{key_base}#normal".encode())
            if rgb_bytes is None or depth_bytes is None or normals_bytes is None:
                raise KeyError(f"Missing keys for {key_base}")
            rgb_img  = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
            depth_np = np.load(io.BytesIO(depth_bytes), allow_pickle=False)
            normals_np= np.load(io.BytesIO(normals_bytes), allow_pickle=False)
        return rgb_img, depth_np, normals_np

    def __getitem__(self, i: int):
        root, shard_idx, scene_id, view_idx = self._index[i]
        env = self._get_env(root, shard_idx)
        rgb_img, depth_np, normals_np = self._load_view(env, scene_id, view_idx)
        sample = self.build_sample(rgb_img, depth_np, normals_np)

        unique_id = f"{root.name}_shard{shard_idx:05d}_{scene_id}_view{view_idx:04d}"
        sample["id"] = unique_id
        return sample