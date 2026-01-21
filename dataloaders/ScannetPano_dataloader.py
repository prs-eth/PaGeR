import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from src.utils.geometry_utils import roll_augment, erp_to_cubemap

class ScannetPano(Dataset):
    HEIGHT, WIDTH = 1284, 2572
    def __init__(self, data_path, split, training=False, log_depth=False, data_augmentation=False, 
                 scenes=None, debug=False):
        self.data_path = data_path / "ScannetPano"
        self.training = training
        self.split = split
        self.log_depth = log_depth
        self.data_augmentation = data_augmentation
        self.debug = debug
        self.data_path = []
        
        tiny_val = False
        if self.split == 'test':
            self.split = 'val'
            split='val'
        if self.split == "tiny_val":
            self.split = "val"
            split="val"
            tiny_val = True
        split_file_path = self.data_path / f"./splits/nvs_sem_{split}.txt"

        self.fetch_data_paths(self.data_path, split_file_path)
        
        if self.debug:
            self.data_path = self.data_path[:100]

        if tiny_val:
            if self.debug:
                self.data_path = self.data_path[::5]
            else:
                self.data_path = self.data_path[::50]

        self.set_depth_ranges()


    def fetch_data_paths(self, data_path: Path, split_file_path: Path):
        with open(split_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                scene_id = line.strip()
                scene_path = data_path / f"data/{scene_id}/panocam/resized_images"
                for path in scene_path.rglob("*.jpg"):
                    self.data_path.append(path)

    def set_depth_ranges(self):
        self.MIN_DEPTH = 1e-2
        self.MAX_DEPTH = 75.0

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
        depth_np = roll_augment(depth_np[..., None], shift_ratio * self.WIDTH)[..., 0]

        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
        mask_tensor = (depth_tensor < (0.99 * self.MAX_DEPTH)) & (depth_tensor > 1.01 * self.MIN_DEPTH)

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

    def __len__(self):
        return len(self.data_path)
    

    def __getitem__(self, idx):
        rgb_path = self.data_path[idx]
        depth_path = rgb_path.parent.parent / "resized_depth" / (rgb_path.stem + ".png")

        rgb_image = Image.open(rgb_path).convert("RGB").resize((self.WIDTH, self.HEIGHT), Image.LANCZOS)
        depth_image = Image.open(depth_path).resize((self.WIDTH, self.HEIGHT), Image.NEAREST)
        depth = np.array(depth_image) / 1000.0

        if self.data_augmentation:
            shift_ratio = np.random.uniform(0, 1)
        else:
            shift_ratio = 0

        rgb_tensor, rgb_cubemap_tensor = self.process_rgb(rgb_image, shift_ratio)
        depth_tensor, depth_cubemap_tensor, mask_tensor, mask_cubemap_tensor = self.process_depth(depth, shift_ratio)

        id = rgb_path.parts[7]+'_'+rgb_path.stem
        return {
            "id": id,
            "rgb": rgb_tensor,
            "rgb_cubemap": rgb_cubemap_tensor,
            "depth": depth_tensor,
            "mask": mask_tensor,
            "depth_cubemap": depth_cubemap_tensor,
            "mask_cubemap": mask_cubemap_tensor,
        }