import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from src.utils.geometry_utils import roll_augment, roll_normal, erp_to_cubemap

class Structured3D(Dataset):
    HEIGHT, WIDTH = 512, 1024
    def __init__(self, data_path, split, training=False, log_depth=False, data_augmentation=False, 
                 scenes=None, debug=False):
        self.data_path = data_path / "Structured3D"
        self.training = training
        self.split = split
        self.log_depth = log_depth
        self.data_augmentation = data_augmentation
        self.debug = debug
        self.rgb_path = []
        self.depth_path = []
        self.normal_path = []
        tiny_val = False

        if self.split == "tiny_val":
            self.split = "val"
            tiny_val = True

        self.fetch_data_paths(self.data_path, self.split)
        
        if self.debug:
            self.rgb_path = self.rgb_path[:100]
            self.depth_path = self.depth_path[:100]
            self.normal_path = self.normal_path[:100]

        if tiny_val:
            if self.debug:
                self.rgb_path = self.rgb_path[::5]
                self.depth_path = self.depth_path[::5]
                self.normal_path = self.normal_path[::5]
            else:
                self.rgb_path = self.rgb_path[::50]
                self.depth_path = self.depth_path[::50]
                self.normal_path = self.normal_path[::50]

        self.set_depth_ranges()


    def fetch_data_paths(self, data_path: Path, split: str):
        if split == "train":
            scene_ids = range(0, 3000)
        elif split == "val":
            scene_ids = range(3000, 3250)
        elif split == "test":
            scene_ids = range(3250, 3500)
        else:
            raise ValueError(f"Unknown split option: {split}")

        for sid in scene_ids:
            scene_dir = data_path / f"scene_{sid:05d}" / "2D_rendering"
            if not scene_dir.is_dir():
                continue

            for img_path in scene_dir.glob("*/panorama/full/rgb_rawlight.png"):
                if (img_path.parent / "depth.png").exists() and (img_path.parent / "normal.png").exists():
                    self.rgb_path.append(img_path)
                    self.depth_path.append(img_path.parent / "depth.png")
                    self.normal_path.append(img_path.parent / "normal.png")


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
                min_depth = torch.quantile(depth_tensor[mask_tensor], 0.001)
                max_depth = torch.quantile(depth_tensor[mask_tensor], 0.999)
                depth_tensor = (depth_tensor - min_depth) / (max_depth - min_depth)
                depth_tensor = torch.clamp(depth_tensor, 0, 1)
            depth_tensor = (depth_tensor * 2) - 1

        depth_cubemap_tensor = erp_to_cubemap(depth_tensor)
        mask_cubemap_tensor = erp_to_cubemap(mask_tensor.float()) > 0.99

        depth_tensor = depth_tensor.repeat(3, 1, 1)
        depth_cubemap_tensor = depth_cubemap_tensor.repeat(1, 3, 1, 1)
        return depth_tensor, depth_cubemap_tensor, mask_tensor, mask_cubemap_tensor


    def process_normal(self, normal_np: np.ndarray, shift_ratio=0.0):
        normal_np = normal_np.astype(np.float32)
        normal_np = roll_augment(normal_np, shift_ratio * self.WIDTH)
        normal_np = roll_normal(normal_np, shift_ratio * self.WIDTH)
        normal_tensor = torch.from_numpy(normal_np).permute(2, 0, 1)
        normal_cubemap_tensor = erp_to_cubemap(normal_tensor)
        return normal_tensor, normal_cubemap_tensor
    

    def __len__(self):
        return len(self.rgb_path)
    

    def __getitem__(self, idx):
        rgb_path = self.rgb_path[idx]
        depth_path = self.depth_path[idx]
        normal_path = self.normal_path[idx]
        try:
            rgb_image = Image.open(rgb_path).convert("RGB")
            depth_image = Image.open(depth_path)
            depth = np.array(depth_image) / 1000.0

            normal_image = Image.open(normal_path)
            normal = np.array(normal_image) / 128.0 - 1.0

            if self.data_augmentation:
                shift_ratio = np.random.uniform(0, 1)
            else:
                shift_ratio = 0

            rgb_tensor, rgb_cubemap_tensor = self.process_rgb(rgb_image, shift_ratio)
            depth_tensor, depth_cubemap_tensor, mask_tensor, mask_cubemap_tensor = self.process_depth(depth, shift_ratio)
            normal_tensor, normal_cubemap_tensor = self.process_normal(normal, shift_ratio)

            id = rgb_path.parts[6]+ "_" + rgb_path.parts[8]

        except Exception as e:
            print(f"Error loading data on path {rgb_path}: {e}")
            id = 0
            rgb_tensor = torch.zeros(3, self.HEIGHT, self.WIDTH)
            rgb_cubemap_tensor = torch.zeros(6, 3, self.HEIGHT//2, self.WIDTH//2)
            depth_tensor = torch.zeros(3, self.HEIGHT, self.WIDTH)
            depth_cubemap_tensor = torch.zeros(6, 3, self.HEIGHT//2, self.WIDTH//2)
            mask_tensor = torch.zeros(1, self.HEIGHT, self.WIDTH).bool()
            mask_cubemap_tensor = torch.zeros(6, 1, self.HEIGHT//2, self.WIDTH//2).bool()
            normal_tensor = torch.zeros(3, self.HEIGHT, self.WIDTH)
            normal_cubemap_tensor = torch.zeros(6, 3, self.HEIGHT//2, self.WIDTH//2)
        return {
            "id": id,
            "rgb": rgb_tensor,
            "rgb_cubemap": rgb_cubemap_tensor,
            "depth": depth_tensor,
            "mask": mask_tensor,
            "depth_cubemap": depth_cubemap_tensor,
            "mask_cubemap": mask_cubemap_tensor,
            "normal": normal_tensor,
            "normal_cubemap": normal_cubemap_tensor
        }