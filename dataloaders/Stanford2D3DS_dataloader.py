import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from src.utils.geometry_utils import roll_augment, erp_to_cubemap

class Stanford2D3DS(Dataset):
    
    HEIGHT, WIDTH = 2048, 4096
    def __init__(self, data_path, training=False, log_depth=False, data_augmentation=False, split=None, 
                 scenes=None, debug=False):
        self.data_path = data_path / "Stanford2D3DS"
        self.training = training
        self.log_depth = log_depth
        self.data_augmentation = data_augmentation
        self.debug = debug
        self.data_path = []
        self.fetch_data_paths(self.data_path)
        
        if self.debug:
            self.data_path = self.data_path[:100]

        self.set_depth_ranges()


    def fetch_data_paths(self, data_path):
        for path in data_path.rglob("*_rgb.png"):
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
        depth_np = roll_augment(depth_np, shift_ratio * self.WIDTH)

        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
        mask_tensor = (depth_tensor < (0.99 * self.MAX_DEPTH)) & (depth_tensor > 1.01 * self.MIN_DEPTH)
        # mask out top and bottom 15% of the image
        mask_tensor[0, :int(0.15*self.HEIGHT), :] = 0
        mask_tensor[0, int(0.85*self.HEIGHT):, :] = 0

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
        depth_path = rgb_path.parent / (rgb_path.name.replace("_rgb.png", "_depth.png"))
        depth_path = depth_path.parent.parent / "depth" / depth_path.name

        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path)
        depth = np.array(depth_image)* 128.0 / 65535.0

        if self.data_augmentation:
            shift_ratio = np.random.uniform(0, 1)
        else:
            shift_ratio = 0

        rgb_tensor, rgb_cubemap_tensor = self.process_rgb(rgb_image, shift_ratio)
        depth_tensor, depth_cubemap_tensor, mask_tensor, mask_cubemap_tensor = self.process_depth(depth, shift_ratio)

        id = rgb_path.name[:-37]
        return {
            "id": id,
            "rgb": rgb_tensor,
            "rgb_cubemap": rgb_cubemap_tensor,
            "depth": depth_tensor,
            "mask": mask_tensor,
            "depth_cubemap": depth_cubemap_tensor,
            "mask_cubemap": mask_cubemap_tensor,
        }