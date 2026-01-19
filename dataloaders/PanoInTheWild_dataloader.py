from pathlib import Path
import numpy as np
from PIL import Image
import torch
from itertools import chain
from torch.utils.data import Dataset
from util.geometry_utils import erp_to_cubemap

class PanoInTheWild(Dataset):
    HEIGHT, WIDTH = 1024, 2048

    def __init__(self,
                 data_path: Path,
                 log_depth: bool = True,
                 split: str = None,
                 data_augmentation = None,
                 training = None,
                 scenes = None,
                 debug=None):
        super().__init__()
        self.data_path = data_path / "PanoInTheWild"

        self._index = [str(p.resolve()) for p in chain(
            self.data_path.glob("*.jpg"),
            self.data_path.glob("*.png")
        )]
        self.set_depth_ranges()


    def set_depth_ranges(self):
        self.MIN_DEPTH = 1e-2
        self.MAX_DEPTH = 75.0

        self.LOG_MIN_DEPTH = np.log(self.MIN_DEPTH)
        self.DEPTH_RANGE = self.MAX_DEPTH - self.MIN_DEPTH
        self.LOG_DEPTH_RANGE = np.log(self.MAX_DEPTH) - np.log(self.MIN_DEPTH)


    def __len__(self):
        return len(self._index)

    def process_rgb(self, rgb_img_pil: Image.Image):
        rgb_np = np.array(rgb_img_pil) / 255.0
        rgb_np = rgb_np * 2 - 1

        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
        rgb_cubemap_tensor = erp_to_cubemap(rgb_tensor)
        return rgb_tensor, rgb_cubemap_tensor


    def build_sample(self, rgb_img_path):
        rgb_img_pil = Image.open(rgb_img_path).convert("RGB")
        rgb_img_pil = rgb_img_pil.resize((self.WIDTH, self.HEIGHT), Image.LANCZOS)
        rgb_tensor, rgb_cubemap = self.process_rgb(rgb_img_pil)
        return {
            "rgb": rgb_tensor,
            "rgb_cubemap": rgb_cubemap,
        }

    def __getitem__(self, i: int):
        sample = self.build_sample(self._index[i])

        sample["id"] = self._index[i].rpartition("/")[-1].split(".")[0]
        return sample
