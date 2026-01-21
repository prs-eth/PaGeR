import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from struct import unpack
from torch.utils.data import Dataset
from src.utils.geometry_utils import roll_augment, erp_to_cubemap

class Replica360_4K(Dataset):
    
    HEIGHT, WIDTH = 2048, 4096
    def __init__(self, data_path, split, training=False, log_depth=False, data_augmentation=False, 
                 scenes=None, debug=False):
        self.data_path = data_path / "Replica360_4K"
        self.training = training
        self.log_depth = log_depth
        self.data_augmentation = data_augmentation
        self.debug = debug
        self.data_paths = []
        
        self.load_data()

        if self.debug:
            self.data_paths = self.data_paths[:100]

        self.set_depth_ranges()


    def set_depth_ranges(self):
        self.MIN_DEPTH = 1e-2
        self.MAX_DEPTH = 75.0

        self.LOG_MIN_DEPTH = np.log(self.MIN_DEPTH)
        self.DEPTH_RANGE = self.MAX_DEPTH - self.MIN_DEPTH
        self.LOG_DEPTH_RANGE = np.log(self.MAX_DEPTH) - np.log(self.MIN_DEPTH)


    def load_data(self):
        scenes_list = [f for f in self.data_path.iterdir() if f.is_dir()]

        for folder in scenes_list:
            self.load_samples(self.data_path /  folder)
    

    def load_samples(self, folder):
        for sample in folder.iterdir():
            if sample.name.endswith(".jpg"):
                self.data_paths.append(sample)

      
    # from https://github.com/manurare/360monodepth/blob/main/code/python/src/utility/depthmap_utils.py
    def read_dpt(self, dpt_file_path):    
        """read depth map from *.dpt file.
        :param dpt_file_path: the dpt file path
        :type dpt_file_path: str
        :return: depth map data
        :rtype: numpy
        """

        TAG_FLOAT = 202021.25  # check for this when READING the file

        dpt_file_path = Path(dpt_file_path)
        ext = dpt_file_path.suffix

        assert len(ext) > 0, "readFlowFile: extension required in fname %s" % dpt_file_path
        assert ext == ".dpt", exit(
            "readFlowFile: fname %s should have extension " ".flo" "" % dpt_file_path
        )

        fid = None
        try:
            fid = open(dpt_file_path, "rb")
        except IOError:
            print("readFlowFile: could not open %s", dpt_file_path)

        tag = unpack("f", fid.read(4))[0]
        width = unpack("i", fid.read(4))[0]
        height = unpack("i", fid.read(4))[0]

        assert tag == TAG_FLOAT, (
            "readFlowFile(%s): wrong tag (possibly due to big-endian machine?)" % dpt_file_path
        )
        assert 0 < width and width < 100000, "readFlowFile(%s): illegal width %d" % (
            dpt_file_path,
            width,
        )
        assert 0 < height and height < 100000, "readFlowFile(%s): illegal height %d" % (
            dpt_file_path,
            height,
        )

        # arrange into matrix form
        depth_data = np.fromfile(fid, np.float32)
        depth_data = depth_data.reshape(height, width)
        fid.close()

        return depth_data
    

    def process_rgb(self, rgb_img_pil: Image.Image, shift_ratio=0.0):
        rgb_np = np.array(rgb_img_pil) / 255.0
        rgb_np = rgb_np * 2 - 1
        rgb_np = roll_augment(rgb_np, shift_ratio * self.WIDTH)
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
        rgb_cubemap_tensor = erp_to_cubemap(rgb_tensor)
        return rgb_tensor.type(torch.float32), rgb_cubemap_tensor.type(torch.float32)


    def process_depth(self, depth_np: np.ndarray, mask, shift_ratio=0.0):
        depth_np = np.clip(depth_np, a_min=self.MIN_DEPTH, a_max=self.MAX_DEPTH).astype(np.float32)
        depth_np = roll_augment(depth_np, shift_ratio * self.WIDTH)

        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
        
        if self.training:
            if self.log_depth:
                depth_tensor = (torch.log(depth_tensor) - self.LOG_MIN_DEPTH) / self.LOG_DEPTH_RANGE
            else:
                min_depth = torch.quantile(depth_tensor[mask], 0.02)
                max_depth = torch.quantile(depth_tensor[mask], 0.98)
                depth_tensor = (depth_tensor - min_depth) / (max_depth - min_depth)
                depth_tensor = torch.clamp(depth_tensor, 0, 1)
            depth_tensor = (depth_tensor * 2) - 1

        depth_cubemap_tensor = erp_to_cubemap(depth_tensor)

        depth_tensor = depth_tensor.repeat(3, 1, 1)
        depth_cubemap_tensor = depth_cubemap_tensor.repeat(1, 3, 1, 1)
        return depth_tensor, depth_cubemap_tensor
    

    def process_mask(self, mask_np: np.ndarray, shift_ratio=0.0):
        mask_np = (mask_np > 0).astype(bool)
        mask_np = roll_augment(mask_np, shift_ratio * self.WIDTH)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        mask_cubemap_tensor = erp_to_cubemap(mask_tensor.float()) > 0.99
        return mask_tensor, mask_cubemap_tensor


    def __len__(self):
        return len(self.data_paths)
    

    def __getitem__(self, idx):
        rgb_path = self.data_paths[idx]
        depth_path = rgb_path.with_name(rgb_path.name.replace("_rgb_pano.jpg", "_depth_pano.dpt"))
        mask_path = rgb_path.with_name(rgb_path.name.replace("_rgb_pano.jpg", "_mask_pano.png"))

        rgb_image = Image.open(rgb_path).convert("RGB").resize((self.WIDTH, self.HEIGHT), Image.LANCZOS)
        depth_image = self.read_dpt(depth_path)
        depth = np.array(depth_image)
        depth = cv2.resize(depth, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
        mask = Image.open(mask_path).convert("L").resize((self.WIDTH, self.HEIGHT), Image.NEAREST)
        mask = np.array(mask)
        
        if self.data_augmentation:
            shift_ratio = np.random.uniform(0, 1)
        else:
            shift_ratio = 0
        rgb_tensor, rgb_cubemap_tensor = self.process_rgb(rgb_image, shift_ratio)
        mask_tensor, mask_cubemap_tensor = self.process_mask(mask, shift_ratio)
        depth_tensor, depth_cubemap_tensor = self.process_depth(depth, mask, shift_ratio)

        id = rgb_path.parts[6] + '_' + rgb_path.parts[7][:4]
        return {
            "id": id,
            "rgb": rgb_tensor,
            "rgb_cubemap": rgb_cubemap_tensor,
            "depth": depth_tensor,
            "mask": mask_tensor,
            "depth_cubemap": depth_cubemap_tensor,
            "mask_cubemap": mask_cubemap_tensor,
        }