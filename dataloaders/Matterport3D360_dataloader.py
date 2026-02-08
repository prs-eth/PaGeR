import torch
import numpy as np
from pathlib import Path
from PIL import Image
from struct import unpack
from torch.utils.data import Dataset
from src.utils.geometry_utils import roll_augment, erp_to_cubemap

class Matterport3D360(Dataset):
    HEIGHT, WIDTH = 1024, 2048
   
    def __init__(self, data_path, split, training=False, log_depth=False, data_augmentation=False, 
                 scenes=None, debug=False):
        self.data_path = data_path / "Matterport3D360"
        self.split = split
        self.training = training
        self.log_depth = log_depth
        self.data_augmentation = data_augmentation
        self.debug = debug
        self.rgb_path = []
        self.gt_path = []
        self.vis_depth_path = []
        self.data_map = {}         
        tiny_val = False

        if self.split == "tiny_val":
            self.split = "val"
            tiny_val = True

        self.load_data(self.split)
        if self.debug:
            self.rgb_path = self.rgb_path[:100]
            self.gt_path = self.gt_path[:100]
            self.vis_depth_path = self.vis_depth_path[:100]

        if tiny_val:
            if self.debug:
                self.rgb_path = self.rgb_path[::3]
                self.gt_path = self.gt_path[::3]
                self.vis_depth_path = self.vis_depth_path[::3]
            else:
                self.rgb_path = self.rgb_path[::10]
                self.gt_path = self.gt_path[::10]
                self.vis_depth_path = self.vis_depth_path[::10]
        
        self.set_depth_ranges()
        assert len(self.rgb_path) == len(self.gt_path) == len(self.vis_depth_path), \
        f"Number of samples in rgb_path: {len(self.rgb_path)}, gt_path: {len(self.gt_path)}, \
        vis_depth_path: {len(self.vis_depth_path)} are not equal."


    def set_depth_ranges(self):
        self.MIN_DEPTH = 1e-2
        self.MAX_DEPTH = 75.0

        self.LOG_MIN_DEPTH = np.log(self.MIN_DEPTH)
        self.DEPTH_RANGE = self.MAX_DEPTH - self.MIN_DEPTH
        self.LOG_DEPTH_RANGE = np.log(self.MAX_DEPTH) - np.log(self.MIN_DEPTH)


    def load_data(self, split):
        scenes_list = []
        split_path = self.data_path / f"scenes_{split}.txt"
        with open(split_path, "r") as f:
            scenes_list = f.readlines()
        scenes_list = [scene.strip() for scene in scenes_list]
        
        dirs = [d.name for d in self.data_path.iterdir() if d.is_dir()]
        for folder in dirs:
            data_root = self.data_path / folder / "data"
            if not data_root.exists():
                continue

            for partition_path in data_root.iterdir():
                if not partition_path.is_dir():
                    continue
                scene_name = partition_path.name
                if scene_name not in scenes_list:
                    continue
                self.load_samples(partition_path)
    
    def load_samples(self, folder):
        folder = Path(folder)
        for sample_path in folder.iterdir():
            sample = sample_path.name
            key = sample.split("_")[0]
            if key not in self.data_map:
                self.data_map[key] = {
                    "rgb": None,
                    "depth": None,
                    "vis_depth": None
                }
            if sample.endswith("_rgb.png"):
                self.data_map[key]["rgb"] = sample_path
            elif sample.endswith(".dpt"):
                self.data_map[key]["depth"] = sample_path
            elif sample.endswith("_vis.png"):
                self.data_map[key]["vis_depth"] = sample_path
        
        self.rgb_path = []
        self.gt_path = []
        self.vis_depth_path = []
        for key in self.data_map.keys():
            self.rgb_path.append(self.data_map[key]["rgb"])
            self.gt_path.append(self.data_map[key]["depth"])
            self.vis_depth_path.append(self.data_map[key]["vis_depth"])
        

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


    def process_depth(self, depth_np: np.ndarray, shift_ratio=0.0):
        depth_np = np.clip(depth_np, a_min=self.MIN_DEPTH, a_max=self.MAX_DEPTH).astype(np.float32)
        depth_np = roll_augment(depth_np, shift_ratio * self.WIDTH)

        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
        mask_tensor = (depth_tensor < (0.99 * self.MAX_DEPTH)) & (depth_tensor > 1.01 * self.MIN_DEPTH)
        # mask out top and bottom 14% of the image
        mask_tensor[0, :int(0.14*self.HEIGHT), :] = 0
        mask_tensor[0, int(0.86*self.HEIGHT):, :] = 0

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
        return len(self.rgb_path)
    

    def __getitem__(self, idx):
        rgb_path = self.rgb_path[idx]
        depth_path = self.gt_path[idx]
        vis_depth_path = self.vis_depth_path[idx]

        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = self.read_dpt(depth_path)
        depth = np.array(depth_image)
        vis_depth = Image.open(vis_depth_path).convert("RGB")
        vis_depth = np.array(vis_depth)

        if self.data_augmentation:
            shift_ratio = np.random.uniform(0, 1)
        else:
            shift_ratio = 0

        rgb_tensor, rgb_cubemap_tensor = self.process_rgb(rgb_image, shift_ratio)
        depth_tensor, depth_cubemap_tensor, mask_tensor, mask_cubemap_tensor = self.process_depth(depth, shift_ratio)

        id = rgb_path.name[:-8]
        return {
            "id": id,
            "rgb": rgb_tensor,
            "rgb_cubemap": rgb_cubemap_tensor,
            "depth": depth_tensor,
            "mask": mask_tensor,
            "depth_cubemap": depth_cubemap_tensor,
            "mask_cubemap": mask_cubemap_tensor,
        }