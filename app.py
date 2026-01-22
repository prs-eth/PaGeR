import sys
import gc
import torch
import numpy as np
import argparse
import logging
import gradio as gr
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from tempfile import NamedTemporaryFile
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from src.pager import Pager
from src.utils.geometry_utils import compute_edge_mask, erp_to_point_cloud_glb, erp_to_cubemap
from src.utils.utils import prepare_image_for_logging

MIN_DEPTH = np.log(1e-2)
DEPTH_RANGE = np.log(75.0)
POINTCLOUD_DOWNSAMPLE_FACTOR = 2
MAX_POINTCLOUD_POINTS = 200000
EXAMPLES_DIR = Path(__file__).parent / "examples"
EXAMPLE_IMAGES = [
    str(p)
    for p in sorted(EXAMPLES_DIR.glob("*"))
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for panorama depth estimation using diffusion models.")

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="A seed for reproducibility."
    )

    parser.add_argument(
        "--depth_checkpoint_path",
        default="prs-eth/PaGeR-depth",
        type=str,
        help="UNet checkpoint to load.",
    )

    parser.add_argument(
        "--normals_checkpoint_path",
        default="prs-eth/PaGeR-normals",
        type=str,
        help="UNet checkpoint to load.",
    )

    parser.add_argument(
        "--enable_xformers", 
        action="store_true", 
        help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    return args

def _release_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def generate_ERP(input_rgb, modality):
    batch = {}
    input_rgb = torch.from_numpy(input_rgb).permute(2,0,1).to(torch.float32) / 255.0
    input_rgb = input_rgb * 2.0 - 1.0
    batch['rgb_cubemap'] = erp_to_cubemap(input_rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred_cubemap = pager(batch, modality)
        if modality == "depth": 
            pred, pred_image = pager.process_depth_output(pred_cubemap, orig_size=(1024, 2048), 
                                                        min_depth=MIN_DEPTH, 
                                                        depth_range=DEPTH_RANGE, 
                                                        log_scale=pager.model_configs["depth"]["config"].log_scale)
            pred, pred_image = pred[0].cpu().numpy(), pred_image.cpu().numpy()
            pred_image = np.clip(pred_image, pred_image.min(), np.quantile(pred_image, 0.99))
            pred_image = prepare_image_for_logging(pred_image)
            pred_image = cmap(pred_image[0,...]/255.0)
            pred_image = (pred_image[..., :3] * 255).astype(np.uint8)
        elif modality == "normal":
            pred = pager.process_normal_output(pred_cubemap, orig_size=(1024, 2048))
            pred = pred.cpu().numpy()
            pred_image = pred.copy()
            pred_image = prepare_image_for_logging(pred_image).transpose(1,2,0)

    return pred_image, pred   

def process_panorama(image_path, output_type, include_pointcloud):
    loaded_image = Image.open(image_path).convert("RGB").resize((2048, 1024))
    input_rgb = np.array(loaded_image)

    modality = "depth" if output_type.lower() == "depth" else "normal"
    is_depth = modality == "depth"
    main_label = "Depth Output" if is_depth else "Surface Normal Output"
    pc_label = (
        "RGB-colored Point Cloud" if is_depth else "Surface Normals-Colored Point Cloud"
    )
    output_image, raw_pred = generate_ERP(input_rgb, modality)

    point_cloud = None
    if include_pointcloud:
        if is_depth:
            depth = np.squeeze(np.array(raw_pred))
            color = (input_rgb.astype(np.float32) / 127.5) - 1.0
        else:
            color = np.array(raw_pred)
            color = np.transpose(color, (1, 2, 0))
            _release_cuda_memory()
            depth = np.squeeze(generate_ERP(input_rgb, "depth", )[1])
        
        edge_filtered_mask = compute_edge_mask(
            depth,
            abs_thresh=0.002,
            rel_thresh=0.002,
        )

        if POINTCLOUD_DOWNSAMPLE_FACTOR > 1:
            depth = depth[::POINTCLOUD_DOWNSAMPLE_FACTOR, ::POINTCLOUD_DOWNSAMPLE_FACTOR]
            color = color[::POINTCLOUD_DOWNSAMPLE_FACTOR, ::POINTCLOUD_DOWNSAMPLE_FACTOR]
            edge_filtered_mask = edge_filtered_mask[::POINTCLOUD_DOWNSAMPLE_FACTOR, ::POINTCLOUD_DOWNSAMPLE_FACTOR]

        tmp = NamedTemporaryFile(suffix=".glb", delete=False)
        erp_to_point_cloud_glb(
            color, depth, edge_filtered_mask, export_path=tmp.name)

        tmp.close()
        point_cloud = tmp.name

    _release_cuda_memory()

    return (
        gr.update(value=output_image, label=main_label),
        gr.update(value=point_cloud, label=pc_label),
    )


def clear_pointcloud():
    return gr.update(value=None)


args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("simple")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
cmap = plt.get_cmap("Spectral")


checkpoint_config = {}
try:
    depth_checkpoint_config_path = hf_hub_download(
        repo_id=args.depth_checkpoint_path,
        filename="config.yaml"
    )
except Exception as e:
    depth_checkpoint_config_path = Path(args.depth_checkpoint_path) / "config.yaml"
depth_config = OmegaConf.load(depth_checkpoint_config_path)
checkpoint_config["depth"] = {"path": args.depth_checkpoint_path, "mode": "trained", "config": depth_config.model}

try:
    normal_checkpoint_config_path = hf_hub_download(
        repo_id=args.normals_checkpoint_path,
        filename="config.yaml"
    )
except Exception as e:
    normal_checkpoint_config_path = Path(args.normals_checkpoint_path) / "config.yaml"
normal_config = OmegaConf.load(normal_checkpoint_config_path)
checkpoint_config["normal"] = {"path": args.normals_checkpoint_path, "mode": "trained", "config": normal_config.model}

pager = Pager(model_configs=checkpoint_config, pretrained_path = depth_config.model.pretrained_path, device=device)
pager.unet["depth"].to(device, dtype=pager.weight_dtype)
pager.unet["depth"].eval()
pager.unet["normal"].to(device, dtype=pager.weight_dtype)
pager.unet["normal"].eval()


with gr.Blocks() as demo:
    gr.Markdown("## 📟 PaGeR: Panoramic Geometry Reconstruction")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="RGB ERP Image",
                type="filepath",
                height=320,
            )
            output_choice = gr.Radio(
                ["Depth", "Surface Normals"],
                value="Depth",
                label="Output Type",
            )
            pointcloud_checkbox = gr.Checkbox(
                label="Generate Point Cloud",
                value=True,
            )
            gr.Examples(
                examples=EXAMPLE_IMAGES,
                inputs=image_input,
                label="Pick an example (or upload your own above)",
                examples_per_page=8,
                cache_examples=False,
            )
            run_button = gr.Button("Run Inference")

        with gr.Column(scale=1):
            rendered_output = gr.Image(
                label="Output",
                type="numpy",
                height=320,
            )

    with gr.Row():
        pointcloud_output = gr.Model3D(
            label="Point Cloud",
            height=360,
            clear_color=[0.0, 0.0, 0.0, 0.0],
        )

    (
        run_button.click(
            fn=clear_pointcloud,
            outputs=pointcloud_output,
            queue=False,
        )
        .then(
            fn=process_panorama,
            inputs=[image_input, output_choice, pointcloud_checkbox],
            outputs=[rendered_output, pointcloud_output],
        )
    )

if __name__ == "__main__":
    _release_cuda_memory()
    demo.launch()
