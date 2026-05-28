# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The PaGeR Authors.
"""Gradio demo: panorama in, metric depth + normals (+ optional point cloud) out."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

# Put src/ on sys.path so depth_anything_3's absolute imports work without `pip install -e .`.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import gradio as gr
import numpy as np
import torch
import trimesh
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from PIL import Image
from scipy.ndimage import median_filter

from src.pager import Pager
from src.utils.geometry_utils import (
    compute_edge_mask,
    cubemap_to_erp,
    erp_to_cubemap,
    erp_to_pointcloud,
)
from src.utils.scene_classifier import get_classifier
from src.utils.utils import (
    prepare_depth_for_logging,
    prepare_normals_for_logging,
)


POINTCLOUD_DOWNSAMPLE_FACTOR = 2

EXAMPLES_DIR = Path(__file__).parent / "assets" / "examples"
EXAMPLE_IMAGES = [
    str(p) for p in sorted(EXAMPLES_DIR.glob("*"))
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
]

SCENE_TO_SCALE_HEAD = {"Indoor": "scale_indoor", "Outdoor": "scale_outdoor"}
MODE_AUTO, MODE_INDOOR, MODE_OUTDOOR = "Auto", "Indoor", "Outdoor"
MODE_CHOICES = [MODE_AUTO, MODE_INDOOR, MODE_OUTDOOR]
FORMAT_MAP, FORMAT_POINTCLOUD = "Map", "Point Cloud"

MODEL_TO_REPO = {
    "pager":              "prs-eth/PaGeR",
    "pager-metric-depth": "prs-eth/PaGeR-metric-depth",
    "pager-normals":      "prs-eth/PaGeR-normals",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaGeR Gradio demo.")
    parser.add_argument("--checkpoint", type=str, default="pager",
                        help="Which checkpoint to run. Short alias "
                             "(pager / pager-metric-depth / pager-normals), a "
                             "HuggingFace Hub repo id, or a local directory. "
                             "Default: pager (the unified prs-eth/PaGeR checkpoint).")
    return parser.parse_args()


def _release_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def _active_skip_heads(scene: str) -> set:
    active = SCENE_TO_SCALE_HEAD[scene]
    return {h for h in SCENE_TO_SCALE_HEAD.values() if h != active}


def _resolve_scene(mode: str, rgb_cubemap_centred: torch.Tensor) -> tuple[str, str]:
    if mode in (MODE_INDOOR, MODE_OUTDOOR):
        return mode, f"**Scale head:** {mode}"
    cube_01 = (rgb_cubemap_centred[0] + 1.0) * 0.5
    label, _ = scene_classifier.classify(cube_01)
    scene = MODE_OUTDOOR if label == "outdoor" else MODE_INDOOR
    return scene, f"**Scale head:** Auto → {scene}"


def _run_inference(input_rgb: np.ndarray, mode: str):
    img = torch.from_numpy(input_rgb).permute(2, 0, 1).to(torch.float32) / 255.0
    img = img * 2.0 - 1.0
    rgb_cubemap = erp_to_cubemap(img, face_w=face_size, fov=cube_fov).unsqueeze(0).to(device)
    scene, decision_md = _resolve_scene(mode, rgb_cubemap)

    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        pred_dict = pager(rgb_cubemap, dtype=torch.float16,
                          skip_heads=_active_skip_heads(scene))

    sky_pred = pred_dict["sky"][0] if "sky" in pred_dict else None
    log_scale = pred_dict.get("scale", None)
    orig_size = (input_rgb.shape[0], input_rgb.shape[1])

    raw_depth, viz_depth = prepare_depth_for_logging(
        pager, pred_dict["depth"][0], sky_pred, orig_size, cmap, log_scale=log_scale,
    )
    _, viz_normals = prepare_normals_for_logging(
        pager, pred_dict["normals"][0], sky_pred, orig_size,
    )

    # Stitch the sky probability to ERP so the point cloud can drop sky pixels
    # directly. Depth-saturation alone (depth >= MAX_DEPTH) misses the smoothstep
    # transition halo around the sky, where the sky-fill blend gave intermediate
    # depths — those survive a depth threshold and look like a ceiling/dome.
    sky_mask_erp = None
    if sky_pred is not None:
        with torch.inference_mode():
            sky_prob_cube = torch.sigmoid(sky_pred.float())
            sky_prob_erp = cubemap_to_erp(sky_prob_cube, *orig_size, fov=cube_fov)
        sky_mask_erp = sky_prob_erp.detach().cpu().numpy().squeeze()

    raw_depth = np.squeeze(raw_depth)
    viz_depth = np.transpose(viz_depth, (1, 2, 0))
    viz_normals = np.transpose(viz_normals, (1, 2, 0))
    return raw_depth, viz_depth, viz_normals, _make_depth_colorbar(raw_depth), decision_md, sky_mask_erp


def _make_depth_colorbar(raw_depth: np.ndarray, height: int = 640) -> np.ndarray:
    """Vertical Spectral colorbar with metric-depth tick labels matching the depth preview."""
    log_d = median_filter(np.log(raw_depth.astype(np.float32)), size=3)
    log_d_min, log_d_max = float(log_d.min()), float(log_d.max())

    fig = plt.figure(figsize=(2.4, height / 100.0), dpi=100)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.06, 0.05, 0.45 / 2.4, 0.82])
    ax.patch.set_alpha(0.0)
    gradient = np.linspace(1.0, 0.0, 512)[:, None]
    ax.imshow(gradient, aspect="auto", cmap="Spectral",
              extent=[0, 1, log_d_min, log_d_max])
    ax.set_xticks([])

    tick_logs = np.linspace(log_d_min, log_d_max, 7)
    tick_meters = np.exp(tick_logs)
    labels = [
        f"{t:.2f}" if t < 1 else (f"{t:.1f}" if t < 10 else f"{int(round(t))}")
        for t in tick_meters
    ]
    ax.set_yticks(tick_logs)
    ax.set_yticklabels(labels, fontsize=18, color="white", weight="500", family="sans-serif")
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", length=0, pad=8, colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.5, 0.965, "Depth [m]", ha="center", fontsize=20,
             color="white", weight="600", family="sans-serif")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=True,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGBA"))


def _export_glb(points: np.ndarray, colors: np.ndarray) -> str:
    tmp = NamedTemporaryFile(suffix=".glb", delete=False)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=points, colors=colors))
    scene.export(tmp.name)
    tmp.close()
    return tmp.name


def _ensure_pointclouds(cache: dict) -> None:
    """Build the RGB- and normals-coloured clouds once and cache them (shared geometry)."""
    if cache.get("rgb_pc_path") is not None and cache.get("normals_pc_path") is not None:
        return

    depth_full = cache["raw_depth"]
    rgb_color_full = cache["input_rgb"].astype(np.float32) / 255.0
    normals_color_full = cache["viz_normals"].astype(np.float32) / 255.0

    edge_mask = compute_edge_mask(depth_full, rel_thresh=0.002)
    sky_mask_full = cache.get("sky_mask_erp")
    if POINTCLOUD_DOWNSAMPLE_FACTOR > 1:
        s = POINTCLOUD_DOWNSAMPLE_FACTOR
        depth = depth_full[::s, ::s]
        rgb_color = rgb_color_full[::s, ::s]
        normals_color = normals_color_full[::s, ::s]
        edge_mask = edge_mask[::s, ::s]
        sky_mask = sky_mask_full[::s, ::s] if sky_mask_full is not None else None
    else:
        depth, rgb_color, normals_color = depth_full, rgb_color_full, normals_color_full
        sky_mask = sky_mask_full

    xyz = erp_to_pointcloud(torch.from_numpy(depth)).permute(1, 2, 0).numpy()
    # Drop sky pixels using the sky probability directly. A depth-saturation
    # filter (depth >= MAX_DEPTH) catches the sky *core* but misses the soft
    # smoothstep halo around it — those pixels get blended intermediate depths
    # and look like a ceiling/dome. The probability mask covers the full sky
    # region including the soft transition.
    if sky_mask is not None:
        sky_keep = sky_mask < 0.3
    else:
        sky_keep = depth < 0.99 * pager.MAX_DEPTH  # fallback
    keep_2d = (depth > 0) & np.asarray(edge_mask, dtype=bool) & sky_keep
    points = xyz[keep_2d]

    def _colors_from(color_image):
        return (np.clip(color_image[keep_2d], 0.0, 1.0) * 255.0).astype(np.uint8)

    cache["rgb_pc_path"] = _export_glb(points, _colors_from(rgb_color))
    cache["normals_pc_path"] = _export_glb(points, _colors_from(normals_color))


def _empty_cache():
    return None


def _colorbar_html(colorbar) -> str:
    # Inline <img> so CSS pins height; gr.Image's own flex sizing fights us inside rows.
    if colorbar is None:
        return ""
    import base64
    pil = Image.fromarray(colorbar, mode="RGBA")
    buf = BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        '<div style="height:320px;display:flex;align-items:flex-start;'
        'justify-content:flex-start;">'
        f'<img src="data:image/png;base64,{b64}" '
        'style="height:320px;width:auto;display:block;" />'
        "</div>"
    )


def _format_outputs(cache, output_format):
    if cache is None:
        return "", None, "", None, None, None
    decision_md = cache.get("decision_md", "")
    if output_format == FORMAT_MAP:
        return (
            decision_md,
            cache["viz_depth"],
            _colorbar_html(cache["depth_colorbar"]),
            cache["viz_normals"],
            None, None,
        )
    _ensure_pointclouds(cache)
    return decision_md, None, "", None, cache["rgb_pc_path"], cache["normals_pc_path"]


def _visibility_updates(output_format):
    show_maps = output_format == FORMAT_MAP
    return gr.update(visible=show_maps), gr.update(visible=not show_maps)


def process_panorama(image_path, mode, output_format, cache):
    # Visibility flip is a separate queue=False event so Model3D is mounted before its GLB lands.
    if image_path is None:
        return ("", None, "", None, None, None, cache)
    cache_key = (str(image_path), mode)
    if cache is None or cache.get("key") != cache_key:
        loaded = Image.open(image_path).convert("RGB").resize((4032, 2016))
        input_rgb = np.array(loaded)
        raw_depth, viz_depth, viz_normals, depth_colorbar, decision_md, sky_mask_erp = _run_inference(input_rgb, mode)
        cache = {
            "key": cache_key,
            "input_rgb": input_rgb,
            "raw_depth": raw_depth,
            "viz_depth": viz_depth,
            "viz_normals": viz_normals,
            "depth_colorbar": depth_colorbar,
            "sky_mask_erp": sky_mask_erp,
            "decision_md": decision_md,
            "rgb_pc_path": None,
            "normals_pc_path": None,
        }
    outputs = _format_outputs(cache, output_format)
    _release_cuda_memory()
    return (*outputs, cache)


def on_format_change_values(image_path, mode, output_format, cache):
    cache_key = (str(image_path) if image_path else None, mode)
    if cache is not None and cache.get("key") == cache_key:
        outputs = _format_outputs(cache, output_format)
    else:
        outputs = ("", None, "", None, None, None)
    return (*outputs, cache)


def on_image_or_scene_change():
    # Invalidate cache + clear outputs so stale results can't be relabelled as the new image.
    return ("", None, "", None, None, None, _empty_cache())


args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("simple")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
cmap = plt.get_cmap("Spectral")

checkpoint_path = MODEL_TO_REPO.get(args.checkpoint, args.checkpoint)

# Fetch config.yaml separately; keep checkpoint_path as the original id so Pager resolves weights.
try:
    checkpoint_config_path = hf_hub_download(
        repo_id=checkpoint_path, filename="config.yaml"
    )
except Exception:
    checkpoint_config_path = Path(checkpoint_path) / "config.yaml"

cfg = OmegaConf.load(checkpoint_config_path)
modalities = list(cfg.modalities)
missing = [m for m in ("scale_indoor", "scale_outdoor") if m not in modalities]
if missing:
    raise RuntimeError(
        f"Checkpoint at {checkpoint_path} does not expose dual scale heads "
        f"(missing {missing}). The Gradio demo expects the unified PaGeR "
        f"checkpoint that ships both indoor and outdoor scale heads."
    )

face_size = int(cfg.face_size)
cube_fov = float(getattr(cfg, "cube_fov", 90.0))

pager = Pager(Path(checkpoint_path), cfg=cfg, device=device)
pager.get_intrinsics_extrinsics(image_size=face_size, fov=cube_fov)
pager.model.to(device, dtype=pager.weight_dtype)
pager.model.eval()

# Eager-init so the first Auto click doesn't pay the weight-load / first-run download.
scene_classifier = get_classifier(device=device)


_CUSTOM_CSS = """
#depth-wrapper { position: relative !important; overflow: visible !important; }
#depth-wrapper > *:not(#depth-colorbar-html) { overflow: visible !important; }
#depth-colorbar-html {
    position: absolute !important;
    top: -16px !important;
    left: 100% !important;
    height: 320px !important;
    width: 120px !important;
    margin-left: 8px !important;
    z-index: 5 !important;
    pointer-events: none !important;
    flex-grow: 0 !important;
    flex-shrink: 0 !important;
}
#depth-colorbar-html > * { height: 100% !important; }
#map-group, #map-group > *, #map-output-column { overflow: visible !important; }
"""


with gr.Blocks() as demo:
    # Gradio 6.0.x dropped Blocks(css=...); inject via gr.HTML instead.
    gr.HTML(f"<style>{_CUSTOM_CSS}</style>")
    gr.Markdown(
        "## 📟 PaGeR — Panoramic Geometry Reconstruction\n\n"
        "Upload a 360° panorama and PaGeR predicts a **metric depth map** "
        "and **surface normals** for the scene. View the results as 2D maps "
        "or as a 3D point cloud."
    )
    inference_cache = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="RGB ERP Image", type="filepath", height=320)
            scene_choice = gr.Radio(
                MODE_CHOICES, value=MODE_AUTO, label="Scene type",
                info="Activates the correct scene-dependent metric scale head for depth prediction."
            )
            format_choice = gr.Radio(
                [FORMAT_MAP, FORMAT_POINTCLOUD], value=FORMAT_MAP, label="Output Format",
                info="Show depth/normals as 2D maps, or as 3D point clouds. "
                     "Switching formats reuses the cached result — no need to re-run inference.",
            )
            gr.Examples(
                examples=EXAMPLE_IMAGES, inputs=image_input,
                label="Pick an example (or upload your own above)",
                examples_per_page=8, cache_examples=False,
            )
            run_button = gr.Button("Run Inference")

        with gr.Column(scale=1, elem_id="map-output-column"):
            decision_md_output = gr.Markdown("")
            with gr.Group(visible=True, elem_id="map-group") as map_group:
                with gr.Group(elem_id="depth-wrapper"):
                    depth_map_output = gr.Image(label="Depth Map", type="numpy", height=320)
                    depth_colorbar_output = gr.HTML(value="", elem_id="depth-colorbar-html")
                normals_map_output = gr.Image(label="Surface Normals Map", type="numpy", height=320)
            with gr.Group(visible=False) as pointcloud_group:
                rgb_pointcloud_output = gr.Model3D(
                    label="RGB-colored Point Cloud", height=320,
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                )
                normals_pointcloud_output = gr.Model3D(
                    label="Surface Normals-colored Point Cloud", height=320,
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                )

    output_components = [
        decision_md_output, depth_map_output, depth_colorbar_output,
        normals_map_output, rgb_pointcloud_output, normals_pointcloud_output,
    ]
    visibility_components = [map_group, pointcloud_group]
    invalidation_outputs = [*output_components, inference_cache]

    (
        format_choice.change(
            fn=lambda fmt: _visibility_updates(fmt),
            inputs=format_choice,
            outputs=visibility_components,
            queue=False,
        )
        .then(
            fn=on_format_change_values,
            inputs=[image_input, scene_choice, format_choice, inference_cache],
            outputs=[*output_components, inference_cache],
        )
    )
    image_input.change(fn=on_image_or_scene_change, outputs=invalidation_outputs, queue=False)
    scene_choice.change(fn=on_image_or_scene_change, outputs=invalidation_outputs, queue=False)
    (
        run_button.click(
            fn=lambda fmt: _visibility_updates(fmt),
            inputs=format_choice,
            outputs=visibility_components,
            queue=False,
        )
        .then(
            fn=process_panorama,
            inputs=[image_input, scene_choice, format_choice, inference_cache],
            outputs=[*output_components, inference_cache],
        )
    )


if __name__ == "__main__":
    _release_cuda_memory()
    demo.launch(server_name="0.0.0.0", share=True)
