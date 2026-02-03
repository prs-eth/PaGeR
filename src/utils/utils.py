import torch
import numpy as np
import cv2
import random
import wandb
from tqdm.auto import tqdm
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

def args_to_omegaconf(args, base_cfg=None):
    cfg = OmegaConf.create(base_cfg)

    def _override_if_provided(container, key):
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                container[key] = value

    def _override_recursively(container):
        if not isinstance(container, DictConfig):
            return

        for key in container.keys():
            node = container[key]
            if isinstance(node, DictConfig):
                _override_recursively(node)
            else:
                _override_if_provided(container, key)

    _override_recursively(cfg)

    return cfg

def _tb_sanitize(v):
    if v is None:
        return "null"
    if isinstance(v, (bool, int, float, str, torch.Tensor)):
        return v
    if isinstance(v, Path):
        return str(v)
    return str(v)

def _flatten_dict(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten_dict(v, key))
            else:
                out[key] = _tb_sanitize(v)
    else:
        out[prefix or "cfg"] = _tb_sanitize(d)
    return out

def convert_paths_to_pathlib(cfg):
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            cfg[key] = convert_paths_to_pathlib(value)
        elif 'path' in key.lower():
            cfg[key] = Path(value) if value is not None else None
    return cfg


def convert_pathlib_to_strings(cfg):
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            cfg[key] = convert_pathlib_to_strings(value)
        elif isinstance(value, Path):
            cfg[key] = str(value)
    return cfg


def prepare_trained_parameters(unet, cfg):
    unet_parameters = []

    if cfg.training.only_train_attention_layers:
        for name, param in unet.named_parameters():
            if (cfg.model.unet_positional_encoding == "uv" and "conv_in" in name) or \
                "transformer_blocks" in name:
                unet_parameters.append(param)
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    else:
        for param in unet.parameters():
            unet_parameters.append(param)
            param.requires_grad_(True)
    
    return unet_parameters


@torch.no_grad()
def validation_loop(accelerator, dataloader, pager, ema_unet, cfg, epoch, global_step, val_type="val"):
    if val_type == "val":
        desc = "Validation"
        x_axis_name = "epoch"
        x_axis = epoch
    elif val_type == "tiny_val":
        desc = "Tiny Validation"
        x_axis_name = "global_step"
        x_axis = global_step
    else:
        raise ValueError(f"Unknown val type {val_type}")
    if cfg.training.use_EMA:
        ema_unet.store(pager.unwrapped_unet.parameters())
        ema_unet.copy_to(pager.unwrapped_unet.parameters())
    val_epoch_loss = 0.0
    log_val_images = {"rgb": [], cfg.model.modality: []}
    log_img_ids = random.sample(range(len(dataloader)), 4)
    progress_bar = tqdm(dataloader, desc=desc, total=len(dataloader), disable=not accelerator.is_main_process)
    for i, batch in enumerate(progress_bar):
        pred_cubemap = pager(batch, cfg.model.modality)
        if cfg.model.modality == "depth":
            min_depth = dataloader.dataset.LOG_MIN_DEPTH if cfg.model.log_scale else dataloader.dataset.MIN_DEPTH
            depth_range = dataloader.dataset.LOG_DEPTH_RANGE if cfg.model.log_scale else dataloader.dataset.DEPTH_RANGE
            loss = pager.calculate_depth_loss(batch, pred_cubemap, min_depth, depth_range, cfg.model.log_scale, cfg.model.metric_depth)
        elif cfg.model.modality == "normal":
            loss = pager.calculate_normal_loss(batch, pred_cubemap)

        avg_loss = accelerator.reduce(loss["total_loss"].detach(), reduction="mean")
        if accelerator.is_main_process:
            progress_bar.set_postfix({"loss": avg_loss.item()})
            val_epoch_loss += avg_loss
        if i in log_img_ids:
            log_val_images["rgb"].append(prepare_image_for_logging(batch["rgb"][0].cpu().numpy()))
            if cfg.model.modality == "depth":
                result_image = pager.process_depth_output(pred_cubemap, orig_size=batch['depth'].shape[2:4], min_depth=min_depth, 
                                                          depth_range=depth_range, log_scale=cfg.model.log_scale)[1].cpu().numpy()
            elif cfg.model.modality == "normal":
                result_image = pager.process_normal_output(pred_cubemap, orig_size=batch['normal'].shape[2:4]).cpu().numpy()
            log_val_images[cfg.model.modality].append(prepare_image_for_logging(result_image))

    val_epoch_loss = val_epoch_loss / len(dataloader)

    if accelerator.is_main_process:
        accelerator.log({x_axis_name: x_axis, f"{val_type}/loss": float(val_epoch_loss)}, step=global_step)

        img_mix_rgb = log_images_mosaic(log_val_images["rgb"])
        img_mix_depth = log_images_mosaic(log_val_images[cfg.model.modality])

        if cfg.logging.report_to == "wandb":
            accelerator.log(
                {x_axis_name: x_axis, f"{val_type}/pred_panorama_rgb": wandb.Image(img_mix_rgb)},
                step=global_step,
            )
            accelerator.log(
                {x_axis_name: x_axis, f"{val_type}/pred_panorama_{cfg.model.modality}": wandb.Image(img_mix_depth)},
                step=global_step,
            )
        elif cfg.logging.report_to == "tensorboard":
            tb_writer = accelerator.get_tracker("tensorboard").writer
            tb_writer.add_image(
                f"{val_type}/pred_panorama_rgb",
                img_mix_rgb,
                global_step,
                dataformats="HWC",
            )
            tb_writer.add_image(
                f"{val_type}/pred_panorama_{cfg.model.modality}",
                img_mix_depth,
                global_step,
                dataformats="HWC",
            )

    if cfg.training.use_EMA:
        ema_unet.restore(pager.unwrapped_unet.parameters())
    return val_epoch_loss


def prepare_image_for_logging(image):
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = (image * 255).astype("uint8")
    return image


def log_images_mosaic(images):
    n = len(images)
    assert 1 <= n <= 4, "Provide between 1 and 4 images (CHW uint8)."

    fullhd_imgs = []
    for img in images:
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[0] in (1, 3), \
            "Each image must be uint8 with shape (C,H,W), C in {1,3}."

        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        img_hwc = np.transpose(img, (1, 2, 0))

        img_fullhd = cv2.resize(img_hwc, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        fullhd_imgs.append(img_fullhd)

    H, W, C = 1080, 1920, 3

    if n == 1:
        return fullhd_imgs[0]

    if n == 2:
        canvas = np.zeros((H, 2*W, C), dtype=np.uint8)
        canvas[:, 0:W, :]   = fullhd_imgs[0]
        canvas[:, W:2*W, :] = fullhd_imgs[1]
        return canvas

    if n == 3:
        canvas = np.zeros((2*H, 2*W, C), dtype=np.uint8)
        x_off = W // 2
        canvas[0:H, x_off:x_off+W, :] = fullhd_imgs[0]
        canvas[H:2*H, 0:W,   :] = fullhd_imgs[1]
        canvas[H:2*H, W:2*W, :] = fullhd_imgs[2]
        return canvas

    canvas = np.zeros((2*H, 2*W, C), dtype=np.uint8)
    canvas[0:H,   0:W,   :] = fullhd_imgs[0]
    canvas[0:H,   W:2*W, :] = fullhd_imgs[1]
    canvas[H:2*H, 0:W,   :] = fullhd_imgs[2]
    canvas[H:2*H, W:2*W, :] = fullhd_imgs[3]
    return canvas


