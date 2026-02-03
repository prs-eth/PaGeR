# @Vukasin Bozic 2026
# Training code for 'Panorama Geometry Estimation using Single-Step Diffusion Models'.
# This training code is a modified version of the original training code for the paper 'Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think',
# https://github.com/VisualComputingInstitute/diffusion-e2e-ft/blob/main/training/train.py.
  
import argparse
import logging
import math

import datasets
import torch

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import hf_hub_download

from tqdm.auto import tqdm
import diffusers
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from dataloaders.Structured3D_dataloader import Structured3D
from dataloaders.Matterport3D360_dataloader import Matterport3D360
from dataloaders.PanoInfinigen_dataloader import PanoInfinigen
from dataloaders.Stanford2D3DS_dataloader import Stanford2D3DS
from dataloaders.ScannetPP_dataloader import ScannetPP
from dataloaders.Structured3D_ScannetPP_dataloader import Structured3D_ScannetPP

from src.pager import Pager
from src.utils.lr_scheduler import IterConstant, IterExponential
from src.utils.utils import (
    prepare_trained_parameters, 
    prepare_image_for_logging, 
    log_images_mosaic, 
    validation_loop, 
    _flatten_dict, 
    convert_paths_to_pathlib, 
    convert_pathlib_to_strings, 
    args_to_omegaconf
)

if is_wandb_available():
    import wandb

check_min_version("0.27.0.dev0")
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Training code for panorama depth and normals estimation models.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--debug",
        action="store_true", 
        default=None,
        help="Whether or not to use a small subset of the dataset. Useful for debugging."
    )

    parser.add_argument(
        "--seed", 
        type=int,
        default=None,
        help="A seed for reproducible training."
    )

    parser.add_argument(
        "--num_train_epochs", 
        type=int,
        default=None,
        help="Total number of training epochs to perform."
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Training checkpoint to resume from. This expects a folder containing the state of the `Accelerator` class.",
    )

    parser.add_argument(
        "--use_EMA",
        action="store_true",
        default=None,
        help="Whether to use Exponential Moving Average (EMA) for model weights.",
    )

    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        choices=["depth", "normals"],
        help="Modality to use for training.",
    )

    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="UNet checkpoint to load. This only loads the savetensors weights of the UNet model.",
    )

    parser.add_argument(
        "--enable_xformers", 
        action="store_true", 
        default=None,
        help="Whether or not to use xformers."
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=None,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--only_train_attention_layers",
        action="store_true", 
        default=None,
        help="Whether or not to only train the attention parameters of the UNet model.",
    )

    parser.add_argument(
        "--unet_positional_encoding",
        type=str,
        default=None,
        choices=["uv", "RoPE"],
        help="Type of positional encoding to use."
    )

    parser.add_argument(
        "--vae_use_RoPE",
        type=bool,
        default=None,
        help="Whether or not to use RoPE positional encoding in the VAE."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Dataset to use for training."
    )

    parser.add_argument(
        "--batch_size", 
        type=int,
        default=None,
        help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["PanoInfinigen", "Matterport3D360", "Structured3D", "ScannetPP", "Structured3D_ScannetPP"],
        help="Dataset to use for training."
    )

    parser.add_argument(
        "--scenes",
        default=None,
        choices=["indoor", "outdoor", "both"], 
        help="Which scenes to use for training. 'indoor' for indoor scenes, 'outdoor' for outdoor scenes, " \
        "'both' for both indoor and outdoor scenes.",
    )

    parser.add_argument(
        "--use_data_augmentation",
        action="store_true", 
        default=None,
        help="Whether or not to use data augmentation, horizontal random rotation."
    )

    parser.add_argument(
        "--metric_depth",
        action="store_true", 
        default=None,
        help="Whether or not to use metric depth, instead of relative."
    )

    parser.add_argument(
        "--log_scale",
        action="store_true", 
        default=None,
        help="Whether to use log scale depth, instead of linear."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--lr_exp_warmup_steps",
        type=float,
        default=None,
        help="Ratio of the total training steps for the learning rate exponential warmup (default: 0.02 = 2%)."
    )

    parser.add_argument(
        "--adam_beta1", 
        type=float,
        default=None,
        help="The beta1 parameter for the Adam optimizer.",
    )
    
    parser.add_argument(
        "--adam_beta2", 
        type=float,
        default=None,
        help="The beta2 parameter for the Adam optimizer."
    )

    parser.add_argument(
        "--adam_weight_decay", 
        type=float,
        default=None,
        help="Weight decay to use."
    )

    parser.add_argument(
        "--adam_epsilon", 
        type=float,
        default=None,
        help="Epsilon value for the Adam optimizer"
    )

    parser.add_argument(
        "--clip_grad_norm", 
        action="store_true",
        default=None,
        help="Whether to use gradient clipping."
    )

    parser.add_argument(
        "--max_grad_norm", 
        type=float,
        default=None,
        help="Max gradient norm."
    )

    parser.add_argument(
        "--l1_loss_weight",
        type=float,
        default=None,
        help="Weight for the L1 loss."
    )

    parser.add_argument(
        "--grad_loss_weight",
        type=float,
        default=None,
        help="Weight for the gradient loss."
    )

    parser.add_argument(
        "--normals_consistency_loss_weight",
        type=float,
        default=None,
        help="Weight for the normals consistency loss."
    )

    parser.add_argument(
        "--invalid_mask_weight",
        type=float,
        default=None,
        help="Weight for the invalid mask loss."
    )

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default=None,
        help=("The project name for the experiment tracker"),
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--save_frequency",
        type=int,
        default=None,
        help="Save the model every X epochs.",
    )

    parser.add_argument(
        "--logging_path",
        type=str,
        default=None,
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *model_save_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        choices=["tensorboard", "wandb"],
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for the wandb run."
    )
    
    parser.add_argument(
        "--loss_report_frequency",
        type=int,
        default=None,
        help="How often to report loss (in steps)."
    )

    parser.add_argument(
        "--img_report_frequency",
        type=int,
        default=None,
        help="How often to report image (in steps)."
    )

    parser.add_argument(
        "--run_validation",
        action="store_true",
        default=None,
        help="Whether to use a validation set."
    )

    parser.add_argument(
        "--run_tiny_validation",
        action="store_true",
        default=None,
        help="Whether to use a smaller validation set for mid-training validation."
    )

    parser.add_argument(
        "--tiny_val_frequency",
        type=int,
        default=None,
        help="How often to use the tiny validation set (in steps)."
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.resume_path:
        cfg = OmegaConf.load(Path(args.resume_path).parent / "config.yaml")
    cfg = args_to_omegaconf(args, cfg)
    cfg = convert_paths_to_pathlib(cfg)

    save_path = cfg.logging.save_path / datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if cfg.logging.report_to == "tensorboard":
        logging_path = save_path / cfg.logging.tensorboard.log_path
        accelerator_project_config = ProjectConfiguration(project_dir=save_path, logging_dir=logging_path)
    else:
        accelerator_project_config = ProjectConfiguration(project_dir=save_path)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with=cfg.logging.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if args.resume_path:
        logger.info("resume_path found in CLI, the original config will be overriden by the config file in the training checkpoint, "
                    "and other CLI args will be used to override the loaded config.")

    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.seed:
        set_seed(cfg.seed)


    checkpoint_config = {}
    if cfg.training.resume_path:
        if cfg.model.checkpoint_path:
            logger.info("Both resume_path and checkpoint_path are set. Resuming training will take precedence.")
        logger.info(f"Resuming from training checkpoint at {cfg.training.resume_path}")
        checkpoint_config[cfg.model.modality] = {"path": cfg.model.pretrained_path, "mode": "resume", 
                                                 "config": cfg.model}
    elif cfg.model.checkpoint_path:
        logger.info(f"Loading UNet weights from checkpoint at {cfg.model.checkpoint_path}")
        try:
            checkpoint_config_path = hf_hub_download(
                repo_id=cfg.model.checkpoint_path,
                filename="config.yaml"
            )
        except Exception as e:
            checkpoint_config_path = Path(cfg.model.checkpoint_path) / "config.yaml"
        checkpoint_cfg = OmegaConf.load(checkpoint_config_path)
        cfg.model = checkpoint_cfg.model
        checkpoint_config[cfg.model.modality] = {"path": cfg.model.checkpoint_path, "mode": "trained", 
                                                 "config": checkpoint_cfg.model}
    else:
        logger.info(f"Loading UNet weights from pretrained model at {cfg.model.pretrained_path}")
        checkpoint_config[cfg.model.modality] = {"path": cfg.model.pretrained_path, "mode": "pretrained", 
                                                 "config": cfg.model}

    pager = Pager(model_configs=checkpoint_config, pretrained_path = cfg.model.pretrained_path, 
                  train_modality=cfg.model.modality, device=accelerator.device)
    pager.prepare_training(accelerator, cfg.training.gradient_checkpointing)
 
    if accelerator.is_main_process:
        save_path.mkdir(parents=True, exist_ok=True)
        cfg_save_path = save_path / "config.yaml"
        save_cfg = convert_pathlib_to_strings(cfg.copy())
        OmegaConf.save(save_cfg, cfg_save_path)

    ema_unet = None
    if cfg.training.use_EMA:
        ema_unet = EMAModel(pager.trained_unet.parameters())
        accelerator.register_for_checkpointing(ema_unet)

    parameters = prepare_trained_parameters(pager.trained_unet, cfg)
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        parameters,
        lr=cfg.optimization.learning_rate,
        betas=(cfg.optimization.adam_beta1, cfg.optimization.adam_beta2),
        weight_decay=cfg.optimization.adam_weight_decay,
        eps=cfg.optimization.adam_epsilon,
    )
    optimizer = accelerator.prepare(optimizer)
    pager.prepare_losses_dict(cfg.loss)

    if cfg.debug:
        logger.info("Running in debug mode, some parameters are changed to default values in order to "
        "accomodate the debug environment.")
        cfg.logging.model_save_frequency = 20
        cfg.logging.img_report_frequency = 10
        cfg.validation.tiny_val_frequency = 15
        num_workers = 1
        prefetch_factor = 1
    else:
        num_workers = 3
        prefetch_factor = 2
        

    dataset_cls = globals()[cfg.data.dataset]
    train_ds = dataset_cls(data_path=cfg.data.data_path, training=True, split="train", scenes=cfg.data.scenes, 
                           log_depth=cfg.model.log_scale, data_augmentation=cfg.data.use_data_augmentation, debug=cfg.debug)
    if cfg.validation.run_validation:
        val_ds = dataset_cls(data_path=cfg.data.data_path, training=True, split="val", scenes=cfg.data.scenes, 
                             log_depth=cfg.model.log_scale, debug=cfg.debug)
    if cfg.validation.run_tiny_validation:
        tiny_val_ds = dataset_cls(data_path=cfg.data.data_path, training=True, split="tiny_val", scenes=cfg.data.scenes, 
                                  log_depth=cfg.model.log_scale, debug=cfg.debug)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.data.batch_size, num_workers=num_workers, pin_memory=True, 
                                                   persistent_workers=True, prefetch_factor=prefetch_factor, shuffle=True, drop_last=True)
    if cfg.validation.run_validation:
        val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.data.batch_size, num_workers=num_workers, pin_memory=True, 
                                                     persistent_workers=True, prefetch_factor=prefetch_factor, shuffle=False)
    if cfg.validation.run_tiny_validation:
        tiny_val_dataloader = torch.utils.data.DataLoader(tiny_val_ds, batch_size=cfg.data.batch_size, num_workers=1, pin_memory=True, 
                                                          persistent_workers=True, prefetch_factor=1, shuffle=False)


    pager.prepare_cubemap_PE(train_ds.HEIGHT, train_ds.WIDTH)
    train_dataloader = accelerator.prepare(train_dataloader)
    if cfg.validation.run_validation:
        val_dataloader = accelerator.prepare(val_dataloader)
    if cfg.validation.run_tiny_validation:
        tiny_val_dataloader = accelerator.prepare(tiny_val_dataloader)

    total_batch_size = cfg.data.batch_size * accelerator.num_processes
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    if cfg.training.max_train_steps is None:
        cfg.training.max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch

    if cfg.debug:
        lr_func = IterConstant(total_iter_length=min(cfg.training.max_train_steps, num_update_steps_per_epoch * 
                                                     cfg.training.num_train_epochs) * accelerator.num_processes)
    else:
        lr_func = IterExponential(total_iter_length=min(cfg.training.max_train_steps, num_update_steps_per_epoch * 
                                                        cfg.training.num_train_epochs) * accelerator.num_processes, final_ratio=0.01, 
                                                        warmup_steps=cfg.optimization.lr_exp_warmup_steps)
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        cfg_container = OmegaConf.to_container(cfg, resolve=True)
        if cfg.logging.report_to == "tensorboard":
            tracker_config = _flatten_dict(cfg_container)
        elif cfg.logging.report_to == "wandb":
            tracker_config = cfg_container
            if cfg.logging.wandb.run_name:
                init_kwargs = {"wandb": {"name": cfg.logging.wandb.run_name}}

        accelerator.init_trackers(
            cfg.logging.tracker_project_name,
            tracker_config,
            init_kwargs=init_kwargs,
        )

        if cfg.logging.report_to == "wandb":
            wandb.define_metric("global_step")
            wandb.define_metric("train/*", step_metric="global_step")
            wandb.define_metric("val/*",   step_metric="global_step")
            wandb.define_metric("tiny_val/*",  step_metric="global_step")

            wandb.define_metric("epoch")
            wandb.define_metric("train/epoch_loss", step_metric="epoch")
            wandb.define_metric("val/epoch_loss",   step_metric="epoch")
        elif cfg.logging.report_to == "tensorboard":
            tb_writer = accelerator.get_tracker("tensorboard").writer


    logger.info(f"  Num batches = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {cfg.training.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.batch_size}")
    logger.info(f"  Train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Max optimization steps = {cfg.training.max_train_steps}")
    logger.info(f"  Number of trained parameters: {sum(p.numel() for p in parameters)}")

    logger.info("***** Running training *****")
    if cfg.training.resume_path:
        resume_path = cfg.training.resume_path
        accelerator.load_state(resume_path)    
        training_custom_states = torch.load(resume_path / "training_custom_states.pt", map_location="cpu")
        global_step = training_custom_states.get("global_step", 0)
        start_epoch = training_custom_states.get("epoch", 0) + 1
        best_val_loss = training_custom_states.get("best_val_loss", float("inf"))
        logger.info(f"Resuming the training from {resume_path}, global step {global_step} and epoch {start_epoch}")
    else:
        global_step = 0
        start_epoch = 0
        best_val_loss = float("inf")

    if cfg.training.use_EMA:
        ema_unet.to(accelerator.device)

    log_train_images = {"rgb": [], cfg.model.modality: []}
    min_depth = train_ds.LOG_MIN_DEPTH if cfg.model.log_scale else train_ds.MIN_DEPTH
    depth_range = train_ds.LOG_DEPTH_RANGE if cfg.model.log_scale else train_ds.DEPTH_RANGE
    for epoch in range(start_epoch, cfg.training.num_train_epochs):
        train_epoch_loss = 0.0
        logger.info(f"Epoch: {epoch}")
        progress_bar = tqdm(train_dataloader, desc=f"Training", total=len(train_dataloader), disable=not accelerator.is_main_process)

        pager.trained_unet.train()
        for batch in progress_bar:
            with accelerator.accumulate(pager.trained_unet):
                pred_cubemap = pager(batch, cfg.model.modality)
                if cfg.model.modality == "depth":
                    loss = pager.calculate_depth_loss(batch, pred_cubemap, min_depth, depth_range, cfg.model.log_scale, cfg.model.metric_depth)
                else:
                    loss = pager.calculate_normals_loss(batch, pred_cubemap)
                accelerator.backward(loss["total_loss"])
                avg_loss = accelerator.reduce(loss["total_loss"].detach(), reduction="mean") / cfg.training.gradient_accumulation_steps
                if accelerator.is_main_process:
                    progress_bar.set_postfix({"loss": avg_loss.item()})
                    train_epoch_loss += avg_loss

                if accelerator.sync_gradients:
                    if cfg.optimization.clip_grad_norm:
                        accelerator.clip_grad_norm_(parameters, cfg.optimization.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    if cfg.training.use_EMA:
                        ema_unet.step(pager.unwrapped_unet.parameters())

                    if accelerator.is_main_process:
                        if global_step % cfg.logging.loss_report_frequency == 0:
                            accelerator.log(
                                {"global_step": global_step, **{f"train/{k}": v.detach().item() for k, v in loss.items()}},
                                step=global_step,
                            )
                            accelerator.log(
                                {"global_step": global_step, "train/lr": lr_scheduler.get_last_lr()[0]},
                                step=global_step,
                            )

                        do_img_collect = (
                            (global_step % cfg.logging.img_report_frequency) == 0
                            or (global_step % cfg.logging.img_report_frequency)
                            > cfg.logging.img_report_frequency - 4
                        )
                        if do_img_collect:
                            with torch.no_grad():
                                log_train_images["rgb"].append(
                                    prepare_image_for_logging(batch["rgb"][0].cpu().numpy())
                                )
                                if cfg.model.modality == "depth":
                                    result_image = pager.process_depth_output(
                                        pred_cubemap,
                                        orig_size=batch["depth"].shape[2:4],
                                        min_depth=min_depth,
                                        depth_range=depth_range,
                                        log_scale=cfg.model.log_scale,
                                    )[1].cpu().numpy()
                                else:
                                    result_image = pager.process_normals_output(
                                        pred_cubemap,
                                        orig_size=batch["normals"].shape[2:4],
                                    ).cpu().numpy()
                                log_train_images[cfg.model.modality].append(
                                    prepare_image_for_logging(result_image)
                                )

                        if global_step % cfg.logging.img_report_frequency == 0:
                            img_mix_rgb = log_images_mosaic(log_train_images["rgb"])
                            img_mix_result = log_images_mosaic(log_train_images[cfg.model.modality])

                            if cfg.logging.report_to == "wandb":
                                accelerator.log(
                                    {
                                        "global_step": global_step,
                                        "train/pred_panorama_rgb": wandb.Image(img_mix_rgb),
                                    },
                                    step=global_step,
                                )
                                accelerator.log(
                                    {
                                        "global_step": global_step,
                                        f"train/pred_panorama_{cfg.model.modality}": wandb.Image(img_mix_result),
                                    },
                                    step=global_step,
                                )
                            elif cfg.logging.report_to == "tensorboard":
                                tb_writer.add_image(
                                    "train/pred_panorama_rgb",
                                    img_mix_rgb,
                                    global_step,
                                    dataformats="HWC",
                                )
                                tb_writer.add_image(
                                    f"train/pred_panorama_{cfg.model.modality}",
                                    img_mix_result,
                                    global_step,
                                    dataformats="HWC",
                                )

                            log_train_images = {"rgb": [], cfg.model.modality: []}
                    global_step += 1

                    if cfg.validation.run_tiny_validation and global_step > 0 and global_step % cfg.validation.tiny_val_frequency == 0:
                        pager.trained_unet.eval()
                        tiny_val_loss = validation_loop(accelerator, tiny_val_dataloader, pager, ema_unet, 
                                                        cfg, epoch, global_step, val_type="tiny_val")
                        if accelerator.is_main_process:
                            logger.info(f"Step {global_step} Tiny Validation loss: {tiny_val_loss: .4f}")
                            accelerator.log({"global_step": global_step, "tiny_val/loss": tiny_val_loss}, step=global_step)
                        pager.trained_unet.train()

                if global_step >= cfg.training.max_train_steps:
                    break 

        train_epoch_loss = train_epoch_loss / len(train_dataloader)


        if cfg.validation.run_validation:
            pager.trained_unet.eval()
            val_epoch_loss = validation_loop(accelerator, val_dataloader, pager, ema_unet, 
                                             cfg, epoch, global_step, val_type="val")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process and val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                model_save_path = save_path / "checkpoint-best"
                pager.save_model(ema_unet, model_save_path)
                logger.info(f"Saved model to {model_save_path}") 

        if epoch % cfg.logging.model_save_frequency == 0 or epoch == cfg.training.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                model_save_path = save_path / "checkpoint-last"
                pager.save_model(ema_unet, model_save_path)
                training_checkpoint_save_path = save_path / "training_checkpoint"
                accelerator.save_state(training_checkpoint_save_path)
                torch.save({"global_step": global_step, "epoch": epoch, "best_val_loss": best_val_loss},
                            training_checkpoint_save_path / "training_custom_states.pt")
                logger.info(f"Saved model to {model_save_path}") 
                logger.info(f"Saved training state to {training_checkpoint_save_path}")


        logger.info(f"Epoch {epoch} train loss: {train_epoch_loss: .4f}")
        if accelerator.is_main_process:
            accelerator.log({"epoch": epoch, "train/epoch_loss": train_epoch_loss}, step=global_step)
        if cfg.validation.run_validation:
            logger.info(f"Epoch {epoch} validation loss: {val_epoch_loss: .4f}")
            if accelerator.is_main_process:
                accelerator.log({"epoch": epoch, "val/epoch_loss": val_epoch_loss}, step=global_step)

    logger.info(f"Finished training.")
    accelerator.end_training()

if __name__ == "__main__":
    main()
