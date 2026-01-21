import torch
from torch import nn
from torch.nn import Conv2d
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from Marigold.unet.unet_2d_condition import UNet2DConditionModel
from Marigold.vae.autoencoder_kl import AutoencoderKL
from src.utils.conv_padding import PaddedConv2d, valid_pad_conv_fn
from src.utils.loss import L1Loss, GradL1Loss, CosineNormalLoss
from src.utils.geometry_utils import (
    get_positional_encoding,
      compute_scale_and_shift,
        compute_shift, 
        depth_to_normals_erp, 
        cubemap_to_erp
    )


class Pager(nn.Module):
    def __init__(self, 
                 model_configs,
                 pretrained_path,
                 train_modality=None,
                 device=torch.device("cpu"),
                 weight_dtype=torch.float32):
        super().__init__()
        self.model_configs = model_configs
        self.weight_dtype = weight_dtype
        self.rgb_latent_scale_factor = 0.18215
        self.depth_latent_scale_factor = 0.18215
        self.train_modality = train_modality
        self.device = device
        self.prepare_model_components(pretrained_path, model_configs)
        self.prepare_empty_encoding()

        self.alpha_prod = self.noise_scheduler.alphas_cumprod.to(device, dtype=weight_dtype)
        self.beta_prod = 1 - self.alpha_prod
        self.num_timesteps = self.noise_scheduler.config.num_train_timesteps - 1
        del self.noise_scheduler


    def prepare_model_components(self, pretrained_path, model_configs):
        vae_use_RoPE = None
        for checkpoint_cfg in model_configs.values():
            if vae_use_RoPE is None:
                vae_use_RoPE = checkpoint_cfg['config'].vae_use_RoPE == "RoPE"
            elif vae_use_RoPE != (checkpoint_cfg['config'].vae_use_RoPE == "RoPE"):
                raise ValueError("All UNet checkpoints must use the same VAE positional encoding configuration.")
            
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_path, subfolder="scheduler", rescale_betas_zero_snr=True)
        self.tokenizer    = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer", revision=None)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder", revision=None, variant=None)
        self.vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae", revision=None, variant=None, 
                                    use_RoPE = vae_use_RoPE)
        self.set_valid_pad_conv(self.vae)

        self.vae.requires_grad_(False)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.vae.eval()

        self.text_encoder.requires_grad_(False)
        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.text_encoder.eval()


        base_in_channels = 8
        pe_channels_size = 0
    
        self.unet = {}
        for modality, checkpoint_cfg in model_configs.items():
            if checkpoint_cfg['config'].unet_positional_encoding == "uv":
                pe_channels_size = 2
            target_in_channels = base_in_channels + pe_channels_size

            self.unet[modality] = UNet2DConditionModel.from_pretrained(
                checkpoint_cfg["path"],
                subfolder="unet",
                revision=None,
                in_channels=target_in_channels if checkpoint_cfg["mode"] == "trained" else base_in_channels,
                use_RoPE=checkpoint_cfg['config'].unet_positional_encoding == "RoPE"
            )
            
            if target_in_channels > base_in_channels and checkpoint_cfg["mode"] != "trained":
                self.extend_unet_conv_in(self.unet[modality], new_in_channels=target_in_channels)
            self.set_valid_pad_conv(self.unet[modality])

        
        if checkpoint_cfg['config'].enable_xformers:
            if is_xformers_available():
                import xformers
                if self.unet.get("depth"):
                    self.unet["depth"].enable_xformers_memory_efficient_attention()
                if self.unet.get("normal"):
                    self.unet["normal"].enable_xformers_memory_efficient_attention()
                self.vae.enable_xformers_memory_efficient_attention()


    def prepare_training(self, accelerator, gradient_checkpointing):
        self.unwrapped_unet = self.unet[self.train_modality]
        self.unet[self.train_modality] = accelerator.prepare(self.unet[self.train_modality])
        self.trained_unet = self.unet[self.train_modality]   

        if gradient_checkpointing:
            self.trained_unet._set_gradient_checkpointing()
            self.vae._set_gradient_checkpointing()
         

    def prepare_cubemap_PE(self, image_height, image_width):
        use_uv_PE = False
        for checkpoint_cfg in self.model_configs.values():
                if checkpoint_cfg['config'].unet_positional_encoding == "uv":
                    use_uv_PE = True
        if use_uv_PE:
            PE_cubemap = get_positional_encoding(image_height, image_width)
            self.PE_cubemap = PE_cubemap.to(device=self.device, dtype=self.weight_dtype)

    def prepare_empty_encoding(self):
        with torch.inference_mode():
            empty_token    = self.tokenizer([""], padding="max_length", truncation=True, return_tensors="pt").input_ids
            empty_token    = empty_token.to(self.device)
            empty_encoding = self.text_encoder(empty_token, return_dict=False)[0]
            self.empty_encoding = empty_encoding.to(self.device, dtype=self.weight_dtype)

        del empty_token
        del self.text_encoder
        del self.tokenizer


    def forward(self, batch, modality):
        with torch.inference_mode():
            c, h, w = batch["rgb_cubemap"].shape[2:]
            rgb_vae_input = batch["rgb_cubemap"].reshape(-1, c, h, w).to(dtype=self.weight_dtype)
            rgb_latents = self.vae.encode(rgb_vae_input, deterministic=True)
            rgb_latents = rgb_latents * self.rgb_latent_scale_factor
            del rgb_vae_input

        timesteps = torch.ones((rgb_latents.shape[0],), device=self.device) * self.num_timesteps
        timesteps = timesteps.long()
        alpha_prod_t = self.alpha_prod[timesteps].view(-1, 1, 1, 1)
        beta_prod_t = self.beta_prod[timesteps].view(-1, 1, 1, 1)

        noisy_latents = torch.zeros_like(rgb_latents).to(self.device)
        encoder_hidden_states = self.empty_encoding.repeat(batch["rgb_cubemap"].shape[0] * 6, 1, 1)
        if self.model_configs[modality]['config'].unet_positional_encoding == "uv":
            batch_PE_cubemap = self.PE_cubemap.repeat(batch["rgb_cubemap"].shape[0], 1, 1, 1)
            unet_input = torch.cat((rgb_latents, noisy_latents, batch_PE_cubemap), dim=1).to(
                self.device
            )
        else:
            unet_input = torch.cat((rgb_latents, noisy_latents), dim=1).to(self.device)

        del rgb_latents
        model_pred = self.unet[modality](
            unet_input,
            timesteps,
            encoder_hidden_states,
            return_dict=False,
        )[0]

        current_latent_estimate = (alpha_prod_t**0.5) * noisy_latents - (beta_prod_t**0.5) * model_pred
        current_scaled_latent_estimate = current_latent_estimate / self.depth_latent_scale_factor
        pred_cubemap = self.vae.decode(current_scaled_latent_estimate, deterministic=True)

        if modality == "depth":
            pred_cubemap = pred_cubemap.mean(dim=1, keepdim=True)
        return pred_cubemap
    

    def prepare_losses_dict(self, loss_cfg):
        self.losses_dict = {}
        if self.train_modality == "depth":
            self.losses_dict["l1_loss"] = {"loss_fn": L1Loss(invalid_mask_weight=loss_cfg.invalid_mask_weight), 
                                           "weight": loss_cfg.l1_loss_weight}
            if loss_cfg.grad_loss_weight > 0.0:
                self.losses_dict["grad_loss"] = {"loss_fn": GradL1Loss(), "weight": loss_cfg.grad_loss_weight}
            if loss_cfg.normals_consistency_loss_weight > 0.0:
                self.losses_dict["normals_consistency_loss"] = {"loss_fn": CosineNormalLoss(), 
                                                                "weight": loss_cfg.normals_consistency_loss_weight}
        else:
            self.losses_dict["cosine_normal_loss"] = {"loss_fn": CosineNormalLoss(), "weight": 1.0} 


    def calculate_depth_loss(self, batch, pred_cubemap, min_depth, depth_range, log_scale, metric_depth):
        loss = {"total_loss": 0.0}

        gt_depth_cubemap =  batch['depth_cubemap'].squeeze(0).mean(dim=1, keepdim=True)
        mask_cubemap = batch["mask_cubemap"].squeeze(0)
        
        if not metric_depth:
            if log_scale:
                scale = compute_shift(pred_cubemap, gt_depth_cubemap, mask_cubemap)
            else:
                scale, shift = compute_scale_and_shift(pred_cubemap, gt_depth_cubemap, mask_cubemap)

            if log_scale:
                pred_cubemap += scale
            else:
                pred_cubemap = pred_cubemap * scale + shift

        for loss_name, loss_params in self.losses_dict.items():
            if loss_name == "normals_consistency_loss":
                gt = batch['normal']
                pred_depth = pred_cubemap
                mask = batch["mask"]
                pred_depth = self.process_depth_output(pred_depth, orig_size=gt.shape[2:], min_depth=min_depth, 
                                                       depth_range=depth_range, log_scale=log_scale)[0]
                pred = depth_to_normals_erp(pred_depth).unsqueeze(0)
            else:
                pred = pred_cubemap
                gt = gt_depth_cubemap
                mask = mask_cubemap
            loss[loss_name] = loss_params["loss_fn"](pred, gt, mask)
            loss["total_loss"] += loss[loss_name] * loss_params["weight"]

        return loss


    def calculate_normal_loss(self, batch, pred_cubemap):
        loss = {"total_loss": 0.0}

        gt_normal_cubemap =  batch['normal_cubemap'].squeeze(0)
        mask_cubemap = batch["mask_cubemap"].squeeze(0)

        for loss_name, loss_params in self.losses_dict.items():
            pred = pred_cubemap
            gt = gt_normal_cubemap
            loss[loss_name] = loss_params["loss_fn"](pred, gt, mask_cubemap)
            loss["total_loss"] += loss[loss_name] * loss_params["weight"]

        return loss
    
    def process_depth_output(self,pred_cubemap, orig_size, min_depth, depth_range, log_scale, mask=None):
        pred_panorama = cubemap_to_erp(pred_cubemap, *orig_size)
        pred_panorama = torch.clamp(pred_panorama, -1, 1)
        pred_panorama = (pred_panorama + 1) / 2
        if mask is not None:
            pred_panorama *= mask
        pred_panorama = pred_panorama * depth_range + min_depth
        if log_scale:
            pred_panorama_viz = pred_panorama.clone()
            pred_panorama = torch.exp(pred_panorama)
        else:
            pred_panorama_viz = torch.log(pred_panorama)
        
        return pred_panorama, pred_panorama_viz


    def process_normal_output(self,pred_cubemap, orig_size):
        pred_panorama = cubemap_to_erp(pred_cubemap, *orig_size)
        pred_panorama = torch.clamp(pred_panorama, -1, 1)
        return pred_panorama
    

    def extend_unet_conv_in(self, unet, new_in_channels: int):
        if new_in_channels < unet.conv_in.in_channels:
            raise ValueError(
                f"new_in_channels ({new_in_channels}) must be >= current "
                f"{unet.conv_in.in_channels}"
            )
        if new_in_channels == unet.conv_in.in_channels:
            return

        old_conv = unet.conv_in
        old_in = old_conv.in_channels
        device, dtype = old_conv.weight.device, old_conv.weight.dtype
        bias_flag = old_conv.bias is not None

        new_conv = Conv2d(
            new_in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=bias_flag,
            padding_mode=old_conv.padding_mode,
        ).to(device=device, dtype=dtype)

        new_conv.weight.zero_()
        new_conv.weight[:, :old_in].copy_(old_conv.weight)
        if bias_flag:
            new_conv.bias.copy_(old_conv.bias)

        unet.conv_in = new_conv
        unet.config["in_channels"] = new_in_channels


    def set_valid_pad_conv(self, module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                if child.padding != (0, 0):
                    setattr(module, name, PaddedConv2d.from_existing(child, valid_pad_conv_fn))
                elif module.__class__.__name__ == "Downsample2D" and module.use_conv:
                    setattr(module, name, PaddedConv2d.from_existing(child, valid_pad_conv_fn, one_side_pad=True))
            else:
                self.set_valid_pad_conv(child)


    def save_model(self, ema_unet, model_save_dir):
        self.unwrapped_unet.save_pretrained(model_save_dir / "original")
        if ema_unet is not None:
            ema_unet.store(self.unwrapped_unet.parameters())
            ema_unet.copy_to(self.unwrapped_unet.parameters())
            self.unwrapped_unet.save_pretrained(model_save_dir / f"EMA")
            ema_unet.restore(self.unwrapped_unet.parameters())


    
