# 📟 PaGeR: Panorama Geometry Estimation using Single-Step Diffusion Models

This project implements **PaGeR**, a Computer Vision method for estimating geometry from monocular panoramic [ERP](https://en.wikipedia.org/wiki/Equirectangular_projection) images implemented in the paper **Panorama Geometry Estimation using Single-Step Diffusion Models**. 

[![Website](assets/badge-website.svg)](website here)
[![Paper](assets/badge-pdf.svg)](paper here)
[![Demo](https://img.shields.io/badge/🤗%20Demo-yellow)](https://huggingface.co/spaces/prs-eth/PaGeR)
[![Dataset and Models](https://img.shields.io/badge/🤗%20Collection-green)](https://huggingface.co/prs-eth/PaGeR)

Team:
[Vukasin Bozic](https://vulus98.github.io/),
[Isidora Slavkovic](https://www.linkedin.com/in/kevin-qu-b3417621b/),
[Dominik Narnhofer](https://scholar.google.com/citations?user=tFx8AhkAAAAJ&hl=en)
[Nando Metzger](https://nandometzger.github.io/),
[Denis Rozumny](https://rozumden.github.io/),
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ),
[Nikolai Kalischek](https://scholar.google.com/citations?user=XwzlnZoAAAAJ&hl=de)

We present **PaGeR**, a diffusion-based model for panoramic geometry reconstruction that extends monocular depth estimation to full 360° scenes. **PaGeR** is a one-step diffusion model trained directly in pixel space, capable of predicting high-resolution panoramic depth and surface normals with strong generalization to unseen environments. Leveraging advances in panorama generation and diffusion fine-tuning, **PaGeR** is trained on **PanoInfinigen**, a newly introduced synthetic dataset of indoor and outdoor scenes with metric depth and normals, producing coherent, metrically accurate geometry. It outperforms prior approaches across standard, few-shot, and zero-shot scenarios.

![teaser_all](assets/teaser.png)


## 📢 News
05-02-2026: Full training, inference, and evaluation code added, along with the [arXiv paper](https://arxiv.org/abs/2312.02145), [interactive demo](https://huggingface.co/spaces/prs-eth/PaGeR) and [depth](https://huggingface.co/prs-eth/PaGeR-depth), [metric depth](https://huggingface.co/prs-eth/PaGeR-depth) and [normals](https://huggingface.co/prs-eth/PaGeR-normals) model checkpoints. Full dataset release coming soon.

## 🚀 Usage

**There are several ways to interact with PaGeR**:

1. A quick start is to use our HF-hosted demo: 
<a href="https://huggingface.co/spaces/prs-eth/PaGeR"><img src="https://img.shields.io/badge/🤗%20Demo-yellow" height="16"></a> 

2. Run the demo locally (requires a 24VRAM GPU) -> see instructions below. 

3. Some interactive examples are also available at our project page: <a href="insert link here"><img src="assets/badge-website.svg" height="16"></a>

4. Finally, local development instructions with this codebase are given below.

## 🛠️ Setup

The code was tested on:

-  Debian GNU/Linux 12, Python 3.10.16,  PyTorch 2.2.0, and CUDA 12.1.

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/prs-eth/PaGeR.git
cd PaGeR
```

### 💻 Dependencies

Create the Conda environment and install the dependencies:

```bash
conda env create -f environment.yaml
```

### 🏁 Prepare the checkpoints

The model checkpoints are hosted on Hugging Face:
- Depth: [prs-eth/PaGeR-depth](https://huggingface.co/prs-eth/PaGeR-depth)
- Metric Depth: [prs-eth/PaGeR-metric-depth](https://huggingface.co/prs-eth/PaGeR-metric-depth)
- Normals: [prs-eth/PaGeR-normals](https://huggingface.co/prs-eth/PaGeR-normals)

Models specialized for indoor scenes are also available:
- Depth Indoor: [prs-eth/PaGeR-depth-indoor](https://huggingface.co/prs-eth/PaGeR-depth-indoor)
- Metric Depth Indoor: [prs-eth/PaGeR-metric-depth-indoor](https://huggingface.co/prs-eth/PaGeR-metric-depth-indoor)

As well as the Surface Normals Estimation model finetuned on Structured3D after the pretraining:
- Normals-Structured3D: [prs-eth/PaGeR-normals-Structured3D](https://huggingface.co/prs-eth/PaGeR-normals-Structured3D)

You can either download them automatically by specifying the HF checkpoint name in the arguments, or download them manually and load from a local path. If you choose the latter, please preserve the original folder structure, as in the Hugging Face repository.

### 📥 Download the datasets

For training, testing or evaluation, you would need to choose and download one or more of the following datasets:
- [PanoInfinigen(coming soon)]()
- [Matterport3D360](https://researchdata.bath.ac.uk/1126/)
- [Stanford2D3DS](https://sdss.redivis.com/datasets/f304-a3vhsvcaf)
- [Scannet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/)
- [Structured3D](https://structured3d-dataset.org/)
- [Replica360_4K](https://github.com/iszihan/replica-dataset)

For download instructions, terms of use, and dataset description, please refer to the webpages of the respective datasets.
We provide the dataloaders for all of these datasets. You just need to choose the respective dataset in the config file or command line argument.

## 📷 Local Gradio Demo

The easiest way to test PaGeR locally is to run the Gradio demo. Make sure you have installed the dependencies as described above, then run:

```bash
python app.py --enable_xformers
``` 
Now you can test the model, explore interactive 3D visualizations on both provided examples and your own images, or download the results.

## 🔧 Configuration settings
We use [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) and [argparse](https://docs.python.org/3/library/argparse.html) for configuration management in all our scripts and models. The parameters for running the script could be influenced by either setting it in the config script, or directly providing a parameter in the CLI. The latter will always take precedence. Note that the model loading parameters will always be loaded from a YAML config file stored along with the model checkpoint, and they won't be overwritten by the local config or CLI args. 
Feel free to set up your own configuration files; the template is given as `configs/base.yaml`.

## 🚀 Run inference

If you want to test models in the regular inference regime
```bash
# Depth
python inference.py \
    --configs "path/to/config" \
    --checkpoint_path "path/to/checkpoint" \
    --enable_xformers \
    --data_path "path/to/dataset" \
    --dataset "dataset-choice" \
    --results_path "path/to/save/results" \
    --pred_only \
```

### ⚙️ Inference settings

The behavior of the code can be customized in the following ways:

| Argument | Description |
|--------|-------------|
| `config` | Path to the YAML configuration file. |
| `checkpoint_path` | Model checkpoint to load (local path or HuggingFace repo ID). |
| `results_path` | Output directory where predictions are saved. |
| `dataset` | Dataset to use (list given above). |
| `data_path` | Root directory of the dataset. |
| `scenes` | Scene type to use: `indoor`, `outdoor`, or `both` (if supported). |
| `img_report_frequency` | Save an example output image every **N** samples. |
| `pred_only` | Save only the prediction image (otherwise saves an RGB + prediction mosaic). |
| `generate_eval` | Save predictions as `.npz` files for later evaluation or *Point Cloud Generation*. |
| `enable_xformers` | Enable memory-efficient attention (**recommended**). |

### 🧊 Point Cloud Generation

Once the inference results are generated, you can also visualize rgb- or surface normals- colored 3D point cloud:
```bash
python generate_point_cloud.py \
    --data_path "path/to/dataset" \
    --dataset "dataset-choice" \
    --color_modality "rgb-or-normals" \
    --depth_path "path/to/depth/predictions" \
    --normals_path "path/to/normals/predictions"
```

Note that you should run inference with `generate_eval` set to *True*, since this code will try to load raw predictions from `eval` folder used for evaluation.

## 📊 Run Evaluation

In order to run depth evaluation of inference results of our (or some other) model with the standard set of depth estimation [metrics](https://huggingface.co/blog/Isayoften/monocular-depth-estimation-guide):

```bash
# Depth
python evaluation/depth_evaluation.py \
    --pred_path "path/to/preds/folder" \
    --data_path "path/to/dataset" \
    --dataset "dataset-choice" \
    --alignment_type "alignment-type-to-apply" \
    --save_error_maps
```

Evaluation of the surface normals estimation could be done, similar to the [PanoNormal](https://arxiv.org/html/2405.18745v1) paper, by running the following command:

```bash
# Normals
python evaluation/normals_estimation.py \
    --pred_path "path/to/preds/folder" \
    --data_path "path/to/dataset" \
    --dataset "dataset-choice" \
```
Finally, edge sharpness evaluation is run as:
```bash
# Edges
python evaluation/edge_estimation.py \
    --pred_path "path/to/preds/folder" \
    --data_path "path/to/dataset" \
    --dataset "dataset-choice" \
```

### Evaluation Settings

The behavior of the code can be customized in the following ways:

| Argument | Description |
|--------|-------------|
| `data_path` | Root directory of the dataset. |
| `dataset` | Dataset to use (list given above). |
| `pred_path` | Directory containing the predicted depth maps to be evaluated. |
| `alignment_type`  | Alignment strategy applied between prediction and ground truth before evaluation. |
| `save_error_maps` | If set, saves per-sample error maps during evaluation. |
| `error_maps_saving_frequency` | Frequency (in number of batches) at which error maps are saved. |

---

## 🏋🏻 Run training 
The training for both depth and surface normals model is run from the single script, for example:

```bash
python train.py \
    --config "path/to/config" \
    --modality "depth" \
    --enable_xformers \
    --data_path "path/to/dataset" \
    --dataset "PanoInfinigen" \
    --log_scale \
    ...
```
Note again that the CLI arguments will overwrite the arguments given in the config file.

### Training settings

Here we provide an exhaustive list of training arguments along with the short description:

#### Global Settings
| Argument | Description |
| :--- | :--- |
| `debug` | Use a small subset of the dataset; useful for quick debugging. |
| `seed` | A seed for reproducible training. |
| `enable_xformers` | Enable memory-efficient attention (**recommended**). |

#### Training Configuration
| Argument | Description |
| :--- | :--- |
| `num_train_epochs` | Total number of training epochs to perform. |
| `max_train_steps` | Total number of training steps (overrides `num_train_epochs`). |
| `gradient_accumulation_steps` | Number of steps to accumulate before a backward/update pass. |
| `only_train_attention_layers` | Train only the attention parameters of the UNet model. |
| `gradient_checkpointing` | Enable to save memory (slower backward pass). |
| `resume_path` | Training checkpoint to resume from (expects an `Accelerator` folder). |
| `use_EMA` | Enable Exponential Moving Average (EMA) for model weights. |

#### Model Configuration
| Argument | Description |
| :--- | :--- |
| `modality` | Modality to use for training: `depth` or `normals`. |
| `pretrained_path` | Path to pretrained model or HuggingFace repo ID. |
| `checkpoint_path` | UNet checkpoint to load (loads `.safetensors` weights only). |
| `unet_positional_encoding` | Type of positional encoding: `uv`, `RoPE`, or `none`. |
| `vae_use_RoPE` | Whether or not to use RoPE positional encoding in the VAE. |
| `metric_depth` | Use metric depth instead of relative depth. Depth only. |
| `log_scale` | Use log scale depth instead of linear. Depth only. |

#### Data Configuration
| Argument | Description |
| :--- | :--- |
| `data_path` | Root directory of the training dataset. |
| `dataset` | Dataset selection (e.g., `PanoInfinigen`, `Matterport3D360`). |
| `scenes` | Scene type to use: `indoor`, `outdoor`, or `both`. |
| `batch_size` | Training batch size per device. |
| `use_data_augmentation` | Enable data augmentation (horizontal random rotation). |

#### Optimization
| Argument | Description |
| :--- | :--- |
| `learning_rate` | Initial learning rate. |
| `lr_exp_warmup_steps` | Ratio of steps for exponential LR warmup (e.g., 0.03 = 3%). |
| `adam_beta1` / `beta2` | Beta parameters for the Adam optimizer. |
| `adam_weight_decay` | Weight decay to use for optimization. |
| `adam_epsilon` | Epsilon value for the Adam optimizer. |
| `clip_grad_norm` | Enable gradient clipping. |
| `max_grad_norm` | Max gradient norm threshold. |

#### Loss Weights (depth training only)
| Argument | Description |
| :--- | :--- |
| `l1_loss_weight` | Weight for the L1 loss term. |
| `grad_loss_weight` | Weight for the gradient loss term. |
| `normals_consistency_loss_weight` | Weight for the normals consistency loss. |
| `invalid_mask_weight` | Weight for the invalid mask loss. |

#### Validation & Logging
| Argument | Description |
| :--- | :--- |
| `run_validation` | Whether to use the full validation set. |
| `run_tiny_validation` | Whether to use a smaller validation set for mid-training checks. |
| `tiny_val_frequency` | Frequency for running the tiny validation (in steps). |
| `tracker_project_name` | Project name for the experiment tracker. |
| `save_path` | Directory where predictions and checkpoints are saved. |
| `save_frequency` | Save the model every **X** epochs. |
| `loss_report_frequency` | How often to report loss (in steps). |
| `img_report_frequency` | How often to report/save image examples (in steps). |
| `report_to` | Logging backend: `tensorboard` or `wandb`. |
| `run_name` | Name for the WandB run. |

#### Resuming training

Along with the regular model checkpointing, full Accelerate checkpoint is saved as well in subfolder `training_checkpoint` inside the checkpointing folder. This enables the continuation of the training - set through the parameter `resume_path`. 

## ✏️ Contributing

Please refer to [this](CONTRIBUTING.md) instruction.

## 🎓 Citation

Please cite our paper:

```bibtex
Put citations here
```

## 🎫 License

This code of this work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

The models are licensed under RAIL++-M License (as defined in the [LICENSE-MODEL](LICENSE-MODEL.txt))

By downloading and using the code and model you agree to the terms in [LICENSE](LICENSE.txt) and [LICENSE-MODEL](LICENSE-MODEL.txt) respectively.

## Acknowledgements

This project builds upon and is inspired by the following repositories and works:

- [Marigold-e2e-ft](https://github.com/VisualComputingInstitute/diffusion-e2e-ft), based on paper [Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think](https://arxiv.org/abs/2409.11355).
- [Marigold](https://github.com/prs-eth/Marigold/tree/main), based on paper [Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://arxiv.org/abs/2312.02145).

We thank the authors and maintainers for making their code publicly available.
