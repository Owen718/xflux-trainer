import argparse
import logging
import math
import os
import re
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from safetensors.torch import save_file

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder, hf_hub_download
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from safetensors.torch import safe_open

from transformers import SiglipVisionModel, AutoProcessor
from diffusers import FluxPriorReduxPipeline

from src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from src.flux.util import (
    configs,
    load_ae,
    load_clip,
    load_flow_model2,
    load_t5,
)
from src.flux.modules.layers import (
    DoubleStreamBlockLoraProcessor,
    SingleStreamBlockLoraProcessor,
)
from src.flux.xflux_pipeline import XFluxSampler

# 使用 JSONL 数据集以获得 face_image_siglip
from torch.utils.data import DataLoader
from image_datasets.dataset import JsonlImageDataset

if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")


def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    clip.requires_grad_(False)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip


def parse_args():
    parser = argparse.ArgumentParser(description="Training script with SigLIP+Redux prior for face guidance.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()
    return args.config


def cfg_get(cfg, key, default=None):
    """兼容 OmegaConf DictConfig 与原生 dict/命名空间的取值助手。"""
    if cfg is None:
        return default
    if hasattr(cfg, 'get'):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    try:
        return getattr(cfg, key)
    except Exception:
        pass
    try:
        return cfg[key]
    except Exception:
        return default


def jsonl_collate_fn(batch):
    """
    自定义 collate：
    - 保证 face_image_siglip 缺失时使用全零 Tensor 占位，避免 DataLoader 因 None 报错
    - 其余键按常规堆叠或列表聚合
    """
    images = [sample["image"] for sample in batch]
    captions = [sample["caption"] for sample in batch]

    # face_image_siglip: [3, 512, 512]，允许 None -> 用全零替代
    face_tensors = []
    for sample in batch:
        t = sample.get("face_image_siglip", None)
        if t is None:
            t = torch.zeros(3, 512, 512, dtype=torch.float32)
        face_tensors.append(t)

    images = torch.stack(images, dim=0)
    face_batch = torch.stack(face_tensors, dim=0)

    # 其他字段通常在训练中不使用，这里聚合保留，便于后续扩展/调试
    paths = [sample.get("path", "") for sample in batch]
    sha1s = [sample.get("sha1", "") for sample in batch]
    faces_meta = [sample.get("face", {}) for sample in batch]

    return {
        "image": images,
        "caption": captions,
        "face_image_siglip": face_batch,
        "path": paths,
        "sha1": sha1s,
        "face": faces_meta,
    }


def build_redux_prior_pipeline(siglip_ckpt: str, redux_repo_id: str, redux_filename: str, device: torch.device, dtype: torch.dtype):
    """
    构建 SigLIP + Redux Image Embedder 管线（与推理保持一致）：
    - SiglipVisionModel + AutoProcessor 负责图像特征抽取
    - ReduxImageEncoder 负责将视觉特征映射为可拼接的 prompt_embeds
    - 权重从 HF 仓库下载并注入
    """
    siglip_model = SiglipVisionModel.from_pretrained(siglip_ckpt)
    siglip_processor = AutoProcessor.from_pretrained(siglip_ckpt)
    from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder

    redux_image_encoder = ReduxImageEncoder()
    pretrained_model_path = hf_hub_download(repo_id=redux_repo_id, filename=redux_filename)
    with safe_open(pretrained_model_path, framework="pt") as f:
        if "redux_up.weight" in f.keys():
            redux_image_encoder.redux_up.weight.data = f.get_tensor("redux_up.weight")
        if "redux_up.bias" in f.keys():
            redux_image_encoder.redux_up.bias.data = f.get_tensor("redux_up.bias")
        if "redux_down.weight" in f.keys():
            redux_image_encoder.redux_down.weight.data = f.get_tensor("redux_down.weight")
        if "redux_down.bias" in f.keys():
            redux_image_encoder.redux_down.bias.data = f.get_tensor("redux_down.bias")

    pipe_prior_redux = FluxPriorReduxPipeline(
        image_encoder=siglip_model,
        feature_extractor=siglip_processor.image_processor,
        image_embedder=redux_image_encoder,
    ).to(device).to(dtype)
    # pipe_prior_redux.set_progress_bar_config(disable=True)
    # pipe_prior_redux.eval()
    return pipe_prior_redux



def main():
    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # 基本日志配置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    dit, vae, t5, clip = get_models(name=args.model_name, device=accelerator.device, offload=False, is_schnell=is_schnell)
    lora_attn_procs = {}

    if args.double_blocks is None:
        double_blocks_idx = list(range(19))
    else:
        double_blocks_idx = [int(idx) for idx in args.double_blocks.split(",")]

    if args.single_blocks is None:
        single_blocks_idx = list(range(38))
    elif args.single_blocks is not None:
        single_blocks_idx = [int(idx) for idx in args.single_blocks.split(",")]

    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks") and layer_index in double_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
              dim=3072, rank=args.rank
            )
        elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = SingleStreamBlockLoraProcessor(
              dim=3072, rank=args.rank
            )
        else:
            lora_attn_procs[name] = attn_processor

    dit.set_attn_processor(lora_attn_procs)

    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit = dit.to(torch.float32)
    dit.train()
    optimizer_cls = torch.optim.AdamW
    for n, param in dit.named_parameters():
        if '_lora' not in n:
            param.requires_grad = False
        else:
            print(n)
    print(sum([p.numel() for p in dit.parameters() if p.requires_grad]) / 1000000, 'parameters')
    optimizer = optimizer_cls(
        [p for p in dit.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ------------------------------
    # 构建 JSONL DataLoader（含 face_image_siglip）
    # 说明：不复用现有 loader()，以便注入自定义 collate_fn 处理 None
    # ------------------------------
    data_cfg = args.data_config
    index_jsonl_path = cfg_get(data_cfg, 'index_jsonl_path')
    img_size = cfg_get(data_cfg, 'img_size', 1024)
    random_ratio = cfg_get(data_cfg, 'random_ratio', True)
    num_workers = cfg_get(data_cfg, 'num_workers', cfg_get(args, 'num_workers', 4))

    dataset = JsonlImageDataset(
        index_jsonl_path=index_jsonl_path,
        img_size=img_size,
        random_ratio=random_ratio,
    )
    train_dataloader = DataLoader(
        dataset,
        # 与原训练脚本保持一致，batch size 使用顶层 args.train_batch_size
        batch_size=args.train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=jsonl_collate_fn,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    dit, optimizer, _, lr_scheduler = accelerator.prepare(
        dit, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = get_schedule(
                999,
                (1024 // 8) * (1024 // 8) // 4,
                shift=True,
            )
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # ------------------------------
    # 构建 SigLIP+Redux 先验（与推理一致）
    # ------------------------------
    use_redux_prior = getattr(args, 'use_redux_prior', True)
    if use_redux_prior:
        siglip_ckpt = getattr(args, 'siglip_ckpt', 'google/siglip2-so400m-patch16-512')
        redux_repo_id = getattr(args, 'redux_repo_id', 'ostris/Flex.1-alpha-Redux')
        redux_filename = getattr(args, 'redux_filename', 'flex1_redux_siglip2_512.safetensors')
        prior_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' or torch.cuda.is_available() else torch.float32
        pipe_prior_redux = build_redux_prior_pipeline(siglip_ckpt, redux_repo_id, redux_filename, accelerator.device, prior_dtype)
    else:
        pipe_prior_redux = None

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(dit):
                # 读取 JSONL DataLoader 输出
                img = batch["image"]                # [B, 3, H, W], 值域约 [-1, 1]
                prompts = batch["caption"]          # List[str]
                face_siglip = batch["face_image_siglip"]  # [B, 3, 512, 512], 已按 SigLIP 规范归一化

                with torch.no_grad():
                    x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))
                    inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)

                    # ------------------------------
                    # 关键改动：将 face_image_siglip 通过 SigLIP+Redux 产生先验嵌入，并拼接到文本嵌入后
                    # - 直接绕过 feature_extractor：batch 的 face_image_siglip 已是 SigLIP 预处理后的张量（[-1,1]、[B,3,512,512]）
                    # - 直接送入 image_encoder(pixel_values=...)，再通过 image_embedder 得到 image_embeds
                    # - 将 image_embeds 与文本嵌入 inp['txt'] 在序列维度拼接，随后重置 txt_ids
                    # ------------------------------
                    if pipe_prior_redux is not None:
                        # 保持值域 [-1,1]，与 SigLIP 期望一致；仅做 dtype/device 对齐
                        siglip_dtype = next(pipe_prior_redux.image_encoder.parameters()).dtype
                        pixel_values = face_siglip.to(device=accelerator.device, dtype=siglip_dtype)

                        # 直接走编码器与 redux 映射（跳过 feature_extractor 的 preprocess）
                        image_latents = pipe_prior_redux.image_encoder(pixel_values=pixel_values).last_hidden_state
                        image_embeds = pipe_prior_redux.image_embedder(image_latents).image_embeds

                        # 对齐 dtype/device，并拼接到现有文本 token embeddings 后
                        image_embeds = image_embeds.to(inp['txt'].dtype).to(inp['txt'].device)
                        inp['txt'] = torch.cat([inp['txt'], image_embeds], dim=1)
                        B, L, _ = inp['txt'].shape
                        inp['txt_ids'] = torch.zeros((B, L, 3), dtype=inp['txt_ids'].dtype, device=inp['txt_ids'].device)

                    # 将 VAE 编码的图像转为序列表示，保持原训练流程
                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                bs = img.shape[0]
                t = torch.tensor([timesteps[random.randint(0, 999)]]).to(accelerator.device)
                x_0 = torch.randn_like(x_1).to(accelerator.device)
                x_t = (1 - t) * x_1 + t * x_0
                bsz = x_1.shape[0]
                guidance_vec = torch.full((x_t.shape[0],), 1, device=x_t.device, dtype=x_t.dtype)

                # 前向预测与损失
                model_pred = dit(img=x_t.to(weight_dtype),
                                img_ids=inp['img_ids'].to(weight_dtype),
                                txt=inp['txt'].to(weight_dtype),
                                txt_ids=inp['txt_ids'].to(weight_dtype),
                                y=inp['vec'].to(weight_dtype),
                                timesteps=t.to(weight_dtype),
                                guidance=guidance_vec.to(weight_dtype),)

                loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                # 分布式日志聚合
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # 反传与优化
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if not args.disable_sampling and global_step % args.sample_every == 0:
                    if accelerator.is_main_process:
                        print(f"Sampling images for step {global_step}...")
                        sampler = XFluxSampler(clip=clip, t5=t5, ae=vae, model=dit, device=accelerator.device)
                        images = []
                        for i, prompt in enumerate(args.sample_prompts):
                            result = sampler(prompt=prompt,
                                             width=args.sample_width,
                                             height=args.sample_height,
                                             num_steps=args.sample_steps
                                             )
                            images.append(wandb.Image(result))
                            print(f"Result for prompt #{i} is generated")
                        wandb.log({f"Results, step {global_step}": images})

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                    accelerator.save_state(save_path)
                    unwrapped_model_state = accelerator.unwrap_model(dit).state_dict()

                    # 仅保存 LoRA 权重到 safetensors
                    lora_state_dict = {k: unwrapped_model_state[k] for k in unwrapped_model_state.keys() if '_lora' in k}
                    save_file(
                        lora_state_dict,
                        os.path.join(save_path, "lora.safetensors")
                    )

                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()


