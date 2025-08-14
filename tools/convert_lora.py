import os
import torch
import safetensors.torch


# 源 xlabs LoRA 权重路径（保持你之前的用法）
Lora_path = "/data/tian/Project/FluxTrainer/x-flux/lora/checkpoint-800/lora.safetensors"


def _set_sd_scripts_lora_module(sds_sd: dict, sds_key: str, down: torch.Tensor, up: torch.Tensor) -> None:
    """将一对 (down, up) 写入 sd-scripts 命名空间，并设置 alpha=rank。"""
    rank = int(down.shape[0])
    sds_sd[sds_key + ".lora_down.weight"] = down
    sds_sd[sds_key + ".lora_up.weight"] = up
    sds_sd[sds_key + ".alpha"] = torch.scalar_tensor(rank, dtype=down.dtype, device=down.device)


def convert_xlabs_to_sd_scripts(xlabs_sd: dict) -> dict:
    """
    将 xlabs LoRA 权重（double_blocks/single_blocks 结构）转换为 sd-scripts 命名空间：
    - double_blocks.i.processor.proj_lora1  -> lora_unet_double_blocks_{i}_img_attn_proj
    - double_blocks.i.processor.proj_lora2  -> lora_unet_double_blocks_{i}_txt_attn_proj
    - double_blocks.i.processor.qkv_lora1   -> lora_unet_double_blocks_{i}_img_attn_qkv
    - double_blocks.i.processor.qkv_lora2   -> lora_unet_double_blocks_{i}_txt_attn_qkv
    - single_blocks.i.processor.proj_lora   -> lora_unet_single_blocks_{i}_linear2
    - single_blocks.i.processor.qkv_lora    -> lora_unet_single_blocks_{i}_linear1（上权重补零到 21504 行）
    仅在对应 down/up 同时存在时才会写入目标键。
    """
    sds_sd: dict[str, torch.Tensor] = {}

    keys = list(xlabs_sd.keys())

    # 解析 double/single 的所有存在的层索引
    double_indices = sorted({int(k.split(".")[1]) for k in keys if k.startswith("double_blocks.")})
    single_indices = sorted({int(k.split(".")[1]) for k in keys if k.startswith("single_blocks.")})

    # 处理 double_blocks
    for i in double_indices:
        base = f"double_blocks.{i}.processor"

        # proj_lora1 -> img_attn_proj
        down_key = f"{base}.proj_lora1.down.weight"
        up_key = f"{base}.proj_lora1.up.weight"
        if down_key in xlabs_sd and up_key in xlabs_sd:
            _set_sd_scripts_lora_module(
                sds_sd,
                f"lora_unet_double_blocks_{i}_img_attn_proj",
                xlabs_sd[down_key],
                xlabs_sd[up_key],
            )

        # proj_lora2 -> txt_attn_proj
        down_key = f"{base}.proj_lora2.down.weight"
        up_key = f"{base}.proj_lora2.up.weight"
        if down_key in xlabs_sd and up_key in xlabs_sd:
            _set_sd_scripts_lora_module(
                sds_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_proj",
                xlabs_sd[down_key],
                xlabs_sd[up_key],
            )

        # qkv_lora1 -> img_attn_qkv
        down_key = f"{base}.qkv_lora1.down.weight"
        up_key = f"{base}.qkv_lora1.up.weight"
        if down_key in xlabs_sd and up_key in xlabs_sd:
            _set_sd_scripts_lora_module(
            sds_sd,
            f"lora_unet_double_blocks_{i}_img_attn_qkv",
                xlabs_sd[down_key],
                xlabs_sd[up_key],
            )

        # qkv_lora2 -> txt_attn_qkv
        down_key = f"{base}.qkv_lora2.down.weight"
        up_key = f"{base}.qkv_lora2.up.weight"
        if down_key in xlabs_sd and up_key in xlabs_sd:
            _set_sd_scripts_lora_module(
            sds_sd,
            f"lora_unet_double_blocks_{i}_txt_attn_qkv",
                xlabs_sd[down_key],
                xlabs_sd[up_key],
            )

    # 处理 single_blocks
    for i in single_indices:
        base = f"single_blocks.{i}.processor"

        # proj_lora -> linear2（直接拷贝）
        down_key = f"{base}.proj_lora.down.weight"
        up_key = f"{base}.proj_lora.up.weight"
        if down_key in xlabs_sd and up_key in xlabs_sd:
            _set_sd_scripts_lora_module(
                sds_sd,
                f"lora_unet_single_blocks_{i}_linear2",
                xlabs_sd[down_key],
                xlabs_sd[up_key],
            )

        # qkv_lora -> linear1（上权重需要补零到 21504 = 3072*3 + 12288）
        down_key = f"{base}.qkv_lora.down.weight"
        up_key = f"{base}.qkv_lora.up.weight"
        if down_key in xlabs_sd and up_key in xlabs_sd:
            down_w = xlabs_sd[down_key]
            up_w = xlabs_sd[up_key]

            # 期望 up_w 为 [9216, rank]，需要填充到 [21504, rank]
            out_dim_expected_qkv = 3072 * 3  # 9216
            out_dim_linear1 = 3072 * 3 + 12288  # 21504

            if up_w.shape[0] != out_dim_expected_qkv:
                raise ValueError(
                    f"single_blocks.{i}.processor.qkv_lora.up.weight 的行数应为 {out_dim_expected_qkv}，实际为 {up_w.shape[0]}"
                )

            rank = int(down_w.shape[0])
            up_padded = torch.zeros((out_dim_linear1, rank), dtype=up_w.dtype, device=up_w.device)
            up_padded[:out_dim_expected_qkv, :] = up_w

            _set_sd_scripts_lora_module(
            sds_sd,
            f"lora_unet_single_blocks_{i}_linear1",
                down_w,
                up_padded,
            )

    return sds_sd


def main() -> None:
    # 读取 xlabs 权重
    print(f"Loading xlabs LoRA: {Lora_path}")
    xlabs_sd = safetensors.torch.load_file(Lora_path)
    print(f"Loaded {len(xlabs_sd)} tensors. Starting conversion (xlabs -> sd-scripts)...")

    # 转换
    sds_sd = convert_xlabs_to_sd_scripts(xlabs_sd)
    print(f"Converted {len(sds_sd)} tensors to sd-scripts format.")

    # 保存到同目录 *_sd_scripts.safetensors
    base, _ = os.path.splitext(Lora_path)
    dst_path = base + "_sd_scripts.safetensors"
    safetensors.torch.save_file(sds_sd, dst_path)
    print(f"Saved sd-scripts LoRA to: {dst_path}")


if __name__ == "__main__":
    main()


