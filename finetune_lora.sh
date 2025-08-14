# export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch --config_file "train_configs/ds_zero2.yaml" train_flux_lora_siglip_deepspeed.py --config "train_configs/test_lora_grpo.yaml"