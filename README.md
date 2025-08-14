# xflux-trainer

## Finetune Lora
```python
accelerate launch --config_file "train_configs/ds_zero2.yaml" train_flux_lora_deepspeed.py --config "train_configs/test_lora.yaml"
```


## Image Caption Generation
First, run SGLang:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang_router.launch_server   --model-path Qwen/Qwen2.5-VL-7B-Instruct   --dp-size 8   --host 0.0.0.0 --port 30000
```
Then, request local api:
```bash
python caption_dataset.py \
  --input-dir /data/tian/Project/FluxTrainer/HQ-Character-Data/images \
  --out-jsonl /data/tian/Project/FluxTrainer/HQ-Character-Data/captions.jsonl \
  --endpoint http://127.0.0.1:30000 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --workers 80 \
  --max-tokens 256 \
  --max-side 1024
```


## Face Detection&Extract Face Embedding
```bash
python face_crop_embed_ray.py \
  --input-dir /data/tian/Project/FluxTrainer/HQ-Character-Data/images  \
  --out-dir /data/tian/Project/FluxTrainer/HQ-Character-Data/out_faces \
  --gpus 8 \
  --det-size 1024 \
  --det-thresh 0.35 \
  --margin 0.12
```
# DanceGRPO
