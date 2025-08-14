import argparse
import os
from omegaconf import OmegaConf

import torch

from src.flux.util import (
    load_t5,
    load_clip,
    load_flow_model2,
    load_ae,
)

from src.flux.xflux_pipeline_faceid import XFluxSampler


def parse_args() -> str:
    parser = argparse.ArgumentParser(description="Predict script for X-Flux")
    parser.add_argument(
        "--config",
        type=str,
        default="train_configs/test_lora.yaml",
        help="path to config yaml",
    )
    args = parser.parse_args()
    return args.config


def get_models(name: str, device: torch.device):
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    clip.requires_grad_(False)

    # 与训练脚本一致，基础模型用 load_flow_model2 加载
    dit = load_flow_model2(name, device="cpu")
    # 推理用 bfloat16，更省显存
    dit = dit.to(device=device, dtype=torch.bfloat16)

    vae = load_ae(name, device=device)
    return dit, vae, t5, clip


@torch.inference_mode()
def main():
    cfg = OmegaConf.load(parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg.model_name

    dit, vae, t5, clip = get_models(name=model_name, device=device)

    sampler = XFluxSampler(clip=clip, t5=t5, ae=vae, model=dit, device=device)

    # 读取采样配置，沿用训练配置字段
    sample_prompts = [
    ""
    # "The image shows a person with vibrant pink hair styled in loose waves. They are wearing a black leather jacket adorned with silver studs and colorful patches, including stars and other designs. Underneath the jacket, a black shirt with white lettering is partially visible. The person is holding a large bouquet of flowers featuring various shades of pink and purple blooms. They are also wearing black fingerless gloves. The background appears to be an outdoor urban setting with a blurred pavement and stone surface.",
    # "The image is a black and white photograph capturing a woman standing barefoot on a pebbled beach. She is facing away from the camera, looking towards the sea. The woman is wearing a long, flowing dress made of a light, translucent fabric that billows gently in the breeze. Her hair is shoulder-length and slightly wavy. The background shows a calm body of water extending to the horizon, with a faint outline of a landmass or cliff in the distance. The overall mood of the image is serene and contemplative.",
    # "The image is a black and white photograph of a woman captured from the shoulders up to just below her chin, with her face partially out of frame. The focus is on her neck, collarbones, and upper chest. She is wearing an open shirt that is slipping off her shoulders, revealing her bare skin underneath. Her hands are gently holding the shirt near her chest. The lighting creates soft shadows that accentuate the contours of her neck and shoulders, giving the image a delicate and intimate feel. The overall mood is elegant and serene.",
    # "The black and white image shows a well-dressed couple walking hand in hand in front of a Tom Ford store. The man is wearing a light-colored shirt tucked into trousers with dark shoes. The woman is dressed in a flowing, long-sleeved dress paired with high heels. She is holding a small bouquet of flowers in her left hand. The background features the store's glass entrance and a large display window showcasing a mannequin dressed in a stylish outfit. The pavement beneath them is patterned with alternating light and dark bricks. The overall scene conveys a sense of elegance and sophistication.",
    # "The image shows a silhouette of a person standing outdoors against a bright, cloudy sky with the sun shining through. The person appears to have curly hair and is wearing a sleeveless dress. They are holding a small bouquet of wildflowers or delicate foliage in one hand, raised slightly. The lighting creates a soft, ethereal glow around the figure, and the details of their face and clothing are obscured by the backlighting, giving the image a dreamy, almost ghostly effect. The overall mood is serene and contemplative.",
    # "The image shows a person from behind with their hands placed on the back of their head. The individual has light skin and shoulder-length hair that appears to be light brown with hints of purple or blue, blowing slightly in the wind. They are wearing a beige, knitted sweater with a wide V-shaped back neckline, exposing the upper back. Around their neck, there is a distinctive necklace featuring an intricate design with a central blue stone and delicate, branch-like metalwork extending from it. The person's fingernails are painted a dark blue or purple color, and they are wearing several rings on their fingers. The background is blurred, focusing attention on the person and their necklace."
    ]
    width = 512
    height = 768
    steps = int(cfg.sample_steps)

    out_dir = getattr(cfg, "output_dir", "outputs")
    out_dir = os.path.join(out_dir, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    ref_image = "/data/tian/Project/FluxTrainer/images/girl_model/4820567_original.jpeg"

    for i, prompt in enumerate(sample_prompts):
        img = sampler(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=steps,
            guidance=3,
            seed=42,
            timestep_to_start_cfg=100,
            input_ref_image=ref_image,
        )
        save_path = os.path.join(out_dir, f"pred_{i}.png")
        img.save(save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()


