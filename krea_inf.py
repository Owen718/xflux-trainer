import os
# os.environ["HF_HOME"] = "/data/tian/Project/FluxTrainer/cache"
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM

prompt = [
"The image shows a person with vibrant pink hair styled in loose waves. They are wearing a black leather jacket adorned with silver studs and colorful patches, including stars and other designs. Underneath the jacket, a black shirt with white lettering is partially visible. The person is holding a large bouquet of flowers featuring various shades of pink and purple blooms. They are also wearing black fingerless gloves. The background appears to be an outdoor urban setting with a blurred pavement and stone surface.",
"The image is a black and white photograph capturing a woman standing barefoot on a pebbled beach. She is facing away from the camera, looking towards the sea. The woman is wearing a long, flowing dress made of a light, translucent fabric that billows gently in the breeze. Her hair is shoulder-length and slightly wavy. The background shows a calm body of water extending to the horizon, with a faint outline of a landmass or cliff in the distance. The overall mood of the image is serene and contemplative.",
"The image is a black and white photograph of a woman captured from the shoulders up to just below her chin, with her face partially out of frame. The focus is on her neck, collarbones, and upper chest. She is wearing an open shirt that is slipping off her shoulders, revealing her bare skin underneath. Her hands are gently holding the shirt near her chest. The lighting creates soft shadows that accentuate the contours of her neck and shoulders, giving the image a delicate and intimate feel. The overall mood is elegant and serene.",
"The black and white image shows a well-dressed couple walking hand in hand in front of a Tom Ford store. The man is wearing a light-colored shirt tucked into trousers with dark shoes. The woman is dressed in a flowing, long-sleeved dress paired with high heels. She is holding a small bouquet of flowers in her left hand. The background features the store's glass entrance and a large display window showcasing a mannequin dressed in a stylish outfit. The pavement beneath them is patterned with alternating light and dark bricks. The overall scene conveys a sense of elegance and sophistication.",
"The image shows a silhouette of a person standing outdoors against a bright, cloudy sky with the sun shining through. The person appears to have curly hair and is wearing a sleeveless dress. They are holding a small bouquet of wildflowers or delicate foliage in one hand, raised slightly. The lighting creates a soft, ethereal glow around the figure, and the details of their face and clothing are obscured by the backlighting, giving the image a dreamy, almost ghostly effect. The overall mood is serene and contemplative.",
"The image shows a person from behind with their hands placed on the back of their head. The individual has light skin and shoulder-length hair that appears to be light brown with hints of purple or blue, blowing slightly in the wind. They are wearing a beige, knitted sweater with a wide V-shaped back neckline, exposing the upper back. Around their neck, there is a distinctive necklace featuring an intricate design with a central blue stone and delicate, branch-like metalwork extending from it. The person's fingernails are painted a dark blue or purple color, and they are wearing several rings on their fingers. The background is blurred, focusing attention on the person and their necklace."
]

# for i,p in enumerate(prompt):
#     image = pipe(
#         p,
#         height=1024,
#         width=1024,
#         guidance_scale=3.0,
#         num_inference_steps=20,
#         generator=torch.Generator(device="cuda").manual_seed(42)
#     ).images[0]
#     image.save(f"flux-krea-dev-{i}.png")

pipe.load_lora_weights("/data/tian/Project/FluxTrainer/x-flux/lora/checkpoint-800/lora_sd_scripts.safetensors",adapter_name="lora")
for i,p in enumerate(prompt):
    image = pipe(
        p,
        height=1024,
        width=1024,
        guidance_scale=3.0,
        num_inference_steps=20,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]
    image.save(f"flux-krea-dev-{i}-lora.png")