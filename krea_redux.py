import torch
from diffusers import FluxPriorReduxPipeline, FluxPipeline
from diffusers.utils import load_image
from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
from transformers import SiglipVisionModel,AutoProcessor
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
ckpt = "google/siglip2-so400m-patch16-512"
siglip_model = SiglipVisionModel.from_pretrained(ckpt)
siglip_processor = AutoProcessor.from_pretrained(ckpt)
print(siglip_processor)
exit()
redux_image_encoder = ReduxImageEncoder()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

pretrained_model_path = hf_hub_download(
    repo_id="ostris/Flex.1-alpha-Redux",
    filename="flex1_redux_siglip2_512.safetensors",
)

with safe_open(pretrained_model_path, framework="pt")  as f:
  # Load weights for redux_up
  if "redux_up.weight" in f.keys():
      redux_image_encoder.redux_up.weight.data = f.get_tensor("redux_up.weight")
  if "redux_up.bias" in f.keys():
      redux_image_encoder.redux_up.bias.data = f.get_tensor("redux_up.bias")
  # Load weights for redux_down
  if "redux_down.weight" in f.keys():
      redux_image_encoder.redux_down.weight.data = f.get_tensor("redux_down.weight")
  if "redux_down.bias" in f.keys():
      redux_image_encoder.redux_down.bias.data = f.get_tensor("redux_down.bias")

pipe_prior_redux = FluxPriorReduxPipeline(
    image_encoder=siglip_model,
    feature_extractor=siglip_processor.image_processor,
    image_embedder=redux_image_encoder
    ).to(device).to(dtype)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-krea-dev" , 
    text_encoder=None,
    text_encoder_2=None,
    torch_dtype=torch.bfloat16
).to("cuda")

image = load_image("/data/tian/Project/FluxTrainer/images/girl_model/4820567_original.jpeg")
width, height = image.size
width = 512
height = 768
pipe_prior_output = pipe_prior_redux(image)
print(pipe_prior_output.keys())
images = pipe(
    guidance_scale=2.5,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0),
    **pipe_prior_output,
    width=width,
    height=height
).images
images[0].save("flux-dev-redux.png")
