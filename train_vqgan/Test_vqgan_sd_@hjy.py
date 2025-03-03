### Step 1: Install Required Libraries
# pip install torch torchvision transformers diffusers

### Step 2: Import Necessary Modules
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler
from PIL import Image
import numpy as np
from diffusers import VQModel,AutoencoderKL,StableDiffusionPipeline # for vqgan
"""
import numpy as np
import PIL
import PIL.Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
pipe = StableDiffusionPipeline.from_pretrained("/Users/jingyinghuang/Code/AIGC/stable-diffusion-v1-4")
prompt="sketch of a cat"
image = pipe(prompt).images[0]
image.save("astronaut_rides_horse.png")
"""

### Step 3: Load Models
# Load the pre-trained components of the Stable Diffusion model: 
# the UNet (for denoising), the CLIP model (for text conditioning), 
# and the scheduler (for diffusion steps).
def load_models(device,pretrained_model_name_or_path):
    # Load the pre-trained UNet for Stable Diffusion
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(device)

    # Load the scheduler for diffusion process
    scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    # Load the CLIP model for text encoding
    clip_model = CLIPTextModel.from_pretrained(pretrained_model_name_or_path,subfolder="text_encoder").to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path,subfolder="tokenizer")

    return unet, scheduler, clip_model, clip_tokenizer


### Step 4: Encode the Text (Prompt)
# We need to encode the text prompt into embeddings using CLIP’s text encoder.
def encode_text(prompt, clip_model, clip_tokenizer, device):
    # Tokenize the prompt
    inputs = clip_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    # Get the text embeddings from CLIP
    with torch.no_grad():
        text_embeddings = clip_model.get_input_embeddings()(inputs['input_ids'])

    return text_embeddings

### Step 5: Sample Latents
# You need to generate the initial noise (latents) which will go through 
# the denoising process. The latents are typically sampled from a Gaussian 
# distribution.
def generate_latents(shape, device):
    # Create random noise latents (Gaussian noise)
    latents = torch.randn(shape, device=device).to(device)

    return latents

# latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

### Step 6: Denoising Loop (Diffusion Process)
# Now you’ll iterate through the denoising steps. This is where the diffusion 
# process happens. You use the `UNet` model to denoise the latents step by step.
def denoise_latents(latents, text_embeddings, unet, scheduler, num_inference_steps, device, do_classifier_free_guidance=True,guidance_scale = 7.5):
    # Initialize the latents with noise (start with the noisy latents)
    timesteps = scheduler.timesteps
    for t in timesteps[:num_inference_steps]:

        print(t)
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        # Get the predicted noise and the model's predicted clean image
        # import pdb 
        # pdb.set_trace()
        print("lmi",latent_model_input.shape) # torch.Size([2, 4, 64, 64])
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents

            # for i, t in enumerate(timesteps):
            #     if self.interrupt:
            #         continue

            #     # expand the latents if we are doing classifier free guidance
            #     latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            #     latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            #     # predict the noise residual
            #     noise_pred = self.unet(
            #         latent_model_input,
            #         t,
            #         encoder_hidden_states=prompt_embeds,
            #         timestep_cond=timestep_cond,
            #         cross_attention_kwargs=self.cross_attention_kwargs,
            #         added_cond_kwargs=added_cond_kwargs,
            #         return_dict=False,
            #     )[0]

            #     # perform guidance
            #     if self.do_classifier_free_guidance:
            #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            #         noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            #     if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
            #         # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            #         noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            #     # compute the previous noisy sample x_t -> x_t-1
            #     latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]


### Step 7: Decode Latents to Image
# After denoising, you’ll have to decode the latents back into an image. 
# If you’re using a VQGAN-based encoder, you’ll need to pass the latents 
# through the decoder, but for the general Stable Diffusion model, 
# the decoder is already built into the VAE, and you can use that to convert 
# the latents into an image.If you want to decode with a VAE, you can add 
# that step, but here’s a simple method using a pre-trained VAE for decoding:

def vae_decode_latents(latents, vae, device):
    with torch.no_grad():
        decoded_image = vae.decode(latents).sample
    return decoded_image
    # image = self.vae.decode(latents, return_dict=False)[0]
    # image = (image / 2 + 0.5).clamp(0, 1)
    # # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    # image = image.cpu().permute(0, 2, 3, 1).float().numpy()

def vqgan_decode_latents(latents, vqgan, device):
    with torch.no_grad():
        decoded_image = vqgan.decode(latents).sample # includes quantize
    return decoded_image

### Step 8: Image Post-Processing
# Once you have the decoded image (latents turned into pixel space), 
# you can post-process and convert the image into something that can 
# be viewed or saved:
def do_norm(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    return  image
# (images / 2 + 0.5).clamp(0, 1)

def post_process_image(image_tensor):
    # Convert the tensor to a numpy array and scale the pixel values
    
    image = image_tensor.squeeze().cpu().numpy()
    image = (image * 255).clip(0, 255).astype(np.uint8)
    # image = np.transpose(image, (1, 2, 0))  # Convert CHW to HWC for PIL
    

        # if images.ndim == 3:
        #     images = images[None, ...]
        # images = (images * 255).round().astype("uint8")
        # if images.shape[-1] == 1:
        #     # special case for grayscale (single channel) images
        #     pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        # else:
        #     pil_images = [Image.fromarray(image) for image in images]
    return Image.fromarray(image)

### Step 9: Putting It All Together (Inference)
# Finally, you can combine everything to perform the inference:
@torch.no_grad()
def run_inference(pretrained_model_name_or_path, prompt, decoder_subfolder_name,num_inference_steps=10, batch_size=1, latent_dim=(4, 64, 64), device='cuda'):
    # Load models

    unet, scheduler, clip_model, clip_tokenizer = load_models(device, pretrained_model_name_or_path)

    # Encode the text prompt into text embeddings
    text_embeddings = encode_text(prompt, clip_model, clip_tokenizer, device)

    # Generate initial latents (random noise)
    latent_shape=(batch_size, latent_dim[0],latent_dim[1],latent_dim[2])
    # print(latent_shape)
    latents = generate_latents(latent_shape, device)

    # Perform the denoising process (the diffusion process)
    latents = denoise_latents(latents, text_embeddings, unet, scheduler, num_inference_steps, device)

    # Decode the latents back to an image
    # Assuming you have a VAE, this is how you'd decode it (if you're using the default VAE from Stable Diffusion)
    # vae = VAE model (Load this model if necessary)

    if decoder_subfolder_name=="vae":
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder=decoder_subfolder_name,)
        vae.to(device)
        decoded_image = vae_decode_latents(latents, vae, device)
    elif decoder_subfolder_name=="vqmodel":
        vqgan = VQModel.from_pretrained(pretrained_model_name_or_path, subfolder=decoder_subfolder_name,)
        decoded_image = vqgan_decode_latents(latents, vqgan, device)
    else:
        raise ValueError("Specific decoder subfolder name: vae or vqmodel")

    # Post-process and return the image
    final_image = post_process_image(do_norm(decoded_image))

    return final_image

def test_vqgan_sd(prompt = "sketch of a cat",
    pretrained_model_name_or_path="/Users/jingyinghuang/Code/AIGC/stable-diffusion-v1-4/",
    decoder_subfolder_name="vae"):

    # pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
    # image = pipe(prompt,num_inference_steps=10).images[0]
    # image.save("StableDiffusionPipeline_cat.png")

    print("start run_inference")
    image = run_inference(pretrained_model_name_or_path, prompt, decoder_subfolder_name,device="cpu")

    # Display or save the image
    # image.show()
    # Optionally, save the image
    image.save("run_inference_cat.png")
    return image

test_vqgan_sd()