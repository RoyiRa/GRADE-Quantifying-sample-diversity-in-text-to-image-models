import random
import pandas as pd
import torch
import os
from diffusers import StableDiffusionXLPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import FluxPipeline
from diffusers.utils.pil_utils import make_image_grid
from PIL import Image
import math
from utils import deepfloyd, load_paths_for_prompt_id

def load_model(model_name: str, device: int):
    device = f"cuda:{device}"
    if model_name == 'sdxl':
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)
    elif model_name == 'sd-3':
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16).to(device)
    elif model_name == 'sdxl-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to(device)
    elif model_name == 'sd-2.1' or model_name == 'laion_sd-2.1':
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, safety_checker=None).to(device)
    elif model_name == 'lcm-sdxl':
        from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
        unet = UNet2DConditionModel.from_pretrained(
            "latent-consistency/lcm-sdxl",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16
        ).to(device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    elif model_name == 'sd-1.4' or model_name == 'laion_sd-1.4':
        from diffusers import StableDiffusionPipeline
        pipe  = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None).to(device)
    elif model_name == 'sd-1.1' or model_name == 'laion_sd-1.1':
        from diffusers import StableDiffusionPipeline
        pipe  = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1", torch_dtype=torch.float16, safety_checker=None).to(device)
    elif model_name == 'deepfloyd-xl':
        from diffusers import DiffusionPipeline
        stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16).to(device)
        stage_2 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
        ).to(device)
        safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": None}#"safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
        stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16).to(device)
    elif model_name == 'deepfloyd-l':
        from diffusers import DiffusionPipeline
        stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-L-v1.0", variant="fp16", torch_dtype=torch.float16).to(device)
        stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16).to(device)
        safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": None} #"safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
        stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16).to(device)
        pipe = [stage_1,stage_2,stage_3]
    elif model_name == 'deepfloyd-m':
        from diffusers import DiffusionPipeline
        stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", variant="fp16", torch_dtype=torch.float16).to(device)
        stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-M-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16).to(device)
        safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": None} #"safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
        stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16).to(device)
        pipe = [stage_1,stage_2,stage_3]
    elif model_name == 'flux-schnell':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(device)
    elif model_name == 'flux-dev':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)

    else:
        raise ValueError(f"Model {model_name} is not supported.")
    

    return pipe


def create_paths_for_image_generation(dirpath, num_images_to_generate, batch_size):
    all_batches = []
    num_full_batches = num_images_to_generate // batch_size
    remainder_images = num_images_to_generate % batch_size

    # Create full batches
    for seed in range(num_full_batches):
        batch_paths = []
        seed_dir = os.path.join(dirpath, str(seed))
        for sub_seed_idx in range(batch_size):
            img_path = os.path.join(seed_dir, f"{seed}_{sub_seed_idx}.png")
            batch_paths.append(img_path)
        all_batches.append(batch_paths)

    # Create the final batch if there are any remainder images
    if remainder_images > 0:
        batch_paths = []
        seed_dir = os.path.join(dirpath, str(num_full_batches))  # This is the new batch for remainder images
        for sub_seed_idx in range(remainder_images):
            img_path = os.path.join(seed_dir, f"{num_full_batches}_{sub_seed_idx}.png")
            batch_paths.append(img_path)
        all_batches.append(batch_paths)

    return all_batches


def generate_images(model_name, dataset_path, num_images_to_generate, device, batch_size):
    df = pd.read_csv(dataset_path)

    prompts = df['prompt'].tolist()
    ids = df['prompt_id'].tolist()
    pipe = load_model(model_name, device)

    dataset_name = dataset_path.split("/")[-1][:-4]
    for prompt_id, prompt in zip(ids, prompts):
        dirpath = os.path.join("generated_images", model_name, dataset_name, str(prompt_id))
        batch_paths = create_paths_for_image_generation(dirpath, num_images_to_generate, batch_size)
        
        for batch in batch_paths:
            seed_dir = os.path.dirname(batch[0])
            if not all(os.path.exists(path) for path in batch):
                os.makedirs(seed_dir, exist_ok=True)
                seed = int(os.path.basename(seed_dir))
            else:
                continue

            if 'deepfloyd' in model_name:
                images = deepfloyd(pipe, prompt, generator=torch.Generator(device).manual_seed(seed), num_images_per_prompt=batch_size)['images']
            elif model_name == 'sdxl-turbo':
                images = pipe(prompt=prompt, guidance_scale=0.0, num_inference_steps=4, num_images_per_prompt=batch_size, generator=torch.Generator(device).manual_seed(seed))['images']
            elif model_name == 'lcm-sdxl':
                images = pipe(prompt=prompt, num_inference_steps=4, generator=torch.Generator(device).manual_seed(seed), guidance_scale=8.0, num_images_per_prompt=batch_size)['images']
            elif model_name == 'sd-3':
                images = pipe(prompt=prompt, negative_prompt="", num_inference_steps=28, guidance_scale=7.0, generator=torch.Generator(device).manual_seed(seed), num_images_per_prompt=batch_size)['images']
            elif model_name == 'flux-schnell':
                images = pipe(prompt, guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256, generator=torch.Generator(device).manual_seed(seed), num_images_per_prompt=batch_size)['images']
            elif model_name == 'flux-dev':
                images = pipe(prompt, height=1024, width=1024,guidance_scale=3.5, num_inference_steps=50, max_sequence_length=512, generator=torch.Generator(device).manual_seed(seed), num_images_per_prompt=batch_size)['images']
            else:
                images = pipe(prompt,generator=torch.Generator(device).manual_seed(seed),num_images_per_prompt=batch_size)['images']
        
            for img, img_path in zip(images, batch):
                img.save(img_path)

        if not os.path.exists(os.path.join(dirpath, "grid_images.jpg")):
            create_img_grid(dirpath)
        

def create_img_grid(dirpath, output_filename='grid_images.jpg'):
    """
    Creates and saves a grid image from all PNG images in the specified directory.

    Args:
        dirpath (str): Path to the directory containing images.
        image_size (tuple): Size to which each image will be resized.
        output_filename (str): Name of the output grid image file.
    """
    images = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if file.lower().endswith('.png'):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {file}: {e}")

    num_images = len(images)
    print(f"Number of images: {num_images}")

    if num_images == 0:
        print("No images found. Exiting.")
        return

    grid_width = math.ceil(math.sqrt(num_images))
    grid_height = math.ceil(num_images / grid_width)
    print(f"Grid dimensions: {grid_width}x{grid_height} (width x height)")

    total_slots = grid_width * grid_height
    if num_images < total_slots:
        blank_image = Image.new('RGB', images[0].size, color=(255, 255, 255))
        images.extend([blank_image] * (total_slots - num_images))

    grid = make_image_grid(images, grid_width, grid_height)
    output_path = os.path.join(dirpath, output_filename)
    grid.save(output_path)
    print(f"Grid image saved to {output_path}")
