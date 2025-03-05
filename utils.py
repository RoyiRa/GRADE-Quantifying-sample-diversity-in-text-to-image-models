import os
from typing import Optional


def load_paths_for_prompt_id(dirpath: str, max_seed: Optional[int] = None):
    imagepaths = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            # a variety of image suffixes
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                if max_seed and int(file.split('_')[0]) >= int(max_seed):
                    print(f"Reached max seed: {max_seed}. Stopping image loading now.")
                    break
                imagepaths.append(os.path.join(root, file))
    print(f"Images found in total: {len(imagepaths)}")
    return imagepaths


def deepfloyd(pipe, prompt, generator):
    stage_1, stage_2, stage_3 = pipe
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
    images = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
    images = stage_2(image=images, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
    images = stage_3(prompt=prompt, image=images, generator=generator, noise_level=100)
    return images

def load_oai_key():
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    file_path = 'oai_key.txt'
    with open(file_path, 'r') as file:
        key = file.read().strip() 
    return key