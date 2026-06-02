from pathlib import Path

from PIL import Image
from PIL.Image import Image as Img

from cppn import CPPN
from render import render_image

import datetime
import random

PRESETS = {'xxxsmall':32, 'xxsmall':64, 'xsmall':128, 'small':256, 'medium':512, 'large':1024, 'huge':2048}
size = PRESETS['large']

def main():
    output_folder = 'output'
    Path(output_folder).mkdir(exist_ok=True)

    tileable = False
    z_dim = 4
    spatial_dim = 4 if tileable else 3

    sigma = random.uniform(0.3, 3)

    model = CPPN(input_dim=spatial_dim + z_dim, weight_sigma=sigma)

    generate_images(model, output_folder, 1, tileable)

def generate_images(model, output_folder: str, quantity: int, tileable: bool): 
    for i in range(quantity):
        r_strength = random.uniform(0.0, 10.0) if not tileable else 1.0
        image: Img = render_image(model, size, size, r_strength=r_strength, tileable=tileable)
        img_name = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) + f"{i:02d}"
        image.save(f'{output_folder}/image-{img_name}.png')

def tile_image(img):
    w, h = img.size
    tiled = Image.new('RGB', (w * 2, h * 2))
    for dx, dy in [(0, 0), (w, 0), (0, h), (w, h)]:
        tiled.paste(img, (dx, dy))
    return tiled

if __name__ == "__main__":
    main()