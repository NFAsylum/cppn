from pathlib import Path

from PIL.Image import Image as Img

from cppn import CPPN
from render import render_image

import datetime

def main():
    output_folder = 'output'
    Path(output_folder).mkdir(exist_ok=True)

    model = CPPN(weight_sigma=0.5)
    generate_images(model, output_folder, 1)

def generate_images(model, output_folder: str, quantity: int): 
    for i in range(quantity):
        image: Img = render_image(model, 480, 480)
        img_name = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) + f"{i:02d}"
        image.save(f'{output_folder}/image-{img_name}.png')


if __name__ == "__main__":
    main()