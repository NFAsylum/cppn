from pathlib import Path

from PIL.Image import Image as Img

from cppn import CPPN
from render import render_image

import datetime

def main():
    model = CPPN()

    image: Img = render_image(model, 480, 480)

    Path('output').mkdir(exist_ok=True)

    img_name = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    image.save(f'output/image-{img_name}.png')

if __name__ == "__main__":
    main()