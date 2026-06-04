from PIL import Image
import math

def tile_image(img):
    w, h = img.size
    tiled = Image.new('RGB', (w * 2, h * 2))
    for dx, dy in [(0, 0), (w, 0), (0, h), (w, h)]:
        tiled.paste(img, (dx, dy))
    return tiled

def psnr(mse):
    return 20* math.log10(1.0 / math.sqrt(mse))