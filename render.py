from PIL import Image
from PIL.Image import Image as Img
import torch

def render_image(model, width, height) -> Img:
    x = torch.linspace(-1, 1, width)
    y = torch.linspace(-1, 1, height)

    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')

    coords = torch.stack([x_grid, y_grid], dim=-1).view(-1, 2)

    with torch.no_grad():
        output = model(coords)

    image_tensor = output.view(height, width, 3)

    image: Img = Image.fromarray((image_tensor.numpy() * 255).astype('uint8'))

    return image