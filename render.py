from PIL import Image
from PIL.Image import Image as Img
import torch

def render_image(model, width, height, z=None, z_dim=4) -> Img:
    x = torch.linspace(-1, 1, width)
    y = torch.linspace(-1, 1, height)

    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
    r_grid = torch.sqrt(x_grid**2 + y_grid**2)

    # Generate random z if not set
    if z is None:
        z = torch.randn(z_dim)

    spatial = torch.stack([x_grid, y_grid, r_grid], dim=-1)

    z_grid = z.view(1, 1, z_dim).expand(height, width, z_dim)

    coords = torch.cat([spatial, z_grid], dim=-1).view(-1, 3 + z_dim)

    with torch.no_grad():
        output = model(coords)

    image_tensor = output.view(height, width, 3)

    image: Img = Image.fromarray((image_tensor.numpy() * 255).astype('uint8'))

    return image