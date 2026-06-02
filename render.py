from PIL import Image
from PIL.Image import Image as Img
import torch
import math

def render_image(model, width, height, z=None, z_dim=4, r_strength=1.0, tileable=False) -> Img:
    x = torch.linspace(-1, 1, width)
    y = torch.linspace(-1, 1, height)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')

    if tileable:
        spatial = torch.stack([
            torch.sin(math.pi * x_grid),
            torch.cos(math.pi * x_grid),
            torch.sin(math.pi * y_grid),
            torch.cos(math.pi * y_grid)
        ], dim=-1)
    else:
        r_grid = torch.sqrt(x_grid**2 + y_grid**2) * r_strength
        spatial = torch.stack([x_grid, y_grid, r_grid], dim=-1)

    spatial_dim = spatial.shape[-1]

    # Generate random z if not set
    if z is None:
        z = torch.randn(z_dim)

    z_grid = z.view(1, 1, z_dim).expand(height, width, z_dim)

    coords = torch.cat([spatial, z_grid], dim=-1).view(-1, spatial_dim + z_dim)

    with torch.no_grad():
        output = model(coords)

    channels = output.shape[-1]
    image_tensor = output.view(height, width, channels)
    if channels == 1:
        image_tensor = image_tensor.repeat(1, 1, 3)

    image: Img = Image.fromarray((image_tensor.numpy() * 255).astype('uint8'))

    return image