import torch
from torch import Tensor
from torch.optim import Adam
from torch.nn.functional import mse_loss
from PIL import Image
from PIL.Image import Image as Img
import matplotlib.pyplot as plt
import math
import os
from pathlib import Path
import numpy as np
import time

from cppn import CPPN
from utils import psnr

TARGET_PATH = 'target/cppn-target-image.png'
TARGET_SIZE = 256
RENDER_SIZE = 512
ITERATIONS = 3000
LEARNING_RATE = 1e-3
SNAPSHOT_ITERS = [0, 50, 200, 500, 1500, 2999]
Z_DIM = 4
OUTPUT_FOLDER = 'training_output'
TILEABLE = False

def main() -> None:
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

    spatial_dim = 4 if TILEABLE else 3
    model = CPPN(input_dim=spatial_dim + Z_DIM, output_channels=3)
    coords = build_coords(size=TARGET_SIZE, tileable=TILEABLE)
    target = load_target(path=TARGET_PATH, size=TARGET_SIZE)


    start = time.perf_counter()

    losses = train(model=model, target=target, coords=coords, iterations=ITERATIONS, lr=LEARNING_RATE,
    snapshot_iters=SNAPSHOT_ITERS, size=TARGET_SIZE, output_folder=OUTPUT_FOLDER)

    elapsed = time.perf_counter() - start

    final_coords = build_coords(size=RENDER_SIZE, tileable=TILEABLE)
    final_render(model=model, coords=final_coords, size=RENDER_SIZE, output_folder=OUTPUT_FOLDER)

    final_mse = losses[-1]
    save_loss_curve(losses=losses, output_folder=OUTPUT_FOLDER)
    report_compression(model=model, target_path=TARGET_PATH, final_mse=final_mse, output_folder=OUTPUT_FOLDER, duration=elapsed)

    torch.save(model.state_dict(), f'{OUTPUT_FOLDER}/model_final.pt')

    print(f'Training finished, outputs in {OUTPUT_FOLDER}')

def load_target(path: str, size: int) -> Tensor:
    target = Image.open(path).convert('RGB')
    target = target.resize((size, size), Image.LANCZOS)

    image_array = np.array(target) / 255.0

    image_tensor = torch.from_numpy(image_array).float()
    image_tensor = image_tensor.view(-1, 3)

    return image_tensor

def build_coords(size: int, tileable: bool) -> Tensor:
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')

    if tileable:
        spatial = torch.stack([
            torch.sin(math.pi * x_grid),
            torch.cos(math.pi * x_grid),
            torch.sin(math.pi * y_grid),
            torch.cos(math.pi * y_grid)
        ], dim=-1)
    else:
        r_grid = torch.sqrt(x_grid**2 + y_grid**2)
        spatial = torch.stack([x_grid, y_grid, r_grid], dim=-1)

    spatial_dim = spatial.shape[-1]

    z = torch.zeros(Z_DIM)

    z_grid = z.view(1, 1, Z_DIM).expand(size, size, Z_DIM)

    coords = torch.cat([spatial, z_grid], dim=-1).view(-1, spatial_dim + Z_DIM)

    return coords

def snapshot(model: CPPN, coords: Tensor, size: int, iter: int, output_folder: str) -> None:
    with torch.no_grad():
        output = model(coords)

    channels = output.shape[-1]
    image_tensor = output.view(size, size, channels)
    if channels == 1:
        image_tensor = image_tensor.repeat(1, 1, 3)

    image: Img = Image.fromarray((image_tensor.numpy() * 255).astype('uint8'))

    img_name = f'snapshot_iter_{iter:05d}'
    image.save(f'{output_folder}/{img_name}.png')

def final_render(model: CPPN, coords: Tensor, size: int, output_folder: str) -> None:
    with torch.no_grad():
        output = model(coords)

    channels = output.shape[-1]
    image_tensor = output.view(size, size, channels)
    if channels == 1:
        image_tensor = image_tensor.repeat(1, 1, 3)

    image: Img = Image.fromarray((image_tensor.numpy() * 255).astype('uint8'))

    img_name = 'final_render'
    image.save(f'{output_folder}/{img_name}.png')

def train(model: CPPN, target: Tensor, coords: Tensor, iterations: int, lr: float, snapshot_iters: list, size: int, output_folder: str) -> list[float]:
    optimizer = Adam(model.parameters(), lr=lr)
    losses = []
    model.train()

    for i in range(iterations):
        optimizer.zero_grad()
        output = model(coords)
        loss = mse_loss(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i in snapshot_iters):
            snapshot(model, coords, size, i, output_folder)

        if i % 100 == 0:
            print(f'Finished iter {i}/{iterations}, loss: {loss.item():.6f}')

    return losses

def save_loss_curve(losses: list[float], output_folder: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.title(f'Training Loss ({len(losses)} iterations)')
    plt.savefig(f'{output_folder}/loss_curve.png')
    plt.close()

def report_compression(model: CPPN, target_path: str, final_mse: float, output_folder: str, duration: float) -> None:
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    target_size_bytes = os.path.getsize(target_path)
    ratio = target_size_bytes / model_size_bytes
    psnr_db = psnr(final_mse)

    model_size_str = f'Model size: {model_size_bytes / 1024:.1f} KB'
    target_size_str = f'Target size: {target_size_bytes / 1024:.1f} KB'
    ratio_str = f'Ratio: {ratio:.2f}x'
    psnr_str = f'PSNR: {psnr_db:.2f} dB'
    duration_str = f'Training duration: {duration:.1f}s'

    print(model_size_str)
    print(target_size_str)
    print(ratio_str)
    print(psnr_str)
    print(duration_str)

    with open(f'{output_folder}/train_log.txt', 'w') as file:
        file.write(model_size_str + '\n')
        file.write(target_size_str + '\n')
        file.write(ratio_str + '\n')
        file.write(psnr_str + '\n')
        file.write(duration_str + '\n')

if __name__ == '__main__':
    main()