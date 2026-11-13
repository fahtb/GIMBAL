"""Download official DINOv2-small code and weights into the workspace cache."""
from pathlib import Path
import torch

if __name__ == '__main__':
    torch.hub.set_dir(str(Path(__file__).resolve().parents[1] / 'runtime/torch_hub'))
    torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14', trust_repo=True)
    print('DINOv2-small ready')
