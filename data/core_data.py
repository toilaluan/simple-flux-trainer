from torch.utils.data import Dataset
from typing import List, Tuple
import torch
import json
import glob
import random
import os
from PIL import Image


class CoreDataset(Dataset):
    def __init__(self, metadata_file: str, root_folder: str):
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)
        self.root_folder = root_folder
        self.bucket_config = self._init_bucket()

    def _init_bucket(
        self,
        base_size: int = 1024,
        min_size: int = 256,
        max_size: int = 1536,
        divisible: int = 32,
    ):
        widths = [
            base_size - divisible * i
            for i in range(1, (base_size - min_size) // divisible)
        ] + [
            base_size + divisible * i
            for i in range(1, (max_size - base_size) // divisible)
        ]
        heights = [
            base_size - divisible * i
            for i in range(1, (base_size - min_size) // divisible)
        ] + [
            base_size + divisible * i
            for i in range(1, (max_size - base_size) // divisible)
        ]
        sizes = {}
        for width in widths:
            for height in heights:
                ratio = width / height
                sizes[ratio] = (width, height)
        print(f"Initialized {len(sizes)} bucket sizes")
        print(sizes)
        return sizes

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        metadata = self.metadata[index]
        caption = metadata["caption"]
        image_path = metadata["image_path"]
        image_path = os.path.join(self.root_folder, image_path)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        ratio = width / height
        # select nearest bucket size
        bucket_ratio = min(self.bucket_config.keys(), key=lambda x: abs(x - ratio))
        bucket_size = self.bucket_config[bucket_ratio]
        image = image.resize(bucket_size)
        return image, caption


class CoreCachedDataset(Dataset):
    def __init__(self, cached_folder: str, max_len: int = 512):
        self.cached_files = glob.glob(f"{cached_folder}/*.pt")
        self.max_len = max_len
        self.max_step = 1000

    def __len__(self):
        return len(self.cached_files)

    def add_noise(self, latent: torch.Tensor):
        sigma = random.random()
        noised_latent = (1 - sigma) * latent + sigma * torch.randn_like(latent)
        return noised_latent, sigma

    def __getitem__(self, index):
        cached_file = self.cached_files[index]
        feeds = torch.load(cached_file)
        latent = feeds["latents"]
        noised_latent, sigma = self.add_noise(latent)
        feeds["timestep"] = torch.Tensor([sigma])
        feeds["latents"] = noised_latent
        step = int(sigma * self.max_step)
        target = latent

        metadata = {
            "step": step,
            "sigma": sigma,
        }

        return feeds, target, metadata


def collate_fn(batch):
    feeds, targets, metadata = zip(*batch)
    feeds = {k: torch.cat([f[k] for f in feeds], dim=0) for k in feeds[0]}
    targets = torch.cat(targets, dim=0)
    return feeds, targets, metadata
