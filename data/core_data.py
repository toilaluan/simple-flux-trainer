from torch.utils.data import Dataset
from typing import List, Tuple
import torch
import json
import glob
import random
import os
from PIL import Image
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
)
from diffusers import FlowMatchEulerDiscreteScheduler

def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas
    schedule_timesteps = noise_scheduler.timesteps
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

class CoreDataset(Dataset):
    def __init__(self, metadata_file: str, root_folder: str):
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)
        self.root_folder = root_folder
        self.bucket_config = self._init_bucket()

    def _init_bucket(
        self,
        base_size: int = 1024,
        min_size: int = 512,
        max_size: int = 1536,
        divisible: int = 32,
    ):
        """
        Build a dictionary that maps aspect-ratio (w / h, rounded) to
        (width, height) tuples whose area lies roughly within
        [1/8 × base_area, 1.1 × base_area].

        Returns
        -------
        dict[float, tuple[int, int]]
        """
        # --- 1. Generate every multiple of `divisible` in [min_size, max_size] ---
        widths  = list(range(min_size, max_size + 1, divisible))
        heights = widths[:]                         # same grid for height

        sizes: dict[float, tuple[int, int]] = {}
        base_area = base_size * base_size           # 1024×1024 by default

        for w in widths:
            for h in heights:
                area = w * h

                # --- 2. Keep pairs whose area is “near” the baseline area ---
                if not (base_area / 8 <= area <= base_area * 1.1):
                    continue

                # --- 3. Use rounded ratio as key; newer entry overwrites older ---
                ratio = round(w / h, 5)
                sizes[ratio] = (w, h)

        # --- 4. Verbose summary ---------------------------------------------------
        print(f"Initialized {len(sizes)} bucket sizes")
        for ratio, wh in sizes.items():
            print(f"Bucket ratio: {ratio:.3f}   size: {wh}")

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
    def __init__(self, cached_folder: str, max_len: int = 512, prefix: str = "cache_"):
        self.cached_files = glob.glob(f"{cached_folder}/{prefix}*.pt")
        self.max_len = max_len
        self.max_step = 1000
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers", subfolder="scheduler",
        )

    def __len__(self):
        return len(self.cached_files)

    def add_noise(self, latent: torch.Tensor, dtype: torch.dtype):
        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=latent.shape[0],
            logit_mean=0,
            logit_std=1,
            mode_scale=1.29,
        )
        # print("u", u)
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        # print("indices", indices)
        timesteps = self.noise_scheduler.timesteps[indices]
        # print("timesteps", timesteps)
        sigmas = get_sigmas(self.noise_scheduler, timesteps)
        # print("sigmas", sigmas)
        noise = torch.randn_like(latent).to(dtype)
        noised_latent = (1 - sigmas) * latent + sigmas * noise
        return noised_latent, sigmas, noise, timesteps

    def __getitem__(self, index):
        cached_file = self.cached_files[index]
        feeds = torch.load(cached_file)
        latent = feeds["latents"]
        dtype = latent.dtype
        noised_latent, sigma, noise, timesteps = self.add_noise(latent, dtype)
        feeds["timestep"] = torch.Tensor([sigma])
        feeds["latents"] = noised_latent
        prompt = feeds.pop("prompt")
        step = int(sigma * self.max_step)
        target = noise - latent
        metadata = {
            "step": timesteps,
            "sigma": sigma,
            "prompt": prompt,
        }

        return feeds, target, metadata


def collate_fn(batch):
    feeds, targets, metadata = zip(*batch)
    feeds = {k: torch.cat([f[k] for f in feeds], dim=0) for k in feeds[0]}
    targets = torch.cat(targets, dim=0)
    return feeds, targets, metadata
