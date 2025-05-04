import torch
import transformers
import diffusers
from data.core_data import CoreDataset
from PIL import Image
import os
from diffusers.image_processor import VaeImageProcessor


class CacheFlux:
    def __init__(
        self,
        pretrained_path: str = "black-forest-labs/FLUX.1-dev",
        save_dir: str = "data/cache",
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.save_dir = save_dir
        self.guidance_scale = 3.5
        self.pretrained_path = pretrained_path
        self.pipeline = diffusers.SanaPipeline.from_pretrained(
            pretrained_path, transformer=None, torch_dtype=torch_dtype
        )
        self.transformer_config = transformers.PretrainedConfig.from_pretrained(
            pretrained_path,
            subfolder="transformer",
        )
        self.vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels))
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.device = "cuda"
        self.pipeline.to(self.device)
        self.torch_dtype = torch_dtype
        os.makedirs(save_dir, exist_ok=True)

    @torch.no_grad()
    def __call__(self, image: Image.Image, prompt: str, filename: str):
        height, width = image.size
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=256,
        )

        num_channels_latents = self.transformer_config.in_channels // 4
        noise_latents, latent_image_ids = self.pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=self.device,
            generator=None,
            latents=None,
        )
        print(noise_latents.shape)
        latents = self.image_processor.preprocess(
            image,
        )
        latents = latents.to(self.device, self.torch_dtype)
        latents = self.pipeline.vae.encode(latents).latent_dist.sample()
        latents = (
            latents - self.pipeline.vae.config.shift_factor
        ) * self.pipeline.vae.config.scaling_factor

        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)
        print(height, width)
        latents = self.pipeline._pack_latents(
            latents,
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
        )
        assert latents.shape == noise_latents.shape
        guidance = (
            torch.tensor([self.guidance_scale]).to(self.torch_dtype).to(self.device)
        )

        feeds = {
            "latents": latents.to(self.torch_dtype).cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(self.torch_dtype).cpu(),
            "prompt_embeds": prompt_embeds.to(self.torch_dtype).cpu(),
            "text_ids": text_ids.to(self.torch_dtype).cpu(),
            "latent_image_ids": latent_image_ids.to(self.torch_dtype).cpu(),
            "guidance": guidance.to(self.torch_dtype).cpu(),
        }

        torch.save(feeds, os.path.join(self.save_dir, f"{filename}.pt"))

    @torch.no_grad()
    def decode_from_latent(self, latents: torch.Tensor, height, width):
        latents = self.pipeline._unpack_latents(
            latents, height, width, self.vae_scale_factor
        )
        latents = latents.to(self.device)
        latents = (
            latents / self.pipeline.vae.config.scaling_factor
        ) + self.pipeline.vae.config.shift_factor

        image = self.pipeline.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")
        return image[0]


if __name__ == "__main__":
    with torch.no_grad():
        cache_flux = CacheFlux(save_dir="debug/cache")
        dataset = CoreDataset(
            root_folder="dataset/itay_test/images",
            metadata_file="dataset/itay_test/metadata.json",
        )
        image, caption = dataset[0]
        height, width = image.size
        image.save("debug/image_2.jpg")
        cache_flux(image, caption, "image")
        feeds = torch.load("data/cache/image.pt")
        image = cache_flux.decode_from_latent(feeds["latents"], height, width)
        image.save("debug/image_2_reconstructed.jpg")
