import torch
import transformers
import diffusers
from data.core_data import CoreDataset
from PIL import Image, ImageOps
import os
from diffusers.image_processor import VaeImageProcessor


class CacheSana:
    def __init__(
        self,
        pretrained_path: str = "",
        save_dir: str = "data/cache",
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.save_dir = save_dir
        self.pretrained_path = pretrained_path
        print(pretrained_path)
        self.pipeline = diffusers.SanaPipeline.from_pretrained(
            pretrained_path, transformer=None, torch_dtype=torch_dtype
        )
        self.transformer_config = transformers.PretrainedConfig.from_pretrained(
            pretrained_path,
            subfolder="transformer",
        )
        self.vae_scale_factor = self.pipeline.vae_scale_factor
        self.image_processor = self.pipeline.image_processor
        self.device = "cuda"
        self.pipeline.to(self.device)
        self.torch_dtype = torch_dtype
        os.makedirs(save_dir, exist_ok=True)

    @torch.no_grad()
    def __call__(self, image: Image.Image, prompt: str, filename: str):
        width, height = image.size
        (
            prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=300,
        )

        num_channels_latents = self.transformer_config.in_channels
        noise_latents = self.pipeline.prepare_latents(
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
        latents = self.pipeline.vae.encode(latents, return_dict=False)[0]
        latents = latents * self.pipeline.vae.config.scaling_factor
        assert latents.shape == noise_latents.shape

        feeds = {
            "latents": latents.to(self.torch_dtype).cpu(),
            "prompt_embeds": prompt_embeds.to(self.torch_dtype).cpu(),
            "prompt_attention_mask": prompt_attention_mask.to(self.torch_dtype).cpu(),
            "prompt": prompt,
        }

        torch.save(feeds, os.path.join(self.save_dir, f"{filename}.pt"))

    @torch.no_grad()
    def decode_from_latent(self, latents: torch.Tensor, height, width):
        # latents = self.pipeline._unpack_latents(
        #     latents, height, width, self.vae_scale_factor
        # )
        latents = latents.to(self.device)
        latents = (
            latents / self.pipeline.vae.config.scaling_factor
        )

        image = self.pipeline.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")
        return image[0]

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

if __name__ == "__main__":
    with torch.no_grad():
        cache_flux = CacheSana(save_dir="debug/cache", pretrained_path="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers")
        dataset = CoreDataset(
            root_folder="dataset/itay_test/images",
            metadata_file="dataset/itay_test/metadata.json",
        )
        os.makedirs("debug/compare", exist_ok=True)



        for i, (image, caption) in enumerate(dataset):
            image = ImageOps.exif_transpose(image)
            cache_flux(image, caption, f"cache_image_{i}")
            feeds = torch.load(f"debug/cache/cache_image_{i}.pt")
            width, height = image.size
            print(width, height)
            print(feeds["latents"].shape)
            decoded_image = cache_flux.decode_from_latent(feeds["latents"], height, width)

            compare_image = get_concat_h(image, decoded_image)
            compare_image.save(f"debug/compare/image_{i}.png")