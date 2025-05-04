import pytorch_lightning as L
import diffusers
import torch
import schedulefree
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
import math
from optimum.quanto import freeze, qfloat8, quantize, qint4
import bitsandbytes as bnb
import gc
import wandb
from torch import nn
from diffusers import SanaPipeline


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


class SanaLightning(nn.Module):
    def __init__(
        self,
        denoiser_pretrained_path: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.torch_dtype = torch_dtype

        self.pipeline = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16,
            text_encoder=None,
        )
        self.denoiser = self.pipeline.transformer
        self.denoiser.to("cuda")
        self.apply_lora(self.denoiser)
        self.denoiser.enable_gradient_checkpointing()
        self.print_trainable_parameters(self.denoiser)

    @staticmethod
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    @staticmethod
    def apply_lora(
        model,
        rank=32,
        alpha=32,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0."],
    ):
        transformer_lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights=init_lora_weights,
            target_modules=target_modules,
        )
        model.add_adapter(transformer_lora_config)

    def forward(
        self,
        latents: torch.Tensor = None,
        timestep: int = None,
        prompt_embeds: torch.Tensor = None,
        **kwargs,
    ):
        noise_pred = self.denoiser(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        return noise_pred.float()

    def loss_fn(self, noise_pred, targets):
        loss = ((noise_pred - targets) ** 2).mean()
        return loss

    def training_step(self, batch, batch_idx):
        feeds, targets, metadata = batch
        for k, v in feeds.items():
            feeds[k] = v.to(self.denoiser.device)
        noise_pred = self(**feeds)
        loss = self.loss_fn(noise_pred, targets)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, lora_path):
        self.pipeline.load_lora_weights(lora_path)
        feeds, targets, metadata = batch
        prompt_embeds = feeds["prompt_embeds"][:1]
        pooled_prompt_embeds = feeds["pooled_prompt_embeds"][:1]
        width = 768
        height = 1024
        steps = 20
        image = self.pipeline(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(42),
        ).images[0]
        image = wandb.Image(image, caption="TODO: Add caption")
        wandb.log({f"Validation image": image})

    def configure_optimizers(self):
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, self.denoiser.parameters())
        )
        optimizer = schedulefree.AdamWScheduleFree(
            params_to_optimize,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def save_lora(self, path: str):
        transformer_lora_layers = get_peft_model_state_dict(self.denoiser)
        diffusers.FluxPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=transformer_lora_layers,
        )
