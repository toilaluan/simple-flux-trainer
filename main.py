from lightning_modules.lightning_sana import SanaLightning
from data.core_data import CoreCachedDataset, collate_fn
import torch
import os
import argparse
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
import wandb
import accelerate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to run training with various options."
    )

    parser.add_argument("--project", default="finetune-flux", help="Wandb project name")
    parser.add_argument("--max_epochs", type=int, default=20, help="Max epochs")
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.9,
        help="Validation check interval",
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--batch_size", default=1, help="Precision", type=int)

    return parser.parse_args()


args = parse_args()

wandb.init(project=args.project)

model = SanaLightning(
    denoiser_pretrained_path="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    learning_rate=1e-4,
    weight_decay=1e-8,
)

cached_dataset = CoreCachedDataset(cached_folder="debug/cache")

train_dataloader = torch.utils.data.DataLoader(
    cached_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
)
val_dataloader = torch.utils.data.DataLoader(
    cached_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
)

optimizer = model.configure_optimizers()

accelerator = accelerate.Accelerator()

model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader
)

model.to(accelerator.device)
model.pipeline.to(accelerator.device)

total_steps = 10000

model.train()
val_batch = next(iter(val_dataloader))

step = 0

lora_save_path = "lora_ckpt"

optimizer.train()

model.denoiser.train()

gradient_accumulation_steps = 8

while total_steps > 0:
    for i, batch in enumerate(train_dataloader):
        feeds, targets, metadata = batch
        for k, v in feeds.items():
            feeds[k] = v.to(model.denoiser.device)
        noise_pred = model(**feeds)
        loss = torch.nn.functional.mse_loss(
            noise_pred.float(), targets.float(), reduction="mean"
        )
        # loss = torch.mean(
        #     ((noise_pred.float() - targets.float()) ** 2).reshape(targets.shape[0], -1),
        #     1,
        # )
        # loss = loss.mean()
        print(f"Step {step} Loss {loss}")
        wandb.log({"loss": loss})
        loss = loss / gradient_accumulation_steps
        total_steps -= 1
        accelerator.backward(loss)
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if step % 150 == 0:
            print("Validating")
            denoiser = accelerator.unwrap_model(model.denoiser)
            model.save_lora(lora_save_path, denoiser)
            model.validation_step(val_batch, lora_save_path)
        step += 1
