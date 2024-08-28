# This file is used to test the cache dataset then use the cache dataset to train the model.

from lightning_modules.lightning_flux import FluxLightning
from data.cache_data import CacheFlux
from data.core_data import CoreDataset, CoreCachedDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

metadata_file = "dataset/itay_test/metadata.json"
root_folder = "dataset/itay_test/images"
dataset = CoreDataset(root_folder=root_folder, metadata_file=metadata_file)
pbar = tqdm(desc=f"Caching root folder: {root_folder}", total=len(dataset))
cache_flux = CacheFlux(save_dir="debug/test_cache")
for item in dataset:
    image, caption = item
    cache_flux(image, caption, "image")
    pbar.update(1)


cached_dataset = CoreCachedDataset(cached_folder="debug/test_cache")

dataloader = DataLoader(
    cached_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
)

flux_lightning = FluxLightning(denoiser_pretrained_path="black-forest-labs/FLUX.1-dev")
flux_lightning.to("cuda")

for batch in dataloader:
    loss = flux_lightning.training_step(batch, 0)
    print(loss)
    break
