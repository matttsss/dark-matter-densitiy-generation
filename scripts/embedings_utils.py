import os

import einops
import torch, pickle
import numpy as np

from tqdm import tqdm
from datasets import Dataset

import torch.nn.functional as F
from torch.utils.data import DataLoader

def fetch_dataset(dataset_path: str):

    cache_file_path = f"cache/{'.'.join(dataset_path.split('/')[-1].split('.')[:-1])}"

    if os.path.isdir(cache_file_path):
        print(f"Loading processed dataset from {cache_file_path}...")
        return Dataset.load_from_disk(cache_file_path)

    with open(dataset_path, 'rb') as f:
        metadata, images = pickle.load(f)

    del metadata["name"]
    del metadata["galaxy_catalogues"]

    images = images[::, [0, 0, 0]]
    ds = Dataset.from_dict({"image": images, **metadata})

    def process_row(idx):
        """This function ensures that the image is tokenised in the same way as
        the pre-trained model is expecting"""

        image = idx["image"]  # [H,W,C] in numpy
        del idx["image"]

        # Upscale to 512x512
        image = F.interpolate(
            torch.Tensor(image, device="cpu").unsqueeze(0), # [C,H,W] -> [1,C,H,W]
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Patchify
        patch_size = 16
        patch_galaxy = einops.rearrange(
            image,
            "c (h p1) (w p2) -> (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        )

        # Normalize patches
        std, mean = torch.std_mean(patch_galaxy, dim=1, keepdim=True)
        patch_galaxy = (patch_galaxy - mean) / (std + 1e-8)

        patch_galaxy = patch_galaxy.to(torch.float)
        galaxy_positions = torch.arange(len(patch_galaxy), dtype=torch.long)
        return {
            "images": patch_galaxy,
            "images_positions": galaxy_positions,
            **idx
        }

    ds = ds.map(process_row).with_format("torch")    
    ds.save_to_disk(cache_file_path)
  
    return ds


def get_embeddings(model, dataset, labels: list[str]):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    has_metals = torch.backends.mps.is_available()
    if has_metals:
        device = 'mps'

    print(f"Generating embeddings on device: {device}")

    model = model.to(device)
    model.eval()


    def batch_to_device(batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: batch_to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, (list,tuple)):
            return type(batch)(batch_to_device(v, device) for v in batch)
        return batch
  
    if has_metals:
        dl = DataLoader(
            dataset,
            batch_size=64,
            num_workers=0
        )
    else:
        dl = DataLoader(
            dataset,
            batch_size=128,
            num_workers=10,
            prefetch_factor=3
        )
        

    zss = []
    yss = {label: [] for label in labels}
    for B in tqdm(dl):
        B = batch_to_device(B, device)
        zs = model.generate_embeddings(B)["images"].detach().cpu().numpy()

        zss.append(zs)

        for label in labels:
            yss[label].append(B[label].detach().cpu().numpy())

    zss = np.concatenate(zss, axis=0)
    yss = {label: np.concatenate(yss[label], axis=0) for label in labels}

    return zss, yss
