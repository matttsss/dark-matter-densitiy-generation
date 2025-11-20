from os.path import isfile
from functools import partial

import einops
import torch, pickle
import numpy as np

from tqdm import tqdm
from datasets import load_dataset, Dataset

import torch.nn.functional as F
from torch.utils.data import DataLoader

def get_datasets(is_astropt_data: bool, nb_points: int, labels: list[str]):
    if is_astropt_data:
        ds = load_dataset("Smith42/galaxies", split="test", revision="v2.0", streaming=True)

        def _process_galaxy_wrapper(idx, func):
            """This function ensures that the image is tokenised in the same way as
            the pre-trained model is expecting"""
            galaxy = func(
                torch.from_numpy(np.array(idx["image"]).swapaxes(0, 2)).to(float)
            ).to(torch.float)
            galaxy_positions = torch.arange(0, len(galaxy), dtype=torch.long)
            return {
                "images": galaxy,
                "images_positions": galaxy_positions,
                **{label: idx[label] for label in labels}
            }

    else:
        with open("data/DarkData/BAHAMAS/bahamas_0.1.pkl", 'rb') as f:
            metadata, images = pickle.load(f)

        images = images[::, [0, 0, 0]]
        ds = Dataset.from_dict({"image": images, **{label: metadata[label] for label in labels}})

        def _process_galaxy_wrapper(idx, func):
            """This function ensures that the image is tokenised in the same way as
            the pre-trained model is expecting"""

            image = F.interpolate(
                torch.Tensor(idx["image"], device="cpu").unsqueeze(0), # [C,H,W] -> [1,C,H,W]
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)     

            galaxy = func(image).to(torch.float)
            galaxy_positions = torch.arange(0, len(galaxy), dtype=torch.long)
            return {
                "images": galaxy,
                "images_positions": galaxy_positions,
                **{label: idx[label] for label in labels}
            }
    
    def process_image(raw_galaxy):
        """Process raw galaxy image into patches and normalize them"""
        # TODO have a parameter?
        patch_size = 16
        patch_galaxy = einops.rearrange(
            raw_galaxy,
            "c (h p1) (w p2) -> (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        )

        std, mean = torch.std_mean(patch_galaxy, dim=1, keepdim=True)
        return (patch_galaxy - mean) / (std + 1e-8)

    ds = (ds.select_columns(["image", *labels])
            .filter(lambda idx: all(idx[label] is not None and np.isfinite(idx[label]) for label in labels))
            .shuffle(seed=42)
            .take(nb_points)
            .map(partial(_process_galaxy_wrapper, func=process_image))
            .with_format("torch")
        )
  
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
    print(labels)
    for B in tqdm(dl):
        B = batch_to_device(B, device)
        zs = model.generate_embeddings(B)["images"].detach().cpu().numpy()

        zss.append(zs)

        for label in labels:
            yss[label].append(B[label].detach().cpu().numpy())

    zss = np.concatenate(zss, axis=0)
    yss = {label: np.concatenate(yss[label], axis=0) for label in labels}

    return zss, yss
