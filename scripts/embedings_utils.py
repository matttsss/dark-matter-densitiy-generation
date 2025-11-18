from os.path import isfile
from functools import partial

import torch, pickle
import numpy as np

from tqdm import tqdm
from datasets import load_dataset, Dataset

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from astropt.local_datasets import GalaxyImageDataset
from astropt.model_utils import load_astropt


def get_embeddings(astro_pt_data, nb_points: int, *labels: str):
    ## Else generate embeddings

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    if torch.backends.mps.is_available():
        device = 'mps'

    print(f"Generating embeddings on device: {device}")

    # set up HF galaxies in test set to be processed
    def normalise(x):
        std, mean = torch.std_mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + 1e-8)

    def data_transforms():
        transform = transforms.Compose(
            [
                transforms.Lambda(normalise),
            ]
        )
        return transform

    def batch_to_device(batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: batch_to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, (list,tuple)):
            return type(batch)(batch_to_device(v, device) for v in batch)
        return batch

    model = load_astropt("Smith42/astroPT_v2.0", path="astropt/095M").to(device)
    model.eval()
    
    galproc = GalaxyImageDataset(
        None,
        spiral=True,
        transform={"images": data_transforms()},
        modality_registry=model.modality_registry,
    )

    # load dataset: we select all lablels present in the argument list
    # as it is easy for this example but there are many values 
    # in Smith42/galaxies v2.0, you can choose from any column in hf.co/datasets/Smith42/galaxies_metadata
    if astro_pt_data:
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

    ds = (ds.select_columns(["image", *labels])
            .filter(lambda idx: all(idx[label] is not None and np.isfinite(idx[label]) for label in labels))
            .map(partial(_process_galaxy_wrapper, func=galproc.process_galaxy))
            .with_format("torch")
            .take(nb_points)
        )
  
    dl = DataLoader(
        ds,
        batch_size=128,
        num_workers=10,
        prefetch_factor=4
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