from os.path import isfile
from functools import partial

import torch
import numpy as np

from tqdm import tqdm
from datasets import load_dataset

from torch.utils.data import DataLoader
from torchvision import transforms

from astropt.local_datasets import GalaxyImageDataset
from astropt.model_utils import load_astropt


def get_embeddings(reset = False, nb_points: int = 1000, *labels: str) -> tuple[np.ndarray, np.ndarray]:
    ## Check for cached embeddings
    if not reset and isfile("cache/zss.npy") and isfile("cache/yss.npy"):
        zss = np.load("cache/zss.npy")
        yss = np.load("cache/yss.npy")

        if zss.shape[0] != nb_points:
            print("Cached embeddings do not match requested number of points, regenerating...")
        else:
            print("Embeddings file (zss.npy) detected so moving straight to linear probe and viz")
            return zss, yss

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
    ds = (
        load_dataset("Smith42/galaxies", split="test", revision="v2.0", streaming=True)
        .select_columns(["image", *labels])
        .filter(lambda idx: all(idx[label] is not None for label in labels))
        .map(partial(_process_galaxy_wrapper, func=galproc.process_galaxy))
        .with_format("torch")
        .take(nb_points)
    )
    dl = DataLoader(
        ds,
        batch_size=32,
        num_workers=10,
        prefetch_factor=4
    )

    zss = []
    yss = []
    for B in tqdm(dl):
        B = batch_to_device(B, device)
        zs = model.generate_embeddings(B)["images"].detach().cpu().numpy()
        zss.append(zs)
        yss.append(np.stack([B[label].detach().cpu().numpy() for label in labels], axis=1))

    zss = np.concatenate(zss, axis=0)
    yss = np.concatenate(yss, axis=0)

    # Cache embeddings
    np.save("cache/zss.npy", zss)
    np.save("cache/yss.npy", yss)

    return zss, yss