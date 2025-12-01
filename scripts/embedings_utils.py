
import numpy as np
import einops, torch, pickle, os

from tqdm.auto import tqdm
from torch.nn.functional import interpolate
from datasets import concatenate_datasets, Dataset

from model_utils import batch_to_device

def spiralise(galaxy):
        """
        Change ViT patch ordering to a 'spiral order'. See Fig 8 in
        https://arxiv.org/pdf/2401.08541.pdf for an illustration.

        Alternate function available here:
        https://www.procook.co.uk/product/procook-spiralizer-black-and-stainless-steel
        """

        def spiral(n):
            """
            generate a spiral index array of side length 'n'
            there must be a better way to do this: any suggestions?
            """
            a = np.arange(n * n)
            b = a.reshape((n, n))
            m = None
            for i in range(n, 0, -2):
                m = np.r_[m, b[0, :], b[1:, -1], b[-1, :-1][::-1], b[1:-1, 0][::-1]]
                b = b[1:-1, 1:-1]
            a[list(m[1:])] = list(a)
            a = abs(a - n * n + 1)
            return a.reshape((n, n))
        
        # Generate a spiralised matrix and then flatten it to the same shape as 'galaxy'
        indices = einops.rearrange(
            spiral(int(np.sqrt(len(galaxy)))),
            "h w -> (h w)",
        )
        assert len(indices) == len(galaxy), (
            "tokenised galaxy must have a square rootable length!"
        )
        spiraled = [ii for _, ii in sorted(zip(indices, galaxy))]
        return (
            torch.stack(spiraled)
            if isinstance(spiraled[0], torch.Tensor)
            else np.stack(spiraled)
        )

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

        idx["norms"] = idx["norms"][0]
    
        # Upscale to 256x256
        image = interpolate(
            torch.Tensor(image, device="cpu").unsqueeze(0), # [C,H,W] -> [1,C,H,W]
            size=(256, 256),
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
        ).to(torch.float)

        # Normalize patches
        std, mean = torch.std_mean(patch_galaxy, dim=1, keepdim=True)
        patch_galaxy = (patch_galaxy - mean) / (std + 1e-8)

        # Spiralise patches
        patch_galaxy = spiralise(patch_galaxy)

        galaxy_positions = torch.arange(len(patch_galaxy), dtype=torch.long)
        return {
            "images": patch_galaxy,
            "images_positions": galaxy_positions,
            **idx
        }

    ds = ds.map(process_row).with_format("torch")    
    ds.save_to_disk(cache_file_path)
  
    return ds

def merge_datasets(datasets: list[str]) -> Dataset:
    return concatenate_datasets([fetch_dataset(path) for path in datasets])

def compute_embeddings(model, dataloader, device: torch.device, label_names: list[str],
                       disable_tqdm: bool = False):
    model.eval()

    all_embeddings = []
    all_labels = {label: [] for label in label_names}
    with torch.no_grad():
        for B in tqdm(dataloader, disable=disable_tqdm):
            B = batch_to_device(B, device)
            embeddings = model.generate_embeddings(B)["images"]
            all_embeddings.append(embeddings)

            for label in label_names:
                all_labels[label].append(B[label])

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = {label: torch.cat(all_labels[label], dim=0) for label in label_names}
    return all_embeddings, all_labels
