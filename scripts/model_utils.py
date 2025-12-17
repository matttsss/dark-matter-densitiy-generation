import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from astropt.model import GPT, GPTConfig
from tqdm.auto import tqdm
from generative_model.vector_field import VectorField, VectorFieldConfig
from scripts.embedings_utils import merge_datasets


@torch.no_grad()
def compute_embeddings(model, dataloader, device: torch.device, label_names: list[str],
                       disable_tqdm: bool = False):
    model.eval()

    all_embeddings = []
    all_labels = {label: [] for label in label_names}

    for B in tqdm(dataloader, disable=disable_tqdm):
        B = batch_to_device(B, device)
        embeddings = model.generate_embeddings(B)["images"]
        all_embeddings.append(embeddings)

        for label in label_names:
            all_labels[label].append(B[label])

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = {label: torch.cat(all_labels[label], dim=0) for label in label_names}
    return all_embeddings, all_labels

def get_datasets(model_path, device, label_names, split_ratio=0.8, nb_points=14000):
    model = load_astropt_model(model_path, device=device, strict=True)
    dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl", 
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl", 
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"],
        feature_names=label_names, stack_features=False) \
            .shuffle(seed=42) \
            .take(nb_points)    

    has_metals = device.type == 'mps'
    dl = DataLoader(
        dataset,
        batch_size = 64 if has_metals else 512,
        num_workers = 0 if has_metals else 4,
        prefetch_factor = None if has_metals else 3
    )

    embeddings, cond_dict = compute_embeddings(model, dl, device, label_names)
    cond = torch.stack([cond_dict[k] for k in label_names], dim=-1)

    # Split into train and val
    nb_train = int(split_ratio * embeddings.size(0))

    train_embeddings = embeddings[:nb_train]
    val_embeddings = embeddings[nb_train:]
    train_cond = cond[:nb_train]
    val_cond = cond[nb_train:]

    return (train_embeddings, val_embeddings), (train_cond, val_cond)

class LinearRegression:

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.weights = None
        self.bias = None

    @staticmethod
    def _append_bias(X):
        return torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim=1)

    @torch.no_grad()
    def fit(self, X, labels):
        X = torch.as_tensor(X, device=self.device)
        labels = torch.as_tensor(labels, device=self.device)

        # Ensure labels is 2D (num_samples, num_outputs)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        if X.dim() == 1:
            X = X.unsqueeze(1)

        new_X = LinearRegression._append_bias(X)
        W = (torch.linalg.pinv(new_X.T @ new_X) @ new_X.T @ labels)

        self.weights = W[1:].T
        self.bias = W[0]

        return self

    def predict(self, X):
        X = torch.as_tensor(X, device=self.device)
        if self.weights is None: raise ValueError("Model has not been fitted yet.")
        return F.linear(X, self.weights, self.bias)
    
    def sample(self, labels):
        if self.weights is None: raise ValueError("Model has not been fitted yet.")

        print(self.weights.shape, (labels - self.bias).shape)

        res = torch.linalg.lstsq(self.weights.cpu(), (labels - self.bias).T.cpu(), driver='gelsy')
        return res.solution.to(self.device)

def load_fm_model(checkpoint_path, device, strict=True, **extra_model_config):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fm_config = VectorFieldConfig(**checkpoint["config"])

    for k, v in extra_model_config.items():
        setattr(fm_config, k, v)

    model = VectorField(fm_config)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    model.to(device)

    return model

def load_astropt_model(checkpoint_path, device, strict=True, get_label_names = False, **extra_model_config):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_args = checkpoint["model_args"]
    modality_registry = checkpoint["modality_registry"]

    # Modify model for finetuning
    config = GPTConfig(**model_args)
    if "target_labels" in checkpoint and get_label_names:
        config.output_dim = len(checkpoint["target_labels"])
    
    for k, v in extra_model_config.items():
        setattr(config, k, v)

    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model = GPT(config, modality_registry)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)

    return (model, checkpoint["target_labels"]) if get_label_names else model

def batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list,tuple)):
        return type(batch)(batch_to_device(v, device) for v in batch)
    return batch

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99, keep_all=False):
        self.momentum = momentum
        self.losses = [] if keep_all else None
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0
        self.losses = [] if self.losses is not None else None

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
    
    def register_loss(self, val):
        if self.losses is not None:
            self.losses.append(val)