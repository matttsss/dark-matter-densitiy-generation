import torch

from astropt.model import GPT, GPTConfig

class LinearRegression(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.W = None

    def fit(self, X, labels):
        X = torch.cat([torch.ones(X.shape[0], 1, device=self.device), X], dim=1)
        self.W = (torch.linalg.pinv(X.T @ X) @ X.T @ labels).detach()

        return X @ self.W

    def predict(self, X):
        if self.W is None:
            raise ValueError("Model has not been fitted yet.")
        
        X = torch.cat([torch.ones(X.shape[0], 1, device=self.device), X], dim=1)
        return X @ self.W
    

def load_model(checkpoint_path, device, strict=True, get_label_names = False, **extra_model_config):
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
