import torch
import torch.nn.functional as F

from astropt.model import GPT, GPTConfig
from generative_model.hash_grid_encoding import MultiResHashGrid
from generative_model.vector_field import VectorField, VectorFieldConfig

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

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
