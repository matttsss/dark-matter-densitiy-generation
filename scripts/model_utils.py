import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field

from astropt.model import GPT, GPTConfig
from torchcfm.conditional_flow_matching import *

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
        # Ensure labels is 2D (num_samples, num_outputs)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

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
    

@dataclass
class VectorFieldConfig:
    sigma: float = 1.0
    dim: int = 128
    encoding_size: int = 64
    hidden: int = 512
    conditions: list[str] = field(default_factory=list)

class VectorField(nn.Module, TargetConditionalFlowMatcher):
    
    def __init__(self, config: VectorFieldConfig):
        nn.Module.__init__(self)
        TargetConditionalFlowMatcher.__init__(self, config.sigma)
    
        self.config = config
        self.encoding_net = nn.Sequential(
            nn.Linear(len(config.conditions), config.encoding_size//2),
            nn.SiLU(),
            nn.Linear(config.encoding_size//2, config.encoding_size),
            nn.SiLU()
        )
    
        self.net = nn.Sequential(
            nn.Linear(config.dim + config.encoding_size + 1, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, cond, t):
        encoding = self.encoding_net(cond)
        return self.net(torch.cat([x, encoding, t], dim=-1))


    def compute_loss(self, x0, x1, cond):
        t, xt, ut = self.sample_location_and_conditional_flow(x0, x1)
        return F.mse_loss(self(xt, cond, t.view(-1, 1)), ut)


    @torch.no_grad()
    def sample_flow(self, cond, steps=1000):
        """
        Integrates dx/dt = v_theta(x, t) from t=0 to 1.
        Start from simple noise distribution.
        """
        dt = 1.0 / steps
        batch_size = cond.size(0)

        t = torch.zeros(batch_size, 1, device=cond.device)
        x = torch.randn(batch_size, self.config.dim, device=cond.device)

        for _ in range(steps):
            x += dt * self(x, cond, t)
            t += dt

        return x

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
