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
    

@dataclass
class VectorFieldConfig:
    sigma: float = 1.0
    dim: int = 768
    encoding_size: int = 128
    ot_method: str = "default"
    conditions: list[str] = field(default_factory=list)
    min_cond: torch.Tensor = None
    max_cond: torch.Tensor = None

class OneBlobEncoding(nn.Module):
    def __init__(self, encoding_size: int):
        super().__init__()
        self.encoding_size = encoding_size

    def forward(self, x):
        x = x.unsqueeze(-1)
        bins = torch.linspace(0, 1, self.encoding_size, device=x.device)
        return torch.exp(-0.5 * ((x - bins.unsqueeze(0)) * self.encoding_size) ** 2)

class VectorField(nn.Module):
    
    def __init__(self, config: VectorFieldConfig, num_threads: int = 4):
        nn.Module.__init__(self)

        match config.ot_method:
            case "default":
                self.solver = ConditionalFlowMatcher(config.sigma)
            case "target":
                self.solver = TargetConditionalFlowMatcher(config.sigma)
            case "variance_preserving":
                self.solver = VariancePreservingConditionalFlowMatcher(config.sigma)
            case "exact":
                self.solver = ExactOptimalTransportConditionalFlowMatcher(config.sigma)
                # Update number of available threads
                self.solver.ot_sampler = OTPlanSampler("exact", num_threads=num_threads)
            case "schrodinger":
                self.solver = SchrodingerBridgeConditionalFlowMatcher(config.sigma)
                # Update number of available threads
                self.solver.ot_sampler = OTPlanSampler(method=self.solver.ot_method, 
                                                       reg=2 * config.sigma**2, num_threads=num_threads)
            case _:
                raise ValueError(f"Unknown ot_method: {config.ot_method}")

        self.encoders = nn.ModuleDict({
            cond: nn.Sequential(
                OneBlobEncoding(encoding_size=2*config.encoding_size//3),
                nn.Linear(2*config.encoding_size//3, config.encoding_size),
                nn.SiLU(),
                nn.Linear(config.encoding_size, config.encoding_size),
            ) for cond in [*config.conditions, "time"]
        })
    
        input_size = config.dim + config.encoding_size
        hidden_size = input_size
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_size, bias=True),
                nn.Linear(input_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Dropout(0.5)
            ) for _ in range(6)
        ])

        self.head = nn.Linear(hidden_size, config.dim)
        
        self._init_weights()
        self.config = config

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, t: torch.Tensor, input: tuple[torch.Tensor, torch.Tensor]):
        x, cond = input

        if self.config.min_cond is None and self.config.max_cond is None:
            self.config.min_cond = torch.min(cond, dim=0).values
            self.config.max_cond = torch.max(cond, dim=0).values

        # Normalize conditions to [0, 1]
        cond = (cond - self.config.min_cond) / (self.config.max_cond - self.config.min_cond + 1e-8)

        encodings = torch.as_tensor(0, device=x.device)
        for key_idx, key in enumerate(self.config.conditions):
            encodings = encodings + self.encoders[key](cond[:, key_idx])
        encodings += self.encoders["time"](t)
        encodings = encodings / (len(self.config.conditions) + 1)

        mlp_input = torch.cat([x, encodings], dim=-1)
        for block in self.blocks:
            mlp_input = mlp_input + block(mlp_input)

        return (self.head(mlp_input), cond)


    def compute_loss(self, x0, x1, cond):
        t, xt, ut = self.solver.sample_location_and_conditional_flow(x0, x1)
        return F.mse_loss(self(t, (xt, cond))[0], ut)


    @torch.no_grad()
    def sample_flow(self, cond, steps=1000):
        """
        Integrates dx/dt = v_theta(x, t) from t=0 to 1.
        Start from simple noise distribution.
        """
        batch_size = cond.size(0)
        dt = 1 / steps

        t = torch.as_tensor(0.0, device=cond.device)
        x = torch.randn(batch_size, self.config.dim, device=cond.device)

        for _ in range(steps):
            x += dt * self(t, (x, cond))[0]
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
