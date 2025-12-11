import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from dataclasses import dataclass, field

from .hash_grid_encoding import MultiResHashGrid

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    OTPlanSampler
)

def divergence_hutchinson(v, x):
    """
    v : vector field output v_theta(x), shape [B, D]
    x : input point, requires_grad=True, shape [B, D]
    """
    eps = torch.randn_like(x)     # Gaussian or Rademacher both work

    # Jv = Jacobian-vector product: (dv/dx) * eps
    Jv = torch.autograd.grad(
        outputs=v,
        inputs=x,
        grad_outputs=eps,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Divergence ≈ eps^T * J * eps
    div = (Jv * eps).sum(dim=1)   # shape [B]
    return div


@dataclass
class VectorFieldConfig:
    sigma: float = 1.0
    dim: int = 768
    encoding_size: int = 64
    ot_method: str = "default"
    conditions: list[str] = field(default_factory=list)
    mlp_depth: int = 4
    hidden_dim: int = 512

class OneBlobEncoding(nn.Module):
    def __init__(self, encoding_size: int):
        super().__init__()
        self.encoding_size = encoding_size

    def forward(self, x):
        x = x.unsqueeze(-1)
        bins = torch.linspace(0, 1, self.encoding_size, device=x.device)
        return torch.exp(-0.5 * ((x - bins.unsqueeze(0)) * self.encoding_size) ** 2)

class SinusoidalEncoding(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even."
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reg = 500
        omega = torch.arange(self.dim // 2, dtype=torch.float32, device=x.device)
        omega /= self.dim / 2.
        omega = 1. / reg**omega
        
        pos = x.reshape(-1)
        out = torch.einsum('m,d->md', pos, omega)
        
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        return torch.cat([emb_sin, emb_cos], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim * 2)
        self.layer_norm_1 = nn.LayerNorm(dim * 2)
        self.activation = nn.SiLU()

        self.fc2 = nn.Linear(dim * 2, dim)
        self.layer_norm_2 = nn.LayerNorm(dim)


    def forward(self, x):

        h = self.fc1(x)
        h = self.layer_norm_1(h)
        h = self.activation(h)

        h = self.fc2(h)
        h = self.layer_norm_2(h)

        h = h + x

        return self.activation(h)


class VectorField(nn.Module):
    def __init__(self, config: VectorFieldConfig, num_threads: int | str ="max"):
        super().__init__()
  
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

        features_per_level = config.encoding_size // 16
        assert features_per_level * 16 == config.encoding_size, "encoding_size must be divisible by 16"
        self.hash_grid = MultiResHashGrid(
            dim=3,
            n_levels=16,
            n_features_per_level=features_per_level,
            log2_hashmap_size=19
        )

        print(f"Hash grid output dim: {self.hash_grid.output_dim}")

        # Build MLP dynamically based on depth and width
        input_dim = self.hash_grid.output_dim + config.dim
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, config.hidden_dim))
        layers.append(nn.SiLU())
        
        # Hidden layers
        for _ in range(config.mlp_depth - 2):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.SiLU())
        
        # Output layer
        layers.append(nn.Linear(config.hidden_dim, config.dim))
        
        self.net = nn.Sequential(*layers)

        self.config = config

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t, x, cond):        
        t = t.unsqueeze(-1) if t.dim() == 1 else t.expand(x.shape[0], 1)
        
        encoded_params = torch.cat([cond, t], dim=-1)
        encoded_params = self.hash_grid(encoded_params)

        return self.net(torch.cat([x, encoded_params], dim=-1))

    @torch.no_grad()
    def sample_flow(self, cond):
        from functools import partial
        """
        Integrates dx/dt = v_theta(x, t) from t=0 to 1.
        Start from simple noise distribution.
        """
        batch_size = cond.size(0)
        x = torch.randn(batch_size, self.config.dim, device=cond.device)
        x = odeint(partial(self, cond=cond), x, torch.as_tensor([0.0, 1.0], device=cond.device), atol=1e-5, rtol=1e-5)
        return x[-1]
