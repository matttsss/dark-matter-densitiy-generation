import torch
import torch.nn as nn

from torchdiffeq import odeint
from dataclasses import dataclass, field

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    OTPlanSampler
)

@dataclass
class VectorFieldConfig:
    sigma: float = 1.0
    dim: int = 128
    encoding_size: int = 64
    hidden: int = 512
    ot_method: str = "default"
    conditions: list[str] = field(default_factory=list)

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

    def forward(self, t, x, cond):
        encoding = self.encoding_net(cond)
        if t.dim() == 0:
            t = t.expand(x.size(0), 1)
        else:
            t = t.unsqueeze(-1)
        
        return self.net(torch.cat([x, encoding, t], dim=-1))

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
