# generative_model/DiT.py

import torch
import torch.nn as nn
import numpy as np

from timm.models.vision_transformer import Attention, Mlp
from .DiT_utils import get_1d_sincos_pos_embed_from_grid


def modulate(x, shift, scale):
    """
    AdaLN modulation: x * (1 + scale) + shift
    x:     (B, N, H)
    shift: (B, H)
    scale: (B, H)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Simple MLP to embed scalar timesteps into hidden_size.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) long or float
        t = t.float().unsqueeze(-1)      # (B, 1)
        return self.mlp(t)               # (B, hidden_size)


class ConditionsEmbedder(nn.Module):
    """
    MLP to embed conditioning vector (e.g. [mass, label]) into hidden_size.
    """
    def __init__(self, hidden_size: int, conditions_size: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(conditions_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        # conditions: (B, conditions_size)
        return self.mlp(conditions)      # (B, hidden_size)


class DiTBlock(nn.Module):
    """
    One DiT transformer block with adaLN-Zero conditioning.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0.0,
        )

        # adaLN modulation producing 6 * H parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, H)
        c: (B, H) conditioning (time + conditions)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-attention branch
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )

        # MLP branch
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class FinalLayer(nn.Module):
    """
    Final DiT layer with adaLN-Zero modulation, maps tokens back to patch space.
    """
    def __init__(self, hidden_size: int, patch_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, H)
        c: (B, H)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)  # (B, N, patch_size)
        return x


class DiT(nn.Module):
    """
    DiT over 1D AstroPT embedding vectors.

    - input_size: embedding dimension (e.g. 768)
    - patch_size: size of 1D patches (e.g. 4 -> 768/4 = 192 tokens)
    - cond_dim:   condition dimension (e.g. 2 for [mass, label])
    """
    def __init__(
        self,
        input_size: int = 712,
        patch_size: int = 4,
        hidden_size: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        cond_dim: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.cond_dim = cond_dim

        assert input_size % patch_size == 0, \
            f"input_size {input_size} must be divisible by patch_size {patch_size}"
        self.num_patches = input_size // patch_size

        # Patch embedding: 1D vector -> N patches of size patch_size
        self.x_embedder = nn.Linear(patch_size, hidden_size)

        # Time & condition embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = ConditionsEmbedder(hidden_size, conditions_size=cond_dim)

        # Fixed sin-cos positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size),
            requires_grad=False,
        )

        # Transformer backbone
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Generic linear init
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Positional embedding (frozen)
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.pos_embed.shape[-1],
            np.arange(self.num_patches)
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # Timestep MLP gets normal init
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN final linear layers for stability (adaLN-Zero)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) -> (B, N, P)
        """
        B = x.shape[0]
        return x.view(B, self.num_patches, self.patch_size)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, P) -> (B, D)
        """
        B = x.shape[0]
        return x.view(B, self.input_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Predict noise ε_θ(x_t, t, cond).

        x:          (B, D)
        t:          (B,)
        conditions: (B, cond_dim)  e.g. [mass, label]
        returns:    (B, D)
        """
        # Patchify & embed
        x = self.patchify(x)             # (B, N, P)
        x = self.x_embedder(x)           # (B, N, H)
        x = x + self.pos_embed           # broadcast (1, N, H)

        # Time + condition embedding -> single context vector c
        t_embed = self.t_embedder(t)               # (B, H)
        c_embed = self.c_embedder(conditions)      # (B, H)
        c = t_embed + c_embed                      # (B, H)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final projection back to patch space, then unpatchify
        x = self.final_layer(x, c)       # (B, N, P)
        x = self.unpatchify(x)           # (B, D)

        return x