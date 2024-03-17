
import math
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange


def is_float_dtype(dtype):
    return any([dtype == float_dtype for float_dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16)])


def exists(val):
    return val is not None


class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        *,
        expansion_factor=2.,
        depth=2,
        norm=False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        def norm_fn(): return nn.LayerNorm(hidden_dim) if norm else nn.Identity()

        layers = [nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.SiLU(),
            norm_fn()
        )]

        for _ in range(depth - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                norm_fn()
            ))

        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        assert is_float_dtype(
            dtype), 'input to sinusoidal pos emb must be a float type'

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(
            half_dim, device=device, dtype=dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1).type(dtype)


class TimeEmbedding(nn.Module):
    def __init__(self, dim,
                 num_timesteps=None,
                 num_time_embeds=1,
                 device=None):
        super(TimeEmbedding, self).__init__()
        self.device = device
        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim),
                                                                                                           MLP(dim, dim * num_time_embeds)),  # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n=num_time_embeds)
        )

    def forward(self, x):
        emb = self.to_time_embeds(x)
        return emb


if __name__ == "__main__":
    # Example usage
    b_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = torch.arange(0, b_size, device=device).float()
    print(t.shape)
    print(t)
    model = TimeEmbedding(dim=8, num_time_embeds=2, device=device)
    model.to(device)
    emb = model(t)
    print(emb.shape)
    print(emb)
