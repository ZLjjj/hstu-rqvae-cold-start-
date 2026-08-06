import gin
import torch

from distributions.gumbel import gumbel_softmax_sample
from einops import rearrange
from enum import Enum
from init.kmeans import kmeans_init_
from modules.loss import QuantizeLoss
from modules.normalize import L2NormalizationLayer
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F


@gin.constants_from_enum
class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3
    EMA = 4


class QuantizeDistance(Enum):
    L2 = 1
    COSINE = 2


class QuantizeOutput(NamedTuple):
    embeddings: Tensor
    ids: Tensor
    loss: Tensor


def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()

    return (
        e -
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) +
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    ).squeeze()


class Quantize(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        do_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        sim_vq: bool = False,  # https://arxiv.org/pdf/2411.02038
        commitment_weight: float = 0.25,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        distance_mode: QuantizeDistance = QuantizeDistance.L2,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dead_code_threshold: float = 1.0,
        dead_code_reset: bool = False,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.dead_code_threshold = dead_code_threshold
        self.dead_code_reset = dead_code_reset

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False) if sim_vq else nn.Identity(),
            L2NormalizationLayer(dim=-1) if codebook_normalize else nn.Identity()
        )

        self.quantize_loss = QuantizeLoss(commitment_weight)

        self.register_buffer("ema_cluster_size", torch.zeros(n_embed))
        self.register_buffer("ema_embed_avg", torch.zeros(n_embed, embed_dim))
        self.register_buffer("last_active_codes", torch.zeros(n_embed, dtype=torch.bool))
        self.register_buffer("last_dead_code_ratio", torch.tensor(0.0))

        if self.forward_mode == QuantizeForwardMode.EMA:
            self.embedding.weight.requires_grad_(False)
        self._init_weights()

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
    
    @torch.no_grad
    def _kmeans_init(self, x) -> None:
        kmeans_init_(self.embedding.weight, x=x)
        self.kmeans_initted = True
        self.ema_embed_avg.copy_(self.embedding.weight.data)
        self.ema_cluster_size.fill_(1.0)

    @torch.no_grad
    def _ema_update(self, x: Tensor, ids: Tensor) -> None:
        x_ema = x.to(self.ema_embed_avg.dtype)
        one_hot = F.one_hot(ids, num_classes=self.n_embed).to(x_ema.dtype)
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.t() @ x_ema

        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=(1.0 - self.ema_decay)
        )
        self.ema_embed_avg.mul_(self.ema_decay).add_(
            embed_sum, alpha=(1.0 - self.ema_decay)
        )

        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.ema_eps)
            / (n + self.n_embed * self.ema_eps)
            * n
        )
        new_weight = self.ema_embed_avg / cluster_size.unsqueeze(1)

        if self.dead_code_reset:
            dead = self.ema_cluster_size < self.dead_code_threshold
            if dead.any() and x_ema.size(0) > 0:
                num_dead = int(dead.sum().item())
                reset_indices = torch.randint(
                    low=0, high=x_ema.size(0), size=(num_dead,), device=x_ema.device
                )
                new_weight[dead] = x_ema[reset_indices]
                self.ema_embed_avg[dead] = x_ema[reset_indices]
                self.ema_cluster_size[dead] = self.dead_code_threshold + 1.0
            self.last_dead_code_ratio.copy_(dead.float().mean())
        else:
            self.last_dead_code_ratio.zero_()

        self.last_active_codes.copy_(cluster_size > self.dead_code_threshold)
        self.embedding.weight.data.copy_(new_weight)

    def get_item_embeddings(self, item_ids) -> Tensor:
        return self.out_proj(self.embedding(item_ids))

    def forward(self, x, temperature) -> QuantizeOutput:
        assert x.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x=x)

        codebook = self.out_proj(self.embedding.weight)
        
        if self.distance_mode == QuantizeDistance.L2:
            dist = (
                (x**2).sum(axis=1, keepdim=True) +
                (codebook.T**2).sum(axis=0, keepdim=True) -
                2 * x @ codebook.T
            )
        elif self.distance_mode == QuantizeDistance.COSINE:
            dist = -(
                x / x.norm(dim=1, keepdim=True) @
                (codebook.T) / codebook.T.norm(dim=0, keepdim=True)
            )
        else:
            raise Exception("Unsupported Quantize distance mode.")

        _, ids = (dist.detach()).min(axis=1)

        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                weights = gumbel_softmax_sample(
                    -dist, temperature=temperature, device=self.device
                )
                emb = weights @ codebook
                emb_out = emb
            elif self.forward_mode == QuantizeForwardMode.STE:
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()
            elif self.forward_mode == QuantizeForwardMode.ROTATION_TRICK:
                emb = self.get_item_embeddings(ids)
                emb_out = efficient_rotation_trick_transform(
                    x / (x.norm(dim=-1, keepdim=True) + 1e-8),
                    emb / (emb.norm(dim=-1, keepdim=True) + 1e-8),
                    x
                )
                emb_out = emb_out * (
                    torch.norm(emb, dim=1, keepdim=True) / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
                ).detach()
            elif self.forward_mode == QuantizeForwardMode.EMA:
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()
                self._ema_update(x=x.detach(), ids=ids.detach())
            else:
                raise Exception("Unsupported Quantize forward mode.")
            
            loss = self.quantize_loss(query=x, value=emb)
        
        else:
            emb_out = self.get_item_embeddings(ids)
            loss = self.quantize_loss(query=x, value=emb_out)

        return QuantizeOutput(
            embeddings=emb_out,
            ids=ids,
            loss=loss
        )
