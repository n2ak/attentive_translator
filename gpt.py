from dataclasses import dataclass
import torch
from torch import nn
from decoder import Decoder


@dataclass
class GPTConfig:
    N: int
    vocab_size: int
    n_embeddings: int
    context_length: int
    n_heads: int
    ff_scale: int


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, device) -> None:
        super().__init__()
        self.decoder = Decoder(
            config.N,
            config.vocab_size,
            config.n_embeddings,
            config.context_length,
            config.n_heads,
            config.ff_scale,
            device,
            False
        )
        self.lin = nn.Linear(config.n_embeddings, config. vocab_size)

    def forward(self, x, y_true=None):
        loss = None
        logits = self.decoder(x)
        logits = self.lin(logits)
        if y_true is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.permute(0, 2, 1), y_true)
        return logits, loss
