import torch.nn as nn
from .attention import Embedding, MultiHeadAttention, FeedForward


class Encoder(nn.Module):
    def __init__(self, N, vocab_size, n_embeddings, time, n_heads, ff_scale, device) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, n_embeddings, time)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(n_heads, n_embeddings, ff_scale, device)
             for _ in range(N)]
        )

    def forward(self, x):
        x = self.embedding(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, n_heads, n_embeddings, ff_scale, device) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.mha = MultiHeadAttention(
            n_heads, n_embeddings, device, masked=False)

        self.ln2 = nn.LayerNorm(n_embeddings)
        self.ff = FeedForward(n_embeddings, ff_scale)

    def forward(self, x):
        x_ = self.ln1(x)
        print(x_.shape)
        x = x + self.mha(x_, x_, x_)
        x = x + self.ff(self.ln2(x))
        return x
