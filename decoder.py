
import torch.nn as nn
from attention import Embedding, FeedForward, MultiHeadAttention


class Decoder(nn.Module):
    def __init__(self, N, vocab_size, n_embeddings, time, n_heads, ff_scale, device, connected_to_encoder=True) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, n_embeddings, time)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(n_heads, n_embeddings, ff_scale, device, connected_to_encoder=connected_to_encoder)
             for _ in range(N)]
        )

    def forward(self, x, V=None, K=None):
        x = self.embedding(x)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, V, K)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_heads, n_embeddings, ff_scale, device, connected_to_encoder=True) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.mmha = MultiHeadAttention(
            n_heads, n_embeddings, device, masked=True)

        if connected_to_encoder:
            self.ln2 = nn.LayerNorm(n_embeddings)
            self.mha = MultiHeadAttention(n_heads, n_embeddings, device)

        self.ln3 = nn.LayerNorm(n_embeddings)
        self.ff = FeedForward(n_embeddings, ff_scale)

        self.connected_to_encoder = connected_to_encoder

    def forward(self, x, V=None, K=None):
        x_ = self.ln1(x)
        x = x + self.mmha(x_, x_, x_)
        if self.connected_to_encoder:
            assert (V is not None) and (K is not None)
            x = x + self.mha(self.ln2(x), V, K)

        x = x + self.ff(self.ln3(x))
        return x
