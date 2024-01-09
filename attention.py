# pylint: disable=E1101,C0103
import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, n_embeddings, head_size, masked, device) -> None:
        super().__init__()
        self.q = nn.Linear(n_embeddings, head_size)
        self.v = nn.Linear(n_embeddings, head_size)
        self.k = nn.Linear(n_embeddings, head_size)
        self.masked = masked
        self.device = device

    def forward(self, Q, V, K):
        x = Q
        Q = self.q(Q)
        V = self.v(V)
        K = self.k(K)
        dk = Q.shape[-1]
        weights = K @ Q.transpose(1, 2) / torch.sqrt(torch.tensor(dk))
        if self.masked:
            tril = torch.tril(torch.ones(weights.shape)).to(self.device)
            weights = weights.masked_fill(tril == 0, float("-inf"))
        weights = torch.softmax(weights, -1)
        x = weights @ V
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embeddings, device, masked=False) -> None:
        super().__init__()
        assert n_embeddings % n_heads == 0, f"{n_embeddings=} must devide {n_heads=}"
        n = n_embeddings // n_heads
        self.heads = nn.ModuleList([Head(n_embeddings, n, masked, device)
                                   for _ in range(n_heads)])
        # TODO projection + dropout

    def forward(self, q, v, k):
        xs = [head(q, v, k) for head in self.heads]
        xs = torch.cat(xs, -1)
        return xs


class FeedForward(nn.Module):
    def __init__(self, num_embeddings, scale, dropout=.1) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(num_embeddings, num_embeddings*scale), nn.ReLU(),
            nn.Linear(num_embeddings*scale, num_embeddings),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size, n_embeddings, time) -> None:
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, n_embeddings)
        self.positional_embedding = nn.Embedding(
            time, n_embeddings
        )
        self.register_buffer(
            "positions",
            torch.arange(0, time)
        )

    def forward(self, x):
        pos = self.positional_embedding(self.positions)
        x = self.input_embedding(x) + pos
        return x
