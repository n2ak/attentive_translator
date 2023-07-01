import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO device


class AttentiveTranslator(nn.Module):

    def __init__(
        self,
        N,
        vocab_size,
        decoder_vocab_size,
        n_embeddings,
        n_heads,
        input_shape,
        output_shape
    ) -> None:
        super().__init__()
        assert (n_embeddings % n_heads) == 0, ""

        B, input_length = input_shape
        B, output_length = output_shape

        head_size = n_embeddings // n_heads

        self.encoder = Encoder(
            N,
            vocab_size,
            n_embeddings,
            input_length,
            head_size
        )
        self.decoder = Decoder(
            N,
            decoder_vocab_size,
            n_embeddings,
            output_length,
            head_size,
        )
        self.linear = nn.Linear(n_embeddings, decoder_vocab_size)

    def forward(self, x, target):
        # print("x", x.shape)
        x = self.encoder(x)
        # print("x", x.shape, "encoder done")
        target = self.decoder(target, x, x)
        target = self.linear(target)

        return target


class Encoder(nn.Module):
    def __init__(self, N, vocab_size, n_embeddings, time, head_size) -> None:
        super().__init__()
        """
        time : Block size
        """
        self.input_embedding = InputEmbedding(vocab_size, n_embeddings)
        self.positional_embedding = PositionalEncoding(
            time, n_embeddings
        )
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(head_size, n_embeddings) for _ in range(N)]
        )
        self.register_buffer(
            "position_buffer",
            torch.arange(0, time, device="cpu")
        )

    def forward(self, x):
        # print(x.shape)
        x = self.input_embedding(x)
        postions = self.positional_embedding(self.position_buffer)
        # print(x.shape, postions.shape)
        x = x + postions
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, head_size, n_embeddings) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(head_size, n_embeddings)
        self.aan1 = AddAndNorm(n_embeddings)

        self.ff = FeedForward(n_embeddings, 1)
        self.aan2 = AddAndNorm(n_embeddings)

    def forward(self, x):
        x = self.aan1(self.mha(x, x, x), x)
        x = self.aan2(self.ff(x), x)
        return x


class Decoder(nn.Module):
    def __init__(self, N, vocab_size, n_embeddings, time, n_heads) -> None:
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size, n_embeddings)
        self.positional_embedding = PositionalEncoding(time, n_embeddings)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(n_heads, n_embeddings) for _ in range(N)]
        )
        self.register_buffer(
            "position_buffer",
            torch.arange(0, time, device="cpu")
        )

    def forward(self, x, V, K):
        x = self.input_embedding(x)
        postions = self.positional_embedding(self.position_buffer)
        x = x + postions
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, V, K)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_heads, n_embeddings) -> None:
        super().__init__()
        self.mmha = MaskedMultiHeadAttention(n_heads, n_embeddings)
        self.aan1 = AddAndNorm(n_embeddings)

        self.mha = MultiHeadAttention(n_heads, n_embeddings)
        self.aan2 = AddAndNorm(n_embeddings)

        self.ff = FeedForward(n_embeddings, 1)
        self.aan3 = AddAndNorm(n_embeddings)

    def forward(self, x, V, K):
        x = self.aan1(self.mmha(x, x, x), x)
        x = self.aan2(self.mha(V, K, x), x)
        x = self.aan3(self.ff(x), x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, n_embeddings, masked=False) -> None:
        super().__init__()

        self.core = nn.Linear(n_embeddings, n_embeddings,
                              bias=False)  # TODO: * 3

        self.sdpa = ScaledDotProductAttention(masked=masked)
        self.concat = torch.cat
        self.linear = nn.Linear(n_embeddings, n_embeddings)  # TODO Not sure

        self.nh = n_embeddings // head_size
        self.hs = head_size

    def forward(self, V, K, Q):
        (B, T, E) = V.shape

        # V.shape=K.shape=Q.shape = (B,T,E)
        yo = torch.concat((V, K, Q), dim=1)  # (B,3*T,E)
        yo = self.core(yo)  # (B,3*T,E)
        _, s, _ = yo.shape
        V, K, Q = yo.split(s//3, dim=1)  # (B,3*T,E)

        V = V.view(B, self.nh, T, self.hs)  # (B,n_heads,T,head_size)
        K = K.view(B, self.nh, T, self.hs)  # (B,n_heads,T,head_size)
        Q = Q.view(B, self.nh, T, self.hs)  # (B,n_heads,T,head_size)

        res = self.sdpa(V, K, Q)  # (B,n_heads,T,head_size)

        res = res.view(B, T, E)  # (B,T,E)
        # res = self.concat(res, 1)
        res = self.linear(res)  # (B,T,E)

        return res


class ScaledDotProductAttention(nn.Module):
    def __init__(self, masked: bool) -> None:
        super().__init__()
        self.scale_f = None
        self.tril = None
        self.masked = masked
        self.matmul = torch.matmul
        self.scale = lambda a: (a / self.scale_f)  # TODO
        self.softmax = F.softmax

    def forward(self, V, K, Q):
        # V.shape=K.shape=Q.shape = (B,n_heads,T,head_size)
        (B, n_heads, T, head_size) = V.shape
        if self.scale_f is None:
            self.scale_f = torch.sqrt(torch.tensor(head_size))

        # (B,n_heads,T,head_size) * (B,n_heads,head_size,T) = (B,n_heads,T,T)
        x = self.matmul(Q, K.transpose(3, 2))  # (B,n_heads,T,T)
        x = self.scale(x)  # (B,n_heads,T,T)
        x = self.mask(x)  # (B,n_heads,T,T)
        x = self.softmax(x, dim=-1)  # (B,n_heads,T,T)
        # (B,n_heads,T,T) * (B,T,head_size) = (B,n_heads,T,head_size)
        x = self.matmul(x, V)  # (B,n_heads,T,head_size)

        return x

    def mask(self, t: torch.Tensor):
        # TODO add mask
        if self.masked:
            if self.tril is None:
                self.tril = torch.tril(torch.ones(t.shape[-2:]))
            t = t.masked_fill(self.tril == 0, float("-inf"))
        return t


class PositionalEncoding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim) -> None:
        super().__init__()
        self.positional_embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.positional_embedding(x)


class InputEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim) -> None:
        super().__init__()
        self.input_embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.input_embedding(x)


class FeedForward(nn.Module):
    def __init__(self, num_embeddings, scale) -> None:
        super().__init__()
        self.linear = nn.Linear(num_embeddings, num_embeddings*scale)
        self.linear2 = nn.Linear(num_embeddings*scale, num_embeddings)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class AddAndNorm(nn.Module):
    def __init__(self, n_embeddings) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_embeddings)

    def forward(self, x, skip_conn):
        x = x + skip_conn
        x = self.norm(x)
        return x


class MaskedMultiHeadAttention(MultiHeadAttention):
    def __init__(self, n_heads, n_embeddings) -> None:
        super().__init__(n_heads, n_embeddings, masked=True)
