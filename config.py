import collections


@collections.dataclass
class ModelConfig:
    N: int
    n_heads: int
    num_lines: int
    block_size: int
    n_embeddings: int
    encoder_ff_scale: int
    decoder_ff_scale: int
    input_block_size: int
    output_block_size: int


@collections.dataclass
class DatasetConfig:
    end_token: str
    start_token: str
    encoder_vocab_size: int
    decoder_vocab_size: int


@collections.dataclass
class TrainingConfig:
    epochs: int
    train_split: float
    batch_size: int

    src_path: str
    dst_path: str
    block_size: int
    last_snapshot_path: str

    save_interval: int = 9999999
