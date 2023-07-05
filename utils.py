
def encode(text: str, dictionary):
    if isinstance(dictionary, dict):
        return [dictionary[c] for c in text]
    if isinstance(dictionary, list):
        return [dictionary.index(c) for c in text]
    raise ""


def decode(encoding, dict: dict) -> str:
    return ''.join([dict[c] for c in encoding])


def split(*arrays, train_size=.8, shuffle=True):
    from sklearn.model_selection import train_test_split
    return train_test_split(*arrays, train_size=train_size, shuffle=shuffle)


def decode_all(array, dict: dict) -> str:
    return list(map(lambda x: decode(x, dict), array))


def load_data(X, Y, block_size, stop_char_index, dst_vocab=None, src_vocab=None):
    import torch
    features = []
    target = []
    chars = []

    def s_e(text, i, block_size, return_last_char=False):
        end = i + block_size
        end = min(len(text), end)
        start = end-block_size

        text_chunk = text[start:end]
        assert len(
            text_chunk) == block_size, f"{len(text_chunk)} != {block_size}"
        # if not return_last_char:
        #           f"{len(text)=}, {i=}, {start=}, {end=}, {end-start=}")

        if return_last_char:
            return text_chunk, (text[end] if end < len(text) else stop_char_index)
        return text_chunk
    for f, t in zip(X, Y):
        max_len = max(len(f)-block_size+1, len(t) - block_size)
        for i in range(max_len):
            features.append(s_e(f, i, block_size))

            tt, c = s_e(t, i, block_size, return_last_char=True)
            target.append(tt)
            chars.append(c)

    features = torch.tensor(features)
    target = torch.tensor(target)
    chars = torch.tensor(chars)
    return features, target, chars


def remove_short(src, dst, block_size):
    new_src, new_dst = [], []
    for i in range(len(src)):
        txt = src[i]
        if len(txt) >= block_size:
            new_src.append(src[i])
            new_dst.append(dst[i])
    return new_src, new_dst


def remove_lines_containing_tokens_(src, dst, start_token, end_token):
    new_src, new_dst = [], []
    for i in range(len(src)):
        in_src = (start_token in src[i]) or (end_token in src[i])
        in_dst = (start_token in dst[i]) or (end_token in dst[i])
        if not (in_src or in_dst):
            new_src.append(src[i])
            new_dst.append(dst[i])
    return new_src, new_dst


def prepare_data(
    src_path,
    dst_path,
    block_size,
    start_token="$",
    end_token="^",
    add_start_end_tokens=True,
    device="cpu",
    num_lines=None,
    encoding="utf8",
    remove_lines_containing_tokens=False
):
    src = load_txt(src_path, num_lines, encoding=encoding)
    dst = load_txt(dst_path, num_lines, encoding=encoding)
    assert len(src) == len(dst), f"{len(src)} != {len(dst)}"
    # print(f"Found {len(src)} lines")
    if remove_lines_containing_tokens:
        src, dst = remove_lines_containing_tokens_(
            src, dst, start_token, end_token)
    assert len(src) == len(dst), f"{len(src)} != {len(dst)}"
    # print(f"Found {len(src)} lines")
    src_vocab = get_vocab(src)
    dst_vocab = get_vocab(dst)

    assert (start_token not in src_vocab +
            dst_vocab), f"'{start_token}' already in vocab"
    assert (end_token not in src_vocab +
            dst_vocab), f"'{end_token}' already in vocab"
    src_vocab.append(start_token)
    src_vocab.append(end_token)
    dst_vocab.append(start_token)
    dst_vocab.append(end_token)

    # add start and end to source lang
    src = [(start_token + txt + end_token) for txt in src]
    dst = [(txt + end_token) for txt in dst]

    src = [encode(txt, src_vocab) for txt in src]
    dst = [encode(txt, dst_vocab) for txt in dst]

    # src = list(filter(lambda l:len(l)==block_size,src))
    # dst = list(filter(lambda l:len(l)==block_size,dst))

    if add_start_end_tokens:
        src = include_tokens(
            src,
            start_token_index=src_vocab.index(start_token),
            amount=block_size-10
        )
        dst = include_tokens(
            dst,
            start_token_index=dst_vocab.index(start_token),
            amount=block_size
        )
    src, dst = remove_short(src, dst, block_size)
    src, dst, chars = load_data(
        src,
        dst,
        block_size,
        dst_vocab.index(end_token),
        dst_vocab=dst_vocab,
        src_vocab=src_vocab,
    )
    src, src_vocab, dst, dst_vocab, chars = src.to(device=device), src_vocab, dst.to(
        device=device), dst_vocab, chars.to(device=device)
    return src, src_vocab, dst, dst_vocab, chars


def load_txt(path, num_lines, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        return f.read().lower().splitlines()[:num_lines]


def get_vocab(texts, start="", end=""):
    if isinstance(texts, list):
        texts = ''.join(texts)
    return list(sorted(set(texts + start + end)))


def load_train_data(X, Y, block_size):
    for x, y in zip(X, Y):
        for i in range(len(X) - block_size):
            x[i:block_size+i]
            y[block_size+i:block_size+i+1]


def include_tokens(targets, start_token_index=None, end_token_index=None, amount=1):
    if (start_token_index is None) and (end_token_index is None):
        raise ""

    for i, t in enumerate(targets):
        if start_token_index is not None:
            targets[i] = ([start_token_index] * amount) + targets[i]
        if end_token_index is not None:
            targets[i] = targets[i] + ([end_token_index] * amount)
    return targets


def save_model(model, epoch, optimizer, path):
    import torch
    d = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss,
    }
    torch.save(d, path)
    print(f"Model weights saved to '{path}'.")


def get_model_trainable_params_count(model):
    return sum([p.nelement() for p in model.parameters()])


def save_params(
    path,
    d
):
    import pickle
    # print("saving dict:")

    with open(path, "wb") as f:
        return pickle.dump(d, f)


def load_params(path) -> dict:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def load_or_create_model(
        N,
        encoder_vocab_size,
        decoder_vocab_size,
        n_embeddings,
        n_head,
        src_shape,
        dst_shape,
        encoder_ff_scale,
        decoder_ff_scale,
        device,
        last_model_checkpoint_path=None,
        optim_fn=None,
):
    from utils import load_weights
    from model import AttentiveTranslator
    # raise ""

    model = AttentiveTranslator(
        N,
        encoder_vocab_size,
        decoder_vocab_size,
        n_embeddings,
        n_head,
        src_shape,
        dst_shape,
        encoder_ff_scale,
        decoder_ff_scale,
        device=device
    ).to(device=device)
    optimizer = None if optim_fn is None else optim_fn(model)
    if last_model_checkpoint_path:
        return load_weights(model,  last_model_checkpoint_path, optimizer=optimizer)
    return model, optimizer, 0


def load_weights(model, path, optimizer=None):
    import torch
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model weights loaded from '{path}'.")
    return model, optimizer, epoch
