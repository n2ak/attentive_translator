
def encode(text: str, dictionary):
    if isinstance(dictionary, dict):
        return [dictionary[c] for c in text]
    if isinstance(dictionary, list):
        return [dictionary.index(c) for c in text]
    raise ""


def decode(encoding, dict: dict) -> str:
    # print(encoding)
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
        #     print(decode(text[start:end], src_vocab),
        #           f"{len(text)=}, {i=}, {start=}, {end=}, {end-start=}")

        if return_last_char:
            return text_chunk, (text[end] if end < len(text) else stop_char_index)
        return text_chunk
    for f, t in zip(X, Y):
        # print(len(f), len(t))
        max_len = max(len(f)-block_size+1, len(t) - block_size)
        for i in range(max_len):
            # if i > len(f):
            #     print(i, len(f))
            features.append(s_e(f, i, block_size))

            tt, c = s_e(t, i, block_size, return_last_char=True)
            target.append(tt)
            chars.append(c)

    features = torch.tensor(features)
    target = torch.tensor(target)
    chars = torch.tensor(chars)
    # print(features.shape, target.shape, chars.shape)
    return features, target, chars


def remove_short(src, dst, block_size):
    new_src, new_dst = [], []
    for i in range(len(src)):
        txt = src[i]
        if len(txt) >= block_size:
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
    if remove_lines_containing_tokens:
        src = [txt for txt in src if (
            (start_token not in txt) and (end_token not in txt))]
        dst = [txt for txt in dst if (
            (start_token not in txt) and (end_token not in txt))]

    # print(len(src))
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
        # print("1", targets[i])
    return targets
