
def encode(text: str, dictionary):
    if isinstance(dictionary, dict):
        return [dictionary[c] for c in text]
    if isinstance(dictionary, list):
        return [dictionary.index(c) for c in text]
    raise ""


def decode(encoding, dict: dict) -> str:
    return ''.join([dict[c] for c in encoding])


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


def include_tokens(targets, start_token_index=None, end_token_index=None):
    if (start_token_index is None) and (end_token_index is None):
        raise ""
    import torch
    num_examples = len(targets)
    final = []
    if start_token_index is not None:
        final.append(torch.ones((num_examples, 1)) * start_token_index)
    final.append(targets)
    if end_token_index is not None:
        final.append(torch.ones((num_examples, 1)) * end_token_index)
    return torch.concat(final, 1).int()
