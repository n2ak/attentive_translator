from utils import encode, decode


def test(
    model,
    input_text,
    block_size,
    src_vocab,
    dst_vocab,
    max_tokens=100,
    start_char="^",
    stop_char="$"
):
    model.eval()
    import torch
    out_text = ""

    for i in range(min(max_tokens, len(input_text) - block_size)):
        text = input_text[i:block_size+i]
        text = encode(text, src_vocab)
        out = encode(start_char*len(text), dst_vocab)
        input = torch.tensor([text])
        output = torch.tensor([out])
        result = model(input, output)
        char = result[:, -1, :].argmax(-1)
        char = decode(char, dst_vocab)
        if char == stop_char:
            print("incountered")
            break
        out_text += char

    return out_text
