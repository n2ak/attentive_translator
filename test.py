from utils import encode, decode


def test(model, x, y):
    model.eval()
    out = model(x, y)
    out = out.argmax(-1)
    return out


def translate(
    model,
    input_text,
    block_size,
    src_vocab,
    dst_vocab,
    device,
    max_tokens=100,
    start_char="^",
    stop_char="$",
):
    import torch.nn.functional as F
    model.eval()
    import torch
    out_text = ""

    context = start_char*(block_size)
    iters = max_tokens
    input_text = start_char + input_text + stop_char
    if len(input_text) < block_size:
        input_text = (start_char*(block_size-len(input_text))) + input_text
    iters = min(max_tokens, len(input_text))
    for i in range(iters):
        _end = min(len(input_text), block_size+i)
        text = input_text[_end-block_size:_end]
        text = encode(text, src_vocab)

        input = torch.tensor([text]).to(device)

        output = encode(context, dst_vocab)
        output = torch.tensor([output]).to(device)

        result = model(input, output)

        char = result.argmax(-1)
        char = decode(char, dst_vocab)

        if char == stop_char:
            # print("incountered")
            break
        context = context[1:] + char
        out_text += char

    return out_text
