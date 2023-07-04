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
    for i in range(min(max_tokens, len(input_text) - block_size)):
        text = input_text[i:block_size+i]
        text = encode(text, src_vocab)

        input = torch.tensor([text]).to(device)

        output = encode(context, dst_vocab)
        output = torch.tensor([output]).to(device)

        result = model(input, output)

        char = result.argmax(-1)
        char = decode(char, dst_vocab)
        print(char)

        if char == stop_char:
            print("incountered")
            break
        context = context[1:] + char
        # break
        out_text += char

    return out_text
