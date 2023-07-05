from utils import encode, decode, load_params


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
            break
        context = context[1:] + char
        out_text += char

    return out_text


def main(
    text,
    args: dict
):
    import torch
    from utils import load_or_create_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_token = args["start_token"]
    end_token = args["end_token"]
    block_size = args["block_size"]
    N = args["N"]
    n_embeddings = args["n_embeddings"]
    n_head = args["n_head"]
    encoder_ff_scale = args["encoder_ff_scale"]
    decoder_ff_scale = args["decoder_ff_scale"]
    src_vocab = args["src_vocab"]
    dst_vocab = args["dst_vocab"]
    load_path = args["load_path"] if "load_path" in args.keys(
    ) else args["save_path"]

    encoder_vocab_size = len(src_vocab)
    decoder_vocab_size = len(dst_vocab)

    model, _, _ = load_or_create_model(
        N,
        encoder_vocab_size,
        decoder_vocab_size,
        n_embeddings,
        n_head,
        block_size,
        block_size,
        encoder_ff_scale,
        decoder_ff_scale,
        device,
        load_path,
        optim_fn=None,
    )
    model.eval()
    output_text = translate(
        model,
        text,
        block_size,
        src_vocab,
        dst_vocab,
        device,
        start_char=start_token,
        stop_char=end_token,
    )

    print(
        f"Source text: '{text}'\n"
        f"Output text: '{output_text}'"
    )


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--params-path", required=True, type=str)
    argparser.add_argument("--text", required=True, type=str)

    args = dict(argparser.parse_args()._get_kwargs())

    params = load_params(args["params_path"])

    main(args["text"], params)
