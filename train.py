from utils import *
from model import AttentiveTranslator
import torch


def arrays_to_loader(
    *arrays,
    batch_size=32,
    shuffle=True
):

    from torch.utils.data import TensorDataset, DataLoader
    my_dataset = TensorDataset(*arrays)  # create your datset
    return DataLoader(
        my_dataset,
        batch_size,
        shuffle=shuffle
    )


def train(
    model: AttentiveTranslator,
    optimizer,
    loss_fn,
    epochs,
    data_loader,
    device,
    valid_loader=None,
    start_epoch=1,
):
    import tqdm
    model.train()
    last_epoch = start_epoch
    print(
        f"- Training is starting, {start_epoch+1}/{epochs+1}.\n"
        f"- Number of model's trainable params: {get_model_trainable_params_count(model)}.\n"
        f"- Train size : {len(data_loader)}.\n"
        f"- Validation size : {len(valid_loader) if (valid_loader is not None) else 0}.\n"
    )
    try:
        for epoch in range(start_epoch, epochs):
            losses = []
            iterator = tqdm.tqdm(data_loader)
            model.train()
            for x, y, t in iterator:
                out, loss = predict(model, x, y, device, t=t, loss_fn=loss_fn)
                losses.append(loss.item())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last_epoch = epoch
                iterator.set_description(
                    f"Epoch {epoch+1}, Loss: {torch.tensor(losses).mean()}")

            if valid_loader is not None:
                valid_loss = validate(
                    model,
                    valid_loader,
                    device,
                    t=t,
                    loss_fn=loss_fn
                )
                print(
                    f"Loss: {torch.tensor(losses).mean()}, Validation Loss: {valid_loss}"
                )
    except KeyboardInterrupt:
        print(
            f"- Stopped trainging at epoch: {last_epoch}\n"
            f"- Model weights will be saved."
        )
    return last_epoch


def validate(model, valid_loader, device, t=None, loss_fn=None):
    model.eval()
    valid_losses = []
    for x, y, t in valid_loader:
        out, loss = predict(model, x, y, device, t=t, loss_fn=loss_fn)
        valid_losses.append(loss.item())
    return torch.tensor(valid_losses).mean()


def predict(model, x, y, device, t=None, loss_fn=None):
    x = x.to(device)
    y = y.to(device)
    out = model(x, y)
    ret = out
    if (loss_fn is not None) and (t is not None):
        loss = loss_fn(out.to(device=device), t.long().to(device=device))
        ret = out, loss
    return ret


def main(
    args: dict,
):
    save_path: str = args["save_path"]

    start_token = args.get("start_token")
    end_token = args.get("end_token")
    num_lines = args.get("num_lines")
    dst_path = args.get("dst_path")
    src_path = args.get("src_path")
    batch_size = args.get("batch_size")
    block_size = args.get("block_size")
    N = args.get("N")
    epochs = args.get("epochs")
    train_size = args.get("train_size")
    n_head = args.get("n_head")
    n_embeddings = args.get("n_embeddings")
    encoder_ff_scale = args.get("encoder_ff_scale")
    decoder_ff_scale = args.get("decoder_ff_scale")
    load_path = args.get("load_path")

    print("Used args:")
    for k, v in args.items():
        print(f"    {k} : {v}")

    assert (save_path is not None) and (len(save_path) > 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Used device:", device)
    loss_fn = torch.nn.functional.cross_entropy
    src, src_vocab, dst, dst_vocab, chars = prepare_data(
        src_path,
        dst_path,
        block_size,
        add_start_end_tokens=True,
        num_lines=num_lines,
        start_token=start_token,
        end_token=end_token,
        remove_lines_containing_tokens=True
    )

    encoder_vocab_size = len(src_vocab)
    decoder_vocab_size = len(dst_vocab)

    model, optim, start_epoch = load_or_create_model(
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
        optim_fn=lambda m: torch.optim.Adam(m.parameters()),
    )
    src_train, src_valid, dst_train, dst_valid, chars_train, chars_valid = split(
        src,
        dst,
        chars,
        train_size=train_size
    )
    len(src_train), len(src_valid), len(dst_train), len(
        dst_valid), len(chars_train), len(chars_valid)
    train_loader = arrays_to_loader(
        src_train,
        dst_train,
        chars_train,
        batch_size=batch_size
    )
    valid_loader = arrays_to_loader(
        src_valid,
        dst_valid,
        chars_valid,
        batch_size=batch_size
    )
    last_epoch = train(
        model,
        optim,
        loss_fn,
        epochs,
        train_loader,
        device,
        valid_loader,
        start_epoch=start_epoch
    )
    save_model(model, last_epoch, optim, save_path)
    args.update({
        "src_vocab": src_vocab,
        "dst_vocab": dst_vocab,
    })
    save_params(save_path.replace(".pt", "_hyper_params.pickle"), args)


def parse_create(args) -> dict:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train-split", default=.8, type=float)
    argparser.add_argument(
        "--src-path", default="./resources/news-commentary-v9.fr-en.en", type=str)
    argparser.add_argument(
        "--dst-path", default="./resources/news-commentary-v9.fr-en.fr", type=str)

    argparser.add_argument("--start-token", default="[", type=str)
    argparser.add_argument("--end-token", default="]", type=str)
    argparser.add_argument("--num-lines", default=None, type=int)
    argparser.add_argument("--epochs", default=1, type=int)
    argparser.add_argument("--N", default=3, type=int)
    argparser.add_argument("--n-embeddings", default=32, type=int)
    argparser.add_argument("--n-head", default=4, type=int)
    argparser.add_argument("--encoder-ff-scale", default=4, type=int)
    argparser.add_argument("--decoder-ff-scale", default=4, type=int)
    argparser.add_argument("--batch-size", default=32, type=int)
    argparser.add_argument("--block-size", default=50, type=int)

    argparser.add_argument("--save-path", required=True, type=str)

    args = argparser.parse_args(args)
    return dict(args._get_kwargs())


def parse_pre(args):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--params-path", required=True, type=str)

    argparser.add_argument("--start-token", type=str)
    argparser.add_argument("--end-token", type=str)
    argparser.add_argument("--num-lines", type=int)
    argparser.add_argument("--batch-size", type=int)
    argparser.add_argument("--epochs", type=int)
    argparser.add_argument("--N", type=int)
    argparser.add_argument("--n-embeddings", type=int)
    argparser.add_argument("--n-head", type=int)
    argparser.add_argument("--encoder-ff-scale", type=int)
    argparser.add_argument("--decoder-ff-scale", type=int)
    argparser.add_argument("--block-size", type=int)
    argparser.add_argument("--save-path", type=str)
    argparser.add_argument("--load-path", type=str)

    args = dict(argparser.parse_args(args)._get_kwargs())

    params_path = args["params_path"]
    training_args = dict(args.items())
    del training_args["params_path"]

    params = load_params(params_path)
    params.update([(k, v) for (k, v) in training_args.items() if v])

    return params


if __name__ == "__main__":
    import argparse

    import sys
    assert sys.argv[1] in ["create", "from-pretrained"]
    if sys.argv[1] == "create":
        args = parse_create(sys.argv[2:])
    if sys.argv[1] == "from-pretrained":
        args = parse_pre(sys.argv[2:])

    main(args)
