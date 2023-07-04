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
        f"- Training is starting, {start_epoch}/{epochs}.\n"
        f"- Train size : {len(data_loader)}.\n"
        f"- Validation size : {len(valid_loader) if (valid_loader is not None) else 0}."
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
                    f"Epoch {epoch}, Loss: {torch.tensor(losses).mean()}")

            if valid_loader is not None:
                valid_loss = validate(
                    model,
                    valid_loader,
                    device,
                    t=t,
                    loss_fn=loss_fn
                )
                print(
                    f"Epoch {epoch}, Loss: {torch.tensor(losses).mean()}, Validation Loss: {valid_loss}"
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
        loss = loss_fn(out, t.long())
        ret = out, loss
    return ret


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
    print(n_embeddings, n_head)
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
    optimizer = optim_fn(model)
    if last_model_checkpoint_path:
        return load_weights(model, optimizer, last_model_checkpoint_path)
    return model, optimizer, 0


def main(
    src_path,
    dst_path,
    start_token,
    end_token,
    num_lines,
    batch_size,
    epochs,
    model_args,
    last_model_checkpoint_path,
    new_model_checkpoint_path,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = torch.nn.functional.cross_entropy
    src, src_vocab, dst, dst_vocab, chars = prepare_data(
        src_path,
        dst_path,
        50,
        add_start_end_tokens=True,
        num_lines=num_lines,
        start_token=start_token,
        end_token=end_token,
        remove_lines_containing_tokens=True
    )
    model, optim, start_epoch = load_or_create_model(
        model_args["N"],
        len(src_vocab),
        len(dst_vocab),
        model_args["n_embeddings"],
        model_args["n_head"],
        src.shape,
        dst.shape,
        model_args["encoder_ff_scale"],
        model_args["decoder_ff_scale"],
        device,
        last_model_checkpoint_path,
        optim_fn=lambda m: torch.optim.Adam(m.parameters()),
    )
    # optim =
    src_train, src_valid, dst_train, dst_valid, chars_train, chars_valid = split(
        src, dst, chars)
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
    save_model(model, last_epoch, optim, new_model_checkpoint_path)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--train-split", default=.8, type=float)
    args.add_argument(
        "--src-path", default="./resources/news-commentary-v9.fr-en.en", type=str)
    args.add_argument(
        "--dst-path", default="./resources/news-commentary-v9.fr-en.fr", type=str)
    args.add_argument("--start-token", default="[", type=str)
    args.add_argument("--end-token", default="]", type=str)
    args.add_argument("--num-lines", default=None, type=int)
    args.add_argument("--batch-size", default=32, type=int)
    args.add_argument("--epochs", default=1, type=int)
    args.add_argument("--N", default=3, type=int)
    args.add_argument("--n-embeddings", default=32, type=int)
    args.add_argument("--n-head", default=4, type=int)
    args.add_argument("--encoder-ff-scale", default=4, type=int)
    args.add_argument("--decoder-ff-scale", default=4, type=int)
    args.add_argument("--load-path", default="", type=str)
    args.add_argument("--save-path", required=True, type=str)

    args = args.parse_args()
    args_dict = dict(args._get_kwargs())

    train_split = args_dict["train_split"]
    src_path = args_dict["src_path"]
    dst_path = args_dict["dst_path"]
    start_token = args_dict["start_token"]
    end_token = args_dict["end_token"]
    num_lines = args_dict["num_lines"]
    batch_size = args_dict["batch_size"]
    epochs = args_dict["epochs"]

    N = args_dict["N"]
    n_embeddings = args_dict["n_embeddings"]
    n_head = args_dict["n_head"]
    encoder_ff_scale = args_dict["encoder_ff_scale"]
    decoder_ff_scale = args_dict["decoder_ff_scale"]

    last_model_checkpoint_path = args_dict["load_path"]
    new_model_checkpoint_path = args_dict["save_path"]

    model_args = {
        "N": N,
        "n_embeddings": n_embeddings,
        "n_head": n_head,
        "encoder_ff_scale": encoder_ff_scale,
        "decoder_ff_scale": decoder_ff_scale,
    }
    main(
        src_path,
        dst_path,
        start_token,
        end_token,
        num_lines,
        batch_size,
        epochs,
        model_args,
        last_model_checkpoint_path,
        new_model_checkpoint_path,
    )
