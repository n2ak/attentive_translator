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
    dst_vocab
):
    import tqdm
    model.train()
    for epoch in (range(epochs)):
        losses = []
        for x, y, t in (pbar2 := tqdm.tqdm(data_loader)):
            x = x.to(device)
            y = y.to(device)
            model.train()
            # print(x.shape, y.shape)
            out = model(x, y)
            B, VS = out.shape

            # print((decode_all(out.argmax(-1), dst_vocab)))

            # out, y = out.view(-1), y.view(-1)
            # out, y = out.cpu(), y.cpu()
            # print(out.shape, y.shape, t.shape)
            loss = loss_fn(out, t.long())
            losses.append(loss.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # print(loss.item())
            pbar2.set_description(f"Loss: {torch.tensor(losses).mean()}")
        # pbar.set_description(f"Loss: {loss}")
        # losses = torch.tensor(losses)
        # print("avg loss", losses.mean())
