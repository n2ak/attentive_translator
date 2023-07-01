from model import AttentiveTranslator
import torch


def arrays_to_loader(
    x,
    y,
    batch_size,
    shuffle=True
):

    from torch.utils.data import TensorDataset, DataLoader
    my_dataset = TensorDataset(x, y)  # create your datset
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
    device

):
    model.train()
    for epoch in range(epochs):
        losses = []
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            model.train()
            out = model(x, y)
            B, T, VS = out.shape
            out, y = out.view(-1, VS), y.view(-1)
            out, y = out.cpu(), y.cpu()
            loss = loss_fn(out, y.long())
            losses.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses = torch.tensor(losses)
        print(losses.mean())
