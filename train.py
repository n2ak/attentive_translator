
from model import AttentiveTranslator


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
    data_loader
):

    for epoch in range(epochs):
        for x, y in data_loader:
            model.train()
            out = model(x, y)
            loss = loss_fn(out.view(-1, 26), y.view(-1))
            print(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
