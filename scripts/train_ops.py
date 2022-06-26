# Train utility

from torch import Tensor
from torch.utils.data import DataLoader

from torch.optim import Optimizer
from torch.optim.lr_scheduler import Optimizer as Scheduler

from sklearn.metrics import accuracy_score
from typing import Tuple, Callable


def train_model(
    model: Callable[[Tensor], Tensor],
    criterion: Callable[[Tensor, Tensor], Tensor],
    metric: Callable[[Tensor, Tensor], Tensor],
    data: Tuple[DataLoader, DataLoader],
    optimizer: Optimizer,
    scheduler: Scheduler,
    epochs: int = 250
) -> dict:
    """
    Train neural network.

    Parameters
    ----------
    model
        Neural architecture to train.
    criterion
        Objective function.
    data
        Tuple of .. train_dataloader:: and .. test_dataloader::
    optimizer
        Optimizer for model weights.
    scheduler
        Learning rate scheduler. Changes learning rate to avoid local optima.
    epochs
        Number of times the network sees the complete dataset.

    Returns
    -------
    ret
        History of training epochs.
    """
    train_loss, test_loss = list(), list()
    train_acc, test_acc = list(), list()

    history = dict()

    train_dl, test_dl = data

    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_dl:
            y_pred = model(X)

            loss = criterion(y_pred, y)
            acc = accuracy_score(y, y_pred.max(dim=-1).indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.append(acc)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            for X_, y_ in test_dl:
                y_p = model(X_)

                acc = accuracy_score(y_, y_p.max(dim=-1).indices)
                loss = criterion(y_p, y_)

                test_loss.append(loss.item())
                test_acc.append(acc)

        history[epoch] = dict()
        history[epoch]["train_loss"] = sum(train_loss) / len(train_loss)
        history[epoch]["train_acc"] = sum(train_acc) / len(train_acc)

        history[epoch]["test_loss"] = sum(test_loss) / len(test_loss)
        history[epoch]["test_acc"] = sum(test_acc) / len(test_acc)

        print(
            f"Epoch {epoch:02d}/{epochs:02d}:",
            f"\n\tTrain loss -> {history[epoch]['train_loss']: .4f} | Test loss -> {history[epoch]['test_loss']: .4f}",
        )
        print(
            f"\tTrain accuracy -> {history[epoch]['train_acc']: .4f} | Test accuracy -> {history[epoch]['test_acc']: .4f}"
        )

        train_loss.clear()
        test_loss.clear()
        train_acc.clear()
        test_acc.clear()

    return history
