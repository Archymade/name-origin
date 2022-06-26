import torch
from torch import Tensor

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from os import listdir
from os.path import join

from typing import Tuple, Optional


class TextDataset(Dataset):
    """
    Text Dataset object.
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_names = list(
            map(lambda x: join(self.root_dir, x), listdir(self.root_dir))
        )
        self.classes = list(map(lambda x: x.split(".")[0], listdir(self.root_dir)))

        self.int2label = dict(enumerate(self.classes))
        self.label2int = {v: k for (k, v) in self.int2label.items()}

        self.files = [self.read_file(f) for f in self.file_names]
        self.data, self.labels = list(), list()

        for file, label in self.files:
            self.data += file
            self.labels += label

        self.unique_characters = self.get_unique_chars()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        string = self.data[ix]
        return self.string2tensor(string), self.labels[ix]

    def string2tensor(self, string):
        string_data = torch.tensor(
            [self.unique_characters[s] for s in string], dtype=torch.int32
        )
        return string_data

    def read_file(self, f):
        """
        Read contents of text file. Returns data and labels.
        """
        with open(f, "rb") as file:
            contents = str(file.read(), encoding="utf-8").split("\n")

        labels = [self.label2int[f.split(".")[0].split("\\")[1]]] * len(contents)

        return contents[:-1], labels[:-1]

    def get_unique_chars(self):
        """
        Obtain unique tokens in Dataset.
        """
        unique_chars = sorted(
            list(set("".join(list(map(lambda x: "".join(x[0]), self.files)))))
        )

        return {v: k for (k, v) in dict(enumerate(unique_chars)).items()}


def pad_and_pack(batch: Tensor) -> Tuple[PackedSequence, Tensor]:
    """
    Pad and pack data batches in DataLoader.

    Parameters
    ----------
    batch
        Specific batch in DataLoader.

    Returns
    -------
    out
        Tuple of (packed data, labels)
    """

    instances = [X for (X, y) in batch]
    if target:
        labels = [y for (X, y) in batch]
    lengths = [X.shape[0] for (X, y) in batch]

    X_padded = pad_sequence(instances, batch_first=False)
    X_packed = pack_padded_sequence(
        X_padded, lengths, batch_first=False, enforce_sorted=False
    )

    return (X_packed, torch.tensor(labels, dtype=torch.int64)) if target else X_packed


def prepare_dataloaders(
    dataset: Dataset,
    test_size: int = 300,
    collate_fn: Callable[[Tensor], Optional[PackedSequence, Tensor]] = None,
    batch_size: int = 1
) -> Tuple[DataLoader, DataLoader]:
    """
    Split Dataset into train and test splits. Convert splits to DataLoaders.

    Parameters
    ----------
    dataset
        Dataset object to split.
    test_size
        Number of instances in test Dataset split.
    collate_fn
        Function to apply batch-wise in DataLoader.
    batch_size
        Number of instances per data batch.

    Returns
    -------
    out
        Tuple of (train_dataloader, test_dataloader)
    """

    train_sampler, test_sampler = random_split(
        dataset, lengths=[len(data) - test_size, test_size]
    )

    train_dl = DataLoader(
        train_sampler, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_dl = DataLoader(
        test_sampler, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dl, test_dl
