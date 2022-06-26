# Network utilities

from torch.nn import Module
from torch.nn import Linear, RNN
from torch.nn import Embedding, Dropout

from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch import log_softmax


# Embedding class for PackedSequence
class PackedEmbedding(Module):
    def __init__(self, embedding_layer: Embedding):
        super(PackedEmbedding, self).__init__()
        self.embedding = embedding_layer

    def forward(self, x):
        if type(x) == PackedSequence:
            unpacked_sequence, lengths = pad_packed_sequence(x, batch_first=True)
            y = self.embedding(unpacked_sequence)
            y = pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
        else:
            y = self.embedding(x)

        return y


class RNNNetwork(Module):
    """
    RNN classification network.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 128,
        num_class: int = 18,
        feature_size: int = 32,
        p: float = 0.3,
    ):
        super(RNNNetwork, self).__init__()

        self.p = p
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_class = num_class
        self.feature_size = feature_size

        self.embedder = PackedEmbedding(
            Embedding(input_size=self.vocab_size, hidden_size=self.feature_size)
        )
        self.rnn = RNN(
            input_size=self.feature_size, hidden_size=self.hidden_size, batch_first=True
        )
        self.fc = Linear(in_features=self.hidden_size, out_features=self.num_class)

        self.dropout = Dropout(p=self.p)

    def forward(self, x):
        x = self.embedder(x)
        output, state = self.rnn(x)
        if type(state) == tuple:
            state = state[0]
        state = state.permute(1, 0, 2).squeeze()
        return log_softmax(self.fc(state), dim=-1)
