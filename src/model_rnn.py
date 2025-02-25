import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, device, n_layers=1, embedding_dimension=50, padding_idx=None):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dimension, padding_idx=padding_idx)
        self.rnn = nn.RNN(input_size=embedding_dimension,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        output, hidden = self.rnn(embedded)
        out = self.fc(hidden[-1])
        return out  # logits for each class

class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)

    def forward(self, input, hidden):
        new_hidden = torch.tanh(self.input2hidden(input) + self.hidden2hidden(hidden))
        return new_hidden

class CustomRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, padding_idx=None):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn_cell = CustomRNNCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        batch_size, seq_length, _ = embedded.size()
        hidden = torch.zeros(batch_size, self.hidden_size).to(inputs.device)
        for t in range(seq_length):
            hidden = self.rnn_cell(embedded[:, t, :], hidden)
        out = self.fc(hidden)
        return out
