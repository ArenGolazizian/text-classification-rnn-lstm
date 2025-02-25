import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, device, bidirectional=False, n_layers=1, embedding_dimension=50, padding_idx=None):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dimension, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=embedding_dimension,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, inputs):
        h0 = torch.zeros(self.n_layers * self.num_directions, inputs.size(0), self.hidden_size).to(inputs.device)
        c0 = torch.zeros(self.n_layers * self.num_directions, inputs.size(0), self.hidden_size).to(inputs.device)
        embedded = self.embedding(inputs)
        lstm_out, (hn, cn) = self.lstm(embedded, (h0, c0))
        if self.bidirectional:
            hn = hn.view(self.n_layers, self.num_directions, inputs.size(0), self.hidden_size)
            hn_forward = hn[-1, 0, :, :]
            hn_backward = hn[-1, 1, :, :]
            hn_combined = torch.cat((hn_forward, hn_backward), dim=1)
            out = self.fc(hn_combined)
        else:
            out = self.fc(hn[-1])
        return out

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden, cell_state):
        combined = torch.cat((input, hidden), dim=1)
        i = torch.sigmoid(self.input_gate(combined))
        f = torch.sigmoid(self.forget_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        g = torch.tanh(self.cell_gate(combined))
        cell_state = f * cell_state + i * g
        hidden = o * torch.tanh(cell_state)
        return hidden, cell_state

class CustomLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, padding_idx=None):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm_cell = CustomLSTMCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        hidden = torch.zeros(batch_size, self.hidden_size).to(inputs.device)
        cell_state = torch.zeros(batch_size, self.hidden_size).to(inputs.device)
        embedded = self.embedding(inputs)
        batch_size, seq_length, _ = embedded.size()
        for t in range(seq_length):
            hidden, cell_state = self.lstm_cell(embedded[:, t, :], hidden, cell_state)
        out = self.fc(hidden)
        return out