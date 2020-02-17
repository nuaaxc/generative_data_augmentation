import torch
import torch.nn as nn


class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim,
                 n_classes, hidden_dim, n_layers, dropout,
                 pretrained_weight=None):
        super(TextBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, scale_grad_by_freq=True, padding_idx=0)
        if pretrained_weight:
            self.embedding.from_pretrained(pretrained_weight)
        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            # dropout=dropout,
                            bidirectional=True,
                            batch_first=True,
                            bias=True)

        self.fc = nn.Linear(hidden_dim * 2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_len):
        x = self.embedding(x)
        x = self.dropout(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        out, _ = self.lstm(x)
        out, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = torch.mean(out, dim=1)
        logit = self.fc(out)

        return logit
