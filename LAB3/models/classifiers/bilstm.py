import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module): 
    '''
    Encoder: Embedding + 5 lớp BiLSTM hai chiều (hidden 256).
    Lớp Linear cuối dự đoán nhãn cho từng token.
    '''
    def __init__(
            self, 
            vocab_size: int, 
            hidden_size: int, 
            n_layer: int, 
            n_labels: int,
            padding_idx: int = 0,
            dropout: float = 0.2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,          # số dòng = số từ trong từ điển
            embedding_dim=hidden_size,          # mỗi từ -> vector 256 chiều 
            padding_idx=padding_idx
        )

        self.bilstm = nn.LSTM(
            input_size=hidden_size,             # đầu vào mỗi bước: vector 256 chiều (từ embedding)
            hidden_size=hidden_size,
            batch_first=True,                   # (bs, len)
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0.0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            in_features=hidden_size * 2, # nhân 2 vì bidirectional
            out_features=n_labels
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        embedded_feats = self.embedding(inputs)         # (bs, len, hidden)
        packed = pack_padded_sequence(embedded_feats, lengths.cpu(), batch_first=True, enforce_sorted=False)

        out, _ = self.bilstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=inputs.size(1))
        return self.classifier(self.dropout(out))   # (bs, len, n_labels)

