"""
2.1. Create a transformer model:
    >>> y_seg = Sigmoid(Linear2(TransformerΘ(S)))
    >>> Θ = {'nhead': 24, 'num_encoder_layers': 5, 'dim_feedforward': 1024}
2.2. Create a loss function:
    >>> loss_fn = binary cross entropy loss
'For the segmentation predictions, 70% of the inner sentences were randomly masked,
while all the begin sentences were not masked in order to address the imbalance class problem.'
basically, remove 70% of the non-boundary sentences for training.
"""
import math
from typing import Optional
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    Taken from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 150) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, 1, d_model)
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pos_encoding[:x.size(0)]
        return self.dropout(x)

class T2_Model(nn.Module):
    """Transformers² Model for Text Segmentation."""
    criterion = nn.CrossEntropyLoss()

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=24,
            dim_feedforward=1024,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=5,
        )
        self.out = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.Sigmoid(),
        )
        self.d_model = d_model

    def forward(self, sent_emb):
        sent_emb = sent_emb * math.sqrt(self.d_model) # scale the embedding (is done in the original paper for no clear reason)
        sent_emb = self.pos_encoder(sent_emb)
        sent_emb = self.transformer_encoder(sent_emb, mask=None, src_key_padding_mask=None) # TODO: implement masks
        sent_emb = self.out(sent_emb)
        return sent_emb

def create_T2_model(input_size: int, use_cuda=True, set_device: Optional[torch.device] = None) -> T2_Model:
    """Create a new T2_Model instance. Uses cuda if available, unless use_cuda=False."""
    global device
    if set_device:
        device = set_device
    elif not use_cuda:
        device = torch.device("cpu")
    return T2_Model().to(device)