import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    Taken from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 300) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(1, max_len, d_model)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pos_encoding[:, :x.size(1), :]
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
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=5,
        )

        self.out = nn.Linear(d_model, 2)
        self.d_model = d_model

    def forward(self, sent_emb, doc_lengths, targets: Optional[torch.TensorType] = None) -> torch.TensorType:
        sent_emb = sent_emb * math.sqrt(self.d_model) # scale the embedding (is done in the 'Attention is all you need' paper for no clear reason)
        sent_emb, key_mask = self.batchify(sent_emb, doc_lengths, targets) # (sentences, d_model) -> (batch_size, max_sentence_length, d_model)
        sent_emb = self.pos_encoder(sent_emb)

        # TODO: Mask out 70% from the non-boundary sentences.
        sent_emb = self.transformer_encoder(sent_emb, mask=self.generate_square_subsequent_mask(sent_emb.shape[1]), src_key_padding_mask=key_mask)
        # sent_emb = self.transformer_encoder(sent_emb, mask=None, src_key_padding_mask=key_mask)

        sent_emb = self.out(sent_emb)
        return self.unbatchify(sent_emb, doc_lengths)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, diag=1) -> torch.TensorType:
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=diag)

    @staticmethod
    def batchify(data: torch.TensorType, doc_lengths: torch.TensorType, targets: Optional[torch.TensorType] = None) -> Tuple[torch.TensorType, torch.TensorType]:
        """
        Create padded batches of the data and a mask for the padded positions.
        Additionally, may mask out 70% of the non-boundary sentences if targets is provided.
        
        TODO: Mask out 70% from the non-boundary sentences correctly. Useful links:
        https://discuss.pytorch.org/t/how-to-add-padding-mask-to-nn-transformerencoder-module/63390/3
        https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask
        
        Args:
            data: Tensor, shape [num sents, embedding_dim]
            doc_lengths: Tensor, shape [batch_size]
        Returns:
            Tensor, shape [batch_size, num_sents, embedding_dim]
        """
        max_len = torch.max(doc_lengths)
        docs = []
        masks = []
        cur_i = 0
        for doc_len in doc_lengths:
            # Pad the sequences to max_len --- pad=(left, right, top, bottom)
            docs.append(nn.functional.pad(data[cur_i: cur_i + doc_len], pad=(0, 0, 0, max_len - doc_len)))

            # FIXME: This mask does not improve things.
            if targets is not None:
                m = (torch.rand(doc_len, device=device) > 0.3) * ~targets[cur_i: cur_i + doc_len].bool() # Mask out 70% of the non-boundary sentences (?)
                m = torch.cat([m, torch.ones(max_len - doc_len, device=device, dtype=bool)], dim=0)

            # Basic mask for the padded positions.
            else:
                m = torch.cat([torch.zeros(doc_len, device=device, dtype=bool), torch.ones(max_len - doc_len, device=device, dtype=bool)])

            masks.append(m)
            cur_i += doc_len
        return torch.stack(docs), torch.stack(masks)

    @staticmethod
    def unbatchify(data: torch.TensorType, doc_lengths: torch.TensorType) -> torch.TensorType:
        """
        Args:
            data: Tensor, shape [batch_size, num_sents, embedding_dim]
            doc_lengths: Tensor, shape [batch_size]
        Returns:
            Tensor, shape [num_sents, embedding_dim]
        """
        docs = []
        for i, doc_len in enumerate(doc_lengths):
            docs.append(data[i, :doc_len])
        return torch.cat(docs, dim=0)


def create_T2_model(input_size: int, use_cuda=True, set_device: Optional[torch.device] = None) -> T2_Model:
    """Create a new T2_Model instance. Uses cuda if available, unless use_cuda=False."""
    global device
    if set_device:
        device = set_device
    elif not use_cuda:
        device = torch.device("cpu")
    return T2_Model(input_size).to(device)