import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import numpy as np
from typing import Optional, List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentenceEncodingRNN(nn.Module):
    """
    Model for sentence encoding.
    """
    def __init__(self, input_size:int, hidden=256, num_layers=2):
        super().__init__()

        self.num_layers = num_layers
        self.hidden = hidden
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden,
                            num_layers=self.num_layers,
                            dropout=0,
                            bidirectional=True)

    def forward(self, x):
        batch_size = x.batch_sizes[0]
        packed_output, _ = self.lstm(x, zero_state(self, batch_size))
        padded_output, lengths = pad_packed_sequence(packed_output, batch_first=True) # (batch, max_sentence_length, hidden * 2) 

        maxes = torch.zeros(batch_size, padded_output.size(2), device=device)
        for i in range(batch_size):
            maxes[i, :] = torch.max(padded_output[i, :lengths[i], :], 0)[0]

        return maxes


class TS_Model(nn.Module):
    """
    Model for Text Segmentation.
    """
    criterion = nn.CrossEntropyLoss()

    def __init__(self, sentence_encoder: SentenceEncodingRNN, hidden=128, num_layers=2):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.sentence_lstm = nn.LSTM(input_size=sentence_encoder.hidden * 2,
                                     hidden_size=hidden,
                                     num_layers=num_layers,
                                     dropout=0,
                                     bidirectional=True)

        # We have two labels
        self.h2s = nn.Linear(hidden * 2, 2)
        self.num_layers = num_layers
        self.hidden = hidden

    def forward(self, sentences: PackedSequence) -> torch.Tensor:
        encoded_sentences = self.sentence_encoder(sentences)
        encoded_sentences, _ = self.sentence_lstm(encoded_sentences)
        res = self.h2s(encoded_sentences)
        return res


def zero_state(module, batch_size):
    # * 2 is for the two directions
    return torch.zeros(module.num_layers * 2, batch_size, module.hidden, device=device), torch.zeros(module.num_layers * 2, batch_size, module.hidden, device=device)

def create_model(input_size: int, use_cuda=True, set_device: Optional[torch.device] = None) -> TS_Model:
    """Create a new TS_Model instance. Uses cuda if available, unless use_cuda=False."""
    global device
    if set_device:
        device = set_device
    elif not use_cuda:
        device = torch.device("cpu")
    sentence_encoder = SentenceEncodingRNN(input_size=input_size, hidden=256, num_layers=2)
    return TS_Model(sentence_encoder, hidden=256, num_layers=2).to(device)