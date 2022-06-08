import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def zero_state(module, batch_size):
    # * 2 is for the two directions
    return torch.zeros(module.num_layers * 2, batch_size, module.hidden).to(device), torch.zeros(module.num_layers * 2, batch_size, module.hidden).to(device)


class SentenceEncodingRNN(nn.Module):
    """
    Model for sentence encoding.
    """
    def __init__(self, input_size=300, hidden=256, num_layers=2):
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
        s = zero_state(self, batch_size)
        packed_output, _ = self.lstm(x, s)
        padded_output, lengths = pad_packed_sequence(packed_output) # (max sentence len, batch, 256) 

        maxes = torch.zeros(batch_size, padded_output.size(2)).to(device)
        for i in range(batch_size):
            maxes[i, :] = torch.max(padded_output[:lengths[i], i, :], 0)[0]

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
                                     batch_first=True,
                                     dropout=0,
                                     bidirectional=True)

        # We have two labels
        self.h2s = nn.Linear(hidden * 2, 2)
        self.num_layers = num_layers
        self.hidden = hidden

    def pad(self, s, max_length):
        s_length = s.size()[0]
        v = s.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0, 0, max_length - s_length))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)


    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0,0, max_document_length - d_length ))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def forward(self, batch):
        doc_lengths = [] # number of sentences in each document
        all_sentences = [] # list of all sentences for all documents
        for document in batch:
            doc_lengths.append(len(document))
            all_sentences += document
        doc_lengths = np.array(doc_lengths)

        # Sort the sentences on length from big to small
        sent_lengths = np.array([s.size()[0] for s in all_sentences]) # number of words in each sentence
        sent_order = np.argsort(sent_lengths)[::-1]
        sorted_sentences = [all_sentences[i] for i in sent_order]
        sorted_lengths = sent_lengths[sent_order]

        # Pad and concatenate the sentences
        max_sent_length = sorted_lengths[0]
        padded_sentences = [self.pad(s, max_sent_length) for s in sorted_sentences]
        big_tensor = torch.cat(padded_sentences, 1)  # (max_sent_length, batch_size, 300)
        packed_tensor = pack_padded_sequence(big_tensor, sorted_lengths)

        # Produce the sentences embeddings and unsort
        encoded_sentences = self.sentence_encoder(packed_tensor)
        unsort_order = torch.LongTensor(inverse_order(sent_order)).to(device)
        unsorted_encodings = encoded_sentences.index_select(0, unsort_order)

        # Seperate sentences back into encoded documents
        index = 0
        encoded_documents = []
        for sentences_count in doc_lengths:
            end_index = index + sentences_count
            encoded_documents.append(unsorted_encodings[index : end_index, :])
            index = end_index

        # Sort the documents
        doc_order = np.argsort(doc_lengths)[::-1]
        ordered_doc_sizes = doc_lengths[doc_order]
        max_doc_length = ordered_doc_sizes[0]
        ordered_documents = [encoded_documents[i] for i in doc_order]

        # Pad and concatenate the documents
        padded_docs = [self.pad_document(d, max_doc_length) for d in ordered_documents]
        docs_tensor = torch.cat(padded_docs, 1)
        packed_docs = pack_padded_sequence(docs_tensor, ordered_doc_sizes)

        # Encode each document
        sentence_lstm_output, _ = self.sentence_lstm(packed_docs, zero_state(self, batch_size=len(batch)))
        padded_output, _ = pad_packed_sequence(sentence_lstm_output)  # (max sentence len, batch, 256)

        # Seperate the documents again, but this time remove the final predictions per document (-1),
        # it's not needed since the predictions denote the END of segments.
        doc_outputs = [padded_output[0:doc_len - 1, i, :] for i, doc_len in enumerate(ordered_doc_sizes)] # unpad and put into list
        
        # Unsort and put through linear layer for outputs
        unsorted_doc_outputs = [doc_outputs[i] for i in inverse_order(doc_order)]
        sentence_outputs = torch.cat(unsorted_doc_outputs, 0)
        return self.h2s(sentence_outputs)

def inverse_order(sort_order: np.ndarray) -> np.ndarray:
    """
    Inverses a given order of indices to use for unsorting.
    """
    inv = np.empty_like(sort_order)
    inv[sort_order] = np.arange(sort_order.size)
    return inv

def create_model(use_cuda=True) -> TS_Model:
    """Create a new TS_Model instance. Uses cuda if available, unless use_cuda=False."""
    if not use_cuda:
        global device
        device = torch.device("cpu")
    sentence_encoder = SentenceEncodingRNN(input_size=300, hidden=256, num_layers=2)
    return TS_Model(sentence_encoder, hidden=256, num_layers=2).to(device)