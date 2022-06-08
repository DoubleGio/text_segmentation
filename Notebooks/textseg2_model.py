from typing import Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SentenceEncodingRNN2(nn.Module):
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
        self.mlp = nn.Sequential(nn.Linear(2*hidden, 2*hidden), nn.Tanh())
        self.context_vector = nn.Parameter(torch.Tensor(2*hidden))
        self.context_vector.data.normal_(0, 0.1) # init context vector with values drawn from a normal dist. with mean 0 and std 0.1

    def forward(self, x):
        batch_size = x.batch_sizes[0]
        s = zero_state(self, batch_size)
        packed_output, _ = self.lstm(x, s)
        padded_output, lengths = pad_packed_sequence(packed_output) # (max sentence len, batch, 256) 

        # attention
        padded_output = padded_output.permute(1, 0, 2)
        word_annotation = self.mlp(padded_output)
        attn_weight = word_annotation.matmul(self.context_vector)
        attended_outputs = torch.stack([F.softmax(attn_weight[i, :lengths[i]], dim=0).matmul(padded_output[i, :lengths[i]]) for i in range(len(lengths))], dim=0)

        return attended_outputs


class TS_Model2(nn.Module):
    """
    Model for Text Segmentation.
    """
    # criterion = nn.CrossEntropyLoss()

    def __init__(self, sentence_encoder: SentenceEncodingRNN2, hidden=128, num_layers=2):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.sentence_lstm = nn.LSTM(input_size=768+2*hidden, # The dimension of BERT embedding + output of BiLSTM hidden state
                                     hidden_size=hidden,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     dropout=0,
                                     bidirectional=True)

        self.sentence_lstm_2 = nn.LSTM(input_size=sentence_encoder.hidden * 4,
                                    hidden_size=hidden,
                                    num_layers=num_layers,
                                    batch_first=True,
                                    dropout=0,
                                    bidirectional=True)

        # We have two labels
        self.h2s = nn.Linear(hidden * 2, 2)
        self.num_layers = num_layers
        self.hidden = hidden

        self.criterion = nn.CrossEntropyLoss()
        module_fwd = [nn.Linear(hidden, hidden), nn.Tanh()]
        self.fwd = nn.Sequential(*module_fwd)

        modules = [nn.Linear(4*hidden+1, 1)]
        self.self_attn = nn.Sequential(*modules)

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

    def compute_sentence_pair_coherence(self, doc_out: torch.TensorType, pair_size=1):
        """
        Does something...
        FIXME: This isnt the σ(cos(e1, e2)) as described in the paper (in the final step)? What do we even do here?
        """
        pad_f = (0,0,pair_size,0); pad_b = (0,0,0,pair_size)
        forward_pair_emb = self.fwd(doc_out[:-1, :self.hidden] - F.pad(doc_out[:-1, :self.hidden], pad_f, 'constant', 0)[:-1,:])
        backward_pair_emb = self.fwd(doc_out[1:, self.hidden:] - F.pad(doc_out[1:, self.hidden:], pad_b, 'constant', 0)[1:,:]).permute(1,0)
        return torch.sigmoid(torch.diag(torch.mm(forward_pair_emb, backward_pair_emb)))

    def compute_sentences_similarity(self, X, sent_i, win_size):
        """
        FIXME: the calc of z is weird, unlike the paper.
        """
        X1 = X.unsqueeze(0)
        Y1 = X.unsqueeze(1)
        X2 = X1.repeat(X.shape[0],1,1)
        Y2 = Y1.repeat(1,X.shape[0],1)

        output = 0

        Z = torch.cat([X2,Y2],-1)
        if sent_i <= win_size:
            a = Z[sent_i,:,0:int(Z.size()[-1]/2)] 
            a_norm = a / a.norm(dim=1)[:, None]
            b = Z[sent_i,:,int(Z.size()[-1]/2):]
            b_norm = b / b.norm(dim=1)[:, None]
            z = torch.cat([Z[sent_i,:], torch.sigmoid(torch.diag(torch.mm(a_norm,b_norm.transpose(0,1)))).unsqueeze(-1)],-1)
            attn_weight = F.softmax(self.self_attn(z), dim=0).permute(1,0)

            output = attn_weight.matmul(Z[sent_i,:,0:int(Z.size()[-1]/2)])
        else:
            a = Z[win_size, :, 0:int(Z.size()[-1]/2)] 
            a_norm = a / a.norm(dim=1)[:, None]
            b = Z[win_size,:,int(Z.size()[-1]/2):]
            b_norm = b / b.norm(dim=1)[:, None]
            z = torch.cat([Z[win_size,:], torch.sigmoid(torch.diag(torch.mm(a_norm,b_norm.transpose(0,1)))).unsqueeze(-1)],-1)
            attn_weight = F.softmax(self.self_attn(z), dim=0).permute(1,0)

            output = attn_weight.matmul(Z[win_size,:,0:int(Z.size()[-1]/2)])

        return output

    def forward(
        self,
        batch: List[List[torch.TensorType]],
        bert_sents: List[torch.TensorType]
        ) -> Union[torch.TensorType, Tuple[torch.TensorType, torch.TensorType]]:
        """
        Forward pass of the model.
        Return:
            Tensor of the model outputs.
            Optionally another tensor of the sentence-pair similarity scores.
        """
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

        # Produce sentence embeddings and unsort
        encoded_sentences = self.sentence_encoder(packed_tensor)
        unsort_order = torch.LongTensor(inverse_order(sent_order)).to(device)
        unsorted_encodings = encoded_sentences.index_select(0, unsort_order)

        # Concatenate the bert embeddings with the att-biLSTM embeddings.
        bert_sents_cat = torch.cat(bert_sents)
        unsorted_encodings = torch.cat((unsorted_encodings, bert_sents_cat),1)

        # Seperate all sentence-encodings back into encoded documents
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

        # Produce biLSTM hidden states for all sentences for each document
        sentence_lstm_output, _ = self.sentence_lstm(packed_docs, zero_state(self, batch_size=len(batch)))
        padded_output, _ = pad_packed_sequence(sentence_lstm_output)  # (max doc len, batch_size, hidden_layers * 2)
        doc_outputs = [padded_output[0:doc_len, i, :] for i, doc_len in enumerate(ordered_doc_sizes)] # unpad and put into list

        # Sentence-Level Restricted Self-Attention
        win_size = 3
        doc_context_outputs = []
        for doc_len, doc_out in zip(ordered_doc_sizes, doc_outputs):
            local_context_embeddings = []
            for i in range(doc_len):
                if i-win_size < 0:
                    local_context_embeddings.append(self.compute_sentences_similarity(doc_out[0:i+win_size+1], i, win_size))
                else:
                    local_context_embeddings.append(self.compute_sentences_similarity(doc_out[i-win_size:i+win_size+1], i, win_size))

            doc_context_outputs.append(torch.stack(local_context_embeddings).squeeze(1))

        # Compute the consecutive sentence pairs' similarities.
        if self.training: # May be skipped since it is not used outside of loss calculation.
            sim_scores = [self.compute_sentence_pair_coherence(doc_out, pair_size=1) for doc_out in doc_outputs]
            unsorted_sim_scores = [sim_scores[i] for i in inverse_order(doc_order)]
            sim_scores = torch.cat(unsorted_sim_scores, 0).unsqueeze(1)

        # Concatenate the outputs and turn them into a single padded tensor
        doc_outputs = [torch.cat([doc_outputs[i], doc_context_outputs[i]], -1) for i in range(len(doc_outputs))]
        padded_docs = [self.pad_document(d, max_doc_length) for d in doc_outputs]
        docs_tensor = torch.cat(padded_docs, 1)
        packed_docs = pack_padded_sequence(docs_tensor, ordered_doc_sizes)

        # Put the outputs through the final biLSTM layer
        sentence_lstm_output, _ = self.sentence_lstm_2(packed_docs, zero_state(self, batch_size=len(batch))) #till now, no sentence is removed
        padded_output, _ = pad_packed_sequence(sentence_lstm_output)  # (max sentence len, batch, 256)

        # Seperate the documents again, but this time remove the final predictions per document (-1),
        # it's not needed since the predictions denote the END of segments.
        doc_outputs = [padded_output[0:doc_len - 1, i, :] for i, doc_len in enumerate(ordered_doc_sizes)] # unpad and put into list
        
        # Unsort and put through linear layer for outputs
        unsorted_doc_outputs = [doc_outputs[i] for i in inverse_order(doc_order)]
        sentence_outputs = torch.cat(unsorted_doc_outputs, 0)
        if self.training:
            return self.h2s(sentence_outputs), sim_scores
        else:
            return self.h2s(sentence_outputs)


def zero_state(module, batch_size):
    # * 2 is for the two directions
    return torch.zeros(module.num_layers * 2, batch_size, module.hidden).to(device), torch.zeros(module.num_layers * 2, batch_size, module.hidden).to(device)

def inverse_order(sort_order: np.ndarray) -> np.ndarray:
    """
    Inverses a given order of indices.
    """
    inv = np.empty_like(sort_order)
    inv[sort_order] = np.arange(sort_order.size)
    return inv

def supervised_cross_entropy(
    pred: torch.TensorType,
    sims: torch.TensorType,
    soft_targets: np.ndarray,
    target_coh_var: torch.TensorType,
    alpha=0.8
    ) -> torch.TensorType:
    """
    Computes the supervised cross entropy loss.
    """
    criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()
    loss_pred = criterion(pred, soft_targets)
    loss_sims = bce_criterion(sims, target_coh_var.unsqueeze(1).type(torch.cuda.FloatTensor))
    loss = alpha*loss_pred + (1-alpha)*loss_sims
    return loss

def create_model(use_cuda=True) -> TS_Model2:
    """Create a new TS_Model2 instance. Uses cuda if available, unless use_cuda=False."""
    if not use_cuda:
        global device
        device = torch.device("cpu")
    sentence_encoder = SentenceEncodingRNN2(input_size=300, hidden=256, num_layers=2)
    return TS_Model2(sentence_encoder, hidden=256, num_layers=2).to(device)