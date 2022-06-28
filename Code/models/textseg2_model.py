from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, PackedSequence
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentenceEncodingRNN2(nn.Module):
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
        self.mlp = nn.Sequential(nn.Linear(2*hidden, 2*hidden), nn.Tanh())
        self.context_vector = nn.Parameter(torch.Tensor(2*hidden))
        self.context_vector.data.normal_(0, 0.1) # init context vector with values drawn from a normal dist. with mean 0 and std 0.1

    def forward(self, x):
        batch_size = x.batch_sizes[0]
        packed_output, _ = self.lstm(x, zero_state(self, batch_size))
        padded_output, lengths = pad_packed_sequence(packed_output, batch_first=True) # (batch, max sentence len, hidden * 2) 

        # attention
        word_annotation = self.mlp(padded_output)
        attn_weight = word_annotation.matmul(self.context_vector)
        attended_outputs = torch.stack([F.softmax(attn_weight[i, :lengths[i]], dim=0).matmul(padded_output[i, :lengths[i]]) for i in range(len(lengths))], dim=0)

        return attended_outputs


class TS2_Model(nn.Module):
    """Model for Text Segmentation."""
    def __init__(self, sentence_encoder: SentenceEncodingRNN2, hidden=128, num_layers=2):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        # input size = hidden * 2 + 768
        self.sentence_lstm = nn.LSTM(input_size=sentence_encoder.hidden * 2 + 768, # Output of BiLSTM hidden state + the dimension of BERT embedding.
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

        return output.squeeze(0)

    def forward(
        self,
        sentences: PackedSequence,
        bert_sents: torch.FloatTensor,
        doc_lengths: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Forward pass of the model.
        """
        encoded_sentences = self.sentence_encoder(sentences)
        encoded_sentences = torch.cat((encoded_sentences, bert_sents), 1) # Add the BERT embeddings to the newly made ones.
        encoded_sentences, _ = self.sentence_lstm(encoded_sentences) # Sentence hidden states.

        # Sentence-Level Restricted Self-Attention
        # TODO: Implement a batched version of this.
        win_size = 3
        doc_outputs = []
        sim_scores = []
        doc_start_index = 0
        for doc_len in doc_lengths:
            doc_out = encoded_sentences[doc_start_index:doc_start_index+doc_len, :]
            doc_context_embedding = []
            for i in range(doc_len):
                doc_context_embedding.append(
                    self.compute_sentences_similarity(
                        doc_out[
                            max(0, i - win_size):min(i + win_size + 1, doc_len)
                        ], i, win_size
                    )
                )
            # Compute the consecutive sentence pairs' similarities.
            sim_scores.append(self.compute_sentence_pair_coherence(doc_out, pair_size=1))
    
            # context_embeddings.append(torch.stack(doc_context_embedding))
            doc_outputs.append(torch.cat([doc_out, torch.stack(doc_context_embedding)], -1))
            doc_start_index += doc_len

        packed_docs = pack_sequence(doc_outputs, enforce_sorted=False)
        del doc_outputs
        torch.cuda.empty_cache()
        encoded_sentences, _ = self.sentence_lstm_2(packed_docs) # Final sentence hidden states.
        encoded_sentences, _ = pad_packed_sequence(encoded_sentences, batch_first=True) # (batch, max sentence len, 512) 
        encoded_sentences = [encoded_sentences[i, :doc_lengths[i], :] for i in range(len(doc_lengths))]
        encoded_sentences = torch.cat(encoded_sentences, 0) # (sentences, 512)

        return self.h2s(encoded_sentences), torch.cat(sim_scores, 0).unsqueeze(1)


def zero_state(module, batch_size):
    # * 2 is for the two directions
    return torch.zeros(module.num_layers * 2, batch_size, module.hidden, device=device), torch.zeros(module.num_layers * 2, batch_size, module.hidden, device=device)

def supervised_cross_entropy(
    pred: torch.TensorType,
    sims: torch.TensorType,
    soft_targets: torch.LongTensor,
    target_coh_var: Optional[torch.LongTensor] = None,
    alpha=0.8
) -> torch.TensorType:
    """
    Computes the supervised cross entropy loss.
    """
    if target_coh_var is None:
        target_coh_var = 1 - soft_targets
    criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()
    loss_pred = criterion(pred, soft_targets)
    loss_sims = bce_criterion(sims, target_coh_var.unsqueeze(1).type(torch.cuda.FloatTensor))
    loss = alpha*loss_pred + (1-alpha)*loss_sims
    return loss

def create_TS2_model(input_size: int, use_cuda=True, set_device: Optional[torch.device] = None) -> TS2_Model:
    """Create a new TS2_Model instance. Uses cuda if available, unless use_cuda=False."""
    global device
    if set_device:
        device = set_device
    elif not use_cuda:
        device = torch.device("cpu")
    sentence_encoder = SentenceEncodingRNN2(input_size=input_size, hidden=256, num_layers=2)
    return TS2_Model(sentence_encoder, hidden=256, num_layers=2).to(device)