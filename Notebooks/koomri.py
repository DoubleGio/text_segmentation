import gensim.downloader as gensim_api
import os
from koomri_model import create_model
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, Dataloader
from typing import List

from utils import ENWIKI_LOC, NLWIKI_LOCK, NLNEWS_LOC, NLAUVI_LOC, get_all_file_names

W2V_NL_PATH = '../word2vec-nl-combined-160.tar.gz' # Taken from https://github.com/clips/dutchembeddings

class BiLSTM:
    """
    """

    def __init__(
        self,
        language = 'en',
        dataset_path = ENWIKI_LOC,
        splits = [0.8, 0.1, 0.1],
        batch_size = 8,
        num_workers = 0,
    ) -> None:
        """
        """
        if language == 'en':
            word2vec = gensim_api.load('word2vec-google-news-300')
        elif language == 'nl':
            word2vec = KeyedVectors.load_word2vec_format(W2V_NL_PATH)
        else:
            raise ValueError(f"Language {language} not supported, try 'en' or 'nl' instead.")

        # Load the data and puts it into split DataLoaders.
        path_list = get_all_file_names(dataset_path)
        train_paths, test_paths = train_test_split(path_list, test_size=1-splits[0])
        dev_paths, test_paths = train_test_split(test_paths, test_size=splits[1]/(splits[1]+splits[2]))

        train_dataset = DatasetMap(train_paths)
        dev_dataset = DatasetMap(dev_paths)
        test_dataset = DatasetMap(test_paths)

        self.train_loader = Dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        self.dev_loader = Dataloader(dev_dataset, batch_size=batch_size, num_workers=num_workers)
        self.test_loader = Dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers)

        self.model = create_model()
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train #TODO: Implement training.
        


class DatasetMap(Dataset):
    """
    Custom Dataset class for use in Dataloaders.
    """
    def __init__(self, files: List[str]) -> None:
        """
        files: List of strings pointing towards files to read.
        """
        self.files = files

    def __getitem__(self, index: int) -> str:
        """
        Returns the file at index.
        """
        with open(self.files[index], 'r') as f:
            return f.read()
    
    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.files)

class Model(nn.Module):
    def __init__(self, sentence_encoder, hidden=128, num_layers=2):
        super(Model, self).__init__()

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

        self.criterion = nn.CrossEntropyLoss()


    def pad(self, s, max_length):
        s_length = s.size()[0]
        v = Variable(maybe_cuda(s.unsqueeze(0).unsqueeze(0)))
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
        batch_size = len(batch)

        sentences_per_doc = []
        all_batch_sentences = []
        for document in batch:
            all_batch_sentences.extend(document)
            sentences_per_doc.append(len(document))

        lengths = [s.size()[0] for s in all_batch_sentences]
        sort_order = np.argsort(lengths)[::-1]
        sorted_sentences = [all_batch_sentences[i] for i in sort_order]
        sorted_lengths = [s.size()[0] for s in sorted_sentences]

        max_length = max(lengths)
        logger.debug('Num sentences: %s, max sentence length: %s', 
                     sum(sentences_per_doc), max_length)

        padded_sentences = [self.pad(s, max_length) for s in sorted_sentences]
        big_tensor = torch.cat(padded_sentences, 1)  # (max_length, batch size, 300)
        packed_tensor = pack_padded_sequence(big_tensor, sorted_lengths)
        encoded_sentences = self.sentence_encoder(packed_tensor)
        unsort_order = Variable(maybe_cuda(torch.LongTensor(unsort(sort_order))))
        unsorted_encodings = encoded_sentences.index_select(0, unsort_order)

        index = 0
        encoded_documents = []
        for sentences_count in sentences_per_doc:
            end_index = index + sentences_count
            encoded_documents.append(unsorted_encodings[index : end_index, :])
            index = end_index

        doc_sizes = [doc.size()[0] for doc in encoded_documents]
        max_doc_size = np.max(doc_sizes)
        ordered_document_idx = np.argsort(doc_sizes)[::-1]
        ordered_doc_sizes = sorted(doc_sizes)[::-1]
        ordered_documents = [encoded_documents[idx] for idx in ordered_document_idx]
        padded_docs = [self.pad_document(d, max_doc_size) for d in ordered_documents]
        docs_tensor = torch.cat(padded_docs, 1)
        packed_docs = pack_padded_sequence(docs_tensor, ordered_doc_sizes)
        sentence_lstm_output, _ = self.sentence_lstm(packed_docs, zero_state(self, batch_size=batch_size))
        padded_x, _ = pad_packed_sequence(sentence_lstm_output)  # (max sentence len, batch, 256)

        doc_outputs = []
        for i, doc_len in enumerate(ordered_doc_sizes):
            doc_outputs.append(padded_x[0:doc_len - 1, i, :])  # -1 to remove last prediction

        unsorted_doc_outputs = [doc_outputs[i] for i in unsort(ordered_document_idx)]
        sentence_outputs = torch.cat(unsorted_doc_outputs, 0)

        x = self.h2s(sentence_outputs)
        return x