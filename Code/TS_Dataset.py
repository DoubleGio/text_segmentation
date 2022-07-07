import os, torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from gensim.models import KeyedVectors
from typing import Optional, List, Tuple
from utils import sent_tokenize_plus, sectioned_clean_text, clean_text, word_tokenize
rng = np.random.default_rng()

class TS_Dataset(Dataset):
    """
    Custom Text Segmentation Dataset class for use in Dataloaders.
    word2vec: KeyedVectors object containing the word2vec model (set by TextSeg class).
    """
    MAX_SECTION_LEN = 70 # Maximum number of sentences in a section.
    word2vec: Optional[KeyedVectors] = None

    def __init__(
        self,
        texts: np.ndarray,
        from_wiki=False,
        labeled=True,
    ) -> None:
        """
        texts: List of strings or paths pointing towards files to read.
        from_wiki: Boolean indicating whether the dataset follows Wikipedia formatting and requires some extra preprocessing.
            # NOTE: The paper says: '... at training time, we removed from each document the first segment, 
            since in Wikipedia it is often a summary that touches many different top- ics, and is therefore less useful for training a seg- mentation model.'
            This is not found in the original code however and is not implemented here.
        labeled: Boolean indicating whether the dataset is labeled or not.
        """
        self.texts = texts
        self.are_paths = True if os.path.exists(self.texts[0]) else False
        self.from_wiki = from_wiki
        self.labeled = labeled

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, np.ndarray]:
        """
        For a document at index, returns:
            data: List of n sentence embeddings (made out of word vectors).

            targets: List of n target labels [0 or 1]; a 1 signifies the END of a section (hence the last sentence/label is discarded later).
             OR
            raw_text: List containing the text of the sentences of the document.
        """
        if self.labeled:
            return self.get_data_targets(index)
        else:
            return self.get_data_raw(index)

    def get_data_targets(self, index: int) -> Tuple[torch.FloatTensor, np.ndarray]:
        """Get the data and targets for a document at index."""
        text = self.get_text(index)
        data = []
        targets = np.array([], dtype=int)

        sections = sectioned_clean_text(text, from_wiki=self.from_wiki)
        suitable_sections_count = 0
        for section in sections:
            sentences = sent_tokenize_plus(section)
            if not (self.MAX_SECTION_LEN > len(sentences) > 1): # Filter too short or too long sections.
                if len(sections) <= 2: # Skip docs that end up with just a single section
                    break
                continue
            
            section_len = 0
            for sentence in sentences:
                sentence_words = word_tokenize(sentence)
                if len(sentence_words) > 0:
                    sentence_emb = self.embed_words(sentence_words)
                    if len(sentence_emb) > 0:
                        section_len += 1
                        data.append(sentence_emb)
                    
            if section_len > 0:
                sentence_labels = np.zeros(section_len, dtype=int)
                sentence_labels[-1] = 1 # Last sentence ends a section --> 1.
                targets = np.append(targets, sentence_labels)
                suitable_sections_count += 1

        # Get a random document if the current one is empty or has less than two suitable sections.
        if len(data) == 0 or suitable_sections_count < 2:
            return self.__getitem__(rng.integers(0, len(self.texts)))

        data = pad_sequence(data, batch_first=True) # (num_sents, max_sent_len, embed_dim)
        return data, targets

    def get_data_raw(self, index: int) -> Tuple[torch.FloatTensor, np.ndarray]:
        """Get the data and raw text for a document at index."""
        text = self.get_text(index)
        data = []
        raw_text = np.array([])

        text = clean_text(text, from_wiki=self.from_wiki)
        sentences = sent_tokenize_plus(text)
        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            if len(sentence_words) > 0:
                sentence_emb = self.embed_words(sentence_words)
                if len(sentence_emb) > 0:
                    data.append(sentence_emb)
                    raw_text = np.append(raw_text, sentence)

        if len(data) == 0:
            return None, None
        data = pad_sequence(data, batch_first=True) # (num_sents, max_sent_len, embed_dim)
        return data, raw_text
        
    def __len__(self) -> int:
        return len(self.texts)

    def get_text(self, index: int) -> str:
        if self.are_paths:
            with open(self.texts[index], 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = self.texts[index]
        return text

    def embed_words(self, words: List[str]) -> torch.FloatTensor:
        res = []
        for word in words:
            if self.word2vec:
                if word in self.word2vec:
                    res.append(self.word2vec[word])
                else:
                    # return TS_Dataset.word2vec['UNK']
                    continue # skip words not in the word2vec model
            else:
                res.append(rng.standard_normal(size=300))
        return torch.FloatTensor(np.stack(res)) if res else []


class TS_Dataset2(TS_Dataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    # Override
    def get_data_targets(self, index: int) -> Tuple[torch.FloatTensor, np.ndarray, torch.FloatTensor]:
        """Get the data, targets and bert embeddings for a document at index."""
        text = self.get_text(index)
        data = []
        targets = np.array([], dtype=int)
        bert_mask = np.array([], dtype=bool) # Keeps track of which sentences are kept

        sections = sectioned_clean_text(text, from_wiki=self.from_wiki)
        suitable_sections_count = 0
        for section in sections:
            sentences = sent_tokenize_plus(section)
            if not (self.MAX_SECTION_LEN > len(sentences) > 1): # Filter too short or too long sections.
                if len(sections) <= 2: # Skip docs that end up with just a single section
                    break
                bert_mask = np.append(bert_mask, np.zeros(len(sentences), dtype=bool))
                continue
            
            section_len = 0
            for sentence in sentences:
                sentence_words = word_tokenize(sentence)
                if len(sentence_words) > 0:
                    sentence_emb = self.embed_words(sentence_words)
                    if len(sentence_emb) > 0:
                        section_len += 1
                        data.append(sentence_emb)
                        bert_mask = np.append(bert_mask, True)
                        continue
                bert_mask = np.append(bert_mask, False)
                    
            if section_len > 0:
                sentence_labels = np.zeros(section_len, dtype=int)
                sentence_labels[-1] = 1 # Last sentence ends a section --> 1.
                targets = np.append(targets, sentence_labels)
                suitable_sections_count += 1

        # Get a random document if the current one is empty or has less than two suitable sections.
        if len(data) == 0 or suitable_sections_count < 2:
            return self.__getitem__(rng.integers(0, len(self.texts)))

        bert_data = torch.load(self.texts[index].replace('data', 'data_bert') + '.pt')
        data = pad_sequence(data, batch_first=True) # (num_sents, max_sent_len, embed_dim)

        return data, targets, bert_data[bert_mask, :]

    def get_data_raw(self, index: int) -> Tuple[torch.FloatTensor, np.ndarray, torch.FloatTensor]:
        """Get the data, raw text and bert embeddings for a document at index."""
        text = self.get_text(index)
        data = []
        raw_text = np.array([])
        bert_mask = np.array([], dtype=bool) # Keeps track of which sentences are kept

        text = clean_text(text, from_wiki=self.from_wiki)
        sentences = sent_tokenize_plus(text)
        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            if len(sentence_words) > 0:
                sentence_emb = self.embed_words(sentence_words)
                if len(sentence_emb) > 0:
                    data.append(sentence_emb)
                    raw_text = np.append(raw_text, sentence)
                    bert_mask = np.append(bert_mask, True)
                    continue
            bert_mask = np.append(bert_mask, False)
            
        if len(data) == 0:
            return None, None
        
        bert_data = torch.load(self.texts[index].replace('data', 'data_bert') + '.pt')
        data = pad_sequence(data, batch_first=True) # (num_sents, max_sent_len, embed_dim)
        return data, raw_text, bert_data[bert_mask, :]
