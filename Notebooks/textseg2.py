import logging, os
from textseg2_model import create_model2, supervised_cross_entropy
from textseg import TextSeg
import gensim.downloader as gensim_api
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from nltk import regexp_tokenize, sent_tokenize
import numpy as np
from transformers import logging as tlogging
from transformers import BertTokenizer, BertModel
rng = np.random.default_rng()
from utils import ENWIKI_LOC, NLWIKI_LOC, NLNEWS_LOC, NLAUVI_LOC_C, NLAUVI_LOC_N, get_all_file_names, sectioned_clean_text, compute_metrics, LoggingHandler

W2V_NL_PATH = '../Datasets/word2vec-nl-combined-160.tar.gz' # Taken from https://github.com/clips/dutchembeddings
CHECKPOINT_BASE = 'checkpoints/textseg2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(LoggingHandler(use_tqdm=True))
tlogging.set_verbosity_error()


class TextSeg2(TextSeg):
    """
    Updated implementation of https://github.com/koomri/text-segmentation.
    Extended with the enhancements from https://github.com/lxing532/improve_topic_seg.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # Override
    def load_data(
        self,
        language: str,
        dataset_path: str,
        from_wiki: bool,
        splits: List[float],
        batch_size: int,
        num_workers: int,
        ) -> None:
        """
        Load the right pretrained models.
        Shuffle and split the data, load it as TS_Datasets and put it into DataLoaders.
        """
        if language == 'en':
            word2vec = gensim_api.load('word2vec-google-news-300')
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased')
        elif language == 'nl':
            word2vec = KeyedVectors.load_word2vec_format(W2V_NL_PATH)
            bert_tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
            bert_model = BertModel.from_pretrained('GroNLP/bert-base-dutch-cased')
        elif language == 'test':
            word2vec = None
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased')
        else:
            raise ValueError(f"Language {language} not supported, try 'en' or 'nl' instead.")

        path_list = get_all_file_names(dataset_path)
        train_paths, test_paths = train_test_split(path_list, test_size=1-splits[0])
        dev_paths, test_paths = train_test_split(test_paths, test_size=splits[1]/(splits[1]+splits[2]))

        train_dataset = TS_Dataset2(files=train_paths, word2vec=word2vec, bert_tokenizer=bert_tokenizer, bert_model=bert_model, from_wiki=from_wiki)
        val_dataset = TS_Dataset2(files=dev_paths, word2vec=word2vec, bert_tokenizer=bert_tokenizer, bert_model=bert_model, from_wiki=from_wiki)
        test_dataset = TS_Dataset2(files=test_paths, word2vec=word2vec, bert_tokenizer=bert_tokenizer, bert_model=bert_model, from_wiki=from_wiki)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers)
        logger.info(f"Loaded {len(train_dataset)} training examples, {len(val_dataset)} validation examples, and {len(test_dataset)} test examples.")

    # Override
    def create_model(self):
        if self.load_from:
            model = torch.load(self.load_from).to(device)
            logger.info(f"Loaded TS_Model2 from {self.load_from}.")
        else:
            model = create_model2()
            logger.info(f"Created new TS_Model2.")
        return model

    # Override
    def train(self, model, optimizer) -> None:
        model.train()
        total_loss = 0.0
        with tqdm(desc='Training', total=len(self.train_loader)*self.train_loader.batch_size, leave=False) as pbar:
            for data, labels, bert_sents in self.train_loader:
                model.zero_grad()
                output, sim_scores = model(data, bert_sents)
                target_var = torch.cat(labels, dim=0).to(device)

                # Generate the ground truth for coherence scores...
                target_list = target_var.cpu().detach().numpy() # convert to numpy array
                target_coh = []
                for t in target_list:
                    if t == 0:
                        target_coh.append(1)
                    else:
                        target_coh.append(0)
                target_coh = torch.LongTensor(target_coh).to(device)

                loss = supervised_cross_entropy(output, sim_scores, target_var, target_coh)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pbar.set_postfix(loss=loss.item())
                pbar.update(len(data))
        avg_loss = total_loss / len(self.train_loader) # Average loss per input.
        logger.info(f"Training Epoch {self.current_epoch + 1} --- Loss = {avg_loss:.4}")
        writer.add_scalar('Loss/Train', avg_loss, self.current_epoch + 1)

    # Override
    def validate(self, model) -> Tuple[float, float, float]:
        model.eval()
        # cm = np.zeros((2, 2), dtype=int)
        thresholds = np.arange(0, 1, 0.05)
        scores = {k: [] for k in thresholds} # (pk, windowdiff) scores for each threshold.

        with tqdm(desc="Validating", total=len(self.val_loader)*self.val_loader.batch_size, leave=False) as pbar:
            for data, labels, bert_sents in self.val_loader:
                output = model(data, bert_sents)
                output_softmax = F.softmax(output, dim=1)
                output_softmax = output_softmax.cpu().detach().numpy() # convert to numpy array.

                # Calculate the Pk and windowdiff per document (in this batch) and append to scores for each threshold.
                doc_start_idx = 0
                for doc_labels in labels:
                    for threshold in thresholds:
                        doc_prediction = (output_softmax[doc_start_idx:doc_start_idx+len(doc_labels)][:, 1] > threshold).astype(int)
                        pk, wd = compute_metrics(doc_prediction, doc_labels)
                        scores[threshold].append((pk, wd))
                    doc_start_idx += len(doc_labels)
                pbar.update(len(data))
        # tn, fp, fn, tp = cm.ravel()
        epoch_best = [np.inf, np.inf] # (pk, windowdiff)
        best_threshold: float = None
        for threshold in thresholds:
            avg_pk_wd = np.mean(scores[threshold], axis=0) # (pk, windowdiff) average for this threshold, over all docs.
            if sum(avg_pk_wd) < sum(epoch_best):
                epoch_best = avg_pk_wd
                best_threshold = threshold
        logger.info(f"Validation Epoch {self.current_epoch + 1} --- For threshold = {best_threshold:.4}, Pk = {epoch_best[0]:.4}, windowdiff = {epoch_best[1]:.4}")
        return epoch_best[0], epoch_best[1], best_threshold

    # Override
    def test(self, model, threshold: float) -> Tuple[float, float]:
        model.eval()
        scores = []
        with tqdm(desc='Testing', total=len(self.test_loader)*self.test_loader.batch_size, leave=False) as pbar:
            for data, labels, bert_sents in self.test_loader:
                output = model(data, bert_sents)
                output_softmax = F.softmax(output, dim=1)
                output_softmax = output_softmax.cpu().detach().numpy() # convert to numpy array.

                # Calculate the Pk and windowdiff per document (in this batch) for the specified threshold.
                doc_start_idx = 0
                for doc_labels in labels:
                    doc_prediction = (output_softmax[doc_start_idx:doc_start_idx+len(doc_labels)][:, 1] > threshold).astype(int)
                    pk, wd = compute_metrics(doc_prediction, doc_labels)
                    scores.append((pk, wd))
                    doc_start_idx += len(doc_labels)
                pbar.update(len(data))
        epoch_pk, epoch_wd = np.mean(scores, axis=0) # (pk, windowdiff) average for this threshold, over all docs.
        logger.info(f"Testing Epoch {self.current_epoch + 1} --- For threshold = {threshold:.4}, Pk = {epoch_pk:.4}, windowdiff = {epoch_wd:.4}")
        return epoch_pk, epoch_wd


def custom_collate(batch):
    """
    Prepare the batch for the model (so we can use data of varying lengths).
    Follows original implementation.
    https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    """
    b_data = []
    b_targets = []
    b_bert_sents = []
    for data, targets, bert_sents in batch:
        tensored_data = []
        for sentence in data:
            tensored_data.append(torch.FloatTensor(np.concatenate(sentence)).to(device))
        b_data.append(tensored_data)
        tensored_targets = torch.LongTensor(targets).to(device)
        b_targets.append(tensored_targets)
        b_bert_sents.append(bert_sents)
    return b_data, b_targets, b_bert_sents

class TS_Dataset2(Dataset):
    """
    Custom Text Segmentation Dataset class for use in Dataloaders.
    Includes BERT sentence embeddings. TODO: Precalculate embeddings?
    """
    def __init__(self, files: List[str], word2vec: KeyedVectors, bert_tokenizer: BertTokenizer, bert_model: BertModel, from_wiki=False) -> None:
        """
        files: List of strings pointing towards files to read.
        word2vec: KeyedVectors object containing the word2vec model.
        from_wiki: Boolean indicating whether the dataset follows Wikipedia formatting and requires some extra preprocessing.
        """
        self.files = files
        self.word2vec = word2vec
        self.from_wiki = from_wiki
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model

    def __getitem__(self, index: int) -> Tuple[List, List[int], torch.FloatTensor]:
        """
        For a document at index, with n sentences, returns:
            data: List of n sentence representations (made out of word vectors).
            labels: List of n-1 labels [0 or 1]; a 1 signifies the END of a section (hence the last label is discarded).
            bert_sents: Tensor of BERT sentence embeddings with shape (n, 768).
        """
        with open(self.files[index], 'r') as f:
            text = f.read()
        sections = sectioned_clean_text(text)
        data = []
        labels = []
        bert_sents = []

        for i, section in enumerate(sections):
            try:
                sentences = sent_tokenize(section)
                sentence_labels = np.zeros(len(sentences), dtype=int)
                for sentence in sentences:
                    sentence_words = self.word_tokenize(sentence)
                    if len(sentence_words) > 0:
                        sentence_representation = [self.model_word(word) for word in sentence_words]
                        data.append(sentence_representation)
                    else:
                        sentence_labels = sentence_labels[:-1]
                sentence_labels[-1] = 1 # Last sentence ends a section --> 1.
                labels += sentence_labels.tolist()
                bert_sents.append(self.bert_embed(sentences))
            except ValueError:
                logging.warning(f"Section {i + 1} in {self.files[index]} is empty.")
            # if len(data) > 0:
                # labels.append(len(data) - 1) # NOTE: this points towards the END of a segment.

        return data, labels[:-1], torch.cat(bert_sents)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.files)

    def bert_embed(self, sentences: Union[str, List[str]], batch_size=8) -> torch.FloatTensor:
        """
        Returns the BERT embedding(s) of a sentence/list of sentences.
        This follows the same procedure as the bert-as-service module (which was originally used).
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        bert_sents = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:min(i+batch_size, len(sentences))]
            tokenized = self.bert_tokenizer(batch, padding=True, return_tensors='pt') # Pad to max seq length in batch.
            # Hidden layers have a shape of (batch_size, max_seq_len, 768) - only the second-to-last layer is used.
            output_layer = self.bert_model(**tokenized, output_hidden_states=True).hidden_states[-2]
            avg_pool = torch.nn.AvgPool2d(kernel_size=(output_layer.shape[1], 1)) # Take the average of the layer on the time (sequence) axis.
            bert_sents.append(avg_pool(output_layer).squeeze(dim=1).detach())

        return torch.cat(bert_sents) # shape = (len(sentences), 768)

    def word_tokenize(self, sentence: str) -> List[str]:
        """
        sentence: String to tokenize.
        wiki_remove: Whether to remove special wiki tokens (e.g. '***LIST***').
        """
        if self.from_wiki:
            for token in ["***LIST***", "***formula***", "***codice***"]:
                sentence = sentence.replace(token, "")
        return regexp_tokenize(sentence, pattern=r'[\wÀ-ÖØ-öø-ÿ\-\']+')

    def model_word(self, word):
        if self.word2vec:
            if word in self.word2vec:
                return self.word2vec[word].reshape(1, 300)
            else:
                return self.word2vec['UNK'].reshape(1, 300)
        else:
            return rng.standard_normal(size=(1,300))
