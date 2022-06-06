import logging
import gensim.downloader as gensim_api
import os
from koomri_model import create_model
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Tuple
from tqdm import tqdm
from nltk import regexp_tokenize, sent_tokenize
from datetime import datetime
import numpy as np
rng = np.random.default_rng()
from utils import ENWIKI_LOC, NLWIKI_LOC, NLNEWS_LOC, NLAUVI_LOC_C, NLAUVI_LOC_N, get_all_file_names, sectioned_clean_text, compute_metrics, LoggingHandler

W2V_NL_PATH = '../Datasets/word2vec-nl-combined-160.tar.gz' # Taken from https://github.com/clips/dutchembeddings
CHECKPOINT_BASE = 'checkpoints/koomri'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(LoggingHandler(use_tqdm=True))

class KoomriImplementation:
    """
    Updated implementation of https://github.com/koomri/text-segmentation.
    """

    def __init__(
        self,
        language = 'en',
        dataset_path = ENWIKI_LOC,
        from_wiki: Optional[bool] = False,
        splits = [0.8, 0.1, 0.1],
        batch_size = 8,
        num_workers = 0,
        use_cuda = True,
        load_from: Optional[str] = None,
    ) -> None:
        """
        """
        if not use_cuda:
            global device
            device = torch.device("cpu")
            logger.info(f"Using device: {device}.")
        else:
            logger.info(f"Using device: {torch.cuda.get_device_name(device)}.")

        if language == 'en':
            word2vec = gensim_api.load('word2vec-google-news-300')
        elif language == 'nl':
            word2vec = KeyedVectors.load_word2vec_format(W2V_NL_PATH)
        elif language == 'test':
            word2vec = None
        else:
            raise ValueError(f"Language {language} not supported, try 'en' or 'nl' instead.")

        # Load the data, shuffle it and put it into split DataLoaders.
        path_list = get_all_file_names(dataset_path)
        train_paths, test_paths = train_test_split(path_list, test_size=1-splits[0])
        dev_paths, test_paths = train_test_split(test_paths, test_size=splits[1]/(splits[1]+splits[2]))

        if from_wiki is None:
            from_wiki = "wiki" in dataset_path.lower()
        train_dataset = TS_Dataset(train_paths, word2vec, from_wiki)
        val_dataset = TS_Dataset(dev_paths, word2vec, from_wiki)
        test_dataset = TS_Dataset(test_paths, word2vec, from_wiki)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers)
        logger.info(f"Loaded {len(train_dataset)} training examples, {len(val_dataset)} validation examples, and {len(test_dataset)} test examples.")

        self.load_from = load_from
        self.current_epoch = 0

    def run(self, epochs=10) -> Tuple[float, float]:
        now = datetime.now().strftime(r'%m-%d_%H-%M')
        checkpoint_path = os.path.join(CHECKPOINT_BASE, now)
        os.makedirs(checkpoint_path, exist_ok=True)

        if self.load_from:
            model = torch.load(self.load_from).to(device)
            logger.info(f"Loaded model from {self.load_from}.")
        else:
            model = create_model()
            logger.info(f"Created new model.")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_scores = [np.inf, np.inf] # (pk, windowdiff)
        with tqdm(desc='Epochs', total=epochs) as pbar:
            for epoch in range(epochs):
                self.current_epoch = epoch
                self.train(model, optimizer)
                torch.save(model, os.path.join(checkpoint_path, f'model_{epoch}.t7'))

                val_pk, val_wd, threshold = self.validate(model)
                if sum([val_pk, val_wd]) < sum(best_val_scores):
                    test_scores = self.test(model, threshold)
                    logger.result(f"Best model from Epoch {epoch} --- For threshold = {threshold:.4}, Pk = {test_scores[0]:.4}, windowdiff = {test_scores[1]:.4}")
                    best_val_scores = [val_pk, val_wd]
                    torch.save(model, os.path.join(checkpoint_path, f'model_best.t7'))
                pbar.update(1)
        return best_val_scores

    def test_run(self, threshold=0.4):
        if self.load_from:
            model = torch.load(self.load_from).to(device)
            logger.info(f"Loaded model from {self.load_from}.")
        else:
            model = create_model()
            logger.info(f"Created new model.")
        return self.test(model, threshold)

    def train(self, model, optimizer) -> None:
        model.train()
        total_loss = 0.0
        with tqdm(desc='Training', total=len(self.train_loader), leave=False) as pbar:
            for data, labels in self.train_loader:
                model.zero_grad()
                output = model(data)
                target_var = torch.cat(labels, dim=0).to(device)
                loss = model.criterion(output, target_var)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
        avg_loss = total_loss / len(self.train_loader) # Average loss per input.
        logger.info(f"Training Epoch {self.current_epoch + 1} --- Loss = {avg_loss:.4}")
        writer.add_scalar('Loss/Train', avg_loss, self.current_epoch + 1)

    def validate(self, model) -> Tuple[float, float, float]:
        model.eval()
        # cm = np.zeros((2, 2), dtype=int)
        thresholds = np.arange(0, 1, 0.05)
        scores = {k: [] for k in thresholds} # (pk, windowdiff) scores for each threshold.

        with tqdm(desc="Validating", total=len(self.val_loader), leave=False) as pbar:
            for data, labels in self.val_loader:
                output = model(data)
                output_softmax = torch.nn.functional.softmax(output, dim=1)
                output_softmax = output_softmax.cpu().detach().numpy() # convert to numpy array.

                # Calculate the Pk and windowdiff per document (in this batch) and append to scores for each threshold.
                doc_start_idx = 0
                for doc_labels in labels:
                    for threshold in thresholds:
                        doc_prediction = (output_softmax[doc_start_idx:doc_start_idx+len(doc_labels)][:, 1] > threshold).astype(int)
                        pk, wd = compute_metrics(doc_prediction, doc_labels)
                        scores[threshold].append((pk, wd))
                    doc_start_idx += len(doc_labels)
                pbar.update(1)
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

    def test(self, model, threshold: float) -> Tuple[float, float]:
        model.eval()
        scores = []
        with tqdm(desc='Testing', total=len(self.test_loader), leave=False) as pbar:
            for data, labels in self.test_loader:
                output = model(data)
                output_softmax = torch.nn.functional.softmax(output, dim=1)
                output_softmax = output_softmax.cpu().detach().numpy() # convert to numpy array.

                # Calculate the Pk and windowdiff per document (in this batch) for the specified threshold.
                doc_start_idx = 0
                for doc_labels in labels:
                    doc_prediction = (output_softmax[doc_start_idx:doc_start_idx+len(doc_labels)][:, 1] > threshold).astype(int)
                    pk, wd = compute_metrics(doc_prediction, doc_labels)
                    scores.append((pk, wd))
                    doc_start_idx += len(doc_labels)
                pbar.update(1)
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
    for data, targets in batch:
        tensored_data = []
        for sentence in data:
            tensored_data.append(torch.FloatTensor(np.concatenate(sentence)).to(device))
        b_data.append(tensored_data)
        tensored_targets = torch.LongTensor(targets).to(device)
        b_targets.append(tensored_targets)
    return b_data, b_targets

class TS_Dataset(Dataset):
    """
    Custom Text Segmentation Dataset class for use in Dataloaders.
    """
    def __init__(self, files: List[str], word2vec: KeyedVectors, from_wiki=False) -> None:
        """
        files: List of strings pointing towards files to read.
        word2vec: KeyedVectors object containing the word2vec model.
        from_wiki: Boolean indicating whether the dataset follows Wikipedia formatting and requires some extra preprocessing.
        """
        self.files = files
        self.word2vec = word2vec
        self.from_wiki = from_wiki

    def __getitem__(self, index: int) -> Tuple[List, List[int]]:
        """
        For a document at index, returns:
            data: List of n sentence representations (made out of word vectors).
            labels: List of n-1 labels [0 or 1]; a 1 signifies the END of a section (hence the last label is discarded).
        """
        with open(self.files[index], 'r') as f:
            text = f.read()
        sections = sectioned_clean_text(text)
        data = []
        labels = []

        for i, section in enumerate(sections):
            try:
                sentences = sent_tokenize(section)
                sentence_labels = np.zeros(len(sentences), dtype=int)
                sentence_labels[-1] = 1 # Last sentence ends a section --> 1.
                for sentence in sentences:
                    sentence_words = self.word_tokenize(sentence)
                    if len(sentence_words) > 0:
                        sentence_representation = [self.model_word(word) for word in sentence_words]
                        data.append(sentence_representation)
                    else:
                        sentence_labels = sentence_labels[:-1]
                        logging.warning(f"Sentence in {self.files[index]} is empty.")
                labels += sentence_labels.tolist()
            except ValueError:
                logging.warning(f"Section {i + 1} in {self.files[index]} is empty.")
            # if len(data) > 0:
                # labels.append(len(data) - 1) # NOTE: this points towards the END of a segment.

        return data, labels[:-1]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.files)

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
