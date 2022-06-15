import logging, os, argparse
from textseg_model import create_model
import gensim.downloader as gensim_api
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Tuple
from tqdm import tqdm
from nltk import regexp_tokenize, sent_tokenize
from datetime import datetime
import numpy as np
rng = np.random.default_rng()
from utils import ENWIKI_LOC, NLWIKI_LOC, NLNEWS_LOC, NLAUVI_LOC_C, NLAUVI_LOC_N, get_all_file_names, sectioned_clean_text, compute_metrics, LoggingHandler

W2V_NL_PATH = '../Datasets/word2vec-nl-combined-160.tar.gz' # Taken from https://github.com/clips/dutchembeddings
CHECKPOINT_BASE = 'checkpoints/textseg'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(LoggingHandler(use_tqdm=True))

class TextSeg:
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
        subset: Optional[int] = None,
    ) -> None:
        """
        """
        if num_workers > 0:
            if torch.multiprocessing.get_start_method() == 'fork':
                logger.warning("Can't use num_workers > 0 with fork start method, setting num_workers to 0.")
                num_workers = 0
        if not use_cuda:
            global device
            device = torch.device("cpu")
            logger.info(f"Using device: {device}.")
        else:
            logger.info(f"Using device: {torch.cuda.get_device_name(device)}.")

        if from_wiki is None:
            from_wiki = "wiki" in dataset_path.lower()
        self.load_data(language, dataset_path, from_wiki, splits, batch_size, num_workers, subset)
        self.load_from = load_from
        self.current_epoch = 0

    def load_data(
        self,
        language: str,
        dataset_path: str,
        from_wiki: bool,
        splits: List[float],
        batch_size: int,
        num_workers: int,
        subset: Optional[int] = None,
        ) -> None:
        """
        Load the right data and pretrained models.
        """
        if language == 'en':
            word2vec_path = gensim_api.load('word2vec-google-news-300', return_path=True)
            word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=100_000)
        elif language == 'nl':
            word2vec = KeyedVectors.load_word2vec_format(W2V_NL_PATH, limit=100_000)
        elif language == 'test':
            word2vec = None
        else:
            raise ValueError(f"Language {language} not supported, try 'en' or 'nl' instead.")

        path_list = get_all_file_names(dataset_path)
        if subset:
            path_list = rng.choice(path_list, size=subset, replace=False).tolist()
        # Split and shuffle the data
        train_paths, test_paths = train_test_split(path_list, test_size=1-splits[0])
        dev_paths, test_paths = train_test_split(test_paths, test_size=splits[1]/(splits[1]+splits[2]))

        train_dataset = TS_Dataset(train_paths, word2vec, from_wiki)
        val_dataset = TS_Dataset(dev_paths, word2vec, from_wiki)
        test_dataset = TS_Dataset(test_paths, word2vec, from_wiki)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers)
        logger.info(f"Loaded {len(train_dataset)} training examples, {len(val_dataset)} validation examples, and {len(test_dataset)} test examples.")

    def get_model(self):
        if self.load_from:
            model = torch.load(self.load_from).to(device)
            logger.info(f"Loaded TS_Model from {self.load_from}.")
        else:
            model = create_model(set_device=device)
            logger.info(f"Created new TS_Model.")
        return model

    def run(self, epochs=10) -> Tuple[float, float]:
        writer = SummaryWriter()
        now = datetime.now().strftime(r'%m-%d_%H-%M')
        checkpoint_path = os.path.join(CHECKPOINT_BASE, now)
        os.makedirs(checkpoint_path, exist_ok=True)

        model = self.get_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_scores = [np.inf, np.inf] # (pk, windowdiff)
        non_improvement_count = 0
        with tqdm(desc='Epochs', total=epochs) as pbar:
            for epoch in range(epochs):
                if non_improvement_count > 5: # Stop early if no improvement for 5 epochs.
                    logger.info("No improvement for 5 epochs, stopping early.")
                    break

                self.current_epoch = epoch
                self.train(model, optimizer, writer)
                torch.save(model, os.path.join(checkpoint_path, f'model_{epoch}.t7'))
                val_pk, val_wd, threshold = self.validate(model, writer)

                if sum([val_pk, val_wd]) < sum(best_val_scores):
                    non_improvement_count = 0
                    test_scores = self.test(model, threshold, writer)
                    logger.result(f"Best model from Epoch {epoch} --- For threshold = {threshold:.4}, Pk = {test_scores[0]:.4}, windowdiff = {test_scores[1]:.4}")
                    best_val_scores = [val_pk, val_wd]
                    torch.save(model, os.path.join(checkpoint_path, f'model_best.t7'))
                else:
                    non_improvement_count += 1
                pbar.update(1)
        writer.close()
        return best_val_scores

    def test_run(self, threshold=0.4):
        if self.load_from:
            model = torch.load(self.load_from).to(device)
            logger.info(f"Loaded model from {self.load_from}.")
        else:
            model = create_model()
            logger.info(f"Created new model.")
        return self.test(model, threshold)

    def train(self, model, optimizer, writer) -> None:
        model.train()
        total_loss = 0.0
        with tqdm(desc='Training', total=len(self.train_loader.dataset), leave=False) as pbar:
            for sents, targets in self.train_loader:
                sents = sents.to(device)
                model.zero_grad()
                output = model(sents)
                output, targets = self.remove_last(output, targets)

                target_var = torch.cat(targets, dim=0).to(device)
                loss = model.criterion(output, target_var)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pbar.set_postfix(loss=loss.item())
                pbar.update(len(targets))
        train_loss = total_loss / len(self.train_loader) # Average loss per batch.
        logger.info(f"Training Epoch {self.current_epoch + 1} --- Loss = {train_loss:.4}")
        writer.add_scalar('Loss/train', train_loss, self.current_epoch)

    def validate(self, model, writer) -> Tuple[float, float, float]:
        model.eval()
        thresholds = np.arange(0, 1, 0.05)
        scores = {k: [] for k in thresholds} # (pk, windowdiff) scores for each threshold.
        total_loss = 0.0

        with tqdm(desc="Validating", total=len(self.val_loader.dataset), leave=False) as pbar:
            for sents, targets in self.val_loader:
                sents = sents.to(device)
                output = model(sents)
                output, targets = self.remove_last(output, targets)
                
                target_var = torch.cat(targets, dim=0).to(device)
                loss = model.criterion(output, target_var)
                total_loss += loss.item()
                output_softmax = F.softmax(output, dim=1)
                output_softmax = output_softmax.cpu().detach().numpy() # convert to numpy array.

                # Calculate the Pk and windowdiff per document (in this batch) and append to scores for each threshold.
                doc_start_idx = 0
                for doc_targets in targets:
                    for threshold in thresholds:
                        doc_prediction = (output_softmax[doc_start_idx:doc_start_idx+len(doc_targets)][:, 1] > threshold).astype(int)
                        pk, wd = compute_metrics(doc_prediction, doc_targets)
                        scores[threshold].append((pk, wd))
                    doc_start_idx += len(doc_targets)

                pbar.set_postfix(loss=loss.item())
                pbar.update(len(targets))

        val_loss = loss.item() / len(self.val_loader) # Average loss per batch.
        best_scores = [np.inf, np.inf] # (pk, windowdiff) scores for the best threshold.
        best_threshold: float = None
        for threshold in thresholds:
            avg_pk_wd = np.mean(scores[threshold], axis=0) # (pk, windowdiff) average for this threshold, over all docs.
            if sum(avg_pk_wd) < sum(best_scores):
                best_scores = avg_pk_wd
                best_threshold = threshold
        logger.info(f"Validation Epoch {self.current_epoch + 1} --- Loss = {val_loss:.4}, For threshold = {best_threshold:.4}, Pk = {best_scores[0]:.4}, windowdiff = {best_scores[1]:.4}")
        writer.add_scalar('Loss/val', val_loss, self.current_epoch)
        writer.add_scalar('pk_score/val', best_scores[0], self.current_epoch)
        writer.add_scalar('wd_score/val', best_scores[1], self.current_epoch)
        return best_scores[0], best_scores[1], best_threshold

    def test(self, model, threshold: float, writer) -> Tuple[float, float]:
        model.eval()
        scores = []
        with tqdm(desc='Testing', total=len(self.test_loader.dataset), leave=False) as pbar:
            for sents, targets in self.test_loader:
                sents = sents.to(device)
                output = model(sents)
                output, targets = self.remove_last(output, targets)

                output_softmax = F.softmax(output, dim=1)
                output_softmax = output_softmax.cpu().detach().numpy() # convert to numpy array.

                # Calculate the Pk and windowdiff per document (in this batch) for the specified threshold.
                doc_start_idx = 0
                for doc_labels in targets:
                    doc_prediction = (output_softmax[doc_start_idx:doc_start_idx+len(doc_labels)][:, 1] > threshold).astype(int)
                    pk, wd = compute_metrics(doc_prediction, doc_labels)
                    scores.append((pk, wd))
                    doc_start_idx += len(doc_labels)
                pbar.update(len(targets))
        epoch_pk, epoch_wd = np.mean(scores, axis=0) # (pk, windowdiff) average for this threshold, over all docs.
        logger.info(f"Testing Epoch {self.current_epoch + 1} --- For threshold = {threshold:.4}, Pk = {epoch_pk:.4}, windowdiff = {epoch_wd:.4}")
        writer.add_scalar('pk_score/test', epoch_pk, self.current_epoch)
        writer.add_scalar('wd_score/test', epoch_wd, self.current_epoch)
        return epoch_pk, epoch_wd

    def remove_last(self, x: torch.Tensor, y: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        For each document in the batch, remove the last sentence prediction/label (as it's unnecessary).
        """
        x_adj = []
        y_adj = []
        first_sent_i = 0
        for target in y:
            final_sent_i = first_sent_i + target.size(0)
            x_adj.append(x[first_sent_i:final_sent_i-1, :])
            y_adj.append(target[:-1])
            first_sent_i = final_sent_i

        return torch.cat(x_adj), y_adj


def custom_collate(batch):
    """
    Prepare the batch for the model (so we can use data of varying lengths).
    https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    """
    all_sents = []
    batched_targets = []
    for data, targets in batch:
        for sentence in data:
            all_sents.append(torch.FloatTensor(sentence))
        batched_targets.append(torch.LongTensor(targets))

    # Pad and pack the sentences into a single tensor. 
    # This function sorts it, but with enfore_sorted=False it gets unsorted again after passing it through a model.
    packed_sents = pack_sequence(all_sents, enforce_sorted=False)
    return packed_sents, batched_targets

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

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], List[int]]:
        """
        For a document at index, returns:
            data: List of n sentence embeddings (made out of word vectors).
            labels: List of n labels [0 or 1]; 
                    A 1 signifies the END of a section (hence the last sentence/label is discarded later).
        """
        with open(self.files[index], 'r') as f:
            text = f.read()
        sections = sectioned_clean_text(text)
        data = []
        labels = []
        suitable_sections_count = 0

        for i, section in enumerate(sections):
            sentences = sent_tokenize(section)
            if not (80 > len(sentences) > 0): # Filter out empty or too long sections.
                if len(sections) <= 2: # Skip docs that end up with just a single section
                    break
                continue

            sentence_labels = np.zeros(len(sentences), dtype=int)
            section_sent_emb = []
            for sentence in sentences:
                sentence_words = self.word_tokenize(sentence)
                if len(sentence_words) > 0:
                    sentence_emb = self.embed_words(sentence_words)
                    if len(sentence_emb) > 0:
                        section_sent_emb.append(sentence_emb)
            
            if len(section_sent_emb) > 0:
                data += section_sent_emb
                sentence_labels = np.zeros(len(section_sent_emb), dtype=int)
                sentence_labels[-1] = 1 # Last sentence ends a section --> 1.
                labels += sentence_labels.tolist()
                suitable_sections_count += 1

        # Get a random document if the current one is empty or has less than two suitable sections.
        if len(data) == 0 or suitable_sections_count < 2:
            logger.warning(f"SKIPPED Document {self.files[index]} - it's empty or has less than two suitable sections.")
            return self.__getitem__(np.random.randint(0, len(self.files)))
        return data, labels

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

    def embed_words(self, words: List[str]) -> np.ndarray:
        res = []
        for word in words:
            if self.word2vec:
                if word in self.word2vec:
                    res.append(self.word2vec[word])
                else:
                    # return self.word2vec['UNK']
                    continue # skip words not in the word2vec model
            else:
                res.append(rng.standard_normal(size=300))
                
        return np.stack(res) if res else res


def main(args):
    ts = TextSeg(
        language=args.lang,
        dataset_path=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cuda= not args.disable_cuda,
        load_from=args.load_from,
        subset=args.subset,
    )
    if args.test:
        ts.test_run()
    else:
        res = ts.run(args.epochs)
    print(res)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="Train/Test a Text Segmentation model.")
    parser.add_argument("--test", action="store_true", help="Test mode.")
    parser.add_argument("--lang", type=str, default="en", help="Language to use.")
    parser.add_argument("--data_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("--subset", type=int, help="Use only a subset of the dataset.")
    parser.add_argument("--disable_cuda", action="store_true", help="Disable cuda (if available).")
    parser.add_argument('--batch_size', help='Batch size', type=int, default=8)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--load_from', help='Where to load an existing model from', type=str, default=None)
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)

    main(parser.parse_args())
