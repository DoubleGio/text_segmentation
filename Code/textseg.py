import logging, os, argparse, gc
from textseg_model import create_model
import gensim.downloader as gensim_api
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Tuple
from tqdm import tqdm
from nltk import sent_tokenize
from datetime import datetime
import numpy as np
rng = np.random.default_rng()
from utils import get_all_file_names, clean_text, sectioned_clean_text, compute_metrics, LoggingHandler, word_tokenize

W2V_NL_PATH = 'text_segmentation/Datasets/word2vec-nl-combined-160.txt' # Taken from https://github.com/clips/dutchembeddings
EARLY_STOP = 4
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
        language: Optional[str] = None,
        word2vec_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        from_wiki: Optional[bool] = None,
        splits = [0.8, 0.1, 0.1],
        batch_size = 8,
        num_workers = 0,
        use_cuda = True,
        load_from: Optional[str] = None,
        subset: Optional[int] = None,
    ) -> None:
        if language:
            if language == 'en':
                word2vec_path = gensim_api.load('word2vec-google-news-300', return_path=True)
                word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=100_000)
            elif language == 'nl':
                word2vec = KeyedVectors.load_word2vec_format(W2V_NL_PATH, limit=100_000)
            elif language == 'test':
                word2vec = None
            else:
                raise ValueError(f"Language {language} not supported, try 'en' or 'nl' instead.")
        elif word2vec_path:
            for is_binary in [True, False]:
                try:
                    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=is_binary, limit=100_000)
                    break
                except FileNotFoundError:
                    raise FileNotFoundError(f"Could not find word2vec file at {word2vec_path}.")
                except UnicodeDecodeError:
                    continue
            if not word2vec:
                raise Exception(f"Invalid encoding, could not load word2vec file from {word2vec_path}.")
        else:
            raise ValueError("Either a word2vec path or a language must be specified.")
        TS_Dataset.word2vec = word2vec
        self.vec_size = word2vec.vector_size if word2vec else 300

        if num_workers > 0:
            if torch.multiprocessing.get_start_method() == 'spawn':
                # Multiprocessing only works on Unix systems which support the fork() system call.
                logger.warning("Can't use num_workers > 0 with spawn start method, setting num_workers to 0.")
                num_workers = 0
        if not use_cuda:
            global device
            device = torch.device("cpu")
            logger.info(f"Using device: {device}.")
            self.use_cuda = False
        else:
            device_name = torch.cuda.get_device_name(device)
            logger.info(f"Using device: {device_name}.")
            self.use_cuda = False if device_name == 'cpu' else True

        if dataset_path:
            self.subsets = [int(subset * split) for split in splits] if subset else None
            if from_wiki is None:
                from_wiki = "wiki" in dataset_path.lower()
            self.load_data(dataset_path, from_wiki, splits, batch_size, num_workers)
        self.load_from = load_from
        self.current_epoch = 0

    def load_data(
        self,
        dataset_path: str,
        from_wiki: bool,
        splits: List[float],
        batch_size: int,
        num_workers: int,
    ) -> None:
        """
        Load the right data and pretrained models.
        """
        self.dataset_name = dataset_path.split('/')[-2]
        path_list = get_all_file_names(dataset_path)

        # Split and shuffle the data
        train_paths, test_paths = train_test_split(path_list, test_size=1-splits[0])
        dev_paths, test_paths = train_test_split(test_paths, test_size=splits[1]/(splits[1]+splits[2]))

        train_dataset = TS_Dataset(np.array(train_paths), from_wiki)
        val_dataset = TS_Dataset(np.array(dev_paths), from_wiki)
        test_dataset = TS_Dataset(np.array(test_paths), from_wiki)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers, pin_memory=self.use_cuda)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers, pin_memory=self.use_cuda)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers, pin_memory=self.use_cuda)
        logger.info(f"Loaded {len(train_dataset)} training examples, {len(val_dataset)} validation examples, and {len(test_dataset)} test examples.")
        if self.subsets:
            logger.info(f"Subsets: train {self.subsets[0]}, validation {self.subsets[1]}, test {self.subsets[2]}")

    def initialize_run(self, resume=False):
        if resume:
            if self.load_from is None:
                raise ValueError("Can't resume without a load_from path.")
            state = torch.load(self.load_from)
            logger.info(f"Loaded state from {self.load_from}.")
            now = state['now']
            writer = SummaryWriter(log_dir=f'runs/textseg/{self.dataset_name}_{now}')
            checkpoint_path = os.path.join(f'checkpoints/textseg/{self.dataset_name}_{now}')
            
            model = create_model(input_size=self.vec_size, set_device=device)
            model.load_state_dict(state['state_dict'])
            logger.info(f"Loaded model from {self.load_from}.")

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.load_state_dict(state['optimizer'])

            best_val_scores = state['best_val_scores']
            non_improvement_count = state['non_improvement_count']
            self.current_epoch = state['epoch']
        else:
            now = datetime.now().strftime(r'%b%d_%H%M')
            writer = SummaryWriter(log_dir=f'runs/textseg/{self.dataset_name}_{now}')
            checkpoint_path = os.path.join(f'checkpoints/textseg/{self.dataset_name}_{now}')
            os.makedirs(checkpoint_path, exist_ok=True)

            model = create_model(input_size=self.vec_size, set_device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            best_val_scores = [np.inf, np.inf] # (pk, windowdiff)
            non_improvement_count = 0
        return now, writer, checkpoint_path, model, optimizer, best_val_scores, non_improvement_count

    def run(self, epochs=10, resume=False) -> Tuple[float, float]:
        """
        Start/Resume training.
        """
        now, writer, checkpoint_path, model, optimizer, best_val_scores, non_improvement_count = self.initialize_run(resume)
        with tqdm(desc='Epochs', total=epochs) as pbar:
            for epoch in range(self.current_epoch, epochs):
                if non_improvement_count > EARLY_STOP:
                    logger.result(f"No improvement for {EARLY_STOP} epochs, stopping early.")
                    break

                self.current_epoch = epoch
                self.train(model, optimizer, writer)
                state = {
                    'now': now,
                    'epoch': epoch,
                    'non_improvement_count': non_improvement_count,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_scores': best_val_scores,
                }
                torch.save(state, os.path.join(checkpoint_path, f'epoch_{epoch}'))
                val_pk, val_wd, threshold = self.validate(model, writer)

                if sum([val_pk, val_wd]) < sum(best_val_scores):
                    non_improvement_count = 0
                    test_scores = self.test(model, threshold, writer)
                    logger.result(f"Best model from Epoch {epoch} --- For threshold = {threshold:.4}, Pk = {test_scores[0]:.4}, windowdiff = {test_scores[1]:.4}")
                    best_val_scores = [val_pk, val_wd]
                    state = {
                        'now': now,
                        'epoch': epoch,
                        'non_improvement_count': non_improvement_count,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_scores': best_val_scores,
                        'threshold': threshold,
                    }
                    torch.save(state, os.path.join(checkpoint_path, f'best_model'))
                else:
                    non_improvement_count += 1
                pbar.update(1)
                gc.collect()
            logger.info(f"Total time elapsed: {pbar.format_dict['elapsed']}")
        writer.close()
        return best_val_scores
    
    def run_test(self, threshold: Optional[float] = None) -> Tuple[float, float]:
        if self.load_from is None:
            raise ValueError("Can't test without a load_from path.")
        state = torch.load(self.load_from)
        model = create_model(input_size=self.vec_size, set_device=device)
        model.load_state_dict(state['state_dict'])
        logger.info(f"Loaded model from {self.load_from}.")
        return self.test(model, threshold if threshold else state['threshold'])

    def segment_text(self, texts: List[str], threshold: Optional[float] = None, from_wiki=False) -> List[str]:
        """
        Load a model and segment text(s).
        """
        if self.load_from is None:
            raise ValueError("Can't segment without a load_from path.")
        state = torch.load(self.load_from)
        model = create_model(input_size=self.vec_size, set_device=device)
        model.load_state_dict(state['state_dict'])
        model.eval()
        logger.info(f"Loaded model from {self.load_from}.")
        if threshold is None:
            threshold = state['threshold']
        logger.info(f"Using threshold {threshold:.2}.")

        text_data = TS_Dataset(np.array(texts), from_wiki, labeled=False)
        text_loader = DataLoader(text_data, batch_size=4, collate_fn=custom_collate, num_workers=0, pin_memory=self.use_cuda)
        logger.info(f"Loaded {len(text_data)} texts.")

        segmented_texts = []
        with tqdm(desc='Segmenting', total=len(text_loader.dataset)) as pbar:
            with torch.no_grad():
                for sents, texts, doc_lengths in text_loader:
                    if sents is None:
                        pbar.update(text_loader.batch_size)
                        continue

                    if self.use_cuda:
                        sents = sents.to(device, non_blocking=True)
                        doc_lengths = doc_lengths.to(device, non_blocking=True)
                    output = model(sents)
                    del sents
                    output_softmax = F.softmax(output, dim=1)
                    output_softmax = output_softmax.cpu().detach().numpy()

                    doc_start_idx = 0
                    for text in texts:
                        predictions = (output_softmax[doc_start_idx:doc_start_idx+len(text)][:, 1] > threshold).astype(int)
                        doc_start_idx += len(text)
                        segmented_text = '======\n'
                        for i, sent in enumerate(text):
                            segmented_text += sent
                            if predictions[i] == 1:
                                segmented_text += '\n======\n'
                        segmented_texts.append(segmented_text)
                        pbar.update(len(texts))
                    torch.cuda.empty_cache()

        return segmented_texts

    def train(self, model, optimizer, writer) -> None:
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        with tqdm(desc=f'Training #{self.current_epoch}', total=min(len(self.train_loader.dataset), self.subsets[0]), leave=False) as pbar:
            for sents, targets, doc_lengths in self.train_loader:
                if pbar.n > self.subsets[0]:
                    break
                if self.use_cuda:
                    sents = sents.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    doc_lengths = doc_lengths.to(device, non_blocking=True)
                model.zero_grad()
                output = model(sents)
                del sents
                output, targets, doc_lengths = self.remove_last(output, targets, doc_lengths)

                loss = model.criterion(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.detach()

                pbar.update(len(doc_lengths))
                torch.cuda.empty_cache()

        train_loss = total_loss.item() / len(self.train_loader) # Average loss per batch.
        logger.info(f"Training Epoch {self.current_epoch + 1} --- Loss = {train_loss:.4}")
        writer.add_scalar('Loss/train', train_loss, self.current_epoch)

    def validate(self, model, writer) -> Tuple[float, float, float]:
        model.eval()
        thresholds = np.arange(0, 1, 0.05)
        scores = {k: [] for k in thresholds} # (pk, windowdiff) scores for each threshold.
        total_loss = torch.tensor(0.0, device=device)

        with tqdm(desc=f"Validating #{self.current_epoch}", total=min(len(self.val_loader.dataset), self.subsets[1]), leave=False) as pbar:
            with torch.no_grad():
                for sents, targets, doc_lengths in self.val_loader:
                    if pbar.n > self.subsets[1]:
                        break
                    if self.use_cuda:
                        sents = sents.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        doc_lengths = doc_lengths.to(device, non_blocking=True)
                    output = model(sents)
                    del sents
                    output, targets, doc_lengths = self.remove_last(output, targets, doc_lengths)

                    loss = model.criterion(output, targets)
                    total_loss += loss.detach()
                    output_softmax = F.softmax(output, dim=1)
                    output_softmax = output_softmax.cpu().detach().numpy() # convert to numpy array.

                    # Calculate the Pk and windowdiff per document (in this batch) and append to scores for each threshold.
                    doc_start_idx = 0
                    for doc_len in doc_lengths:
                        for threshold in thresholds:
                            doc_prediction = (output_softmax[doc_start_idx:doc_start_idx+doc_len][:, 1] > threshold).astype(int)
                            # doc_prediction = (output_softmax[doc_start_idx:doc_start_idx+len(doc_targets)][:, 1] > threshold).astype(int)
                            pk, wd = compute_metrics(doc_prediction, targets[doc_start_idx:doc_start_idx+doc_len])
                            scores[threshold].append((pk, wd))
                        doc_start_idx += doc_len
                    pbar.update(len(targets))
                    torch.cuda.empty_cache()

        val_loss = total_loss.item() / len(self.val_loader) # Average loss per batch.
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

    def test(self, model, threshold: float, writer: Optional[SummaryWriter] = None) -> Tuple[float, float]:
        model.eval()
        scores = []
        with tqdm(desc=f'Testing #{self.current_epoch}', total=min(len(self.test_loader.dataset), self.subsets[2]), leave=False) as pbar:
            with torch.no_grad():
                for sents, targets, doc_lengths in self.test_loader:
                    if pbar.n > self.subsets[2]:
                        break
                    if self.use_cuda:
                        sents = sents.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        doc_lengths = doc_lengths.to(device, non_blocking=True)
                    output = model(sents)
                    del sents
                    output, targets, doc_lengths = self.remove_last(output, targets, doc_lengths)

                    output_softmax = F.softmax(output, dim=1)
                    output_softmax = output_softmax.cpu().detach().numpy() # convert to numpy array.

                    # Calculate the Pk and windowdiff per document (in this batch) for the specified threshold.
                    doc_start_idx = 0
                    for doc_len in doc_lengths:
                        doc_prediction = (output_softmax[doc_start_idx:doc_start_idx+doc_len][:, 1] > threshold).astype(int)
                        # doc_prediction = (output_softmax[doc_start_idx:doc_start_idx+len(doc_labels)][:, 1] > threshold).astype(int)
                        pk, wd = compute_metrics(doc_prediction, targets[doc_start_idx:doc_start_idx+doc_len])
                        scores.append((pk, wd))
                        doc_start_idx += doc_len
                    pbar.update(len(targets))
                    torch.cuda.empty_cache()

        epoch_pk, epoch_wd = np.mean(scores, axis=0) # (pk, windowdiff) average for this threshold, over all docs.
        if writer:
            logger.info(f"Testing Epoch {self.current_epoch + 1} --- For threshold = {threshold:.4}, Pk = {epoch_pk:.4}, windowdiff = {epoch_wd:.4}")
            writer.add_scalar('pk_score/test', epoch_pk, self.current_epoch)
            writer.add_scalar('wd_score/test', epoch_wd, self.current_epoch)
        return epoch_pk, epoch_wd

    def remove_last(self, x: torch.Tensor, y: torch.LongTensor, lengths: torch.LongTensor) -> Tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
        """
        For each document in the batch, remove the last sentence prediction/label (as it's unnecessary).
        """
        x_adj = torch.tensor([], device=device)
        y_adj = torch.tensor([], dtype=torch.int, device=device)
        first_sent_i = 0
        for i in range(lengths.size(0)):
            final_sent_i = first_sent_i + lengths[i]
            x_adj = torch.cat((x_adj, x[first_sent_i:final_sent_i-1, :]), dim=0)
            y_adj = torch.cat((y_adj, y[first_sent_i:final_sent_i-1]), dim=0)
            first_sent_i = final_sent_i
            lengths[i] -= 1
        
        return x_adj, y_adj, lengths


def custom_collate(batch):
    all_sents = []
    batched_targets = np.array([])
    doc_lengths = torch.LongTensor([])
    for data, targets in batch:
        if data is not None:
            for i in range(data.shape[0]):
                all_sents.append(data[i][data[i].sum(dim=1) != 0]) # remove padding
            doc_lengths = torch.cat((doc_lengths, torch.LongTensor([len(targets)])))
            batched_targets = np.concatenate((batched_targets, targets))
    if len(all_sents) == 0:
        return None, None, None
    packed_sents = pack_sequence(all_sents, enforce_sorted=False)
    if batched_targets.dtype == float:
        batched_targets = torch.from_numpy(batched_targets).long()
    return packed_sents, batched_targets, doc_lengths

def custom_collate_old(batch):
    """
    Prepare the batch for the model (so we can use data of varying lengths).
    https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    """
    all_sents = []
    batched_targets = []
    for data, targets in batch:
        all_sents += data
        if targets.dtype == int: # If targets is a list of ints, then it are the ground truths.
            batched_targets.append(torch.LongTensor(targets))
        else:                       # Else it is the raw text.
            batched_targets.append(targets)
    del batch
    # Pad and pack the sentences into a single tensor. 
    # This function sorts it, but with enfore_sorted=False it gets unsorted again after passing it through a model.
    packed_sents = pack_sequence(all_sents, enforce_sorted=False)
    return packed_sents, batched_targets

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
        labeled=True
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
            sentences = sent_tokenize(section)
            if not (TS_Dataset.MAX_SECTION_LEN > len(sentences) > 1): # Filter too short or too long sections.
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
            doc_id = self.texts[index].split('/', 2)[-1] if self.are_paths else index
            logger.warning(f"SKIPPED Document {doc_id} - empty/unsuitable.")
            return self.__getitem__(rng.integers(0, len(self.texts)))

        data = pad_sequence(data, batch_first=True) # (num_sents, max_sent_len, embed_dim)
        return data, targets

    def get_data_raw(self, index: int) -> Tuple[torch.FloatTensor, np.ndarray]:
        """Get the data and raw text for a document at index."""
        text = self.get_text(index)
        data = []
        raw_text = np.array([])

        text = clean_text(text, from_wiki=self.from_wiki)
        sentences = sent_tokenize(text)
        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            if len(sentence_words) > 0:
                sentence_emb = self.embed_words(sentence_words)
                if len(sentence_emb) > 0:
                    data.append(sentence_emb)
                    raw_text = np.append(raw_text, sentence)

        if len(data) == 0:
            doc_id = self.texts[index].split('/', 2)[-1] if self.are_paths else index
            logger.warning(f"SKIPPED Document {doc_id} - it's empty.")
            return None, None
        data = pad_sequence(data, batch_first=True) # (num_sents, max_sent_len, embed_dim)
        return data, raw_text
        
    def __len__(self) -> int:
        return len(self.texts)

    def get_text(self, index: int) -> str:
        if self.are_paths:
            with open(self.texts[index], 'r') as f:
                text = f.read()
        else:
            text = self.texts[index]
        return text

    def embed_words(self, words: List[str]) -> torch.FloatTensor:
        res = []
        for word in words:
            if TS_Dataset.word2vec:
                if word in TS_Dataset.word2vec:
                    res.append(TS_Dataset.word2vec[word])
                else:
                    # return TS_Dataset.word2vec['UNK']
                    continue # skip words not in the word2vec model
            else:
                # logger.warning("No word2vec model loaded.")
                res.append(rng.standard_normal(size=300))
        return torch.FloatTensor(np.stack(res)) if res else []


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
        ts.run_test()
    else:
        res = ts.run(args.epochs, args.resume)
    print(f'Best Pk = {res[0]} --- Best windowdiff = {res[1]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Test a Text Segmentation model.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test", action="store_true", help="Test mode.")
    mode.add_argument("--resume", action="store_true", help="Resume training from a checkpoint.")
    parser.add_argument("--lang", type=str, default="en", help="Language to use.")
    parser.add_argument("--data_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("--subset", type=int, help="Use only a subset of the dataset.")
    parser.add_argument("--disable_cuda", action="store_true", help="Disable cuda (if available).")
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--load_from', help='Where to load an existing model from', type=str, default=None)
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=16)

    main(parser.parse_args())
