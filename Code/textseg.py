import logging, os, argparse, gc
from models import create_TS_model
import gensim.downloader as gensim_api
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Generator, List, Optional, Tuple, Type, Union
from TS_Dataset import TS_Dataset
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np
rng = np.random.default_rng()
from utils import get_all_file_names, compute_metrics, LoggingHandler, SECTION_MARK

W2V_NL_PATH = 'Datasets/word2vec-nl-combined-160.txt' # Taken from https://github.com/clips/dutchembeddings
EARLY_STOP = 3
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
        subset = np.inf,
    ) -> None:      
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

        self.init_dataset_class(language=language, word2vec_path=word2vec_path)
        if dataset_path:
            if from_wiki is None:
                from_wiki = "wiki" in dataset_path.lower()
            self.load_data(
                dataset_path=dataset_path,
                from_wiki=from_wiki,
                splits=splits,
                batch_size=batch_size,
                num_workers=num_workers,
                subset=subset
            )
        self.load_from = load_from
        self.current_epoch = 0

    def init_dataset_class(self, language: Optional[str] = None, word2vec_path: Optional[str] = None, dataset_class: Type[TS_Dataset] = TS_Dataset):
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
            self.language = language
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
            self.language = 'nl' if 'nl' in word2vec_path else 'en'
        else:
            raise ValueError("Either a word2vec path or a language must be specified.")
        self.vec_size = word2vec.vector_size if word2vec else 300
        dataset_class.word2vec = word2vec

    def get_dataset_class(self) -> Type[TS_Dataset]:
        return TS_Dataset

    def load_data(
        self,
        dataset_path: str,
        from_wiki: bool,
        splits: List[float],
        batch_size: int,
        num_workers: int,
        subset = np.inf,
    ) -> None:
        """
        Load the right data and pretrained models.
        """
        dataset_class = self.get_dataset_class()
        self.dataset_name = dataset_path.split('/')[-2]
        path_list = get_all_file_names(dataset_path)

        # Split and shuffle the data
        train_paths, test_paths = train_test_split(path_list, test_size=1-splits[0])
        val_paths, test_paths = train_test_split(test_paths, test_size=splits[1]/(splits[1]+splits[2]))
        
        train_dataset = dataset_class(np.array(train_paths), from_wiki)
        val_dataset = dataset_class(np.array(val_paths), from_wiki)
        test_dataset = dataset_class(np.array(test_paths), from_wiki)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=self.custom_collate, num_workers=num_workers, pin_memory=self.use_cuda)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=self.custom_collate, num_workers=num_workers, pin_memory=self.use_cuda)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=self.custom_collate, num_workers=num_workers, pin_memory=self.use_cuda)
        logger.info(f"Loaded {len(train_dataset)} training examples, {len(val_dataset)} validation examples, and {len(test_dataset)} test examples.")
        if subset < len(path_list):
            self.sizes = {'train': int(subset * splits[0]), 'val': int(subset * splits[1]), 'test': int(subset * splits[2])}
            logger.info(f"Subsets: train {self.sizes['train']}, validate {self.sizes['val']}, test {self.sizes['test']}")
        else:
            self.sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    def initialize_run(self, resume=False, model_creator: Callable = create_TS_model):
        cname = self.__class__.__name__.lower()
        if resume:
            if self.load_from is None:
                raise ValueError("Can't resume without a load_from path.")
            state = torch.load(self.load_from)
            logger.info(f"Loaded state from {self.load_from}.")
            now = state['now']
            writer = SummaryWriter(log_dir=f'runs/{cname}/{self.dataset_name}_{now}')
            checkpoint_path = os.path.join(f'checkpoints/{cname}/{self.dataset_name}_{now}')
            
            model = model_creator(input_size=self.vec_size, set_device=device)
            model.load_state_dict(state['state_dict'])
            logger.info(f"Loaded model from {self.load_from}.")

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.load_state_dict(state['optimizer'])

            best_val_scores = state['best_val_scores']
            non_improvement_count = state['non_improvement_count']
            self.current_epoch = state['epoch'] + 1 # +1 because we're resuming from the epoch after the last one
        else:
            now = datetime.now().strftime(r'%b%d_%H%M')
            writer = SummaryWriter(log_dir=f'runs/{cname}/{self.dataset_name}_{now}')
            checkpoint_path = os.path.join(f'checkpoints/{cname}/{self.dataset_name}_{now}')
            os.makedirs(checkpoint_path, exist_ok=True)

            model = model_creator(input_size=self.vec_size, set_device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            best_val_scores = [np.inf, np.inf] # (pk, windowdiff)
            non_improvement_count = 0
        return now, writer, checkpoint_path, model, optimizer, best_val_scores, non_improvement_count

    def run(self, epochs=10, resume=False) -> Tuple[float, float]:
        """
        Start/Resume training.
        """
        now, writer, checkpoint_path, model, optimizer, best_val_scores, non_improvement_count = self.initialize_run(resume)
        with tqdm(desc='Epochs', total=epochs, initial=self.current_epoch) as pbar:
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
                gc.collect()
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
            logger.info(f"Total time elapsed: {str(timedelta(seconds=int(pbar.format_dict['elapsed'])))}")
        writer.close()
        return best_val_scores
    
    def run_test(
        self,
        texts: Optional[List[str]] = None,
        from_wiki = False,
        batch_size = 8,
        num_workers = 4,
        threshold: Optional[float] = None
    ) -> Tuple[float, float, float]:
        if self.load_from is None:
            raise ValueError("Can't test without a load_from path.")
        if texts is not None:
            dataset_class = self.get_dataset_class()
            test_dataset = dataset_class(np.array(texts), from_wiki=from_wiki)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=self.custom_collate, num_workers=num_workers, pin_memory=self.use_cuda)
            self.sizes = {'test': len(test_dataset)}
        state = torch.load(self.load_from)
        model = create_TS_model(input_size=self.vec_size, set_device=device)
        model.load_state_dict(state['state_dict'])
        logger.info(f"Loaded model from {self.load_from}.")
        return self.test(model, threshold if threshold else state['threshold'], return_acc=True)

    def segment_text(self, texts: List[str], threshold: Optional[float] = None, from_wiki=False) -> List[str]:
        """
        Load a model and segment text(s).
        """
        if self.load_from is None:
            raise ValueError("Can't segment without a load_from path.")
        state = torch.load(self.load_from)
        model = create_TS_model(input_size=self.vec_size, set_device=device)
        model.load_state_dict(state['state_dict'])
        model.eval()
        logger.info(f"Loaded model from {self.load_from}.")
        if threshold is None:
            threshold = state['threshold']
        logger.info(f"Using threshold {threshold:.2}.")
        del state

        text_data = TS_Dataset(np.array(texts), from_wiki, labeled=False)
        text_loader = DataLoader(text_data, batch_size=4, collate_fn=self.custom_collate, num_workers=0, pin_memory=self.use_cuda)
        logger.info(f"Loaded {len(text_data)} texts.")
        del texts

        segmented_texts = []
        with tqdm(desc='Segmenting', total=len(text_loader.dataset)) as pbar:
            with torch.inference_mode():
                for sents, texts, doc_lengths in text_loader:
                    if sents is None:
                        segmented_texts.append(None)
                        pbar.update(text_loader.batch_size)
                        continue
                    if self.use_cuda:
                        sents, doc_lengths = self.move_to_cuda(sents, doc_lengths)
                    output = model(sents)
                    del sents
                    output_softmax = F.softmax(output, dim=1)

                    predictions = (output_softmax[:, 1] > threshold).long()
                    doc_start_idx = 0
                    for doc_len in doc_lengths:
                        segmented_text = f'{SECTION_MARK}\n'
                        for i, sent in enumerate(texts[doc_start_idx:doc_start_idx+doc_len]):
                            segmented_text += ' ' + sent
                            if predictions[i] == 1 and i < doc_len-1:
                                segmented_text += f'\n{SECTION_MARK}\n'
                        doc_start_idx += doc_len
                        segmented_texts.append(segmented_text)
                        pbar.update(1)
                    torch.cuda.empty_cache()

        return segmented_texts

    def train(self, model, optimizer, writer) -> None:
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        with tqdm(desc=f'Training #{self.current_epoch}', total=self.sizes['train'], leave=False) as pbar:
            for sents, targets, doc_lengths in self.train_loader:
                if pbar.n > self.sizes['train']:
                    break
                if self.use_cuda:
                    sents, targets, doc_lengths = self.move_to_cuda(sents, targets, doc_lengths)
                model.zero_grad()
                output = model(sents)
                del sents
                output, targets, doc_lengths = self.remove_last(output, targets, doc_lengths)

                loss = model.criterion(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += (loss.detach() * len(doc_lengths))

                if pbar.n % 10 == 0:
                    pbar.set_postfix(loss=loss.item())
                pbar.update(len(doc_lengths))
                torch.cuda.empty_cache()

            train_loss = total_loss.item() / self.sizes['train'] # Average loss per doc.
            # logger.info(f"Skipped {self.train_loader.dataset.skipped} docs.")
        logger.info(f"Training Epoch {self.current_epoch + 1} --- Loss = {train_loss:.4}")
        writer.add_scalar('Loss/train', train_loss, self.current_epoch)

    def validate(self, model, writer) -> Tuple[float, float, float]:
        model.eval()
        thresholds = np.arange(0, 1, 0.05)
        scores = {k: [] for k in thresholds} # (pk, windowdiff) scores for each threshold.
        total_loss = torch.tensor(0.0, device=device)

        with tqdm(desc=f"Validating #{self.current_epoch}", total=self.sizes['val'], leave=False) as pbar:
            with torch.inference_mode():
                for sents, targets, doc_lengths in self.val_loader:
                    if pbar.n > self.sizes['val']:
                        break
                    if self.use_cuda:
                        sents, targets, doc_lengths = self.move_to_cuda(sents, targets, doc_lengths)
                    output = model(sents)
                    del sents
                    output, targets, doc_lengths = self.remove_last(output, targets, doc_lengths)

                    loss = model.criterion(output, targets)
                    total_loss += (loss.detach() * len(doc_lengths))
                    output_softmax = F.softmax(output, dim=1)

                    # Calculate the Pk and windowdiff per document (in this batch) and append to scores for each threshold.
                    for threshold in thresholds:
                        predictions = (output_softmax[:, 1] > threshold).long()
                        doc_start_idx = 0
                        for doc_len in doc_lengths:
                            pk, wd = compute_metrics(predictions[doc_start_idx:doc_start_idx+doc_len], targets[doc_start_idx:doc_start_idx+doc_len])
                            scores[threshold].append((pk, wd))
                            doc_start_idx += doc_len
                    
                    if pbar.n % 10 == 0:
                        pbar.set_postfix(loss=loss.item())
                    pbar.update(len(doc_lengths))
                    torch.cuda.empty_cache()

            val_loss = total_loss.item() / self.sizes['val'] # Average loss per doc.
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

    def test(self, model, threshold: float, writer: Optional[SummaryWriter] = None, return_acc=False) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        model.eval()
        scores = []
        with tqdm(desc=f'Testing #{self.current_epoch}', total=self.sizes['test'], leave=False) as pbar:
            with torch.inference_mode():
                for sents, targets, doc_lengths in self.test_loader:
                    if pbar.n > self.sizes['test']:
                        break
                    if self.use_cuda:
                        sents, targets, doc_lengths = self.move_to_cuda(sents, targets, doc_lengths)
                    output = model(sents)
                    del sents
                    output, targets, doc_lengths = self.remove_last(output, targets, doc_lengths)
                    output_softmax = F.softmax(output, dim=1)

                    # Calculate the Pk and windowdiff per document (in this batch) for the specified threshold.
                    predictions = (output_softmax[:, 1] > threshold).long()
                    doc_start_idx = 0
                    for doc_len in doc_lengths:
                        res = compute_metrics(predictions[doc_start_idx:doc_start_idx+doc_len], targets[doc_start_idx:doc_start_idx+doc_len], return_acc=return_acc)
                        scores.append(res)
                        doc_start_idx += doc_len

                    pbar.update(len(doc_lengths))
                    torch.cuda.empty_cache()

        epoch_avg = np.mean(scores, axis=0) # (pk, windowdiff) average for this threshold, over all docs.
        logger.info(f"Testing Epoch {self.current_epoch + 1} --- For threshold = {threshold:.4}, Pk = {epoch_avg[0]:.4}, windowdiff = {epoch_avg[1]:.4}")
        if writer:
            writer.add_scalar('pk_score/test', epoch_avg[0], self.current_epoch)
            writer.add_scalar('wd_score/test', epoch_avg[1], self.current_epoch)
        return (epoch_avg[0], epoch_avg[1], epoch_avg[2]) if return_acc else (epoch_avg[0], epoch_avg[1])
    

    def remove_last(self, x: torch.Tensor, y: torch.LongTensor, lengths: torch.LongTensor) -> Tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
        """For each document in the batch, remove the last sentence prediction/label (as it's unnecessary)."""
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

    @staticmethod
    def move_to_cuda(*tensors: torch.TensorType) -> Generator[torch.TensorType, None, None]:
        """Moves all given tensors to device."""
        for t in tensors:
            yield t.to(device, non_blocking=True)

    @staticmethod
    def custom_collate(batch) -> Tuple[torch.Tensor, torch.LongTensor, torch.LongTensor]:
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
    parser.add_argument('--epochs', help='Number of epochs to run', type=int)
    parser.add_argument('--load_from', help='Where to load an existing model from', type=str)
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=16)

    main(parser.parse_args())
