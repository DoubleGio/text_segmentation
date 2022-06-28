"""
[pdf](https://arxiv.org/pdf/2110.07160.pdf)
Transformer² model (without S_single & L_topic):
    1. Obtain the sentence embeddings (using pretrained BERT).
        >>> sentence1 = 'sentence one'; sentence2 = 'yet another one'
        1.1. Pairwise tokenize (skip last sentence I guess?): 
            >>> tokens = tokenizer(text=sentence1, textpair=sentence2, padding=True, return_tensors='pt')
            >>> ['[CLS]', 'sentence', 'one', '[SEP]', 'yet', 'another', 'one', '[SEP]']
        1.2. Get sentence embedding from CLS token:
            >>> out = model(**tokens)
            >>> out.last_hidden_state[:, 0, :] # (sentences, tokens, hidden_size)
    2. Train a transformer model to classify whether each sentence is a segment boundary.
        2.1. Create a transformer model:
            >>> y_seg = Sigmoid(Linear2(TransformerΘ(S)))
            >>> Θ = {'nhead': 24, 'num_encoder_layers': 5, 'dim_feedforward': 1024}
        2.2. Create a loss function:
            >>> loss_fn = binary cross entropy loss
        'For the segmentation predictions, 70% of the inner sentences were randomly masked,
        while all the begin sentences were not masked in order to address the imbalance class problem.'
        basically, remove 70% of the non-boundary sentences for training.
"""
import os, logging, torch, gc
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple, Any, Callable, Generator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
from transformers import logging as tlogging
from TS_Pipeline import TS_Pipeline
# from transformers2_model import create_T2model
from models import create_T2model
from utils import get_all_file_names, sectioned_clean_text, sent_tokenize_plus, LoggingHandler

EARLY_STOP = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(LoggingHandler(use_tqdm=True))
tlogging.set_verbosity_error()

class Transformer2:

    def __init__(
        self,
        language: Optional[str] = None,
        bert_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        from_wiki: Optional[bool] = None,
        splits = [0.8, 0.1, 0.1],
        batch_size = 8,
        num_workers = 0,
        use_cuda = True,
        load_from: Optional[str] = None,
        subset: Optional[int] = None,
    ) -> None:
        if bert_name:
            bert_tokenizer = BertTokenizerFast.from_pretrained(bert_name)
            bert_model = BertModel.from_pretrained(bert_name)
        elif language:
            if language == 'en' or language == 'test':
                bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
                bert_model = BertModel.from_pretrained('bert-base-uncased')
            elif language == 'nl':
                bert_tokenizer = BertTokenizerFast.from_pretrained('GroNLP/bert-base-dutch-cased')
                bert_model = BertModel.from_pretrained('GroNLP/bert-base-dutch-cased')
            else:
                raise ValueError(f"Language {language} not supported.")

        if not use_cuda:
            global device 
            device = torch.device('cpu')
            logger.info(f"Using device: {device}.")
            self.use_cuda = False
        else:
            device_name = torch.cuda.get_device_name(device)
            logger.info(f"Using device: {device_name}.")
            self.use_cuda = False if device_name == 'cpu' else True
        
        if dataset_path:
            self.subsets = {'train': int(subset * splits[0]), 'val': int(subset * splits[1]), 'test': int(subset * splits[2])} if subset else None
            if from_wiki is None:
                from_wiki = "wiki" in dataset_path.lower()
            self.load_data(dataset_path, from_wiki, splits, batch_size, num_workers, bert_tokenizer, bert_model)
        self.load_from = load_from
        self.current_epoch = 0

    def load_data(self, dataset_path, from_wiki, splits, batch_size, num_workers, bert_tokenizer, bert_model):
        self.dataset_name = dataset_path.split('/')[-2]
        path_list = get_all_file_names(dataset_path)

        # Split and shuffle the data
        train_paths, test_paths = train_test_split(path_list, test_size=1-splits[0])
        val_paths, test_paths = train_test_split(test_paths, test_size=splits[1]/(splits[1]+splits[2]))

        pipe = Transformer2Pipeline(bert_tokenizer, bert_model, device, batch_size, num_workers, from_wiki=from_wiki)
        self.train_loader = pipe(train_paths)
        self.val_loader = pipe(val_paths)
        self.test_loader = pipe(test_paths)

    def initialize_run(self, resume=False):
        cname = self.__class__.__name__.lower()
        if resume:
            if self.load_from is None:
                raise ValueError("Can't resume without a load_from path.")
            state = torch.load(self.load_from)
            logger.info(f"Loaded state from {self.load_from}.")
            now = state['now']
            writer = SummaryWriter(log_dir=f'runs/{cname}/{self.dataset_name}_{now}')
            checkpoint_path = os.path.join(f'checkpoints/{cname}/{self.dataset_name}_{now}')
            
            model = create_T2model(input_size=self.vec_size, set_device=device)
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

            model = create_T2model(input_size=self.vec_size, set_device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            best_val_scores = [np.inf, np.inf] # (pk, windowdiff)
            non_improvement_count = 0
        return now, writer, checkpoint_path, model, optimizer, best_val_scores, non_improvement_count

    def run(self, epochs=10, resume=False):
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
                gc.collect()
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

    def train(self, model, optimizer, writer) -> None:
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        with tqdm(desc=f'Training #{self.current_epoch}', total=self.sizes['train'], leave=False) as pbar:
            for sent_embeddings, targets, doc_lengths in self.train_loader:
                if pbar.n > self.sizes['train']:
                    break
                if self.use_cuda:
                    sent_embeddings, targets, doc_lengths = self.move_to_cuda(sent_embeddings, targets, doc_lengths)
                model.zero_grad()
                output = model(sent_embeddings, targets)
                # del sents, bert_sents

                loss = model.criterion(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += (loss.detach() * len(doc_lengths))

                if pbar.n % 10 == 0:
                    pbar.set_postfix(loss=loss.item())
                pbar.update(len(doc_lengths))
                torch.cuda.empty_cache()

            train_loss = total_loss.item() / pbar.n # Average loss per doc.
        logger.info(f"Training Epoch {self.current_epoch + 1} --- Loss = {train_loss:.4}")
        writer.add_scalar('Loss/train', train_loss, self.current_epoch)

    def validate(self, model, writer) -> Tuple[float, float, float]:
        model.eval()
        thresholds = np.arange(0, 1, 0.05)
        scores = {k: [] for k in thresholds} # (pk, windowdiff) scores for each threshold.
        total_loss = torch.tensor(0.0, device=device)

        with tqdm(desc=f"Validating #{self.current_epoch}", total=self.sizes['val'], leave=False) as pbar:
            with torch.no_grad():
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

            val_loss = total_loss.item() / pbar.n # Average loss per doc.
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
        with tqdm(desc=f'Testing #{self.current_epoch}', total=self.sizes['test'], leave=False) as pbar:
            with torch.no_grad():
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
                        pk, wd = compute_metrics(predictions[doc_start_idx:doc_start_idx+doc_len], targets[doc_start_idx:doc_start_idx+doc_len])
                        scores.append((pk, wd))
                        doc_start_idx += doc_len

                    pbar.update(len(doc_lengths))
                    torch.cuda.empty_cache()

        epoch_pk, epoch_wd = np.mean(scores, axis=0) # (pk, windowdiff) average for this threshold, over all docs.
        logger.info(f"Testing Epoch {self.current_epoch + 1} --- For threshold = {threshold:.4}, Pk = {epoch_pk:.4}, windowdiff = {epoch_wd:.4}")
        if writer:
            writer.add_scalar('pk_score/test', epoch_pk, self.current_epoch)
            writer.add_scalar('wd_score/test', epoch_wd, self.current_epoch)
        return epoch_pk, epoch_wd
            
    
    @staticmethod
    def move_to_cuda(*tensors: torch.TensorType) -> Generator[torch.TensorType, None, None]:
        """Moves all given tensors to device."""
        for t in tensors:
            yield t.to(device, non_blocking=True)

class Transformer2Pipeline(TS_Pipeline):

    def _sanitize_parameters(self, from_wiki=False, max_sent_len=30) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        preprocess_params, forward_params = {}, {}
        preprocess_params["from_wiki"] = from_wiki
        preprocess_params["max_sent_len"] = max_sent_len
        return preprocess_params, forward_params

    def preprocess(self, path: str, from_wiki: bool, max_sent_len: int) -> Tuple[Dict[str, torch.TensorType], np.ndarray]:
        with open(path, 'r') as f:
            text = f.read()
        sections = sectioned_clean_text(text, from_wiki=from_wiki)
        sentences = []
        targets = np.array([], dtype=int)
        for section in sections:
            s = sent_tokenize_plus(section)
            sentences += s
            section_targets = np.zeros(len(s), dtype=int)
            section_targets[0] = 1
            targets = np.append(targets, section_targets)
        # Pairwise tokenizing (last sentence is skipped)
        model_inputs = self.tokenizer(
            text=sentences[:-1], 
            text_pair=sentences[1:], 
            padding=True, 
            truncation=True, 
            max_length=max_sent_len * 2, # *2 because of pairwise
            return_tensors='pt'
        )
        return model_inputs, targets[:-1]

    def _forward(self, input_tensors: Dict[str, torch.TensorType]) -> torch.TensorType:
        """Returns the [CLS] token."""
        return self.model(**input_tensors).last_hidden_state[:, 0, :] # (sentences, hidden_size)

if __name__ == "__main__":
    t = Transformer2(language='en', batch_size=3, dataset_path='text_segmentation/Datasets/ENWiki/data_subset', from_wiki=True)
    t.run()
