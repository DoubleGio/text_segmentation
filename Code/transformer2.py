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
import os, logging, torch, gc, argparse
import numpy as np
from datetime import datetime, timedelta
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple, Any, Callable, Generator, Union
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BatchEncoding, BertTokenizerFast, BertModel
from transformers import logging as tlogging
from TS_Pipeline import TS_Pipeline
from models import create_T2_model
from utils import compute_metrics, get_all_file_names, sectioned_clean_text, sent_tokenize_plus, LoggingHandler

EARLY_STOP = 3
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
            self.language = 'nl' if 'nl' in bert_name else 'en'
        elif language:
            if language == 'en' or language == 'test':
                bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
                bert_model = BertModel.from_pretrained('bert-base-uncased')
            elif language == 'nl':
                bert_tokenizer = BertTokenizerFast.from_pretrained('GroNLP/bert-base-dutch-cased')
                bert_model = BertModel.from_pretrained('GroNLP/bert-base-dutch-cased')
            else:
                raise ValueError(f"Language {language} not supported.")
            self.language = language
        self.emb_size = bert_model.config.hidden_size

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
            if from_wiki is None:
                from_wiki = "wiki" in dataset_path.lower()
            self.load_data(dataset_path, from_wiki, splits, batch_size, num_workers, bert_tokenizer, bert_model, subset)
        self.load_from = load_from
        self.current_epoch = 0

    def load_data(
        self,
        dataset_path: str,
        from_wiki: bool,
        splits: List[float],
        batch_size: int,
        num_workers: int,
        bert_tokenizer,
        bert_model,
        subset = np.inf,
    ):
        self.dataset_name = dataset_path.split('/')[-2]
        path_list = get_all_file_names(dataset_path)

        # Split and shuffle the data
        train_paths, test_paths = train_test_split(path_list, test_size=1-splits[0])
        val_paths, test_paths = train_test_split(test_paths, test_size=splits[1]/(splits[1]+splits[2]))

        pipe = Transformer2Pipeline(bert_tokenizer, bert_model, device, batch_size, num_workers, from_wiki=from_wiki)
        self.train_loader = pipe(train_paths)
        self.val_loader = pipe(val_paths)
        self.test_loader = pipe(test_paths)
        logger.info(f"Loaded {len(self.train_loader)} training examples, {len(self.val_loader)} validation examples, and {len(self.test_loader)} test examples.")

        if subset < len(path_list):
            self.sizes = {'train': int(subset * splits[0]), 'val': int(subset * splits[1]), 'test': int(subset * splits[2])}
            logger.info(f"Subsets: train {self.sizes['train']}, validate {self.sizes['val']}, test {self.sizes['test']}")
        else:
            self.sizes = {'train': len(self.train_loader), 'val': len(self.val_loader), 'test': len(self.test_loader)}

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
            
            model = create_T2_model(input_size=self.vec_size, set_device=device)
            model.load_state_dict(state['state_dict'])
            logger.info(f"Loaded model from {self.load_from}.")

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            optimizer.load_state_dict(state['optimizer'])

            best_val_scores = state['best_val_scores']
            non_improvement_count = state['non_improvement_count']
            self.current_epoch = state['epoch'] + 1 # +1 because we're resuming from the epoch after the last one
        else:
            now = datetime.now().strftime(r'%b%d_%H%M')
            writer = SummaryWriter(log_dir=f'runs/{cname}/{self.dataset_name}_{now}')
            checkpoint_path = os.path.join(f'checkpoints/{cname}/{self.dataset_name}_{now}')
            os.makedirs(checkpoint_path, exist_ok=True)

            model = create_T2_model(input_size=self.emb_size, set_device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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
                val_pk, val_wd = self.validate(model, writer)
                gc.collect()
                if sum([val_pk, val_wd]) < sum(best_val_scores):
                    non_improvement_count = 0
                    test_scores = self.test(model, writer)
                    logger.result(f"Best model from Epoch {epoch} --- Pk = {test_scores[0]:.4}, windowdiff = {test_scores[1]:.4}")
                    best_val_scores = [val_pk, val_wd]
                    state = {
                        'now': now,
                        'epoch': epoch,
                        'non_improvement_count': non_improvement_count,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_scores': best_val_scores,
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
            if self.language == 'nl':
                bert_name = 'GroNLP/bert-base-dutch-cased'
            else:
                bert_name = 'bert-base-uncased'
            bert_tokenizer = BertTokenizerFast.from_pretrained(bert_name)
            bert_model = BertModel.from_pretrained(bert_name)
            pipe = Transformer2Pipeline(bert_tokenizer, bert_model, device, batch_size, num_workers, from_wiki=from_wiki)
            self.test_loader = pipe(texts)
            self.sizes = {'test': len(self.test_loader)}
        state = torch.load(self.load_from)
        model = create_T2_model(input_size=self.vec_size, set_device=device)
        model.load_state_dict(state['state_dict'])
        logger.info(f"Loaded model from {self.load_from}.")
        return self.test(model, threshold if threshold else state['threshold'], return_acc=True)

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
                output = model(sent_embeddings, doc_lengths)

                loss = model.criterion(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += (loss.detach() * len(doc_lengths))

                if pbar.n % 10 == 0:
                    pbar.set_postfix(loss=loss.item())
                pbar.update(len(doc_lengths))
                torch.cuda.empty_cache()

            train_loss = total_loss.item() / self.sizes['train'] # Average loss per doc.
        logger.info(f"Training Epoch {self.current_epoch} --- Loss = {train_loss:.4}")
        writer.add_scalar('Loss/train', train_loss, self.current_epoch)

    def validate(self, model, writer) -> Tuple[float, float]:
        model.eval()
        scores = []
        total_loss = torch.tensor(0.0, device=device)

        with tqdm(desc=f"Validating #{self.current_epoch}", total=self.sizes['val'], leave=False) as pbar:
            with torch.inference_mode():
                for sent_embeddings, targets, doc_lengths in self.val_loader:
                    if pbar.n > self.sizes['val']:
                        break
                    if self.use_cuda:
                        sent_embeddings, targets, doc_lengths = self.move_to_cuda(sent_embeddings, targets, doc_lengths)
                    output = model(sent_embeddings, doc_lengths)

                    loss = model.criterion(output, targets)
                    total_loss += (loss.detach() * len(doc_lengths))

                    predictions = output.argmax(dim=1)
                    doc_start_idx = 0
                    for doc_len in doc_lengths:
                        pk, wd = compute_metrics(predictions[doc_start_idx:doc_start_idx+doc_len], targets[doc_start_idx:doc_start_idx+doc_len])
                        scores.append((pk, wd))
                        doc_start_idx += doc_len

                    if pbar.n % 10 == 0:
                        pbar.set_postfix(loss=loss.item())
                    pbar.update(len(doc_lengths))
                    torch.cuda.empty_cache()

            val_loss = total_loss.item() / self.sizes['val'] # Average loss per doc.
        avg_pk_wd = np.mean(scores, axis=0)
        logger.info(f"Validation Epoch {self.current_epoch} --- Loss = {val_loss:.4}, Pk = {avg_pk_wd[0]:.4}, windowdiff = {avg_pk_wd[1]:.4}")
        writer.add_scalar('Loss/val', val_loss, self.current_epoch)
        writer.add_scalar('pk_score/val', avg_pk_wd[0], self.current_epoch)
        writer.add_scalar('wd_score/val', avg_pk_wd[1], self.current_epoch)
        return avg_pk_wd

    def test(self, model, writer: Optional[SummaryWriter] = None, return_acc=False) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        model.eval()
        scores = []
        with tqdm(desc=f'Testing #{self.current_epoch}', total=self.sizes['test'], leave=False) as pbar:
            with torch.inference_mode():
                for sent_embeddings, targets, doc_lengths in self.test_loader:
                    if pbar.n > self.sizes['test']:
                        break
                    if self.use_cuda:
                        sent_embeddings, targets, doc_lengths = self.move_to_cuda(sent_embeddings, targets, doc_lengths)
                    output = model(sent_embeddings, doc_lengths)
                    
                    predictions = output.argmax(dim=1)
                    doc_start_idx = 0
                    for doc_len in doc_lengths:
                        res = compute_metrics(predictions[doc_start_idx:doc_start_idx+doc_len], targets[doc_start_idx:doc_start_idx+doc_len], return_acc=return_acc)
                        scores.append(res)
                        doc_start_idx += doc_len

                    pbar.update(len(doc_lengths))
                    torch.cuda.empty_cache()

        epoch_avg = np.mean(scores, axis=0) # (pk, windowdiff) average for this threshold, over all docs.
        logger.info(f"Testing Epoch {self.current_epoch + 1} --- Pk = {epoch_avg[0]:.4}, windowdiff = {epoch_avg[1]:.4}")
        if writer:
            writer.add_scalar('pk_score/test', epoch_avg[0], self.current_epoch)
            writer.add_scalar('wd_score/test', epoch_avg[1], self.current_epoch)
        return (epoch_avg[0], epoch_avg[1], epoch_avg[2]) if return_acc else (epoch_avg[0], epoch_avg[1])
            
    
    @staticmethod
    def move_to_cuda(*tensors: torch.TensorType) -> Generator[torch.TensorType, None, None]:
        """Moves all given tensors to device."""
        for t in tensors:
            yield t.to(device, non_blocking=True)

class Transformer2Pipeline(TS_Pipeline):
    def _sanitize_parameters(self, from_wiki=False, max_sent_len=30, max_text_len = 300) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        preprocess_params, forward_params = {}, {}
        preprocess_params["from_wiki"] = from_wiki
        preprocess_params["max_sent_len"] = max_sent_len
        preprocess_params["max_text_len"] = max_text_len
        return preprocess_params, forward_params

    def preprocess(self, path: str, from_wiki: bool, max_sent_len: int, max_text_len: int) -> Tuple[Dict[str, torch.TensorType], np.ndarray]:
        with open(path, 'r') as f:
            text = f.read()
        sections = sectioned_clean_text(text, from_wiki=from_wiki)
        sentences = []
        targets = np.array([], dtype=int)
        for section in sections:
            s = sent_tokenize_plus(section)
            if not len(s) > 1: # Skip sections with only one sentence.
                if len(sections) <= 2: # Skip docs that end up with only one section.
                    break
                continue
            sentences += s
            section_targets = np.zeros(len(s), dtype=int)
            section_targets[0] = 1
            targets = np.append(targets, section_targets)

        if not (max_text_len > len(sentences) - 1 > 0):
            return None
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

    @staticmethod
    def no_collate_fn(item):
        item = item[0]
        return *item, torch.LongTensor([len(item[-1])]) # (*items, doc_length)

    @staticmethod
    def cat_collate_fn(tokenizer) -> Callable:
        """Returns the cat_collate function."""
        t_padding_value = tokenizer.pad_token_id

        def cat_collate(items) -> Tuple[torch.TensorType, torch.TensorType, torch.TensorType]:
            """Concatenates all items together and addiotnally returns the length of the individual items."""
            all_model_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
            all_targets = np.array([], dtype=int)
            doc_lengths = torch.LongTensor([])
            max_length = max(item[0]['input_ids'].shape[1] for item in items)
            for model_input, targets in items:
                all_targets = np.concatenate((all_targets, targets), axis=0)
                doc_lengths = torch.cat((doc_lengths, torch.LongTensor([len(targets)])), dim=0)
                for key, value in model_input.items():
                    if key == 'input_ids':
                        if value.shape[1] < max_length:
                            value = torch.cat((value, torch.zeros(value.shape[0], max_length - value.shape[1], dtype=int) + t_padding_value), dim=1)
                    else: # key == 'attention_mask' or key == 'token_type_ids'
                        if value.shape[1] < max_length:
                            value = torch.cat((value, torch.zeros(value.shape[0], max_length - value.shape[1], dtype=int)), dim=1)
                    all_model_inputs[key].append(value)
            
            for key, value in all_model_inputs.items():
                all_model_inputs[key] = torch.cat(value, dim=0)
            return BatchEncoding(all_model_inputs), torch.from_numpy(all_targets), doc_lengths
            
        return cat_collate

def main(args):
    t2 = Transformer2(
        language=args.lang,
        bert_name=args.bert_name,
        dataset_path=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cuda= not args.disable_cuda,
        load_from=args.load_from,
        subset=args.subset,
    )
    if args.test:
        t2.run_test()
    else:
        res = t2.run(args.epochs, args.resume)
    print(f'Best Pk = {res[0]} --- Best windowdiff = {res[1]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Test a Text Segmentation model.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test", action="store_true", help="Test mode.")
    mode.add_argument("--resume", action="store_true", help="Resume training from a checkpoint.")
    parser.add_argument("--bert_name", type=str, help="Name of the BERT model to use (if not using pre-embedded texts).")
    parser.add_argument("--lang", type=str, default="en", help="Language to use.")
    parser.add_argument("--data_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("--subset", type=int, help="Use only a subset of the dataset.")
    parser.add_argument("--disable_cuda", action="store_true", help="Disable cuda (if available).")
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--load_from', help='Where to load an existing model from', type=str, default=None)
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=8)

    main(parser.parse_args())
