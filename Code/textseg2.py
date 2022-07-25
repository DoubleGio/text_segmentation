import logging, argparse, torch
from models import create_TS2_model, supervised_cross_entropy
from textseg import TextSeg
from TS_Dataset import TS_Dataset2
from TS_Pipeline import TS_Pipeline
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, List, Optional, Tuple, Dict, Any, Union, Type, Generator
from tqdm import tqdm
import numpy as np
from transformers import logging as tlogging
from transformers import BertTokenizerFast, BertModel, BatchEncoding
from gensim.models import KeyedVectors
rng = np.random.default_rng()
from utils import SECTION_MARK, sent_tokenize_plus, get_all_file_names, sectioned_clean_text, compute_metrics, LoggingHandler, clean_text, word_tokenize

W2V_NL_PATH = 'Datasets/word2vec-nl-combined-160.txt' # Taken from https://github.com/clips/dutchembeddings
EARLY_STOP = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(LoggingHandler(use_tqdm=True))
tlogging.set_verbosity_error()


class TextSeg2(TextSeg):
    """
    Updated implementation of https://github.com/koomri/text-segmentation.
    Extended with the enhancements from https://github.com/lxing532/improve_topic_seg.
    """
  
    def __init__(self, bert_name: Optional[str] = None, **kwargs) -> None:
        if bert_name:
            # 'bert-base-uncased' = EN
            # 'GroNLP/bert-base-dutch-cased' = NL
            self.load_data = self.load_pipeline(bert_name)
        if kwargs.get('quiet', False):
            logger.setLevel(logging.WARNING)
        super().__init__(**kwargs)

    def load_pipeline(self, bert_name: str) -> Callable:
        bert_tokenizer = BertTokenizerFast.from_pretrained(bert_name)
        bert_model = BertModel.from_pretrained(bert_name)

        def _load_pipeline(
            dataset_path: str,
            from_wiki: bool,
            splits: List[float],
            batch_size: int,
            num_workers: int,
            word2vec: Optional[KeyedVectors] = None,
            subset = np.inf
        ) -> None:
            self.dataset_name = dataset_path.split('/')[-2]
            path_list = get_all_file_names(dataset_path)

            # Split and shuffle the data
            train_paths, test_paths = train_test_split(path_list, test_size=1-splits[0])
            val_paths, test_paths = train_test_split(test_paths, test_size=splits[1]/(splits[1]+splits[2]))

            pipe = Textseg2Pipeline(word2vec, bert_tokenizer, bert_model, device, batch_size, num_workers, from_wiki=from_wiki)
            self.train_loader = pipe(train_paths)
            self.val_loader = pipe(val_paths)
            self.test_loader = pipe(test_paths)

            if subset < len(path_list):
                self.sizes = {'train': int(subset * splits[0]), 'val': int(subset * splits[1]), 'test': int(subset * splits[2])}
                logger.info(f"Subsets: train {self.sizes['train']}, validate {self.sizes['val']}, test {self.sizes['test']}")
            else:
                self.sizes = {'train': len(self.train_loader), 'val': len(self.val_loader), 'test': len(self.val_loader)}
            
        return _load_pipeline

    # Override
    def init_dataset_class(self, **kwargs) -> Type[TS_Dataset2]:
        return super().init_dataset_class(**kwargs, dataset_class=TS_Dataset2)

    # Override
    def get_dataset_class() -> Type[TS_Dataset2]:
        return TS_Dataset2

    # Override
    def initialize_run(self, resume=False, model_creator: Callable = create_TS2_model):
        return super().initialize_run(resume, model_creator)

    # Override
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
            pipe = Textseg2Pipeline(TS_Dataset2.word2vec, bert_tokenizer, bert_model, device, batch_size, num_workers, from_wiki=from_wiki)
            self.test_loader = pipe(texts)
            self.sizes = {'test': len(self.test_loader)}
        state = torch.load(self.load_from)
        model = create_TS2_model(input_size=self.vec_size, set_device=device)
        model.load_state_dict(state['state_dict'])
        logger.info(f"Loaded model from {self.load_from}.")
        return self.test(model, threshold if threshold else state['threshold'], return_acc=True)

    # Override
    def segment_texts(self, texts: List[str], threshold: Optional[float] = None, from_wiki=False) -> Generator[str, None, None]:
        """
        Load a model and segment text(s).
        """
        if self.load_from is None:
            raise ValueError("Can't segment without a load_from path.")
        if texts is not None:
            if self.language == 'nl':
                bert_name = 'GroNLP/bert-base-dutch-cased'
            else:
                bert_name = 'bert-base-uncased'
            bert_tokenizer = BertTokenizerFast.from_pretrained(bert_name)
            bert_model = BertModel.from_pretrained(bert_name)
            pipe = Textseg2Pipeline(TS_Dataset2.word2vec, bert_tokenizer, bert_model, device, 2, 0, from_wiki=from_wiki, labeled=False)
            text_loader = pipe(texts)
            logger.info(f"Loaded {len(text_loader)} texts.")

        state = torch.load(self.load_from)
        model = create_TS2_model(input_size=self.vec_size, set_device=device)
        model.load_state_dict(state['state_dict'])
        model.eval()
        logger.info(f"Loaded model from {self.load_from}.")
        if threshold is None:
            threshold = state['threshold']
        logger.info(f"Using threshold {threshold:.2}.")
        del state

        # segmented_texts = []
        with tqdm(desc='Segmenting', total=len(text_loader), leave=False) as pbar:
            with torch.inference_mode():
                for bert_sents, sents, texts, doc_lengths in text_loader:
                    if sents is None:
                        # segmented_texts.append(None)
                        pbar.update(text_loader.batch_size)
                        continue
                    if self.use_cuda:
                        bert_sents, sents, doc_lengths = self.move_to_cuda(bert_sents, sents, doc_lengths)
                    output, _ = model(sents, bert_sents, doc_lengths)
                    del sents, bert_sents
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
                        # segmented_texts.append(segmented_text)
                        yield segmented_text
                        pbar.update(1)
                    torch.cuda.empty_cache()

    # Override
    def train(self, model, optimizer, writer) -> None:
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        with tqdm(desc=f'Training #{self.current_epoch}', total=self.sizes['train'], leave=False) as pbar:
            for bert_sents, sents, targets, doc_lengths in self.train_loader:
                if pbar.n > self.sizes['train']:
                    break
                if self.use_cuda:
                    sents, targets, bert_sents, doc_lengths = self.move_to_cuda(sents, targets, bert_sents, doc_lengths)
                model.zero_grad()
                output, sim_scores = model(sents, bert_sents, doc_lengths)
                del sents, bert_sents
                output, targets, doc_lengths = self.remove_last(output, targets, doc_lengths)

                loss = supervised_cross_entropy(output, sim_scores, targets)
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

    # Override
    def validate(self, model, writer: SummaryWriter) -> Tuple[float, float, float]:
        model.eval()
        thresholds = np.arange(0, 1, 0.05)
        scores = {k: [] for k in thresholds} # (pk, windowdiff) scores for each threshold.
        total_loss = torch.tensor(0.0, device=device)
        with tqdm(desc=f"Validating #{self.current_epoch}", total=self.sizes['val'], leave=False) as pbar:
            with torch.inference_mode():
                for bert_sents, sents, targets, doc_lengths in self.val_loader:
                    if pbar.n > self.sizes['val']:
                        break
                    if self.use_cuda:
                        sents, targets, bert_sents, doc_lengths = self.move_to_cuda(sents, targets, bert_sents, doc_lengths)
                    output, sim_scores = model(sents, bert_sents, doc_lengths)
                    del sents, bert_sents
                    output, targets, doc_lengths = self.remove_last(output, targets, doc_lengths)

                    loss = supervised_cross_entropy(output, sim_scores, targets)
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

    # Override
    def test(self, model, threshold: float, writer: Optional[SummaryWriter] = None, return_acc=False) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        model.eval()
        scores = []
        with tqdm(desc=f'Testing #{self.current_epoch}', total=self.sizes['test'], leave=False) as pbar:
            with torch.inference_mode():
                for bert_sents, sents, targets, doc_lengths in self.test_loader:
                    if pbar.n > self.sizes['test']:
                        break
                    if self.use_cuda:
                        sents, targets, bert_sents, doc_lengths = self.move_to_cuda(sents, targets, bert_sents, doc_lengths)
                    output, _ = model(sents, bert_sents, doc_lengths)
                    del sents, bert_sents
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

    # Override
    @staticmethod
    def custom_collate(batch) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.LongTensor]:
        """
        Prepare the batch for the model (so we can use data of varying lengths).
        Follows original implementation.
        https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
        """
        all_bert = []
        all_data = []
        all_targets = np.array([])
        doc_lengths = torch.LongTensor([])
        for data, targets, bert in batch:
            for i in range(data.shape[0]):
                all_data.append(data[i][data[i].sum(dim=1) != 0]) # remove padding
            all_targets = np.concatenate((all_targets, targets))
            all_bert.append(bert)
            doc_lengths = torch.cat((doc_lengths, torch.LongTensor([len(targets)])))

        all_data = pack_sequence(all_data, enforce_sorted=False)
        if all_targets.dtype == float:
            all_targets = torch.from_numpy(all_targets).long()
        return torch.cat(all_bert), all_data, all_targets, doc_lengths


class Textseg2Pipeline(TS_Pipeline):
    def __init__(self, word2vec: Optional[KeyedVectors] = None, *args, **kwargs) -> None:
        self.word2vec = word2vec
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(
        self,
        from_wiki: Optional[bool] = None,
        max_sent_len: Optional[int] = None,
        max_sec_len: Optional[int] = None,
        labeled: Optional[bool] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        preprocess_params, forward_params = {}, {}
        if from_wiki is not None: preprocess_params["from_wiki"] = from_wiki
        if max_sent_len is not None: preprocess_params["max_sent_len"] = max_sent_len
        if max_sec_len is not None: preprocess_params["max_sec_len"] = max_sec_len
        if labeled is not None: preprocess_params["labeled"] = labeled
        return preprocess_params, forward_params

    def preprocess(self, path: str, from_wiki = False, max_sent_len = 30, max_sec_len = 70, labeled = True) -> Tuple[Dict[str, torch.TensorType], torch.FloatTensor, np.ndarray]:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        if labeled:
            return self.get_data_targets(text, from_wiki, max_sent_len, max_sec_len)
        else:
            return self.get_data_raw(text, from_wiki, max_sent_len)

    def get_data_targets(self, text:str, from_wiki: bool, max_sent_len: int, max_sec_len: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, np.ndarray]:
        data = []
        raw_text = []
        targets = np.array([], dtype=int)

        sections = sectioned_clean_text(text, from_wiki=from_wiki)
        suitable_sections_count = 0
        for section in sections:
            sentences = sent_tokenize_plus(section)
            if not (max_sec_len > len(sentences) > 1): # Filter too short or too long sections.
                if len(sections) <= 2: # Skip docs that end up with just a single section
                    break
                continue
            
            section_sentences = []
            for sentence in sentences:
                sentence_words = word_tokenize(sentence)
                if len(sentence_words) > 0:
                    sentence_emb = self.embed_words(sentence_words)
                    if len(sentence_emb) > 0:
                        section_sentences.append(sentence)
                        data.append(sentence_emb)
                        continue
                    
            if len(section_sentences) > 0:
                raw_text += section_sentences
                sentence_labels = np.zeros(len(section_sentences), dtype=int)
                sentence_labels[-1] = 1 # Last sentence ends a section --> 1.
                targets = np.append(targets, sentence_labels)
                suitable_sections_count += 1

        # Get a random document if the current one is empty or has less than two suitable sections.
        if len(data) == 0 or suitable_sections_count < 2:
            return None

        model_inputs = self.tokenizer(
            text=raw_text,
            padding=True,
            truncation=True,
            max_length=max_sent_len,
            return_tensors='pt'
        )
        data = pad_sequence(data, batch_first=True) # (num_sents, max_sent_len, embed_dim)
        return model_inputs, data, targets

    def get_data_raw(self, text:str, from_wiki: bool, max_sent_len: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, np.ndarray]:
        data = []
        raw_text = []
        text = clean_text(text, from_wiki=from_wiki)
        sentences = sent_tokenize_plus(text)
        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            if len(sentence_words) > 0:
                sentence_emb = self.embed_words(sentence_words)
                if len(sentence_emb) > 0:
                    data.append(sentence_emb)
                    raw_text.append(sentence)

        if len(data) == 0:
            return None
        
        model_inputs = self.tokenizer(
            text=raw_text,
            padding=True,
            truncation=True,
            max_length=max_sent_len,
            return_tensors='pt'
        )
        data = pad_sequence(data, batch_first=True) # (num_sents, max_sent_len, embed_dim)
        return model_inputs, data, np.array(raw_text)

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

    def _forward(self, input_tensors: Dict[str, torch.TensorType]) -> torch.TensorType:
        output_layer = self.model(**input_tensors, output_hidden_states=True).hidden_states[-2]
        avg_pool = torch.nn.AvgPool2d(kernel_size=(output_layer.shape[1], 1)) # Take the average of the layer on the time (sequence) axis.
        return avg_pool(output_layer).squeeze(dim=1).detach()

    @staticmethod
    def no_collate_fn(item):
        item = item[0]
        model_input, data, targets = item
        if targets.dtype == int:
            targets = torch.from_numpy(targets)
        return model_input, data, targets, torch.LongTensor([len(targets)])
        
    @staticmethod
    def cat_collate_fn(tokenizer) -> Callable:
        """Returns the cat_collate function2."""
        t_padding_value = tokenizer.pad_token_id

        def cat_collate(items) -> Tuple[torch.TensorType, torch.TensorType, torch.TensorType]:
            """Concatenates all items together and addiotnally returns the length of the individual items."""
            all_model_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
            all_data = []
            all_targets = np.array([])
            doc_lengths = torch.LongTensor([])
            max_length = max(item[0]['input_ids'].shape[1] for item in items)
            for model_input, data, targets in items:
                all_targets = np.concatenate((all_targets, targets), axis=0)
                for i in range(data.shape[0]):
                    all_data.append(data[i][data[i].sum(dim=1) != 0]) # remove padding
                doc_lengths = torch.cat((doc_lengths, torch.LongTensor([len(targets)])), dim=0)
                
                for key, value in model_input.items():
                    if key == 'input_ids':
                        if value.shape[1] < max_length:
                            value = torch.cat((value, torch.zeros(value.shape[0], max_length - value.shape[1], dtype=int) + t_padding_value), dim=1)
                    else: # key == 'attention_mask' or key == 'token_type_ids'
                        if value.shape[1] < max_length:
                            value = torch.cat((value, torch.zeros(value.shape[0], max_length - value.shape[1], dtype=int)), dim=1)
                    all_model_inputs[key].append(value)

            all_data = pack_sequence(all_data, enforce_sorted=False)
            for key, value in all_model_inputs.items():
                all_model_inputs[key] = torch.cat(value, dim=0)
            if all_targets.dtype == float:
                all_targets = torch.from_numpy(all_targets).long()
            return BatchEncoding(all_model_inputs), all_data, all_targets, doc_lengths
            
        return cat_collate

def main(args):
    ts = TextSeg2(
        bert_name=args.bert_name,
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
