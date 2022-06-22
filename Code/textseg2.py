import logging, os
from datetime import datetime
from textseg2_model import create_model2, supervised_cross_entropy
from textseg import TextSeg, TS_Dataset
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, List, Optional, Tuple, Union
from tqdm import tqdm
from nltk import regexp_tokenize, sent_tokenize
import numpy as np
from transformers import logging as tlogging
from transformers import BertTokenizer, BertModel
rng = np.random.default_rng()
from utils import get_all_file_names, sectioned_clean_text, compute_metrics, LoggingHandler, clean_text, word_tokenize

W2V_NL_PATH = 'text_segmentation/Datasets/word2vec-nl-combined-160.txt' # Taken from https://github.com/clips/dutchembeddings
EARLY_STOP = 4
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
        # TODO: precalculate embeddings
        language = kwargs.get('language')
        if bert_name:
            bert_tokenizer = BertTokenizer.from_pretrained(bert_name)
            bert_model = BertModel.from_pretrained(bert_name)
        elif language:
            if language == 'en' or language == 'test':
                bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                bert_model = BertModel.from_pretrained('bert-base-uncased')
            elif language == 'nl':
                bert_tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
                bert_model = BertModel.from_pretrained('GroNLP/bert-base-dutch-cased')
            else:
                raise ValueError(f"Language {language} not supported.")
        super().__init__(**kwargs)
        if self.use_cuda:
            # bert_tokenizer = bert_tokenizer.to(device)
            bert_model = bert_model.to(device)
        TS_Dataset2.bert_tokenizer = bert_tokenizer
        TS_Dataset2.bert_model = bert_model
        logger.info(f"Loaded BERT model: '{bert_model.name_or_path}'.")
    
    # Override
    def load_data(
        self,
        dataset_path: str,
        from_wiki: bool,
        splits: List[float],
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().load_data(dataset_path, from_wiki, splits, batch_size, num_workers, TS_Dataset2, custom_collate)

    # Override
    def initialize_run(self, resume=False, model_creator: Callable = create_model2):
        return super().initialize_run(resume, model_creator)

    # Override
    def train_old(self, model, optimizer, writer) -> None:
        model.train()
        total_loss = 0.0
        with tqdm(desc=f'Training #{self.current_epoch}', total=len(self.train_loader.dataset), leave=False) as pbar:
            for data, labels, bert_sents, doc_lengths in self.train_loader:
                data, bert_sents = data.to(device), bert_sents.to(device)
                model.zero_grad()
                output, sim_scores = model(data, bert_sents, doc_lengths)
                del data, bert_sents
                torch.cuda.empty_cache()
                #TODO:
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
        train_loss = total_loss / len(self.train_loader) # Average loss per batch.
        logger.info(f"Training Epoch {self.current_epoch + 1} --- Loss = {train_loss:.4}")
        writer.add_scalar('Loss/train', train_loss, self.current_epoch + 1)

    # Override
    def train(self, model, optimizer, writer) -> None:
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        with tqdm(desc=f'Training #{self.current_epoch}', total=min(len(self.train_loader.dataset), self.subsets['train']), leave=False) as pbar:
            for sents, targets, bert_sents, doc_lengths in self.train_loader:
                if pbar.n > self.subsets['train']:
                    break
                if self.use_cuda:
                    sents = sents.to(device, non_blocking=True)
                    bert_sents = bert_sents.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    doc_lengths = doc_lengths.to(device, non_blocking=True)
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
    def validate(self, model, writer) -> Tuple[float, float, float]:
        model.eval()
        thresholds = np.arange(0, 1, 0.05)
        scores = {k: [] for k in thresholds} # (pk, windowdiff) scores for each threshold.
        total_loss = torch.tensor(0.0, device=device)
        with tqdm(desc=f"Validating #{self.current_epoch}", total=min(len(self.val_loader.dataset), self.subsets['val']), leave=False) as pbar:
            with torch.no_grad():
                for sents, targets, bert_sents, doc_lengths in self.val_loader:
                    if pbar.n > self.subsets['val']:
                        break
                    if self.use_cuda:
                        sents = sents.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        doc_lengths = doc_lengths.to(device, non_blocking=True)
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
    def test(self, model, threshold: float, writer: Optional[SummaryWriter]) -> Tuple[float, float]:
        model.eval()
        scores = []
        with tqdm(desc=f'Testing #{self.current_epoch}', total=min(len(self.test_loader.dataset), self.subsets['test']), leave=False) as pbar:
            with torch.no_grad():
                for sents, targets, bert_sents, doc_lengths in self.test_loader:
                    if pbar.n > self.subsets['test']:
                        break
                    if self.use_cuda:
                        sents = sents.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        doc_lengths = doc_lengths.to(device, non_blocking=True)
                    output, _ = model(sents, bert_sents, doc_lengths)
                    del sents, bert_sents
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


def custom_collate(batch) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.LongTensor]:
    """
    Prepare the batch for the model (so we can use data of varying lengths).
    Follows original implementation.
    https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    """
    all_sents = []
    batched_targets = np.array([])
    all_bert = []
    doc_lengths = torch.LongTensor([])
    for data, targets, bert in batch:
        for i in range(data.shape[0]):
            all_sents.append(data[i][data[i].sum(dim=1) != 0]) # remove padding
        batched_targets = np.concatenate((batched_targets, targets))
        all_bert.append(bert)
        doc_lengths = torch.cat((doc_lengths, torch.LongTensor([len(targets)])))

    packed_sents = pack_sequence(all_sents, enforce_sorted=False)
    if batched_targets.dtype == float:
        batched_targets = torch.from_numpy(batched_targets).long()
    return packed_sents, batched_targets, torch.cat(all_bert), doc_lengths

class TS_Dataset2(TS_Dataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    # Override
    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, np.ndarray, torch.FloatTensor]:
        """
        For a document at index, returns:
            data: List of n sentence embeddings (made out of word vectors).

            targets: List of n target labels [0 or 1]; a 1 signifies the END of a section (hence the last sentence/label is discarded later).
             OR
            raw_text: List containing the text of the sentences of the document.

            bert_sents: List of n BERT sentence embeddings.
        """
        text = self.get_text(index)
        data = []
        targets = np.array([], dtype=int)
        raw_text = np.array([])
        
        sections = sectioned_clean_text(text, from_wiki=self.from_wiki)
        suitable_sections_count = 0
        for section in sections:
            sentences = sent_tokenize(section)
            if not (TS_Dataset2.MAX_SECTION_LEN > len(sentences) > 1): # Filter out empty or too long sections.
                if len(sections) <= 2: # Skip docs that end up with just a single section
                    break
                continue
            
            section_len = 0
            for sentence in sentences:
                sentence_words = word_tokenize(sentence)
                if len(sentence_words) > 0:
                    sentence_emb = self.embed_words(sentence)
                    if len(sentence_emb) > 0:
                        section_len += 1
                        data.append(sentence_emb)
                        raw_text = np.append(raw_text, sentence)
                    
            if section_len > 0:
                sentence_labels = np.zeros(section_len, dtype=int)
                sentence_labels[-1] = 1 # Last sentence ends a section --> 1.
                targets = np.append(targets, sentence_labels)
                suitable_sections_count += 1
            
        # Get a random document if the current one is empty or has less than two suitable sections.
        if len(data) == 0 or suitable_sections_count < 2:
            return self.__getitem__(rng.integers(0, len(self.texts)))

        bert_data = self.bert_embed(raw_text)
        data = pad_sequence(data, batch_first=True) # (num_sents, max_sent_len, embed_dim)
        return (data, targets, bert_data) if self.labeled else (data, raw_text, bert_data)

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
            sentences = sent_tokenize(section)
            if not (TS_Dataset.MAX_SECTION_LEN > len(sentences) > 1): # Filter too short or too long sections.
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
        text = self.get_text(index)
        data = []
        raw_text = np.array([])
        bert_mask = np.array([], dtype=bool) # Keeps track of which sentences are kept

        text = clean_text(text, from_wiki=self.from_wiki)
        sentences = sent_tokenize(text)
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


def _bert_embed(self, sentences: Union[str, List[str], np.ndarray], batch_size=4) -> torch.FloatTensor:
    """
    Returns the BERT embedding(s) of a sentence/list of sentences.
    This follows the same procedure as the bert-as-service module (which was originally used).
    """
    if isinstance(sentences, str):
        sentences = [sentences]
    elif isinstance(sentences, np.ndarray):
        sentences = sentences.tolist()
    bert_sents = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:min(i+batch_size, len(sentences))]
        tokenized = TS_Dataset2.bert_tokenizer(batch, padding=True, truncation=True, max_length=TS_Dataset2.MAX_SECTION_LEN, return_tensors='pt').to(device)
        # Hidden layers have a shape of (batch_size, max_seq_len, 768) - only the second-to-last layer is used.
        output_layer = TS_Dataset2.bert_model(**tokenized, output_hidden_states=True).hidden_states[-2]
        avg_pool = torch.nn.AvgPool2d(kernel_size=(output_layer.shape[1], 1)) # Take the average of the layer on the time (sequence) axis.
        bert_sents.append(avg_pool(output_layer).squeeze(dim=1).detach())

    return torch.cat(bert_sents).cpu() # shape = (len(sentences), 768)
