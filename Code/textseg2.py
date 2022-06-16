import logging, os
from datetime import datetime
from textseg2_model import create_model2, supervised_cross_entropy
from textseg import TextSeg, TS_Dataset
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from nltk import regexp_tokenize, sent_tokenize
import numpy as np
from transformers import logging as tlogging
from transformers import BertTokenizer, BertModel
rng = np.random.default_rng()
from utils import get_all_file_names, sectioned_clean_text, compute_metrics, LoggingHandler, clean_text, word_tokenize

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

    def __init__(self, bert_name: Optional[str] = None, **kwargs) -> None:
        # TODO: precalculate embeddings
        language = kwargs.get('language')
        if bert_name:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_name)
            self.bert_model = BertModel.from_pretrained(bert_name)
        elif language:
            if language == 'en' or language == 'test':
                self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            elif language == 'nl':
                self.bert_tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
                self.bert_model = BertModel.from_pretrained('GroNLP/bert-base-dutch-cased')
            else:
                raise ValueError(f"Language {language} not supported.")
        logger.info(f"Loaded BERT model: '{self.bert_model.name_or_path}'.")
        super().__init__(**kwargs)
        
    # Override
    def load_data(
        self,
        dataset_path: str,
        from_wiki: bool,
        splits: List[float],
        batch_size: int,
        num_workers: int,
        subset: Optional[int] = None,
    ) -> None:
        """
        Load the right pretrained models.
        Shuffle and split the data, load it as TS_Datasets and put it into DataLoaders.
        """
        self.dataset_name = dataset_path.split('/')[-2]
        path_list = get_all_file_names(dataset_path)
        if subset:
            if subset < len(path_list):
                path_list = rng.choice(path_list, size=subset, replace=False).tolist()
        train_paths, test_paths = train_test_split(path_list, test_size=1-splits[0])
        dev_paths, test_paths = train_test_split(test_paths, test_size=splits[1]/(splits[1]+splits[2]))

        train_dataset = TS_Dataset2(texts=train_paths, word2vec=self.word2vec, bert_tokenizer=self.bert_tokenizer, bert_model=self.bert_model, from_wiki=from_wiki)
        val_dataset = TS_Dataset2(texts=dev_paths, word2vec=self.word2vec, bert_tokenizer=self.bert_tokenizer, bert_model=self.bert_model, from_wiki=from_wiki)
        test_dataset = TS_Dataset2(texts=test_paths, word2vec=self.word2vec, bert_tokenizer=self.bert_tokenizer, bert_model=self.bert_model, from_wiki=from_wiki)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=num_workers)
        logger.info(f"Loaded {len(train_dataset)} training examples, {len(val_dataset)} validation examples, and {len(test_dataset)} test examples.")

    # Override
    def initialize_run(self, resume=False):
        if resume:
            if self.load_from is None:
                raise ValueError("Can't resume without a load_from path.")
            state = torch.load(self.load_from)
            logger.info(f"Loaded state from {self.load_from}.")
            now = state['now']
            writer = SummaryWriter(log_dir=f'runs/textseg2/{self.dataset_name}_{now}')
            checkpoint_path = os.path.join(f'checkpoints/textseg2/{self.dataset_name}_{now}')
            
            model = create_model2(input_size=self.vec_size, set_device=device)
            model.load_state_dict(state['state_dict'])
            logger.info(f"Loaded model from {self.load_from}.")

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.load_state_dict(state['optimizer'])

            best_val_scores = state['best_val_scores']
            non_improvement_count = state['non_improvement_count']
            self.current_epoch = state['epoch']
        else:
            now = datetime.now().strftime(r'%b%d_%H%M')
            writer = SummaryWriter(log_dir=f'runs/textseg2/{self.dataset_name}_{now}')
            checkpoint_path = os.path.join(f'checkpoints/textseg2/{self.dataset_name}_{now}')
            os.makedirs(checkpoint_path, exist_ok=True)

            model = create_model2(input_size=self.vec_size, set_device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            best_val_scores = [np.inf, np.inf] # (pk, windowdiff)
            non_improvement_count = 0
        return now, writer, checkpoint_path, model, optimizer, best_val_scores, non_improvement_count


    # Override
    def train(self, model, optimizer, writer) -> None:
        model.train()
        total_loss = 0.0
        with tqdm(desc=f'Training #{self.current_epoch}', total=len(self.train_loader.dataset), leave=False) as pbar:
            for data, labels, bert_sents in self.train_loader:
                data, bert_sents = data.to(device), bert_sents.to(device)
                model.zero_grad()
                output, sim_scores = model(data, bert_sents)
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
    all_sents = []
    batched_targets = []
    batched_bert = []
    for data, targets, bert in batch:
        for sentence in data:
            all_sents.append(torch.FloatTensor(sentence))
        if type(targets[0]) == int: # If targets is a list, then it are the ground truths.
            batched_targets.append(torch.LongTensor(targets))
        else:                       # Else it is the raw text.
            batched_targets.append(targets)
        batched_bert.append(bert)
        
    # Pad and pack the sentences into a single tensor. 
    # This function sorts it, but with enfore_sorted=False it gets unsorted again after passing it through a model.
    packed_sents = pack_sequence(all_sents, enforce_sorted=False)
    return packed_sents, batched_targets, torch.cat(batched_bert)


    # b_data = []
    # b_targets = []
    # b_bert_sents = []
    # for data, targets, bert_sents in batch:
    #     tensored_data = []
    #     for sentence in data:
    #         tensored_data.append(torch.FloatTensor(np.concatenate(sentence)).to(device))
    #     b_data.append(tensored_data)
    #     tensored_targets = torch.LongTensor(targets).to(device)
    #     b_targets.append(tensored_targets)
    #     b_bert_sents.append(bert_sents)
    # return b_data, b_targets, b_bert_sents

class TS_Dataset2(TS_Dataset):
    def __init__(
        self,
        bert_tokenizer: BertTokenizer,
        bert_model: BertModel,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
    
    # Override
    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], Union[List[int], List[str]], torch.FloatTensor]:
        """
        For a document at index, returns:
            data: List of n sentence embeddings (made out of word vectors).
            labels: List of n labels [0 or 1]; 
                    A 1 signifies the END of a section (hence the last sentence/label is discarded later).
        """
        text = self.get_text(index)
        data = []
        labels = []
        raw_text = []
        
        sections = sectioned_clean_text(text, from_wiki=self.from_wiki)
        suitable_sections_count = 0
        for section in sections:
            sentences = sent_tokenize(section)
            if not (80 > len(sentences) > 0): # Filter out empty or too long sections.
                if len(sections) <= 2: # Skip docs that end up with just a single section
                    break
                continue
            
            section_sent_emb = []
            for sentence in sentences:
                sentence_words = word_tokenize(sentence)
                if len(sentence_words) > 0:
                    sentence_emb = self.embed_words(sentence)
                    if len(sentence_emb) > 0:
                        section_sent_emb.append(sentence_emb)
                        raw_text.append(sentence)
                    
            if len(section_sent_emb) > 0:
                data += section_sent_emb
                sentence_labels = np.zeros(len(section_sent_emb), dtype=int)
                sentence_labels[-1] = 1 # Last sentence ends a section --> 1.
                labels += sentence_labels.tolist()
                suitable_sections_count += 1
            
        bert_data = self.bert_embed(raw_text)
        # Get a random document if the current one is empty or has less than two suitable sections.
        if len(data) == 0 or suitable_sections_count < 2:
            doc_id = self.texts[index] if self.are_paths else index
            logger.warning(f"SKIPPED Document {doc_id} - it's empty or has less than two suitable sections.")
            return self.__getitem__(np.random.randint(0, len(self.texts)))

        return data, labels, bert_data if self.labeled else data, raw_text, bert_data


    def bert_embed(self, sentences: Union[str, List[str]], batch_size=4) -> torch.FloatTensor:
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


class TS_Dataset22(Dataset):
    """
    Custom Text Segmentation Dataset class for use in Dataloaders.
    Includes BERT sentence embeddings. TODO: Precalculate embeddings?
    """
    def __init__(
        self,files: List[str],
        word2vec: KeyedVectors,
        bert_tokenizer: BertTokenizer,
        bert_model: BertModel,
        from_wiki=False,
        labeled=True
        ) -> None:
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
        self.labeled = labeled

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
