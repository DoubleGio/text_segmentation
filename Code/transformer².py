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
import logging, torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset, DataLoader
from typing import List, Optional
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertModel, FeatureExtractionPipeline
from transformers import logging as tlogging
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
        PipelineDataset.tokenizer = bert_tokenizer

        if not use_cuda:
            global device 
            device = torch.device('cpu')
            logger.info(f"Using device: {device}.")
            self.use_cuda = False
        else:
            PipelineIterator.model = bert_model.to(device)
            device_name = torch.cuda.get_device_name(device)
            logger.info(f"Using device: {device_name}.")
            self.use_cuda = False if device_name == 'cpu' else True
        
        if dataset_path:
            self.subsets = {'train': int(subset * splits[0]), 'val': int(subset * splits[1]), 'test': int(subset * splits[2])} if subset else None
            if from_wiki is None:
                from_wiki = "wiki" in dataset_path.lower()
            self.load_data(dataset_path, from_wiki, splits, batch_size, num_workers)

    def load_data(self, dataset_path, from_wiki, splits, batch_size, num_workers):
        self.dataset_name = dataset_path.split('/')[-2]
        path_list = get_all_file_names(dataset_path)

        # Split and shuffle the data
        train_paths, test_paths = train_test_split(path_list, test_size=1-splits[0])
        dev_paths, test_paths = train_test_split(test_paths, test_size=splits[1]/(splits[1]+splits[2]))

        self.train_loader = self.get_iterator(train_paths)

    def get_iterator(self, paths):
        dataset = PipelineDataset(paths)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=no_collate_fn)
        model_iterator = PipelineIterator(dataloader)
        return model_iterator

    def run(self):
        for x, y in self.train_loader:
            # x.shape = (sentences, bertsize), y.shape = (sentences,)
            break

class PipelineDataset(Dataset):
    def __init__(self, paths: List[str], from_wiki: bool = False):
        self.paths = paths
        self.from_wiki = from_wiki

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.preprocess(self.paths[i])

    def preprocess(self, path: str):
        with open(path, 'r') as f:
            text = f.read()
        sections = sectioned_clean_text(text, from_wiki=self.from_wiki)
        sentences = []
        targets = np.array([], dtype=int)
        for section in sections:
            s = sent_tokenize_plus(section)
            sentences += s
            section_targets = np.zeros(len(s), dtype=int)
            section_targets[0] = 1
            targets = np.append(targets, section_targets)
        # Pairwise tokenizing (last sentence is skipped)
        model_inputs = self.tokenizer(text=sentences[:-1], text_pair=sentences[1:], padding=True, truncation=True, max_length=60, return_tensors='pt')
        return model_inputs, targets[:-1]

class PipelineIterator(IterableDataset):
    def __init__(self, loader, loader_batch_size: Optional[int] = None):
        self.loader = loader
        self.loader_batch_size = loader_batch_size

    def len(self):
        return len(self.loader)
    
    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        item, targets = next(self.iterator)
        return self.forward(item), targets

    def forward(self, model_inputs):
        with torch.inference_mode():
            model_inputs = model_inputs.to(device)
            model_outputs = self.model(**model_inputs).last_hidden_state[:, 0, :] # (sentences, tokens, hidden_size)
            model_outputs = model_outputs.to('cpu')
        return model_outputs

def no_collate_fn(item):
    return item[0]

class BertPipeline(FeatureExtractionPipeline):
    MAX_SENT_LENGTH = 30 

    def _sanitize_parameters(self, from_wiki=False, **kwargs):
        preprocess_params = {}
        preprocess_params["from_wiki"] = from_wiki
        return preprocess_params, {}, {}

    def preprocess(self, doc_path, from_wiki=False):
        with open(doc_path, 'r') as f:
            text = f.read()
        sections = sectioned_clean_text(text, from_wiki=from_wiki)
        sentences = np.array([])
        targets = np.array([], dtype=int)
        for section in sections:
            s = sent_tokenize_plus(section)
            sentences = np.append(sentences, s)
            section_targets = np.zeros(len(s), dtype=int)
            section_targets[0] = 1
            targets = np.append(targets, section_targets)
        return_tensors = self.framework
        # Pairwise tokenizing (last sentence is skipped)
        model_inputs = self.tokenizer(text=sentences[:-1], text_pair=sentences[1:], padding=True, truncation=True, max_length=BertPipeline.MAX_SENT_LENGTH, return_tensors=return_tensors)
        return model_inputs, targets

    def forward(self, model_inputs, **forward_params):
        with self.device_placement():
            if self.framework == "tf":
                model_inputs["training"] = False
                model_outputs = self._forward(model_inputs, **forward_params)
            elif self.framework == "pt":
                inference_context = self.get_inference_context()
                with inference_context():
                    model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                    model_outputs = self._forward(model_inputs, **forward_params)
                    model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
            else:
                raise ValueError(f"Framework {self.framework} is not supported")
        return model_outputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        # Return the CLS tokens
        return model_outputs.last_hidden_state[:, 0, :] # (sentences, tokens, hidden_size)
         
    def postprocess(self, model_outputs):
        return model_outputs

if __name__ == "__main__":
    t = Transformer2(language='en', dataset_path='text_segmentation/Datasets/ENWiki/data_subset', from_wiki=True)
    t.run()
