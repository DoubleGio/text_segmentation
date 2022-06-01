import logging
import gensim.downloader as gensim_api
import os
from koomri_model import create_model
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import torch
from torch.utils.data import Dataset, Dataloader
from typing import List, Optional, Tuple
from tqdm import tqdm
from nltk import regexp_tokenize, sent_tokenize
from tensorboard_logger import configure, log_value
from datetime import datetime
import numpy as np
from utils import ENWIKI_LOC, NLWIKI_LOCK, NLNEWS_LOC, NLAUVI_LOC, get_all_file_names, sectioned_clean_text

W2V_NL_PATH = '../Datasets/word2vec-nl-combined-160.tar.gz' # Taken from https://github.com/clips/dutchembeddings
CHECKPOINT_BASE = 'checkpoints/koomri'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# TODO: Make fancier logger with colors and whatnot: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

class KoomriImplementation:
    """
    Updated implementation of https://github.com/koomri/text-segmentation.
    """

    def __init__(
        self,
        language = 'en',
        dataset_path = ENWIKI_LOC,
        splits = [0.8, 0.1, 0.1],
        batch_size = 8,
        num_workers = 0,
        use_cuda = True,
        load_from: Optional[str] = None,
    ) -> None:
        """
        """
        if language == 'en':
            word2vec = gensim_api.load('word2vec-google-news-300')
        elif language == 'nl':
            word2vec = KeyedVectors.load_word2vec_format(W2V_NL_PATH)
        else:
            raise ValueError(f"Language {language} not supported, try 'en' or 'nl' instead.")

        # Load the data and puts it into split DataLoaders.
        path_list = get_all_file_names(dataset_path)
        train_paths, test_paths = train_test_split(path_list, test_size=1-splits[0])
        dev_paths, test_paths = train_test_split(test_paths, test_size=splits[1]/(splits[1]+splits[2]))

        train_dataset = DatasetMap(train_paths)
        val_dataset = DatasetMap(dev_paths)
        test_dataset = DatasetMap(test_paths)

        self.train_loader = Dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        self.val_loader = Dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers)
        self.test_loader = Dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers)

        if load_from:
            self.model = torch.load(load_from).to(device)
        else:
            self.model = create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.current_epoch = 0

        if not use_cuda:
            global device
            device = torch.device("cpu")

    def run(self, epochs=10):
        now = datetime.now().strftime(r'%m-%d_%H-%M')
        try: # Try-except block to get around the mistaken (?) "code unreachable error".
            configure(f"./logs/koomri/{now}")
        except:
            pass
        checkpoint_path = os.path.join(CHECKPOINT_BASE, now)
        os.makedirs(checkpoint_path, exist_ok=True)

        best_pk = 1.0
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.train()
            # with(os.path.join(checkpoint_path, f'model{epoch}.t7'))
            torch.save(self.model, os.path.join(checkpoint_path, f'model_{epoch}.t7'))

            val_pk, threshold = self.validate()
            if val_pk < best_pk:
                test_pk = self.test(threshold)
                logging.debug(f"Current best model from epoch {epoch} with Pk={test_pk:.4} and threshold {threshold:.4}")
                best_pk = val_pk
                torch.save(self.model, os.path.join(checkpoint_path, f'model_best.t7'))

    def train(self) -> None:
        self.model.train()
        total_loss = 0.0
        with tqdm(desc='Training', total=len(self.train_loader)) as pbar:
            for i, (data, labels) in enumerate(self.train_loader):
                self.model.zero_grad()
                output = self.model(data)
                # target_var = Variable(torch.cat(labels, dim=0).to(device), requires_grad=False) # Variable is deprecated.
                target_var = torch.cat(labels, dim=0).to(device)
                loss = self.model.criterion(output, target_var)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                pbar.set_postfix(loss=loss.data[0])
                pbar.update(1)
        avg_loss = total_loss / len(self.train_loader) # Average loss per input.
        logging.debug(f"Training Epoch {self.current_epoch + 1} --- Loss = {avg_loss:.4}")
        log_value('Training Loss', avg_loss, self.current_epoch + 1)

    def validate(self):
        self.model.eval()
        with tqdm(desc="Validating", total=len(self.val_loader)) as pbar:
            cm = np.zeros((2, 2), dtype=int)
            for i, (data, labels) in enumerate(self.val_loader):
                output = self.model(data)
                output_softmax = torch.nn.functional.softmax(output, dim=1)
                # target_var = Variable(torch.cat(labels, dim=0).to(device), requires_grad=False) # Variable is deprecated.
                target_var = torch.cat(labels, dim=0).to(device)
                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = target_var.data.cpu().numpy()
                cm += confusion_matrix(target_seg, output_seg)
                acc.update(output_softmax.data.cpu().numpy(), target) # TODO: implement this.
                pbar.update(1)
        # tn, fp, fn, tp = cm.ravel()
        logging.info(f"Validation Epoch {self.current_epoch + 1} ---  ") # TODO: stuff.
        return epoch_pk, threshold

    def test(self, threshold):
        model.eval()
        with tqdm(desc='Testing', total=len(dataset)) as pbar:
            acc = accuracy.Accuracy()
            for i, (data, target, paths) in enumerate(dataset):
                if True:
                    if i == args.stop_after:
                        break
                    pbar.update()
                    output = model(data)
                    output_softmax = F.softmax(output, 1)
                    targets_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
                    output_seg = output.data.cpu().numpy().argmax(axis=1)
                    target_seg = targets_var.data.cpu().numpy()
                    preds_stats.add(output_seg, target_seg)

                    current_idx = 0

                    for k, t in enumerate(target):
                        document_sentence_count = len(t)
                        to_idx = int(current_idx + document_sentence_count)

                        output = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > threshold)
                        h = np.append(output, [1])
                        tt = np.append(t, [1])

                        acc.update(h, tt)

                        current_idx = to_idx

        epoch_pk, epoch_windiff = acc.calc_accuracy()

        logger.debug('Testing Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                          preds_stats.get_accuracy(),
                                                                                                          epoch_pk,
                                                                                                          epoch_windiff,
                                                                                                          preds_stats.get_f1()))
        preds_stats.reset()

        return epoch_pk
class DatasetMap(Dataset):
    """
    Custom Dataset class for use in Dataloaders.
    """
    def __init__(self, files: List[str], word2vec: KeyedVectors) -> None:
        """
        files: List of strings pointing towards files to read.
        word2vec: KeyedVectors object containing the word2vec model.
        """
        self.files = files
        self.word2vec = word2vec

    def __getitem__(self, index: int) -> Tuple[List, List[int]]:
        """
        Returns data and labels of the file at index.
        """
        with open(self.files[index], 'r') as f:
            text = f.read()
        sections = sectioned_clean_text(text)
        data = []
        labels = []

        for section in sections:
            sentences = sent_tokenize(section)
            for sentence in sentences:
                sentence_words = regexp_tokenize(sentence, pattern=r'[\wÀ-ÖØ-öø-ÿ\-\']+') # This pattern is more inclusive and better than '\w+'.
                if len(sentence_words) > 0:
                    sentence_representation = [self.model_word(word) for word in sentence_words]
                    data.append(sentence_representation)
                else:
                    logging.warning(f"Sentence in {self.files[index]} is empty.")
            if len(data) > 0:
                labels.append(len(data) - 1) # NOTE: this points towards the END of a segment.

        return data, labels

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.files)

    def model_word(self, word):
        if word in self.word2vec:
            return self.word2vec[word].reshape(1, 300)
        else:
            return self.word2vec['UNK'].reshape(1, 300)
