"""
Script to test what the optimal amount of workers to use for textseg(2) DataLoaders are.
0 is the only one that works 🙃
"""
from time import time
import os
# import multiprocessing as mp
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from textseg import TS_Dataset, custom_collate
from utils import ENWIKI_LOC, get_all_file_names
import gensim.downloader as gensim_api
from gensim.models import KeyedVectors
from tqdm import tqdm
# from transformers import BertTokenizer, BertModel
if __name__ == '__main__':
    mp.set_start_method('spawn')
    # word2vec_path = gensim_api.load('word2vec-google-news-300', return_path=True)
    # word2vec = KeyedVectors.load_word2vec_format(os.path.join('../../../..', word2vec_path), binary=True, limit=100000)
    train_paths = get_all_file_names('text_segmentation/Datasets/ENWiki/data')[:10000]
    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bert_model = BertModel.from_pretrained('bert-base-uncased')
    # train_dataset = TS_Dataset(files=train_paths, word2vec=word2vec, bert_tokenizer=bert_tokenizer, bert_model=bert_model, from_wiki=True)
    train_dataset = TS_Dataset(train_paths, None, True)
    print(f'Loaded {len(train_dataset)} files')
    cpu_count = mp.cpu_count()
    print(f'CPU count: {cpu_count}')
    print('')
    for num_workers in range(0, cpu_count // 2, 2):
        train_loader = DataLoader(train_dataset,collate_fn=custom_collate,num_workers=num_workers,batch_size=8)
        start = time()
        for epoch in range(1):
            with tqdm(total=len(train_loader) * 8, desc='Files processed', leave=False) as t:
                for i, data in enumerate(train_loader):
                    t.update(8)
                    pass
        end = time()
        print(f'Num workers: {num_workers:2}, time: {end-start:.4}')
