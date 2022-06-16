import torch.multiprocessing as mp
from textseg2 import TextSeg2
from textseg import TextSeg
from utils import *
import numpy as np
rng = np.random.default_rng()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # l = TextSeg(language='en', dataset_path='text_segmentation/Datasets/ENWiki/data', num_workers=4, subset=1000, batch_size=2)
    # res = l.run(2)
    # k = TextSeg2(language='test', dataset_path="text_segmentation/Datasets/NLWiki/data", num_workers=0, subset=500, batch_size=2)
    # res = k.run(3)
    paths = get_all_file_names('text_segmentation/Datasets/NLNews/datax/')
    sample = rng.choice(paths, size=3, replace=False)
    model_path = "checkpoints/textseg/NLNews_15-06_18-50/best_model"
    ts = TextSeg(word2vec_path='text_segmentation/Datasets/word2vec-nl-combined-160.txt', load_from=model_path)
    res = ts.segment_text(sample)
    print(res)