from textseg2 import TextSeg2
from textseg import TextSeg
from utils import *
import numpy as np
rng = np.random.default_rng()

if __name__ == '__main__':
    # l = TextSeg(word2vec_path='../Datasets/word2vec-nl-combined-160.txt', dataset_path='../Datasets/NLNews/data', num_workers=0, subset=100, batch_size=4)
    # res = l.run(2)

    # l = TextSeg(load_from='checkpoints/textseg/NLNews_Jun20_1453/best_model', language='nl', num_workers=0, batch_size=4)
    # paths = get_all_file_names('text_segmentation/Datasets/NLNews/data')
    # sample = rng.choice(paths, size=4, replace=False)
    # res = l.segment_text(sample)

    k = TextSeg2(language='nl', dataset_path="text_segmentation/Datasets/NLWiki/data", num_workers=0, subset=1000, batch_size=2)
    res = k.run(2)

    # paths = get_all_file_names('text_segmentation/Datasets/NLNews/datax/')
    # sample = rng.choice(paths, size=3, replace=False)
    # model_path = "checkpoints/textseg/NLNews_15-06_18-50/best_model"
    # ts = TextSeg(language='nl', load_from=model_path)
    # res = ts.segment_text(sample)
    print(res)