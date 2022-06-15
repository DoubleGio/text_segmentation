import torch.multiprocessing as mp
from textseg import TextSeg

if __name__ == '__main__':
    mp.set_start_method('spawn')
    k = TextSeg(language='en', dataset_path="text_segmentation/Datasets/ENWiki/data", num_workers=2, subset=10_000)
    res = k.run(3)
    print(res)