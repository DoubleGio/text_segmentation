import torch.multiprocessing as mp
from textseg import TextSeg

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # k = TextSeg(language='nl', dataset_path="text_segmentation/Datasets/NLWiki/data", num_workers=2, subset=1_000)
    # res = k.run(3)
    l = TextSeg(language='nl', dataset_path="text_segmentation/Datasets/NLWiki/data",
    num_workers=0, subset=1_000, load_from="checkpoints/textseg/NLWiki_15-06_15-43/best_model")	
    res = l.run_test()
    print(res)