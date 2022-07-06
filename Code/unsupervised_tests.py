import utils, os, argparse, gc
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from texttiling import TextTiling
from graphseg import run_graphseg
from berttiling import BertTiling
rng = np.random.default_rng()

# os.environ["TOKENIZERS_PARALLELISM"] = "true"
BATCH_SIZE = 10

def main(n=100):
    """
    w: pseudosentence length (approximate avg. sentence length)
    k: block comparison size (approximate avg. section size)
    """
    datasets = {
        'ENWiki': {"loc": utils.ENWIKI_LOC, "lang": "en", "from_wiki": True, "w": 20, "k": 5},
        'NLWiki': {"loc": utils.NLWIKI_LOC, "lang": "nl", "from_wiki": True, "w": 15, "k": 5},
        'NLNews': {"loc": utils.NLNEWS_LOC, "lang": "nl", "from_wiki": False, "w": 17, "k": 10},
        'NLAuVi_N': {"loc": utils.NLAUVI_LOC_N, "lang": "nl", "from_wiki": False, "w": 14, "k": 40},
        'NLAuVi_C': {"loc": utils.NLAUVI_LOC_C, "lang": "nl", "from_wiki": False, "w": 15, "k": 20},
    }
    methods = [
        'TextTiling',
        # 'GraphSeg',
        'BertTiling',
    ]
    results = pd.DataFrame(
        [[[] for _ in range(len(methods))] for _ in range(len(datasets.keys()))], # Initializes empty lists for each cell
        index=datasets.keys(), columns=methods
    )

    with tqdm(desc=f'Processing...', total=len(datasets.keys())) as pbar:
        for dataset, params in datasets.items():
            pbar.set_description(f"Testing {dataset}")
            # paths = rng.choice(utils.get_all_file_names(params['loc']), n, replace=False).tolist()
            paths = utils.get_all_file_names(params['loc'])[:n]
            # pbar.set_postfix_str("processing GraphSeg")
            # with tqdm(desc="Files processed", total=len(paths), leave=False) as pbar2:
            #     for i in range(0, len(paths), BATCH_SIZE):
            #         pk, wd, acc = run_graphseg(paths[i:i+BATCH_SIZE], lang=params['lang'], n=n, from_wiki=params['from_wiki'], relatedness_threshold=0.2, minimal_seg_size=2)
            #         if pk is not None:
            #             results.loc[dataset, 'GraphSeg'] += list(zip(pk, wd, acc))
            #         pbar2.update(len(paths[i:i+BATCH_SIZE]))
            #     gc.collect()
            pbar.set_postfix_str("initializing [Text,Bert]Tiling")
            tt = TextTiling(lang=params['lang'], w=params['w'], k=params['k'])
            bt = BertTiling(lang=params['lang'], k=params['k'])
            pbar.set_postfix_str("")
            for path in tqdm(paths, desc="Files processed", leave=False):
                with open(path, 'r') as f:
                    text = f.read()
                try:
                    tt_res = tt.evaluate(text, from_wiki=params['from_wiki'])
                    bt_res = bt.evaluate(text, from_wiki=params['from_wiki'])
                except:
                    continue
                results.loc[dataset, 'TextTiling'].append(tt_res)
                results.loc[dataset, 'BertTiling'].append(bt_res)
            pbar.update(1)

    results_avg = results.applymap(np.mean, axis=0)
    results_avg.to_csv('results_avg.csv')
    print(results_avg)

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, default=100, help='Number of files to process')
    args = parser.parse_args()
    main(n=args.n)
    