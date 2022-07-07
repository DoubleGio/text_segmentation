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
from_wiki = False
tt: TextTiling = None

def tt_multi(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    try:
        return tt.evaluate(text, from_wiki=from_wiki)
    except:
        return None

def main(n: int):
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
        'BertTiling',
    ]
    results = pd.DataFrame(
        [[[] for _ in range(len(methods))] for _ in range(len(datasets.keys()))], # Initializes empty lists for each cell
        index=datasets.keys(), columns=methods
    )

    with tqdm(total=len(datasets.keys())) as pbar:
        for dataset, params in datasets.items():
            pbar.set_description(f"Testing {dataset}")
            paths = utils.get_all_file_names(os.path.join(params['loc'], '0-999'))[:n]
            global from_wiki
            from_wiki = params['from_wiki']
            # global tt
            # tt = TextTiling(lang=params['lang'], w=params['w'], k=params['k'])

            # with mp.Pool(processes=4) as p:
            #     with tqdm(desc="Processing TextTiling", total=len(paths), leave=False) as pbar2:
            #         for tt_res in p.imap_unordered(tt_multi, paths):
            #             if tt_res is not None:
            #                 results.loc[dataset, 'TextTiling'].append(tt_res)
            #             pbar2.update(1)
            # gc.collect()

            bt = BertTiling(lang=params['lang'], k=params['k'], batch_size=1, num_workers=2)
            bt_res = bt.eval_multi(paths, from_wiki=params['from_wiki'])
            results.loc[dataset, 'BertTiling'] = bt_res
            pbar.update(1)

    results_avg = results.applymap(np.mean, axis=1)
    results_avg.to_csv('results_avg.csv')
    print(results_avg)

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=np.inf, help='Number of files to process')
    args = parser.parse_args()
    main(n=args.n)
    