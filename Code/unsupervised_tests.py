import utils, os, argparse, gc
import pandas as pd
import numpy as np
import multiprocessing as mp
from torch.cuda import empty_cache
from typing import Optional
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
        return (tt.evaluate(text, from_wiki=from_wiki), os.path.basename(path))
    except:
        return (None, None)

def tt_multi2(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    try:
        return (utils.SECTION_MARK + '\n' + f'\n{utils.SECTION_MARK}\n'.join(tt.tokenize(text, from_wiki=from_wiki)), os.path.basename(path))
    except:
        return (None, None)

def main(n: Optional[int] = None, output_texts = False):
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
            global tt
            tt = TextTiling(lang=params['lang'], w=params['w'], k=params['k'])
            tt_multi_ = tt_multi2 if output_texts else tt_multi
            with mp.Pool(processes=4) as p:
                with tqdm(desc="Processing TextTiling", total=len(paths), leave=False) as pbar2:
                    for tt_res, fname in p.imap_unordered(tt_multi_, paths):
                        if tt_res is not None:
                            if isinstance(tt_res, str):
                                out_path = os.path.join('../example_outputs', 'TextTiling', dataset, f'{fname}.txt')
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                with open(out_path, 'w', encoding='utf-8') as f:
                                    f.write(tt_res)
                            else:
                                results.loc[dataset, 'TextTiling'].append(tt_res)
                        pbar2.update(1)
            gc.collect()

            bt = BertTiling(lang=params['lang'], k=params['k'], batch_size=1, num_workers=2)
            if output_texts:
                for t, fname in bt.segment_multi(paths, from_wiki=params['from_wiki']):
                    out_path = os.path.join('../example_outputs', 'BertTiling', dataset, f'{fname}.txt')
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(t)
            else:
                bt_res = bt.eval_multi(paths, from_wiki=params['from_wiki'])
                results.loc[dataset, 'BertTiling'] = bt_res
            empty_cache()
            pbar.update(1)

    if not output_texts:
        avg_tt = results['TextTiling'].apply(np.mean, axis=0)
        avg_bt = results['BertTiling'].apply(np.mean, axis=1)
        results_avg = pd.concat([avg_tt, avg_bt], axis=1)
        results_avg.to_csv('unsupervised_tests.csv')
        print(results_avg)

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, help='Number of files to process')
    parser.add_argument('--texts', action='store_true', help='Output texts')
    args = parser.parse_args()
    main(n=args.n, output_texts=args.texts)
    