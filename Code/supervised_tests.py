import os, utils, argparse
import pandas as pd
import numpy as np
from textseg import TextSeg
from textseg2 import TextSeg2
from transformer2 import Transformer2
from tqdm import tqdm

def main(bs: int, nw: int, n: int):
    datasets = {
        'ENWiki': {"loc": utils.ENWIKI_LOC, "checkpoints": ["ENWiki_Jun21_1358", "ENWiki_Jun26_1851", "ENWiki_Jul05_0952"]},
        'NLWiki': {"loc": utils.NLWIKI_LOC, "checkpoints": ["NLWiki_Jun21_1937", "NLWiki_Jun28_1745", "NLWiki_Jul06_1223"]},
        'NLNews': {"loc": utils.NLNEWS_LOC, "checkpoints": ["NLNews_Jun21_2335", "NLNews_Jun28_0946", "NLNews_Jul06_1545"]},
        # 'NLAuVi_N': {"loc": utils.NLAUVI_LOC_N, "checkpoints": ["ENWiki_Jun21_1358", ]},
        # 'NLAuVi_C': {"loc": utils.NLAUVI_LOC_C, "checkpoints": ["ENWiki_Jun21_1358", ]},
    }
    methods = [
        'TextSeg',
        'TextSeg2',
        'Transformers2',
    ]
    results = pd.DataFrame(
        [[[] for _ in range(len(methods))] for _ in range(len(datasets.keys()))], # Initializes empty lists for each cell
        index=datasets.keys(), columns=methods
    )

    with tqdm(total=len(datasets.keys())) as pbar:
        for dataset, params in datasets.items():
            pbar.set_description(f"Testing {dataset}")
            paths = utils.get_all_file_names(os.path.join(params['loc'], '0-999'))[:n]
            from_wiki = 'wiki' in dataset.lower()
            lang = 'nl' if 'NL' in dataset else 'en'
            for name, method in zip(methods, [TextSeg, TextSeg2, Transformer2]):
                ts = method(language=lang, load_from=os.path.join('../checkpoints/textseg/', params['checkpoints'][0], 'best_model'))
                results.loc[dataset, name] = ts.run_test(paths, batch_size=bs, num_workers=nw, from_wiki=from_wiki)
            # ts = TextSeg(language=lang, load_from=os.path.join('../checkpoints/textseg/', params['checkpoints'][0], 'best_model'))
            # ts2 = TextSeg2(language=lang, load_from=os.path.join('../checkpoints/textseg2/', params['checkpoints'][0], 'best_model'))
            # t2 = Transformer2(language=lang, load_from=os.path.join('../checkpoints/transformer2/', params['checkpoints'][0], 'best_model'))
            # results.loc[dataset]['TextSeg'] = ts.run_test(paths, batch_size=bs, num_workers=nw, from_wiki=from_wiki)
            # results.loc[dataset]['TextSeg2'] = ts2.run_test(paths, batch_size=bs, n_workers=nw, from_wiki=from_wiki)
            # results.loc[dataset]['Transformers2'] = t2.run_test(paths, batch_size=bs, n_workers=nw, from_wiki=from_wiki)
            pbar.update(1)
    results.to_csv('supervised_tests.csv')
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--nw', type=int, default=0)
    parser.add_argument('--n', type=int, default=np.inf)
    args = parser.parse_args()
    main(args.bs, args.nw, args.n)
