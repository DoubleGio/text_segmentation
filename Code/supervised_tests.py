import os, utils, argparse
import pandas as pd
import numpy as np
from typing import Optional
from textseg import TextSeg
from textseg2 import TextSeg2
from transformer2 import Transformer2
from tqdm import tqdm
from tabulate import tabulate

DATASETS = {
    'ENWiki': {"loc": utils.ENWIKI_LOC, "checkpoints": ["ENWiki_Jun21_1358", "ENWiki_Jun26_1851", "ENWiki_Jul05_0952"]},
    'NLWiki': {"loc": utils.NLWIKI_LOC, "checkpoints": ["NLWiki_Jun21_1937", "NLWiki_Jun28_0946", "NLWiki_Jul06_1223"]},
    'NLNews': {"loc": utils.NLNEWS_LOC, "checkpoints": ["NLNews_Jun21_2335", "NLNews_Jun28_1745", "NLNews_Jul06_1545"]},
    'NLAuVi_N': {"loc": utils.NLAUVI_LOC_N, "checkpoints": ["NLAuVi_Jul09_1022", "NLAuVi_Jul09_1410", "NLAuVi_Jul07_1011"]},
    # 'NLAuVi_C': {"loc": utils.NLAUVI_LOC_C, "checkpoints": ["NLAuVi_Jul08_1155", "NLAuVi_Jul08_1155", "NLAuVi_Jul08_1155"]},
}
METHODS = [
    'TextSeg',
    'TextSeg2',
    'Transformer2',
]

def main(bs: int, nw: int, n: Optional[int] = None):
    """Main supervised tests."""
    save_as = '../Results/supervised_tests'
    results = []
    with tqdm(total=len(METHODS)) as pbar:
        for i, method_name in enumerate(METHODS):   # for each method
            pbar.set_description(f"Testing {method_name}")
            res = pd.DataFrame(np.nan, index=DATASETS.keys(), columns=DATASETS.keys())
            with tqdm(total=len(DATASETS.keys()), leave=False) as pbar2:
                for dataset, params in DATASETS.items():    # for each dataset 
                    pbar2.set_description(f'    Trained on {dataset}')
                    if dataset == 'ENWiki':
                        ts = eval(method_name)(language='en', load_from=os.path.join('../checkpoints', method_name.lower(), params['checkpoints'][i], 'best_model'), quiet=True)
                        paths = utils.get_all_file_names(os.path.join(params['loc'], '0-999'))[:n]
                        test_dict = {'ENWiki': np.around(ts.run_test(paths, batch_size=bs, num_workers=nw, from_wiki=True), decimals=DECIMALS)}
                        res.loc[dataset] = test_dict
                    else:
                        test_dict = dict((a, [0,0,0]) for a in list(DATASETS.keys())[1:]) # [Pk, Wd, Acc]
                        ts = eval(method_name)(language='nl', load_from=os.path.join('../checkpoints', method_name.lower(), params['checkpoints'][i], 'best_model'), quiet=True)
                        with tqdm(total=len(list(DATASETS.items())[1:]), leave=False) as pbar3:
                            pbar3.set_description(f'        Testing on {dataset}')
                            for nl_dataset, params_ in list(DATASETS.items())[1:]:
                                paths = utils.get_all_file_names(os.path.join(params_['loc'], '0-999'))[:n]
                                fw = 'Wiki' in nl_dataset
                                test_dict[nl_dataset] = np.around(ts.run_test(paths, batch_size=bs, num_workers=nw, from_wiki=fw), decimals=DECIMALS)
                                pbar3.update(1)
                        res.loc[dataset] = test_dict
                    pbar2.update(1)
            pbar.update(1)
            results.append(res)
    results = pd.concat(results, keys=METHODS)
    results.rename_axis(['Method', 'Trained on ⇩'], inplace=True)
    results.to_csv(f'{save_as}.csv')
    results.reset_index().to_markdown(f'{save_as}.md')
    results_pretty = tabulate(results.reset_index(), headers='keys', tablefmt='github')
    print(results_pretty)

def t2(bs: int, nw: int, n: Optional[int] = None):
    """Testing to compare NLAuVi normal vs. concatenated."""
    save_as = '../Results/t2_tests'
    methods = {
        'NLAuVi_N': {"checkpoint": "NLAuVi_Jul07_1011"},
        'NLAuVi_C': {"checkpoint": "NLAuVi_Jul08_1155"},
    }
    nl_datasets = list(DATASETS.keys())[1:]
    results = pd.DataFrame(np.nan, index=methods.keys(), columns=DATASETS.keys())
    with tqdm(total=len(methods.keys()), leave=False) as pbar:
        for dataset, params in methods.items():    # for each method 
            pbar.set_description(f'    Trained on {dataset}')
            ts = Transformer2(language='nl', load_from=os.path.join('../checkpoints', 'transformer2', params['checkpoint'], 'best_model'), quiet=True)
            res_dict = dict((a, [0,0,0]) for a in nl_datasets) # [Pk, Wd, Acc]
            with tqdm(total=len(nl_datasets), leave=False) as pbar2: # for each dataset
                for nl_dataset in nl_datasets:
                    paths = utils.get_all_file_names(os.path.join(DATASETS[nl_dataset]['loc'], '0-999'))[:n]
                    fw = 'Wiki' in nl_dataset
                    res_dict[nl_dataset] = np.around(ts.run_test(paths, batch_size=bs, num_workers=nw, from_wiki=fw), decimals=DECIMALS)
                    pbar2.update(1)
            results.loc[dataset] = res_dict
            pbar.update(1)
    results.rename_axis(['Trained on ⇩'], inplace=True)
    results.to_csv(f'{save_as}.csv')
    results.reset_index().to_markdown(f'{save_as}.md')
    results_pretty = tabulate(results.reset_index(), headers='keys', tablefmt='github')
    print(results_pretty)

def mixed(bs: int, nw: int, n: Optional[int] = None):
    """Testing models trained on the MIXED dataset."""
    save_as = '../Results/mixed_tests'
    methods = ['TextSeg', 'Transformer2']
    results = pd.DataFrame(np.nan, index=methods, columns=list(DATASETS.keys())[1:])
    with tqdm(total=len(methods)) as pbar:
        for method_name, loc in zip(methods, ['MIXED_Jul10_1107', 'MIXED_Jul10_2204']):   # for each method
            pbar.set_description(f"Testing {method_name}")
            ts = eval(method_name)(language='nl', load_from=os.path.join('../checkpoints', method_name.lower(), loc, 'best_model'), quiet=True)
            res_dict = dict((a, [0,0,0]) for a in list(DATASETS.keys())[1:]) # [Pk, Wd, Acc]
            with tqdm(total=len(DATASETS.keys()), leave=False) as pbar2:
                for dataset, params in list(DATASETS.items())[1:]:    # for each NL dataset
                    paths = utils.get_all_file_names(os.path.join(params['loc'], '0-999'))[:n]
                    fw = 'Wiki' in dataset
                    res_dict[dataset] = np.around(ts.run_test(paths, batch_size=bs, num_workers=nw, from_wiki=fw), decimals=DECIMALS)
                    pbar2.update(1)
            results.loc[method_name] = res_dict
            pbar.update(1)
    results.to_csv(f'{save_as}.csv')
    results.reset_index().to_markdown(f'{save_as}.md')
    results_pretty = tabulate(results.reset_index(), headers='keys', tablefmt='github')
    print(results_pretty)

def write_texts(n: Optional[int] = None):
    with tqdm(total=len(DATASETS.keys())) as pbar:
        for dataset, params in DATASETS.items():
            pbar.set_description(f'Testing {dataset}')
            from_wiki = 'Wiki' in dataset
            language = 'nl' if 'NL' in dataset else 'en'
            paths = utils.get_all_file_names(os.path.join(params['loc'], '0-999'))[:n]
            for i, method_name in enumerate(METHODS):
                ts = eval(method_name)(language=language, load_from=os.path.join('../checkpoints', method_name.lower(), params['checkpoints'][i], 'best_model'), quiet=True)
                for j, text in enumerate(ts.segment_texts(paths, from_wiki=from_wiki)):
                    out_path = os.path.join('../example_outputs', method_name, dataset, f'{os.path.basename(paths[j])}.txt')
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(text)
            pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=2, help="Batch size") # FIXME: does not work for batch_size = 1
    parser.add_argument('--nw', type=int, default=0, help="Number of workers")
    parser.add_argument('--n', type=int, help="Size of the test set")
    parser.add_argument('--decimals', type=int, default=4)
    parser.add_argument('--mixed', action='store_true', help='Test mixed model')
    parser.add_argument('--t2', action='store_true', help='Test normal vs. concat')
    parser.add_argument('--get_avgs', action='store_true')
    parser.add_argument('--texts', action='store_true', help='Output texts')
    args = parser.parse_args()

    global DECIMALS
    DECIMALS = args.decimals

    if args.mixed:
        mixed(args.bs, args.nw, args.n)
    elif args.t2:
        t2(args.bs, args.nw, args.n)
    elif args.get_avgs:
        df = pd.read_csv(f'supervised_tests.csv')
        df1 = df.iloc[1:, 3:].applymap(lambda x: np.array([float(y) for y in list(filter(None, x[1:-1].split(' ')))]), 'ignore')
        avg1 = df1.apply(np.mean, axis=1) # average (pk, wd, acc) across all datasets per method
        a = np.invert(np.diag([True]*3)); b = np.array([[True]*3])
        mask = np.concatenate((a,b,b,a,b,b,a,b))
        avg2 = df1.where(mask, np.nan).apply(np.mean, axis=1) # average (pk, wd, acc) across all datasets (EXCL. where it has been trained on) per method
        print('Inclusive:')
        print(pd.concat([df.iloc[1:, 0:2], avg1.apply(np.around, decimals=4)], axis=1).dropna())
        print('Exclusive:')
        print(pd.concat([df.iloc[1:, 0:2], avg2.apply(np.around, decimals=4)], axis=1).dropna())
    elif args.texts:
        write_texts(args.n)
    else:
        main(args.bs, args.nw, args.n)

