import os, subprocess, shutil, re, utils, argparse
import numpy as np
from typing import List, Tuple, Optional, Union
from utils import SECTION_MARK, clean_text, get_all_file_names

ORIG_DIR = 'text_segmentation/GraphSeg/data/input_orig'
INPUT_DIR = 'text_segmentation/GraphSeg/data/input'
OUTPUT_DIR = 'text_segmentation/GraphSeg/data/output'
SEG_EN = 'text_segmentation/GraphSeg/binary/graphseg_en.jar'
SEG_NL = 'text_segmentation/GraphSeg/binary/graphseg_nl.jar'

def copy_data(location: Union[str, List[str]], n: Optional[int] = np.inf, wiki=False):
    """
    Copy n files from location to input_orig directory.
    Removes wiki-header if wiki=True.
    """
    if isinstance(location, str):
        for root, _, files in os.walk(location):
            for file in files:
                if n == 0:
                    return
                if not os.path.exists(os.path.join(ORIG_DIR, file)):
                    shutil.copy(os.path.join(root, file), ORIG_DIR)
                    if wiki:
                        with open(os.path.join(ORIG_DIR, file), 'r+') as orig:
                            data = orig.read()
                            orig.seek(0)
                            orig.write(re.sub(r'^(?:(<doc)|(\n<\/doc)).*\n', '', data, flags=re.MULTILINE))
                            orig.truncate()
                    n -= 1
    else:
        for file in location:
            if n == 0:
                return
            if not os.path.exists(os.path.join(ORIG_DIR, os.path.basename(file))):
                shutil.copy(file, ORIG_DIR)
                if wiki:
                    with open(os.path.join(ORIG_DIR, os.path.basename(file)), 'r+') as orig:
                        data = orig.read()
                        orig.seek(0)
                        orig.write(re.sub(r'^(?:(<doc)|(\n<\/doc)).*\n', '', data, flags=re.MULTILINE))
                        orig.truncate()
                n -= 1

def clean_data(from_wiki=False):
    """
    Clean up docs, place them in input directory.
    """
    for root, _, files in os.walk(ORIG_DIR):
        for file in files:
            with open(os.path.join(root, file), 'r') as f:
                doc = f.read()
            with open(os.path.join(INPUT_DIR, file), 'w') as f2:
                f2.write(clean_text(doc, from_wiki=from_wiki))

def reset_data_folder():
    """
    Reset the input_orig, input and output directories.
    """
    for folder in [ORIG_DIR, INPUT_DIR, OUTPUT_DIR]:
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except OSError as e:
                print(f'Failed to delete {path}. Reason: {e}')

def calculate_results(from_wiki=False, return_mean=False) -> Union[Tuple[float, float, float], Tuple[List[float], List[float], List[float]]]:
    """
    Generate ground truths from original files.
    Generate predictions from output.
    Compute metrics and return average (pk, wd).
    """
    scores = {"pk": [], "windiff": [], "acc": []}
    files = os.listdir(OUTPUT_DIR)
    if len(files) == 0:
        return None, None, None
    for file in files:
        with open(os.path.join(OUTPUT_DIR, file), 'r') as f:
            pred_text = f.read()
        with open(os.path.join(ORIG_DIR, file), 'r') as f:
            orig_text = f.read()
        pred_text = SECTION_MARK + utils.clean_text(pred_text, mark_sections=True, from_wiki=from_wiki)
        orig_text = utils.clean_text(orig_text, mark_sections=True, from_wiki=from_wiki)
        pred = utils.generate_boundary_list(pred_text)[:-1] # outputs get an extra section marker at the end
        orig = utils.generate_boundary_list(orig_text)
        try:
            pk_wd = utils.compute_metrics(pred, orig, return_acc=True)
        except ValueError:
            continue
        scores["pk"].append(pk_wd[0])
        scores["windiff"].append(pk_wd[1])
        scores["acc"].append(pk_wd[2])
    if return_mean:
        return np.mean(scores["pk"]), np.mean(scores["windiff"]), np.mean(scores["acc"])
    else:
        return scores["pk"], scores["windiff"], scores["acc"]

def run_graphseg(location: Union[str, List[str]], lang='en', n: Optional[int] = None, from_wiki=False, relatedness_threshold = 0.25, minimal_seg_size = 2):
    """
    Run GraphSeg for <n> files in <location>.

    Parameters
    ===================
    location: str
        Path to directory containing files or paths to files to be segmented.
    n: int
        Max number of files to be segmented.
    from_wiki: bool
        If True, the files are assumed to be in the wiki format.
    relatedness_threshold: float (default: 0.25)
        The threshold to be used in the construction of the relatedness graph: 
        larger values will give large number of small segments, whereas the smaller treshold values will provide a smaller number of coarse segments.
    minimal_seg_size: int (default: 2)
        The minimum size of a segment (in sentences).
    """
    reset_data_folder()
    copy_data(location, n, from_wiki)
    clean_data(from_wiki)
    if lang == 'en':
        jar_loc = SEG_EN
    elif lang == 'nl':
        jar_loc = SEG_NL
    else:
        raise ValueError(f"Language {lang} not supported.")
    try:
        subprocess.run(
            ['java', '-jar', jar_loc, INPUT_DIR, OUTPUT_DIR, f'{relatedness_threshold}', f'{minimal_seg_size}', ],
            # stdout=subprocess.DEVNULL, # suppress output
            # stderr=subprocess.DEVNULL,
            # timeout=60, # seconds
        )
    except subprocess.TimeoutExpired:
        pass
    res = calculate_results(from_wiki=from_wiki, return_mean=True)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GraphSeg")
    parser.add_argument("--prepare", action="store_true", help="Prepare data for GraphSeg.")
    parser.add_argument("--results", action="store_true", help="Calculate results.")
    parser.add_argument("--n", type=int, default=1000, help="Number of files to segment")
    parser.add_argument("--relatedness_threshold", type=float, default=0.25, help="The threshold to be used in the construction of the relatedness graph: larger values will give large number of small segments, whereas the smaller treshold values will provide a smaller number of coarse segments.")
    parser.add_argument("--minimal_seg_size", type=int, default=2, help="The minimum size of a segment (in sentences).")
    parser.add_argument("location", type=str, help="Path to directory containing files or paths to files to be segmented.")
    args = parser.parse_args()

    from_wiki = "wiki" in args.location.lower()
    locations = get_all_file_names(args.location)
    if 'EN' in args.location:
        lang = 'en'
        jar_loc = SEG_EN
    elif 'NL' in args.location:
        lang = 'nl'
        jar_loc = SEG_NL

    if args.prepare:
        reset_data_folder()
        copy_data(locations, args.n, from_wiki)
        clean_data(from_wiki)
        print("Run the jar seperately:")
        print(f"java -jar {jar_loc} {INPUT_DIR} {OUTPUT_DIR} {args.relatedness_threshold} {args.minimal_seg_size}")
    elif args.results:
        res = calculate_results(from_wiki, True)
        with open("graphseg_results.txt", "a+") as f:
            f.write(f"{args.location} {args.n} files --- Pk: {res[0]}, windowdiff: {res[1]}, accuracy: {res[2]}")
        print(f"Pk: {res[0]}, windowdiff: {res[1]}, accuracy: {res[2]}")
    else:
        res = run_graphseg(locations, lang, args.n, from_wiki=from_wiki, relatedness_threshold=args.relatedness_threshold, minimal_seg_size=args.minimal_seg_size)
        with open("graphseg_results.txt", "a+") as f:
            f.write(f"{args.location} {args.n} files --- Pk: {res[0]}, windowdiff: {res[1]}, accuracy: {res[2]}")
        print(f"Pk: {res[0]}, windowdiff: {res[1]}, accuracy: {res[2]}")
