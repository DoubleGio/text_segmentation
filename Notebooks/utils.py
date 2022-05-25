import re, os, shutil
from typing import List, Optional, Union, Tuple
from nltk.tokenize import sent_tokenize
from nltk.metrics.segmentation import pk, windowdiff

ENWIKI_LOC = '../ENWiki/data'
NLWIKI_LOC = '../NLWiki/data'
NLNEWS_LOC = '../NLNews/data_2404-1301'
NLAUVI_LOC = '../NLAuVi/data_2604-1516'
SECTION_MARK = '==='

def clean_text(text: str, mark_sections=False) -> str:
    """
    Removes wiki-header and footer.
    Removes headers (= lines starting with SECTION_MARK) from string.
    If mark_sections=True, add SECTION_MARK to mark section beginnings.
    """
    t = re.sub(r'^(?:(<doc)|(\n<\/doc)).*\n+', '', text, flags=re.MULTILINE)
    if mark_sections:
        t = re.sub(r'^=+.*\n+', SECTION_MARK, t, flags=re.MULTILINE)
    else:
        t = re.sub(r'^=+.*\n+', '', t, flags=re.MULTILINE)
    return t

def compute_metrics(predictions: List[int], ground_truth: List[int], k: Optional[int] = None, quiet=True) -> Tuple[float, float]:
    """
    Turns predictions/ground_truth List[int] into Strings for nltk pk & windowdiff functions.
    >>> [1,0,1] --> "101"
    Returns pk and windowdiff scores.
    """
    # Turn List[int] into String
    # [1,0,1] --> "101"
    predictions = "".join(map(str, predictions))
    ground_truth = "".join(map(str, ground_truth))

    if not k:
        window_size = int(round(len(ground_truth) / (ground_truth.count("1") * 2.0)))
    pk_score = pk(ground_truth, predictions, k=window_size)
    windiff_score = windowdiff(ground_truth, predictions, k=window_size)
    
    if not quiet:
        print(f'Pk      = {pk_score}')
        print(f'Windiff = {windiff_score}')
    return pk_score, windiff_score

def get_truth(clean_text: Union[str, List[str]]) -> List[int]:
    """
    Returns a list containing a 1 or 0 for each sentence, 
    signifying whether a section starts here or not, respectively.
    """
    if isinstance(clean_text, str):
        clean_text = sent_tokenize(clean_text)
    if not clean_text[0].startswith(SECTION_MARK):
        raise ValueError(f"Text sections_marks are not found/marked with {SECTION_MARK}")
    return [1 if s.startswith(SECTION_MARK) else 0 for s in clean_text]

def subdivide_dir(root: str, N=1000):
    """
    Put files from a directory into subdirectories containing N files.
    Also filters out empty files.
    """
    files = [
        f for f in os.listdir(root) 
        if os.path.isfile(os.path.join(root, f)) 
        and os.path.getsize(os.path.join(root,f)) > 0
    ]
    n_files = len(files)

    for i, file in enumerate(files):
        # create new subdir if necessary
        if i % N == 0:
            until = min(n_files, i+N-1)
            subdir_name = os.path.join(root, f'{i}-{until}')
            if not os.path.exists(subdir_name):
                os.mkdir(subdir_name)
        shutil.move(os.path.join(root, file), os.path.join(subdir_name, file))

