import re, os, shutil, logging
from functools import partial, partialmethod
from tqdm import tqdm
from typing import List, Optional, Union, Tuple, Iterable
from nltk.tokenize import sent_tokenize
from nltk.metrics.segmentation import pk, windowdiff

ENWIKI_LOC = '../Datasets/ENWiki/data'
NLWIKI_LOC = '../Datasets/NLWiki/data'
NLNEWS_LOC = '../Datasets/NLNews/data'
NLAUVI_LOC_C = '../Datasets/NLAuVi/data_concat'
NLAUVI_LOC_N = '../Datasets/NLAuVi/data_normal'
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

def sectioned_clean_text(text: str) -> List[str]:
    """
    Removes wiki-header and footer.
    Split text into sections (without header marks).
    """
    t = re.sub(r'^(?:(<doc)|(\n<\/doc)).*\n+', '', text, flags=re.MULTILINE)
    split = re.split(r'^=+.*\n+', t, flags=re.MULTILINE)
    return list(filter(None, split))

def compute_metrics(predictions: Iterable[int], ground_truth: Iterable[int], k: Optional[int] = None, quiet=True) -> Tuple[float, float]:
    """
    Turns predictions/ground_truth Iterable[int] into Strings for use in nltk pk & windowdiff functions.
    >>> [1,0,1] --> "101"
    Returns pk and windowdiff scores.
    """
    def try_iter_to_list(x):
        try:
            return x.tolist()
        except AttributeError:
            pass
        try:
            return list(x)
        except TypeError:
            raise TypeError(f"{x} is not iterable.")

    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length.")

    if not isinstance(predictions, list): predictions = try_iter_to_list(predictions)
    if not isinstance(ground_truth, list): ground_truth = try_iter_to_list(ground_truth)
    
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

def get_all_file_names(dir: str) -> List[str]:
    """
    Return a list of all files in a directory and its subdirectories.
    """
    return [
        os.path.join(root, file) 
        for root, _, files in os.walk(dir) 
        for file in files 
        if os.path.isfile(os.path.join(root, file))
    ]

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

class LoggingHandler(logging.Handler):

    def __init__(self, use_tqdm=False, level = logging.NOTSET) -> None:
        super().__init__(level)
        self.use_tqdm = use_tqdm

        # Initialize custom 'result' logging level
        logging.RESULT = logging.INFO + 5
        logging.addLevelName(logging.RESULT, "RESULT")
        logging.Logger.result = partialmethod(logging.Logger.log, logging.RESULT)
        logging.result = partial(logging.log, logging.RESULT)

        # Initialize colours
        grey = "\x1b[38;20m"
        bold_green = "\x1b[32;1m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
        self.FORMATS = {
            logging.DEBUG: grey + fmt + reset,
            logging.INFO: grey + fmt + reset,
            logging.RESULT: bold_green + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
            logging.CRITICAL: bold_red + fmt + reset
        }

    def emit(self, record: logging.LogRecord) -> None:
        if self.use_tqdm:
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)
        else:
            super().emit(record)

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)