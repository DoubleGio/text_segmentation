import re, os, shutil, logging, nltk
from functools import partial, partialmethod
from tqdm import tqdm
from typing import Generator, List, Optional, Union, Tuple, Iterable
from nltk.tokenize import sent_tokenize, regexp_tokenize
from nltk.metrics.segmentation import pk, windowdiff
from sklearn.metrics import accuracy_score
import numpy as np
nltk.download('punkt', quiet=True)

ENWIKI_LOC = 'Datasets/ENWiki/data'
NLWIKI_LOC = 'Datasets/NLWiki/data'
NLNEWS_LOC = 'Datasets/NLNews/data'
NLAUVI_LOC_C = 'Datasets/NLAuVi/data_concat'
NLAUVI_LOC_N = 'Datasets/NLAuVi/data_normal'
SECTION_MARK = '==='

def clean_text(text: str, mark_sections=False, from_wiki=False) -> str:
    """
    Removes wiki-header and footer.
    Removes headers (= lines starting with equals signs) from string.
    If mark_sections=True, add SECTION_MARK to mark section beginnings.
    """
    if from_wiki:
        text = remove_wiki_markup(text)
    sections = re.split(fr'^=+.*\n[^=\w]*', text, flags=re.MULTILINE)
    text = ''
    for s in sections:
        if s:
            s = re.sub(r'[^\w\d]+$', '.\n', s) # Makes sure there's no weird section endings
            if mark_sections:
                s = SECTION_MARK + s
            text += s
    return text

def sectioned_clean_text(text: str, from_wiki=False) -> List[str]:
    """
    Removes wiki-header and footer.
    Split text into sections (without header marks).
    """
    if from_wiki:
        text = remove_wiki_markup(text)
    split = re.split(fr'^=+.*\n[^=\w]*', text, flags=re.MULTILINE)
    return list(filter(None, split))

def remove_wiki_markup(text: str) -> str:
    """Removes wiki-header and footer."""
    text = re.sub(r'^(?:(<doc)|(\n<\/doc)).*\n+', '', text, flags=re.MULTILINE)
    for token in ["LIST", "formula", "codice"]:
        text = re.sub(fr'\*\*\*{token}\*\*\*.*\n+', '', text, flags=re.MULTILINE)
    return text

def sent_tokenize_plus(text: str) -> List[str]:
    """Extends nltk sent_tokenize by also splitting sentences on newlines."""
    text = re.split(r'\n', text, flags=re.MULTILINE)
    res = []
    for t in text:
        res += sent_tokenize(t)
    return res

def word_tokenize(sentence: str) -> List[str]:
    """sentence: String to tokenize."""
    return regexp_tokenize(sentence.lower(), pattern=r'[\wÀ-ÖØ-öø-ÿ\-\']+')

def compute_metrics(predictions: Iterable[int], ground_truth: Iterable[int], k: Optional[int] = None, return_acc=False, quiet=True) -> Union[Tuple[float, float], Tuple[float, float, float]]:
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
        raise ValueError(f"Predictions ({len(predictions)}) and ground truth ({len(ground_truth)}) must have same length.")

    if not isinstance(predictions, list): predictions = try_iter_to_list(predictions)
    if not isinstance(ground_truth, list): ground_truth = try_iter_to_list(ground_truth)
    
    if return_acc:
        acc = accuracy_score(ground_truth, predictions)
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
        if return_acc: print(f'Acc     = {acc}')
    return (pk_score, windiff_score, acc) if return_acc else (pk_score, windiff_score)

def get_all_file_names(dir: str) -> List[str]:
    """
    Return a list of all files in a directory and its subdirectories.
    """
    res = [
        os.path.join(root, file) 
        for root, _, files in os.walk(dir) 
        for file in files 
        if os.path.isfile(os.path.join(root, file))
    ]
    if len(res) == 0:
        raise ValueError(f"No files found in {dir}")
    return res

def yield_all_file_names(dir: str) -> Generator[str, None, None]:
    for root, _, files in os.walk(dir):
        for file in files:
            if os.path.isfile(os.path.join(root, file)):
                yield os.path.join(root, file)

def generate_boundary_list(marked_text: Union[str, List[str]]) -> List[int]:
    """
    marked_text = text or list of sentences.
    Returns a list containing a 1 or 0 for each sentence, 
    signifying whether a section starts here or not, respectively.
    """
    if isinstance(marked_text, str):
        marked_text = sent_tokenize_plus(marked_text)
    if not marked_text[0].startswith(SECTION_MARK):
        raise ValueError(f"Text sections_marks are not found/marked with {SECTION_MARK}")
    return [1 if s.startswith(SECTION_MARK) else 0 for s in marked_text]

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

# Pasted from the SciPy cookbook: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x, window_len=11, window='flat'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        # raise ValueError("Input vector needs to be bigger than window size.")
        return x

    if window_len<3 or x.size<=3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len, 'd')
    else:
        w=eval('np.' + window + '(window_len)')

    y=np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1 : -window_len + 1]

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