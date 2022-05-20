import re
from typing import List, Union
from nltk.tokenize import sent_tokenize

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
    