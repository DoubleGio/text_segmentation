import re, os
import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Union, Callable
from matplotlib import pyplot as plt
from nltk.tokenize.texttiling import TextTilingTokenizer, BLOCK_COMPARISON, VOCABULARY_INTRODUCTION, LC, HC, DEFAULT_SMOOTHING, TokenTableField, TokenSequence
from nltk.corpus import stopwords
from utils import sent_tokenize_plus, clean_text, smooth, compute_metrics, generate_boundary_list, SECTION_MARK


class TextTiling(TextTilingTokenizer):
    """
    Extension of nltk.texttiling module.
    ====
    (includes some fixes)

    Tokenize a document into topical sections using the TextTiling algorithm.
    This algorithm detects subtopic shifts based on the analysis of lexical
    co-occurrence patterns.

    The process starts by tokenizing the text into pseudosentences of
    a fixed size w. Then, depending on the method used, similarity
    scores are assigned at sentence gaps. The algorithm proceeds by
    detecting the peak differences between these scores and marking
    them as boundaries. The boundaries are normalized to the closest
    paragraph break and the segmented text is returned.
    """
    
    def __init__(
        self,
        lang='en',
        section_mark=SECTION_MARK,
        w=20,
        k=10,
        similarity_method=BLOCK_COMPARISON,
        smoothing_method=DEFAULT_SMOOTHING,
        smoothing_width=2,
        smoothing_passes=1,
        cutoff_policy=HC,
    ) -> None:
        """
        Parameters:
        ===========
        lang: str
            Language of the text (and which stopwords to load).
        section_mark: str (default: '===')
            Marker for section boundaries.
        w: int (default: 20)
            Size (in words) of the pseudosentences.
        k: int (default: 10)
            Size (in sentences) of the block used in the block comparison method. Approximates average section length.
        similarity_method: str (default: 'block_comparison')
            Method used to calculate the similarity scores.
        smoothing_method: str (default: 'default_smoothing')
            Method used to smooth the gap scores.
        smoothing_width: int (default: 2)
            Width of the smoothing window.
        smoothing_passes: int (default: 1)
            Number of passes of the smoothing method.
        cutoff_policy: str (default: 'HC')
            Policy used to determine the boundaries.
        """
        if lang == 'en':
            self.stopwords = stopwords.words('english')
        elif lang == 'nl':
            self.stopwords = stopwords.words('dutch')
        else:
            raise ValueError(f'Language {lang} not supported.')
        self.section_mark = section_mark
        self.w = w
        self.k = k
        self.similarity_method = similarity_method
        self.smoothing_method = smoothing_method
        self.smoothing_width = smoothing_width
        self.smoothing_passes = smoothing_passes
        self.cutoff_policy = cutoff_policy

    def evaluate(self, text: str, from_wiki=False, quiet=True) -> Tuple[float, float]:
        """Evaluate the texttiling algorithm on a given text, returns the Pk and Windiff scores."""
        pred_text = self.tokenize(text, from_wiki=from_wiki, return_predictions=False)
        predictions = generate_boundary_list(SECTION_MARK + SECTION_MARK.join(pred_text))
        ground_truth = generate_boundary_list(clean_text(text, mark_sections=True, from_wiki=from_wiki))
        return compute_metrics(predictions[1:], ground_truth[1:], return_acc=True, quiet=quiet)
    
    def get_eval_multi(self, from_wiki: bool) -> Callable:
        """Wrapper for the evaluate function that can be used with multiprocessing."""
        def eval_multi(path):
            with open(path, 'r') as f:
                text = f.read()
            try:
                return self.evaluate(text, from_wiki=from_wiki)
            except Exception:
                return None
        return eval_multi
        
    def evaluate_batch(self, texts: List[str], batch_size: int, from_wiki=False, quiet=True) -> List[Tuple[float, float]]:
        """Evaluate the texttiling algorithm on a list of texts, returns the Pk and Windiff scores."""
        # return [self.evaluate(text, from_wiki=from_wiki, quiet=quiet) for text in texts]

    def tokenize(self, text: str, from_wiki=False, return_predictions=False) -> Union[Tuple[List[float], List[float], List[float], List[int]], List[str]]:
        """Return a tokenized copy of *text*, where each "token" represents a separate topic."""

        marked_text = clean_text(text, mark_sections=True, from_wiki=from_wiki)
        cleaned_text = clean_text(text, from_wiki=from_wiki)

        # Remove punctuation
        sents = sent_tokenize_plus(marked_text)
        nopunct_text = ""
        nopunct_sent_word_counts = []
        nopunct_par_breaks = []
        for sent in sents:
            nopunct_sent = ""
            for c in re.finditer(fr"[\wÀ-ÖØ-öø-ÿ\-\' ]|([\n\t]+)|({self.section_mark}+)", sent.lower()):
                if c.group(2): # If it's a section marker
                    nopunct_par_breaks.append(len(nopunct_text) + c.start() - len(nopunct_par_breaks) * len(self.section_mark) + 1)
                elif c.group(1): # If it's a newline or tab
                    nopunct_sent += ' '
                else:
                    nopunct_sent += c.group(0)
            nopunct_text += nopunct_sent + ' '
            nopunct_sent_word_counts.append(len(re.findall(r"[\wÀ-ÖØ-öø-ÿ\-\']+", nopunct_sent)))

        tokseqs = self._divide_to_tokensequences(nopunct_text)

        # Filter stopwords
        for ts in tokseqs:
            ts.wrdindex_list = [wi for wi in ts.wrdindex_list if wi[0] not in self.stopwords]

        token_table = self._create_token_table(tokseqs, nopunct_par_breaks)

        # Lexical score determination
        if self.similarity_method == BLOCK_COMPARISON:
            gap_scores = self._block_comparison(tokseqs, token_table)
        elif self.similarity_method == VOCABULARY_INTRODUCTION:
            raise NotImplementedError("Vocabulary introduction not implemented")
        else:
            raise ValueError(f"Similarity method {self.similarity_method} not recognized")

        if self.smoothing_method == DEFAULT_SMOOTHING:
            smooth_scores = self._smooth_scores(gap_scores)
        else:
            raise ValueError(f"Smoothing method {self.smoothing_method} not recognized")

        # Boundary identification
        depth_scores = self._depth_scores(smooth_scores)
        segment_boundaries = self._identify_boundaries(depth_scores)

        if return_predictions:
            return gap_scores, smooth_scores, depth_scores, segment_boundaries
        else:
            segmented_text = []
            # The (nopunct) word count locations of the boundaries; shifted right, since each boundary stands between two sequences
            block_boundaries = [(self.w * (i+1)) for i, b in enumerate(segment_boundaries) if b == 1]
            sents = sent_tokenize_plus(cleaned_text)
            word_count = 0
            sent_idx = 0
            for bb in block_boundaries:
                if bb == 0:
                    continue
                segment = ""
                best_distance = np.inf
                while sent_idx < len(sents):
                    sent_wc = nopunct_sent_word_counts[sent_idx]
                    if abs(bb - (word_count + sent_wc)) < best_distance: # If adding the sentence to the segment would make the distance smaller
                        word_count += sent_wc
                        best_distance = abs(bb - word_count)
                        segment += sents[sent_idx] + " "
                    else:
                        break
                    sent_idx += 1
                segmented_text.append(segment)
            segmented_text.append(" ".join(sents[sent_idx:]))
            return segmented_text

    def _depth_scores(self, scores) -> List[float]:
        """Calculates the depth of each gap, i.e. the average difference
        between the left and right peaks and the gap's score"""

        depth_scores = [0] * len(scores)

        for i, gapscore in enumerate(scores):
            lpeak = gapscore
            for score in scores[i::-1]:
                if score >= lpeak:
                    lpeak = score
                else:
                    break
            rpeak = gapscore
            for score in scores[i:]:
                if score >= rpeak:
                    rpeak = score
                else:
                    break
            depth_scores[i] = lpeak + rpeak - 2 * gapscore

        return depth_scores

    def _divide_to_tokensequences(self, text):
        "Divides the text into pseudosentences of fixed size"
        w = self.w
        wrdindex_list = []
        matches = re.finditer(r"[\wÀ-ÖØ-öø-ÿ\-']+", text) # This captures things as "don't" and words with hyphens or accented characters as single words
        for match in matches:
            wrdindex_list.append((match.group(), match.start()))
        return [
            TokenSequence(i / w, wrdindex_list[i : i + w])
            for i in range(0, len(wrdindex_list), w)
        ]

    def _identify_boundaries(self, depth_scores):
        """Identifies boundaries at the peaks of similarity score
        differences"""

        boundaries = [0 for _ in depth_scores]

        avg = sum(depth_scores) / len(depth_scores)
        stdev = np.std(depth_scores)

        if self.cutoff_policy == LC:
            cutoff = avg - stdev
        else:
            cutoff = avg - stdev / 2.0

        depth_tuples = sorted(zip(depth_scores, range(len(depth_scores))))
        depth_tuples.reverse()
        hp = list(filter(lambda x: x[0] > cutoff, depth_tuples))

        for dt in hp:
            boundaries[dt[1]] = 1
            for dt2 in hp:  # undo if there is a boundary close already
                if (
                    dt[1] != dt2[1]
                    and abs(dt2[1] - dt[1]) < 4
                    and boundaries[dt2[1]] == 1
                ):
                    boundaries[dt[1]] = 0
        return boundaries

    def _normalize_boundaries(self, text, boundaries, paragraph_breaks):
        """
        Normalize the boundaries identified to the closest sentence start.
        This differs from the original implementation, where the boundaries were
        normalized to the closest paragraph start.
        """
        normalized_boundaries = []
        prevb = 0
        for b in boundaries:
            if b == 0:
                continue
            if b in paragraph_breaks:
                normalized_boundaries.append(b)
                prevb = b
            else:
                normalized_boundaries.append(prevb)
        if prevb < len(text):
            normalized_boundaries.append(len(text))
        return normalized_boundaries

    def _smooth_scores(self, gap_scores):
        "Wraps the smooth function from the SciPy Cookbook"
        smoothed_scores = np.array(gap_scores[:])
        for _ in range(self.smoothing_passes):
            smoothed_scores = smooth(smoothed_scores, window_len=self.smoothing_width + 1)
        return smoothed_scores

    def word_tokenize(self):
        """
        Returns all the words in the text (in the same fashion as nltk texttiling).
        """
        t = re.sub(fr'^{self.section_mark}.*\n', '', self.text, flags=re.MULTILINE) # Remove headers
        # Don't instantly use re.findall() to find words - nltk package removes just punctuation first as follows:
        nopunct_t = "".join(re.findall(r"[\wÀ-ÖØ-öø-ÿ\-\' \n\t]", t.lower()))
        return re.findall(r"[\wÀ-ÖØ-öø-ÿ\-\']+", nopunct_t)

    def get_truth(self, text: str):
        # Remove tabs/newlines and punctuation
        text = re.sub(r"[\n]+", " ", text).lower()
        text = "".join(re.findall(r"[\wÀ-ÖØ-öø-ÿ\-\' \t]", text)) # A bit redundant, but still
        word_list = re.findall(r"[\wÀ-ÖØ-öø-ÿ\-\'\t]+", text)
        truths = []
        for i in range(self.w, len(word_list), self.w):
            truths.append(int(any(word.startswith(self.section_mark) for word in word_list[i:i+self.w])))
        return truths

    def get_truth_per_word(self, text:str):
        text = re.sub(r"[\n]+", " ", text)
        nopunct_t = "".join(re.findall(fr"(?:[\wÀ-ÖØ-öø-ÿ\-\' ]|{self.section_mark})", text.lower()))
        sep_t = re.findall(fr"(?:[\wÀ-ÖØ-öø-ÿ\-\']|{self.section_mark})+", nopunct_t) # List all words (inc. starting with '==='); ignore punctuation etc.
        return [1 if s.startswith(f'{self.section_mark}') else 0 for s in sep_t]

    def plot(self, s, ss, d, b):
        """
        The scores represent the score of token-sequence GAPS.
        So a boundary score of 1 at i=0 means that a boundary occurs between sequences 0 and 1.
        To plot this out per words, we translate the token-sequence gaps to the beginning of a new sequence by shifting everything right and adding a 0 at the start.
        Additionally adds a 0 at the end to signify the very last word.
        """
        # Shift to the right + add ending 0
        s_adj = [0] + s + [0]
        ss_adj = [0] + ss + [0]
        d_adj = [0] + d + [0]
        b_adj = [1] + b + [0] # Start of text is always a boundary, kinda
        ground_truth = self.get_truth_per_word()

        word_n = len(self.word_tokenize())
        word_indices = np.append(np.arange(len(s_adj[:-1])) * self.w, word_n)

        # Setup the labels for xticks to point towards the start of sentences (for use in ax.set_xticks())
        labels = []
        labels_loc = []
        last_loc = 0
        cur_loc = 0
        max_len = len(str(word_n))
        min_distance = int(np.rint(word_n * 0.02)) # Max distance between labels is ~2% of max words
        t = re.sub(r'^===.*\n', '', self.text, flags=re.MULTILINE)
        for s in sent_tokenize_plus(t):
            nopunct_s = "".join(re.findall(r"[\wÀ-ÖØ-öø-ÿ\-\' ]", s))
            sep_s = re.findall(r"[\wÀ-ÖØ-öø-ÿ\-\']+", nopunct_s)
            if cur_loc - last_loc > min_distance: # If enough distance between ticks
                first_word = sep_s[0]
                labels.append(f'{first_word[:3]}... - {cur_loc:{max_len}}') if len(first_word) > 6 else labels.append(f'{first_word} - {cur_loc:{max_len}}')
                last_loc = cur_loc
            else:
                labels.append('')
            labels_loc.append(cur_loc)
            cur_loc += len(sep_s)

        # Make the plot
        fig, ax = plt.subplots(figsize=(20,5),)
        ax.set_xlabel("Word index")
        ax.set_ylabel("Scores")
        ax.set_ylim(0, 1)
        ax.set_xticks(labels_loc, labels, rotation=45, ha='right')
        ax.plot(word_indices, s_adj, label="Gap Scores", color='b')
        ax.plot(word_indices, ss_adj, label="Smoothed Gap scores", color='g')
        ax.plot(word_indices, d_adj, label="Depth scores", color='y')
        ax.vlines(range(len(ground_truth)), ymin=0, ymax=ground_truth, color='r', label="Ground Truth")
        _, _, baseline = ax.stem(word_indices, b_adj, linefmt='--', markerfmt=' ', label="Predicted boundaries")
        plt.setp(baseline, color='k', alpha=0.5)
        ax.legend()
        fig.show()
