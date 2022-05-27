#!/usr/bin/env python3
from typing import List, Union, Tuple
from utils import compute_metrics
from nltk.tokenize import sent_tokenize
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from matplotlib import pyplot as plt
import numpy as np
import torch

PARALLEL_INFERENCE_INSTANCES = 20

class BertTiling:
    """
    """

    def __init__(
        self,
        k=15,
        smoothing_passes=1,
        smoothing_width=2,
        cutoff_policy='HC',
        language='en'
    ) -> None:
        if cutoff_policy in ['HC', 'LC']:
            self.cutoff_policy = cutoff_policy
        else:
            raise ValueError(f"Cutoff-policy '{cutoff_policy}' not supported, try 'HC' or 'LC'.")

        if language == 'en':
            roberta_config = RobertaConfig.from_pretrained('roberta-base')
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', config=roberta_config)
            self.roberta_model = RobertaModel.from_pretrained('roberta-base', config=roberta_config)
        elif language == 'nl':
            roberta_config = RobertaConfig.from_pretrained('pdelobelle/robbert-v2-dutch-base')
            self.roberta_tokenizer = RobertaTokenizer.from_pretrained('pdelobelle/robbert-v2-dutch-base', config=roberta_config)
            self.roberta_model = RobertaModel.from_pretrained('pdelobelle/robbert-v2-dutch-base', config=roberta_config)
        else:
            raise ValueError(f"Language '{language}' not recognized, try 'en' or 'nl' instead.")
        
        self.k = k
        self.smoothing_passes = smoothing_passes
        self.smoothing_width = smoothing_width

    def evaluate(self, input: Union[str, List[str]], ground_truth: List[int]):
        """
        Takes (list of) sentences, segments it, plots the results and prints the pk and windowdiff scores.
        """
        if isinstance(input, str):
            input = sent_tokenize(input)
        res = self.topic_segmentation_bert(input, return_all=True)
        self.plot(*res, ground_truth)
        pk, wd = compute_metrics([1] + res[-1], ground_truth) # Add 1 to start of predictions to signify start of text (which counts as boundary)
        print(f'Pk-score      = {pk:.3}')
        print(f'Windiff-score = {wd:.3}')

    def topic_segmentation_bert(self, input: Union[str, List[str]], return_all=False) -> List[int]:
        """
        Takes list of sentences and produces list of binary predictions.
        """
        if isinstance(input, str):
            input = sent_tokenize(input)

        # parallel inference; creat batches of sentences to process
        batches_features = []
        for batch_sentences in self._split_list(input, PARALLEL_INFERENCE_INSTANCES):
            # Create sentence representations
            batches_features.append(self._get_features_from_sentence(batch_sentences))
        features = [feature for batch in batches_features for feature in batch] # Flattens batches

        block_score = self._block_comparion(features)
        block_score_smooth = self._smooth_scores(block_score)

        depth_score = self._depth_calc(block_score_smooth)
        predictions = self._identify_boundaries(depth_score)
        if return_all:
            return block_score, block_score_smooth, depth_score, predictions
        return predictions

    def plot(
        self,
        block_s: List[float],
        smooth_s: List[float],
        depth_s: List[float],
        boundaries: List[int],
        ground_truth: List[int]
    ):
        """
        Plots scores, predictions and ground truth PER SENTENCE.
        The scores represent the score of sentence GAPS: a boundary score of 1 at i=0 means that a boundary occurs between sentences 0 and 1.
        Hence the scores are shifted to the right to allign with the next sentence.
        """
        # Shift to the right
        block_s = [0] + block_s
        smooth_s = [0] + smooth_s
        depth_s = [0] + depth_s
        boundaries = [1] + boundaries # We consider the start of a text as a boundary

        # Make the plot
        fig, ax = plt.subplots(figsize=(20,5),)
        ax.set_xlabel("Sentence index")
        ax.set_ylabel("Scores")
        ax.set_ylim(0, 1)
        ax.plot(block_s, label="Gap Scores", color='b')
        ax.plot(smooth_s, label="Smoothed Gap scores", color='g')
        ax.plot(depth_s, label="Depth scores", color='y')
        ax.vlines(range(len(ground_truth)), ymin=0, ymax=ground_truth, color='r', label="Ground Truth")
        _, _, baseline = ax.stem(boundaries, linefmt='--', markerfmt=' ', label="Predicted boundaries")
        plt.setp(baseline, color='k', alpha=0.5)
        ax.legend()
        fig.show()

    def _block_comparion(self, features: List[torch.Tensor]) -> List[float]:
        """
        Implements the block comparison method.
        Follows NLTK implementation, to get scores for EACH gap.
        """
        gap_scores = []
        num_gaps = len(features) - 1
        similarity_metric = torch.nn.CosineSimilarity()

        for curr_gap in range(num_gaps):
            # adjust window size for boundary conditions
            if curr_gap < self.k - 1 and num_gaps - curr_gap > curr_gap:
                window_size = curr_gap + 1
            elif curr_gap > num_gaps - self.k:
                window_size = num_gaps - curr_gap
            else:
                window_size = self.k
            
            b1 = self._compute_window(features, curr_gap - window_size + 1, curr_gap + 1)
            b2 = self._compute_window(features, curr_gap + 1, curr_gap + window_size + 1)
            gap_scores.append(float(similarity_metric(b1[0], b2[0])))
        
        return gap_scores

    def _compute_window(self, features: List[torch.Tensor], start_index: int, end_index: int):
        """
        Given start and end index of embedding, compute pooled window value.
        Adjusted pooling kernel, from (stack_size-1, 1) to (stack_size, 1) compared to core.py
        [window_size, 768] -> [1, 768]
        """
        stack = torch.stack([features[0] for features in features[start_index:end_index]])
        stack = stack.unsqueeze(0)  # https://jbencook.com/adding-a-dimension-to-a-tensor-in-pytorch/
        stack_size = end_index - start_index
        pooling = torch.nn.MaxPool2d((stack_size, 1))
        return pooling(stack)

    def _depth_calc(self, block_scores: List[float]) -> List[float]:
        """
        The depth score corresponds to how strongly the cues for a subtopic changed on both sides of a
        given token-sequence gap and is based on the distance from the peaks on both sides of the valleyto that valley.
        Changed to follow NLTK implementation.
        returns depth_scores
        """
        depth_scores = [0 for _ in block_scores]
        clip = 1
        index = clip

        for gapscore in block_scores[clip:-clip]:
            lpeak = gapscore
            for score in block_scores[index::-1]:
                if score >= lpeak:
                    lpeak = score
                else:
                    break
            rpeak = gapscore
            for score in block_scores[index:]:
                if score >= rpeak:
                    rpeak = score
                else:
                    break
            depth_scores[index] = lpeak + rpeak - 2 * gapscore
            index += 1

        return depth_scores

    def _get_features_from_sentence(self, batch_sentences, layer=-2):
        """
        extracts the BERT semantic representation
        from a sentence, using an averaged value of
        the `layer`-th layer

        returns a 1-dimensional tensor of size 768
        """
        batch_features = []
        for sentence in batch_sentences:
            tokens = self.roberta_tokenizer.encode(sentence, return_tensors="pt")
            hidden_layers = self.roberta_model(tokens, output_hidden_states=True).hidden_states
            pooling = torch.nn.AvgPool2d((tokens.shape[1], 1)) # FIXME: shouldnt this be maxpool (according to paper)?
            sentence_features = pooling(hidden_layers[layer])
            batch_features.append(sentence_features[0])
        return batch_features

    def _identify_boundaries(self, depth_scores: List[float]) -> List[int]:
        """
        Identify boundaries (method taken from NLTK texttiling implementation).
        """
        boundaries = [0 for _ in depth_scores]

        avg = sum(depth_scores) / len(depth_scores)
        stdev = np.std(depth_scores)
        if self.cutoff_policy == "LC":
            cutoff = avg - stdev
        else:
            cutoff = avg - stdev / 2.0
        
        depth_tuples = sorted(zip(depth_scores, range(len(depth_scores))))
        depth_tuples.reverse()
        hp = list(filter(lambda x: x[0] > cutoff, depth_tuples))

        for dt in hp:
            boundaries[dt[1]] = 1
            # undo if there is a boundary close already
            for dt2 in hp:
                if (
                    dt[1] != dt2[1]
                    and abs(dt2[1] - dt[1]) < 4
                    and boundaries[dt2[1]] == 1
                ):
                    boundaries[dt[1]] = 0
        return boundaries


    def _smooth_scores(self, gap_scores, smoothing_passes=1):
        "Wraps the smooth function from the SciPy Cookbook"
        smoothed_scores = np.array(gap_scores[:])
        for _ in range(smoothing_passes):
            smoothed_scores = smooth(smoothed_scores, window_len=self.smoothing_width + 1)
        return smoothed_scores
    
    def _split_list(self, l: List, n: int) -> List[List]:
        """
        Splits a list l into n, relatively even, instances.
        """
        k, m = divmod(len(l), n)
        return (
            l[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(min(len(l), n))
        )

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
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
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
