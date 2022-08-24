from typing import Any, Dict, List, Generator, Tuple
from utils import compute_metrics, sent_tokenize_plus, smooth, clean_text, generate_boundary_list, SECTION_MARK
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaModel
from transformers import logging as tlogging
from matplotlib import pyplot as plt
from TS_Pipeline import TS_Pipeline
from tqdm import tqdm
import numpy as np
import torch, os

PARALLEL_INFERENCE_INSTANCES = 20
tlogging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BertTiling:
    """
    Updated implementation of https://github.com/gdamaskinos/unsupervised_topic_segmentation.
    """
    def __init__(
        self,
        k=15,
        smoothing_passes=1,
        smoothing_width=2,
        cutoff_policy='HC',
        lang='en',
        batch_size=1,
        num_workers=0,
    ) -> None:
        if cutoff_policy in ['HC', 'LC']:
            self.cutoff_policy = cutoff_policy
        else:
            raise ValueError(f"Cutoff-policy '{cutoff_policy}' not supported, try 'HC' or 'LC'.")

        if lang == 'en':
            roberta_config = RobertaConfig.from_pretrained('roberta-base')
            roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', config=roberta_config)
            roberta_model = RobertaModel.from_pretrained('roberta-base', config=roberta_config)
        elif lang == 'nl':
            roberta_config = RobertaConfig.from_pretrained('pdelobelle/robbert-v2-dutch-base')
            roberta_tokenizer = RobertaTokenizerFast.from_pretrained('pdelobelle/robbert-v2-dutch-base', config=roberta_config)
            roberta_model = RobertaModel.from_pretrained('pdelobelle/robbert-v2-dutch-base', config=roberta_config)
        else:
            raise ValueError(f"Language '{lang}' not recognized, try 'en' or 'nl' instead.")
        
        self.pipeline = BertTilingPipeline(roberta_tokenizer, roberta_model, device, batch_size, num_workers, k=k)
        self.smoothing_passes = smoothing_passes
        self.smoothing_width = smoothing_width

    def evaluate(self, input: str, from_wiki=False, quiet=True, plot=False):
        """
        Takes a text, segments it, plots the results and prints the pk and windowdiff scores.
        """
        res = self.topic_segmentation_bert(input, from_wiki=from_wiki, return_all=True)
        ground_truth = generate_boundary_list(clean_text(input, mark_sections=True, from_wiki=from_wiki))
        if plot:
            self.plot(*res, ground_truth)
        pk, wd, acc = compute_metrics(res[-1], ground_truth[1:], return_acc=True, quiet=quiet)
        return pk, wd, acc

    def eval_multi(self, input: List[str], from_wiki=False, quiet=True) -> Tuple[List[float], List[float], List[float]]:
        pk, wd, acc = [], [], []
        bs_pipeline = self.pipeline(input, from_wiki=from_wiki, path=True)
        with tqdm(desc="Processing BertTiling", total=len(input), leave=False) as pbar:
            for i, block_scores in enumerate(bs_pipeline):
                if block_scores.numel() != 0:
                    block_scores_smooth = self._smooth_scores(block_scores)
                    depth_scores = self._depth_calc(block_scores_smooth)
                    predictions = self._identify_boundaries(depth_scores)
                    with open(input[i], 'r', encoding='utf-8') as f:
                        true_text = f.read()
                    ground_truth = generate_boundary_list(clean_text(true_text, mark_sections=True, from_wiki=from_wiki))
                    try:
                        pk_, wd_, acc_ = compute_metrics(predictions, ground_truth[1:], return_acc=True, quiet=quiet)
                        pk.append(pk_)
                        wd.append(wd_)
                        acc.append(acc_)
                    except:
                        pass
                pbar.update(1)
        return pk, wd, acc

    def segment_multi(self, input: List[str], from_wiki=False) -> Generator[str, None, None]:
        bs_pipeline = self.pipeline(input, from_wiki=from_wiki, path=True, return_sents=True)
        with tqdm(desc="Processing BertTiling", total=len(input), leave=False) as pbar:
            for block_scores, sents, fname in bs_pipeline:
                if block_scores.numel() != 0:
                    block_scores_smooth = self._smooth_scores(block_scores)
                    depth_scores = self._depth_calc(block_scores_smooth)
                    predictions = self._identify_boundaries(depth_scores)
                    text = SECTION_MARK + '\n'
                    for sent, pred in zip(sents[1:], predictions):
                        if pred == 1:
                            text += sent + '\n' + SECTION_MARK + '\n'
                        else:
                            text += sent + ' '
                    yield (text, fname)
                pbar.update(1)

    def topic_segmentation_bert(self, input: str, from_wiki=False, return_all=False) -> List[int]:
        """
        Takes a string and produces list of binary predictions.
        """
        block_scores = self.pipeline(input, from_wiki=from_wiki)
        block_scores_smooth = self._smooth_scores(block_scores)

        depth_scores = self._depth_calc(block_scores_smooth)
        predictions = self._identify_boundaries(depth_scores)
        if return_all:
            return block_scores, block_scores_smooth, depth_scores, predictions
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

    def _identify_boundaries(self, depth_scores: List[float], threshold=0.6) -> List[int]:
        """
        Identifies boundaries. Follows original implementation.
        """
        boundaries = [0] * len(depth_scores)
        local_maxima = []
        local_maxima_i = []
        threshold = threshold * max(depth_scores)
        # threshold = np.mean(depth_scores) - np.std(depth_scores)
        for i in range(1, len(depth_scores) - 1):
            if depth_scores[i - 1] < depth_scores[i] > depth_scores[i + 1]:
                local_maxima.append(depth_scores[i])
                local_maxima_i.append(i)
        for lm, lmi in zip(local_maxima, local_maxima_i):
            if lm > threshold:
                boundaries[lmi] = 1
        return boundaries

    def _identify_boundaries_nltk(self, depth_scores: List[float]) -> List[int]:
        """
        Identify boundaries. Follows NLTK implementation.
        """
        boundaries = [0] * len(depth_scores)

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

    def _smooth_scores(self, gap_scores: torch.FloatTensor, smoothing_passes=1):
        "Wraps the smooth function from the SciPy Cookbook"
        smoothed_scores = gap_scores.numpy()
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

class BertTilingPipeline(TS_Pipeline):

    def _sanitize_parameters(self, from_wiki=False, path=False, return_sents=False, k=15) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        preprocess_params, forward_params = {}, {}
        preprocess_params["from_wiki"] = from_wiki
        preprocess_params["path"] = path
        preprocess_params["return_sents"] = return_sents
        forward_params["k"] = k
        return preprocess_params, forward_params

    def preprocess(self, text: str, from_wiki: bool, path: bool, return_sents: bool) -> Dict[str, torch.TensorType]:
        if path:
            fname = os.path.basename(text)
            with open(text, "r", encoding='utf-8') as f:
                text = f.read()
        text = clean_text(text, from_wiki=from_wiki)
        sentences = sent_tokenize_plus(text)
        model_inputs = self.tokenizer(
            text=sentences,
            padding=True,
            return_tensors="pt",
        )
        return model_inputs if not return_sents else (model_inputs, sentences, fname)

    def _forward(self, input_tensors: Dict[str, torch.TensorType], k: int) -> torch.TensorType:
        try:
            hidden_layer = self.model(**input_tensors, output_hidden_states=True).hidden_states[-2]
        except RuntimeError:
            return torch.tensor([], device=self.device)
        pooling = torch.nn.AvgPool2d((input_tensors["input_ids"].shape[1], 1))
        features = pooling(hidden_layer).squeeze()

        # Block comparison
        # Follows NLTK implementation, to get scores for EACH gap.
        num_gaps = features.shape[0] - 1
        gap_scores = torch.empty(num_gaps, device=self.device)
        similarity_metric = torch.nn.CosineSimilarity(dim=0) # calculate over dim 0
        for curr_gap in range(num_gaps):
            if curr_gap < k - 1 and num_gaps - curr_gap > curr_gap:
                window_size = curr_gap + 1
            elif curr_gap > num_gaps - k:
                window_size = num_gaps - curr_gap
            else:
                window_size = k

            b1 = self.compute_window(features[curr_gap - window_size + 1 : curr_gap + 1, :])
            b2 = self.compute_window(features[curr_gap + 1 : curr_gap + window_size + 1, :])
            gap_scores[curr_gap] = similarity_metric(b1, b2)
        return gap_scores

    def compute_window(self, features):
        pooling = torch.nn.MaxPool1d(features.shape[0]) # (dim, n) -> (dim, 1)
        res = pooling(features.transpose(1, 0))         # hence the transpose: (n, dim) -> (dim, n) -> (dim, 1)
        return res
    
    @staticmethod
    def no_collate_fn(item):
        return item[0]
