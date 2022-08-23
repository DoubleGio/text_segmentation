import os, argparse, torch
from transformers import logging as tlogging
from transformers import BertTokenizer, BertModel, FeatureExtractionPipeline
from tqdm import tqdm
from utils import sent_tokenize_plus, clean_text, get_all_file_names, yield_all_file_names, ENWIKI_LOC, NLWIKI_LOC, NLNEWS_LOC, NLAUVI_LOC_N, NLAUVI_LOC_C

"""
For each document in each dataset, create and save a BERT embedding.
This follows the same procedure as the bert-as-service module (which was originally used).
Use with caution, as the embeddings are quite large.
"""

tlogging.set_verbosity_error()
datasets = [ENWIKI_LOC, NLWIKI_LOC, NLNEWS_LOC, NLAUVI_LOC_N, NLAUVI_LOC_C]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SENT_LENGTH = 30 # See dataset analysis notebook; sentences generally don't contain more than 25 words.
PIPE = None

class CustomPipeline(FeatureExtractionPipeline):

    def _sanitize_parameters(self, from_wiki=False, **kwargs):
        preprocess_params = {}
        preprocess_params["from_wiki"] = from_wiki
        return preprocess_params, {}, {}

    def preprocess(self, doc_path, from_wiki=False):
        with open(doc_path, 'r') as f:
            text = f.read()
        text = clean_text(text, from_wiki=from_wiki)
        sentences = sent_tokenize_plus(text)
        return_tensors = self.framework
        model_inputs = self.tokenizer(sentences, padding=True, truncation=True, max_length=MAX_SENT_LENGTH, return_tensors=return_tensors)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs, output_hidden_states=True).hidden_states[-2]
        avg_pool = torch.nn.AvgPool2d(kernel_size=(model_outputs.shape[1], 1))
        t = avg_pool(model_outputs).squeeze(dim=1).detach()
        return t

    def postprocess(self, model_outputs):
        return model_outputs

    def __call__(self, inputs, *args, **kwargs):
        return zip(super().__call__(inputs, *args, **kwargs), inputs)


def get_pipe(lang='en'):
    global PIPE
    if PIPE is None:
        if lang == 'en':
            bt = BertTokenizer.from_pretrained('bert-base-uncased')
            bm = BertModel.from_pretrained('bert-base-uncased')
        elif lang == 'nl':
            bt = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
            bm = BertModel.from_pretrained('GroNLP/bert-base-dutch-cased')
        PIPE = CustomPipeline(bm, bt, framework='pt', device=0)
    return PIPE

def reset_pipe():
    global PIPE
    PIPE = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Turn a folder with texts into BERT embeddings.')
    parser.add_argument('--data_dir', type=str, default='text_segmentation/Datasets/ENWiki/data_subset')
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--wiki', action='store_true')
    parser.add_argument('--num_processes', type=int, default=2)
    args = parser.parse_args()

    total_files = len(get_all_file_names(args.data_dir))
    file_gen = yield_all_file_names(args.data_dir)
    p = get_pipe(lang=args.lang)

    with tqdm(desc='Docs processed', total=total_files) as pbar:
        for bert_emb, doc_path in p(file_gen, num_workers=args.num_processes, batch_size=1, from_wiki=args.wiki):
            new_loc = doc_path.replace('data', 'data_bert') + '.pt'
            new_folder = new_loc.rsplit('/', 1)[0]
            if not os.path.exists(new_folder):
                os.makedirs(new_folder, exist_ok=True)
            torch.save(bert_emb, new_loc)
            pbar.update()

    print(f"DONE! Processed {total_files} files.")
