from textseg2 import TextSeg2
from textseg import TextSeg
from transformers import BertModel, BertTokenizer
from texttiling import TextTiling
from utils import *
import numpy as np
rng = np.random.default_rng()

short_text = "========,1,preface.\nBlack Beauty is an 1877 novel by English author Anna Sewell. It was composed in the last years of her life, during which she remained in her house as an invalid. The novel became an immediate best-seller, with Sewell dying just five months after its publication, but having lived long enough to see her only novel become a success. With fifty million copies sold, Black Beauty is one of the best-selling books of all time. While forthrightly teaching animal welfare, it also teaches how to treat people with kindness, sympathy, and respect. In 2003, the novel was listed at number 58 on the BBC's survey The Big Read.\n========,2,Background.\nAnna Sewell was born in Great Yarmouth, England, and had a brother named Philip, who was an engineer in Europe. At the age of 14, Anna fell while walking home from school in the rain and injured both ankles. Through mistreatment of the injury, she became unable to walk or stand for any length of time for the rest of her life. Disabled and unable to walk, she began learning about horses, spending many hours driving her father to and from the station from which he commuted to work. Her dependence on horse-drawn transportation fostered her respect for horses. Sewells introduction to writing began in her youth when she helped edit the works of her mother, Mary Wright Sewell (1797–1884), a deeply religious, popular author of juvenile best-sellers. Anna Sewell never married or had children.\n===asdf\n"

if __name__ == '__main__':
    # l = TextSeg(word2vec_path='../Datasets/word2vec-nl-combined-160.txt', dataset_path='../Datasets/NLNews/data', num_workers=0, subset=100, batch_size=4)
    # res = l.run(2)

    # l = TextSeg(load_from='checkpoints/textseg/NLNews_Jun20_1453/best_model', language='nl', num_workers=0, batch_size=4)
    # paths = get_all_file_names('text_segmentation/Datasets/NLNews/data')
    # sample = rng.choice(paths, size=4, replace=False)
    # res = l.segment_text(sample)

    # k = TextSeg2(load_from='checkpoints/textseg2/ENWiki_Jun26_1851/epoch_1', bert_name='bert-base-uncased', language='en', dataset_path="text_segmentation/Datasets/ENWiki/data_subset", num_workers=2, subset=1000, batch_size=2)
    # res = k.run(3, resume=True)

    # text_segmentation/Datasets/NLWiki/data/213000-213999/wiki_213555

    # paths = get_all_file_names('text_segmentation/Datasets/NLNews/datax/')
    # sample = rng.choice(paths, size=3, replace=False)
    # model_path = "checkpoints/textseg/NLNews_15-06_18-50/best_model"
    # ts = TextSeg(language='nl', load_from=model_path)
    # res = ts.segment_text(sample)
    # print(res)

    # with open('text_segmentation/Datasets/NLWiki/data/213000-213999/wiki_213555', 'r') as f:
    #     text = f.read()
    # t1 = clean_text(text, from_wiki=True)
    # print(len(sent_tokenize(t1)))

    tt = TextTiling(lang='en')
    with open("text_segmentation/Datasets/ENWiki/data_subset/338825", 'r') as f:
        text = f.read()
    tt.evaluate(text, True)
