import torch.multiprocessing as mp
from textseg2 import TextSeg2
from textseg import TextSeg
from utils import *
import numpy as np
rng = np.random.default_rng()

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    l = TextSeg(language='nl', dataset_path='text_segmentation/Datasets/NLNews/data', num_workers=0, subset=100, batch_size=4)
    res = l.run(2)
    # l = TextSeg(load_from='checkpoints/textseg/NLNews_15-06_18-50/best_model', language='nl', dataset_path='text_segmentation/Datasets/NLNews/data', num_workers=0, subset=10000, batch_size=4)
    # res = l.segment_text(["Toen ik vorige week donderdagavond met een koud biertje achter mn pc ging zitten om de Summer Game Fest Kickoff-stream te bekijken, had ik goede hoop. Mijn verwachtingen waren niet ontzettend hooggespannen, maar toch had ik het gevoel dat we wat toffe aankondigingen gingen krijgen. Ruim twee uur later was ik ondanks mijn lage verwachtingen toch teleurgesteld. Een dozijn aan ruimtehorrorgames, een schaamteloze advertentiecampagne van Dwayne The Rock Johnson en een remake van een remaster als uitsmijter waren nou niet bepaald waar ik graag twee uur van mn vrije avond aan had besteedt. Gelukkig ben ik na de teleurstellende stream van Geoff Keighley toch nog even blijven hangen en daar heb ik allesbehalve spijt van! Aansluitend op de Summer Game Fest-stream was het namelijk tijd voor een korte presentatie van Devolver Digital. Je weet wel, de uitgever van zeer vermakelijk spelletjes zoals Fall Guys, Deaths Door en Loop Hero. Daar kon nog wel eens een flinke verrassing uit de hoge hoed worden getoverd en ja hoor, daar was ie dan: The Plucky Squire!"])
    # k = TextSeg2(language='test', dataset_path="text_segmentation/Datasets/NLWiki/data", num_workers=0, subset=1000, batch_size=2)
    # res = k.run(3)
    # paths = get_all_file_names('text_segmentation/Datasets/NLNews/datax/')
    # sample = rng.choice(paths, size=3, replace=False)
    # model_path = "checkpoints/textseg/NLNews_15-06_18-50/best_model"
    # ts = TextSeg(language='nl', load_from=model_path)
    # res = ts.segment_text(sample)
    print(res)
