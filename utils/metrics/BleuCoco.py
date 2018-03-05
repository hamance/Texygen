import json
import os
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from utils.metrics.Metrics import Metrics
from utils.pycocotools.coco import COCO


class BleuCoco(Metrics):
    def __init__(self, test_text='', annotation_file='', gram=3):
        super().__init__()
        self.name = 'BleuCoco'
        self.coco = COCO(annotation_file)
        self.test_data = json.load(open(test_text, 'r'))
        self.gram = gram
    
    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        return self.get_bleu_parallel()

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        weight = tuple((1. / ngram for _ in range(ngram)))
        for hypothesis in self.test_data:
            annIds = self.coco.getAnnIds(imgIds=hypothesis['id'])
            anns = self.coco.loadAnns(annIds)
            bleu.append(self.calc_bleu(anns, hypothesis['caption'], weight))
        return sum(bleu) / len(bleu)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                annIds = self.coco.getAnnIds(imgIds=hypothesis['id'])
                anns = self.coco.loadAnns(annIds)
                result.append(pool.apply_async(self.calc_bleu, args=(anns, hypothesis['caption'], weight)))
        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt
