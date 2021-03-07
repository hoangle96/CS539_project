import fasttext 
import pkg_resources
from typing import List, Tuple
import kenlm
import math

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

class Evaluator(object):
    def __init__(self, dataset = 'yelp'):
        acc_path = 'hoang/acc_'+str(dataset)+'.bin' 
        ppl_path = 'hoang/ppl_'+str(dataset)+'.bin'       
        self.classifier = fasttext.load_model(acc_path)
        self.ppl_model = kenlm.Model(ppl_path)
        self.dataset = dataset

    def style_check(self, text_transfered, style_origin):
        text_transfered = ' '.join(text_transfered)
        if text_transfered == '':
            return False
        label = self.classifier.predict([text_transfered])
        if self.dataset == 'yelp':
            style_transfered = label[0][0] == '__label__positive'
        elif self.dataset == 'amazon':
            style_transfered = label[0][0] == '__label__2'
        return (style_transfered != style_origin)
    
    def score_style(self, texts: List[list], styles_origin: List[int]) -> float:
        """
        texts: list of tokenized sentences, e.g. [['great', 'food', 'great', 'decor'], ['the', 'food', 'is', 'always', 'fresh']]
        styles_origin: list of the sentiment of the original sentences e.g. [1,1]
        """
        assert len(texts) == len(styles_origin), 'Size of inputs does not match!'
        count = 0
        for text, style in zip(texts, styles_origin):
            if self.style_check(text, style):
                count += 1
        return count / len(texts)
    
    def ppl(self, texts_transfered):
        """
        texts_transfered: list of tokenized sentences, e.g. [['great', 'food', 'great', 'decor'], ['the', 'food', 'is', 'always', 'fresh']]
        """
        texts_transfered = [' '.join(itm) for itm in texts_transfered]
        sum = 0
        words = []
        length = 0
        for _, line in enumerate(texts_transfered):
            words += [word for word in line.split()]
            length += len(line.split())
            score = self.ppl_model.score(line)
            sum += score
        return math.pow(10, -sum / length)
    
    def self_bleu(self, texts_origin: List[list], texts_transfered: List[list]) -> float:
        """
        texts_origin: list of tokenized original sentences, e.g. [['great', 'food', 'great', 'decor'], ['the', 'food', 'is', 'always', 'fresh']]
        texts_transfered: list of tokenized transferred sentences, e.g. [['horrible', 'food', 'horrendous', 'decor'], ['the', 'food', 'is', 'never', 'fresh']]
        """
        assert len(texts_origin) == len(texts_transfered), 'Size of inputs does not match!'
        sum = 0
        n = len(texts_origin)
        for x, y in zip(texts_origin, texts_transfered):
            sum += sentence_bleu(references = [x], hypothesis = y) * 100
        return sum / n

    def ref_bleu(self, texts_ref: List[list], texts_transfered: List[list]) -> float:
        """
        texts_ref: list of tokenized human-annotated sentences, e.g. [['great', 'food', 'great', 'decor'], ['the', 'food', 'is', 'always', 'fresh']]
        texts_transfered: list of tokenized transferred sentences, e.g. [['horrible', 'food', 'horrendous', 'decor'], ['the', 'food', 'is', 'never', 'fresh']]
        """
        assert len(texts_ref) == len(texts_transfered), 'Size of inputs does not match!'
        sum = 0
        n = len(texts_ref)
        for x, y in zip(texts_ref, texts_transfered):
            sum += sentence_bleu(references = [x], hypothesis = y) * 100
        return sum / n