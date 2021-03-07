import fasttext 
import pkg_resources
from typing import List, Tuple

class SentimentAnalyzer(object):
    def __init__(self):
        yelp_acc_path = 'hoang/acc_yelp.bin'        
        self.classifier = fasttext.load_model(yelp_acc_path)
    
    def style_check(self, text_transfered, style_origin):
        text_transfered = ' '.join(text_transfered)
        if text_transfered == '':
            return False
        label = self.classifier.predict([text_transfered])
        style_transfered = label[0][0] == '__label__positive'
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