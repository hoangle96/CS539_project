import editdistance
from collections import Counter
import math
import numpy as np 
from pathlib import Path
import csv 
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import fasttext 
import pkg_resources
from typing import List, Tuple
from sentiment_analyzer import SentimentAnalyzer
# https://github.com/fastnlp/style-transformer/blob/master/evaluator/evaluator.py


def self_bleu(texts_origin: List[list], texts_transfered: List[list]) -> float:
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

def ref_bleu(texts_ref: List[list], texts_transfered: List[list]) -> float:
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

def read_test_file(dataset = "amazon"):
    assert (dataset == 'amazon' or dataset == 'yelp'), "Input amazon or yelp"
    
    human_pos_file_path = Path('./data/ref/'+ dataset +'/sentiment.test.1.human')
    label_pos_file_path = Path('./data/ref/'+ dataset +'/sentiment.test.1.label')
    human_neg_file_path = Path('./data/ref/'+ dataset +'/sentiment.test.0.human')
    label_neg_file_path = Path('./data/ref/'+ dataset +'/sentiment.test.0.label')

    pos_input_list, pos_labels_list, pos_human_list = [], [], []
    neg_input_list, neg_labels_list, neg_human_list = [], [], []

    with open(human_pos_file_path, 'r') as f:
        human_pos = csv.reader(f, delimiter='\n')
        with open(label_pos_file_path, 'r') as f_2:
            label_pos = csv.reader(f_2, delimiter='\n')

            for human_row, label_row in zip(human_pos, label_pos):
                input, human_label  = human_row[0].split('\t')
                input, label, _ = label_row[0].split('\t')

                pos_input_list.append(input)
                pos_labels_list.append(label)
                pos_human_list.append(human_label)
    
    with open(human_neg_file_path, 'r') as f:
        human_neg = csv.reader(f, delimiter='\n')
        with open(label_neg_file_path, 'r') as f_2:
            label_neg = csv.reader(f_2, delimiter='\n')

            for human_row, label_row in zip(human_neg, label_neg):
                input, human_label = human_row[0].split('\t')
                input, label, _ = label_row[0].split('\t')

                neg_input_list.append(input)
                neg_labels_list.append(label)
                neg_human_list.append(human_label)

    return pos_input_list, pos_labels_list, pos_human_list, neg_input_list, neg_labels_list, neg_human_list

def cal_perplexity(sentence, model):
    pass # how ????

if __name__ == '__main__':

    pos_input_list, pos_labels_list, pos_human_list, neg_input_list, neg_labels_list, neg_human_list = read_test_file(dataset = 'yelp')
    
    tokenizer = RegexpTokenizer(r'\w+')

    pos_input_tokens_list = [tokenizer.tokenize(sentence) for sentence in pos_input_list]
    pos_labels_tokens_list = [tokenizer.tokenize(sentence) for sentence in pos_labels_list]
    pos_human_tokens_list = [tokenizer.tokenize(sentence) for sentence in pos_human_list]

    neg_input_tokens_list = [tokenizer.tokenize(sentence) for sentence in neg_input_list]
    neg_labels_tokens_list = [tokenizer.tokenize(sentence) for sentence in neg_labels_list]
    neg_human_tokens_list = [tokenizer.tokenize(sentence) for sentence in neg_human_list]

    ######### Self BLEU ###########
    print("---- BLEU score of the transferred texts against the original ones ----")

    pos_to_neg_self = self_bleu(pos_labels_tokens_list, pos_input_tokens_list)
    print("Positive to negative: ", pos_to_neg_self)

    neg_to_pos_self = self_bleu(neg_labels_tokens_list, neg_input_tokens_list)
    print("Negative to positive: ", neg_to_pos_self)
    
    ######### Sentiment ############
    sen_analyzer = SentimentAnalyzer()

    print("---- BLEU score of the transferred texts against the original ones ----")

    original_style = [1]*len(pos_input_tokens_list)
    pos_to_neg_sent_acc = sen_analyzer.score_style(pos_human_tokens_list, original_style)
    print("Positive to negative: ", pos_to_neg_sent_acc)

    original_style = [0]*len(neg_input_tokens_list)
    neg_to_pos_sent_acc = sen_analyzer.score_style(neg_human_tokens_list, original_style)
    print("Negative to positive: ", neg_to_pos_sent_acc)

    ######### Against human ############
    sen_analyzer = SentimentAnalyzer()

    print("---- BLEU score of the transferred texts against human-annotated ones ----")

    pos_to_neg_self = self_bleu(pos_human_tokens_list, pos_input_tokens_list)
    print("Positive to negative: ", pos_to_neg_self)

    neg_to_pos_self = self_bleu(neg_human_tokens_list, neg_input_tokens_list)
    print("Negative to positive: ", neg_to_pos_self)

