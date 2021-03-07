import numpy as np 
from pathlib import Path
import csv 
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

from Evaluator import Evaluator
import fasttext
import warnings
warnings. filterwarnings("ignore") 
# https://github.com/fastnlp/style-transformer/blob/master/evaluator/evaluator.py


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


if __name__ == '__main__':
    
    dataset = 'amazon'

    pos_input_list, pos_labels_list, pos_human_list, neg_input_list, neg_labels_list, neg_human_list = read_test_file(dataset = dataset)
    
    pos_input_tokens_list = [word_tokenize(sentence) for sentence in pos_input_list]
    pos_labels_tokens_list = [word_tokenize(sentence) for sentence in pos_labels_list]
    pos_human_tokens_list = [word_tokenize(sentence) for sentence in pos_human_list]

    neg_input_tokens_list = [word_tokenize(sentence) for sentence in neg_input_list]
    neg_labels_tokens_list = [word_tokenize(sentence) for sentence in neg_labels_list]
    neg_human_tokens_list = [word_tokenize(sentence) for sentence in neg_human_list]

    sen_analyzer = Evaluator(dataset = dataset)

    ######### Self BLEU ###########
    print("---- BLEU score of the transferred texts against the original ones ----")

    pos_to_neg_self = sen_analyzer.self_bleu(pos_labels_tokens_list, pos_input_tokens_list)
    print("Positive to negative: ", pos_to_neg_self)

    neg_to_pos_self = sen_analyzer.self_bleu(neg_labels_tokens_list, neg_input_tokens_list)
    print("Negative to positive: ", neg_to_pos_self)
    
    ######### Sentiment ############

    print("---- Accuracy in style (%) ----")

    original_style = [1]*len(pos_input_tokens_list)
    pos_to_neg_sent_acc = sen_analyzer.score_style(pos_human_tokens_list, original_style)
    print("Positive to negative: ", pos_to_neg_sent_acc*100)

    original_style = [0]*len(neg_input_tokens_list)
    neg_to_pos_sent_acc = sen_analyzer.score_style(neg_human_tokens_list, original_style)
    print("Negative to positive: ", neg_to_pos_sent_acc*100)

    ######## Perplexity ##########

    print("---- Perplexity score of the transferred texts ----")

    pos_to_neg_ppl = sen_analyzer.ppl(pos_input_tokens_list)
    print("Positive to negative: ", pos_to_neg_ppl)

    pos_to_neg_ppl = sen_analyzer.ppl(neg_input_tokens_list)
    print("Negative to positive: ", pos_to_neg_ppl)

    ######### Against human ############

    print("---- BLEU score of the transferred texts against human-annotated ones ----")

    pos_to_neg_self = sen_analyzer.self_bleu(pos_human_tokens_list, pos_input_tokens_list)
    print("Positive to negative: ", pos_to_neg_self)

    neg_to_pos_self = sen_analyzer.self_bleu(neg_human_tokens_list, neg_human_tokens_list)
    print("Negative to positive: ", neg_to_pos_self)

