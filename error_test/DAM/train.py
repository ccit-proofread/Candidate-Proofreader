#coding=utf-8
import os
import random
import math
import time
from collections import defaultdict
#from utils import *
from paras import *

#Katz backoff smoothing technique
TRIGRAM = "count_trigram"
BIGRAM = "count_bigram"
UNIGRAM = "count_unigram"
TRIGRAM_A = "trigram_A"
BIGRAM_A = "bigram_A"
WORD_SET = "word_set"

START_SYMBOL = "START"
END_SYMBOL = "END"
DISCOUNTED = 0.5

def get_trained_data(LOAD_PATH):
    print("loading '%s'..." % LOAD_PATH)
    with open(LOAD_PATH, 'r', encoding='utf-8') as f:
        dict = f.read()
        return eval(dict)


def save_to_file(content, SAVE_PATH):
    print("saving '%s'..." % SAVE_PATH)
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        f.write(str(content))
    print("length = %d" % len(content))

class BackoffModel:
    count_trigram = {}
    count_bigram = {}
    count_unigram = {}
    trigram_A = defaultdict(set)  #解决KeyError异常
    bigram_A = defaultdict(set)
    word_set = set()
    total_words = 1

    def __init__(self, training_corpus_dir, dict_dir):   #dict_dir数据根目录
        self.training_corpus_dir = training_corpus_dir
        self.dict_dir = dict_dir

        self.word_set_dir = self.dict_dir + WORD_SET
        self.trigram_dir = self.dict_dir + TRIGRAM
        self.bigram_dir = self.dict_dir + BIGRAM
        self.unigram_dir = self.dict_dir + UNIGRAM
        self.trigram_A_dir = self.dict_dir + TRIGRAM_A
        self.bigram_A_dir = self.dict_dir + BIGRAM_A

        print("initializing Corpus object for '%s'..." % training_corpus_dir)

        print('training...')
        start_training = time.time()

        os.makedirs(self.dict_dir)
        for filename in os.listdir(self.training_corpus_dir):
            with open(self.training_corpus_dir + filename, 'r+') as file:
                count = 0
                lines = file.readlines()
                for line in lines:
                     if line.strip():
                        penult_word = START_SYMBOL
                        last_word = START_SYMBOL

                        for word in line.split():
                            count += 1
                            if(penult_word != START_SYMBOL):
                                self.count_trigram[penult_word, last_word, word] = self.count_trigram.get((
                                                                                penult_word,last_word, word), 0) + 1
                            if(last_word != START_SYMBOL):
                                self.count_bigram[last_word, word] = self.count_bigram.get((last_word, word), 0) + 1
                            self.count_unigram[word] = self.count_unigram.get(word, 0) + 1
                            if (penult_word != START_SYMBOL):
                                self.trigram_A[penult_word, last_word].add(word)
                            if (last_word != START_SYMBOL):
                                self.bigram_A[last_word].add(word)
                                self.word_set.add(word)

                            penult_word = last_word
                            last_word = word
                            if count % 100000 == 0: print(count)
                file.close()

            end_training = time.time()
            print('===> time for training: ~%ss' % round(end_training - start_training))

            print('saving trained data to files...')
            start_saving_to_files = time.time()

            save_to_file(self.word_set, self.word_set_dir)
            save_to_file(self.count_trigram, self.trigram_dir)
            save_to_file(self.count_bigram, self.bigram_dir)
            save_to_file(self.count_unigram, self.unigram_dir)
            save_to_file(self.trigram_A, self.trigram_A_dir)
            save_to_file(self.bigram_A, self.bigram_A_dir)

            end_saving_to_files = time.time()
            print('===> time for saving to files: ~%ss' % round(end_saving_to_files - start_saving_to_files))

        self.total_words = sum(self.count_unigram.values())
        print("num_total_words = %d" % self.total_words)
        #if计算start就要减去self.count_unigram[START_SYMBOL]

    def unigram_mle(self, word):
        unique_count = len(self.count_unigram) - 1
        return self.count_unigram.get(word, unique_count) / self.total_words  # TODO unknown word ???

    def unigram_back_off_model(self, last_word, word):
        unigram_mle_others = 1 - sum(self.unigram_mle(w) for w in self.bigram_A[last_word])
        return self.bigram_alpha(last_word) * self.unigram_mle(word) / unigram_mle_others

    def bigram_discounted_count(self, last_word, word):
        return self.count_bigram[last_word, word] - DISCOUNTED

    def bigram_alpha(self, last_word):
        return 1 - sum(self.bigram_discounted_model(last_word, w) for w in self.bigram_A[last_word])

    def bigram_discounted_model(self, last_word, word):
        return self.bigram_discounted_count(last_word, word) / self.count_unigram.get(last_word, 1)

    def bigram_back_off_model(self, last_word, word):
        if word in self.bigram_A[last_word]:
            return self.bigram_discounted_model(last_word, word)
        else:
            return self.unigram_back_off_model(last_word, word)

    def trigram_discounted_count(self, penult_word, last_word, word):
        return self.count_trigram[penult_word, last_word, word] - DISCOUNTED

    def trigram_alpha(self, penult_word, last_word):
        return 1 - sum(self.trigram_discounted_model(penult_word, last_word, w)
                       for w in self.trigram_A[penult_word, last_word])

    def trigram_discounted_model(self, penult_word, last_word, word):
        return self.trigram_discounted_count(penult_word, last_word, word) / self.count_bigram[penult_word, last_word]

    def trigram_back_off_model(self, penult_word, last_word, word):
        if word in self.trigram_A[penult_word, last_word]:
            return self.trigram_discounted_model(penult_word, last_word, word)
        else:
            bigram_mle_others = 1 - sum(self.bigram_back_off_model(last_word, w)
                                        for w in self.trigram_A[penult_word, last_word])
            return self.trigram_alpha(penult_word, last_word) * self.bigram_back_off_model(last_word, word) / \
                bigram_mle_others

if __name__ == '__main__':
    ngram = BackoffModel('./origin_train_data/', './ngram_data/')
