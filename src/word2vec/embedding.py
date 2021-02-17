import pandas as pd
import os
import numpy as np
from src.utils.logger import creat_logger
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

logger = creat_logger('../../logs/embedding.log')
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(curPath)[0])[0]


class SingletonMetaclass(type):
    '''
    @description: singletion
    '''
    def __init__(self,*args,**kwargs):
        self.__instance = None
        super.__init__(*args,**kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass,self).__call__(*args,**kwargs)
            return self.__instance
        else:
            return self.__instance


class Embedding(metaclass=SingletonMetaclass):
    def __init__(self):
        self.stopwords = open(root_path+'/data/stopwords.txt',encoding='utf-8').readlines()
        # self.ar = Autoencoder()

    def load_data(self):
        logger.info('load data')
        self.data = pd.concat([pd.read_csv('../data/traincsv',sep = '\t'),
             pd.read_csv('../../data/test.csv', sep ='\t'),
             pd.read_csv('../../data/dev.csv', sep ='\t')
                        ])
        self.datatext = self.data['text'].apply(lambda x: x.split(' '))
        logger.info('load data finish')


    def train(self):
        count_vect = TfidfVectorizer(
                    stop_words=self.stopwords,
                    max_df = 0.6,
                    ngram_range = (1,2)
        )
        self.tfidf = count_vect.fit(self.data["text"])
        logger.info('train tfidf_embedding')


        logger.info('start train word2vec word2vec')
        self.w2v = models.Word2Vec(min_count = 2,
                              window = 3,
                              size = 300,
                              sample = 6e-5,
                              alpha = 0.03,
                              min_alpha = 0.0007,
                              negative = 15,
                              workers = 4,
                              iter = 10,
                              max_vocab_size = 50000)
        self.w2v.build_vocab(self.datatext)
        logger.info('train word2vec word2vec')
        self.w2v.train(self.datatext,
                  total_examples=self.w2v.corpus_count,
                  epochs = 15,
                  report_delay = 1)
        logger.info('word2vec word2vec finish')


        logger.info('train fasttext word2vec')
        self.fast = models.FastText(self.datatext,
                               size = 300,
                               window = 2,
                               alpha = 0.03,
                               min_count = 2,
                               iter = 10,
                               min_n = 1,
                               max_n = 3,
                               word_ngrams = 2,
                               max_vocab_size = 500000)
        logger.info('fasttext finish')


    def saver(self):
        logger.info('save tfidf model')
        joblib.dump(self.tfidf,root_path + '/model/word2vec/tfidf_model')
        logger.info('save word2vec model')
        self.w2v.wv.save_word2vec_format(root_path + '/model/word2vec/w2v_model.bin',binary=False)
        logger.info('save fasttext model')
        self.w2v.wv.save_word2vec_format(root_path + '/model/word2vec/fast_model.bin',binary=False)


    def loader(self):
        logger.info('load tfidf word2vec model')
        self.tfidf = joblib.load(root_path + '/model/word2vec/tfidf_model')
        logger.info('load w2v word2vec model')
        self.w2v = models.KeyedVectors.load_word2vec_format(root_path + '/model/word2vec/w2v_model.bin',binary=False)
        logger.info('load fasttext word2vec model')
        self.fast = models.KeyedVectors.load_word2vec_format(root_path + '/model/word2vec/fast_model.bin',binary=False)


if __name__ == '__main__':
    em = Embedding()
    em.load_data()
    em.train()
    em.saver()