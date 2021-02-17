import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.externals import joblib
from bayes_opt import BayesianOptimization
from gensim import models
from logger import creat_logger
logger = creat_logger('../logs/word2vec.log')


max_length = 500 #表示样本的最大长度，降维之后的维度
sentence_max_lenght = 1500 #表示句子/样本在降维之前的维度
Train_features3,Test_features3,Train_label3,Test_labele3 = [],[],[],[]

def load_model():
    fast_embedding = models.KeyedVectors.load('fast_model')
    w2v_embedding = models.KeyedVectors.load('w2v_model')
    logger.info('fast_embedding输出词表的个数{},w2v_embedding输出词表的个数{}'.format(len(fast_embedding.wv.voab.keys()),len(w2v_embedding.wv.vocab.keys())))
    logger.info('load word2vec models finished!')
    train