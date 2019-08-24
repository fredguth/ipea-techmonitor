# coding: utf-8
import re
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from scipy.stats import poisson

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
# Gensim
import gensim
import gensim.corpora as corpora
import gensim.models as models


def setup():
    tqdm.pandas()

def flatNestedList(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]

def getSemesterTermFrequencyMatrixFrom(dataframe, column='Unigrams', min_freq=12, max_freq=80):
    df = pd.DataFrame(dataframe[column])
    df = df.resample('D',closed='left', label='left').apply(flatNestedList)
    cv = CountVectorizer(tokenizer=(lambda x: x), preprocessor=(lambda x: x), min_df=min_freq, max_df=max_freq)
    table = cv.fit_transform(df[column])
    docterm=pd.DataFrame(table.todense())
    docterm.index = df.index
    semterm = docterm.resample('2QS',closed='left', label='left').sum()
    semterm.columns = cv.get_feature_names()
    semterm=semterm.T
    columns = semterm.columns.strftime(date_format='%Y-%b')
    semterm.columns = np.arange(1,len(semterm.columns)+1).astype(int)
    return semterm, columns

def normalize(df):
    return df.div(df.sum(axis=0), axis=1)*100000

def getPoisson(df):
    index = df.index
    columns = df.columns
    p = pd.DataFrame(poisson.cdf(k=df.loc[:,2:len(df.columns)],mu=df.loc[:,1:len(df.columns)-1]))
    p.columns = columns[1:]
    p.index = index
    return p


def generateTrends(df, columns, size):
    ll=[]
    for c in df.columns:
        ll.append(np.array(df[df.loc[:,c] >0].sort_values(by=[c],ascending=True)[:size].loc[:,c].index))
    trends = pd.DataFrame(ll).T
    trends.columns = columns[1:]
    return trends


setup()
tweets = pd.read_pickle('./data/tokenized.data')
dataframe = tweets
column = 'Bigrams'
output = './data/trends.xls'
semterm, columns = getSemesterTermFrequencyMatrixFrom(tweets, 'Bigrams')
semterm = normalize(semterm)
p = getPoisson(semterm)
trends = generateTrends(p, columns, 1000)
trends.to_excel(output)
