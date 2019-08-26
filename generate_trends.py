# coding: utf-8
import re
import sys
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from scipy.stats import poisson
from configparser import ConfigParser, ExtendedInterpolation
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
# Gensim
import gensim
import gensim.corpora as corpora
import gensim.models as models


def setup():
    tqdm.pandas()

def flatNestedList(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]

def getSemesterTermFrequencyMatrixFrom(dataframe, column='Unigrams', min_freq=2, max_freq=500, max_features=100000):
    print('Counting term frequency')
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
    print('Normalizing')
    return df.div(df.sum(axis=0), axis=1)*100000

def getPoisson(df):
    print ('Calculating poisson percentages')
    index = df.index
    columns = df.columns
    p = pd.DataFrame(poisson.cdf(k=df.loc[:,2:len(df.columns)],mu=df.loc[:,1:len(df.columns)-1]))
    p.columns = columns[1:]
    p.index = index
    return p


def generateTrends(df, columns, size, threshold):
    print('Creating xls file')
    ll=[]
    for c in df.columns:
        ll.append(np.array(df[df.loc[:,c] < threshold].sort_values(by=[c],ascending=True)[:size].loc[:,c].index))
    trends = pd.DataFrame(ll).T
    trends.columns = columns[1:]
    return trends

def readData(filename):
    print('Reading data....')
    start = time.time()
    df = pd.read_pickle(filename)
    end = time.time()
    print(f'Read finished in {end-start:.2f} seconds.\n')
    return df



print('Generating Trends')
start = time.time()
setup()
config = ConfigParser(inline_comment_prefixes="#;", interpolation=ExtendedInterpolation())
config.read('config.ini')
inputfile = config['Text Cleaning']['tokenized_file']
output = config['General']['output_file']
writer = pd.ExcelWriter(output, engine='xlsxwriter')
df= readData(inputfile)
for column in ['Unigrams', 'Bigrams']:
    semterm, columns = getSemesterTermFrequencyMatrixFrom(df, column)
    semterm = normalize(semterm)
    p = getPoisson(semterm)
    trends = generateTrends(p, columns, 1000, 0.05)
    trends.to_excel(writer, sheet_name=column)
writer.save()
end = time.time()
print(f'Excel file generated in {end-start:.2f} seconds.\n')
