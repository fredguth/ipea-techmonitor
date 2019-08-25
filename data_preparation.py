# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import re
import unicodedata
import time
from tqdm import tqdm
import nltk
from nltk import word_tokenize, pos_tag
from nltk import bigrams
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import TweetTokenizer
from string import punctuation
from stops import stop_words

def setup():
    start = time.time()
    tqdm.pandas()
    print ('Running setup...')
    resources = ['taggers/averaged_perceptron_tagger', 'corpora/wordnet', 'corpora/stopwords', 'tokenizers/punkt']
    for path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(path.split('/')[1])
    end = time.time()
    print (f'Setup finished in {end-start:.2f} seconds.\n')

def days_to_date(srl_no, first=datetime.datetime(2010,1,1,0,0)):
    days = int(srl_no-1)
    new_date = first + datetime.timedelta(days)
    return new_date


def read_data(filename):
    print('Reading data...')
    start = time.time()
    tweets = pd.read_csv(filename, encoding='latin-1', sep=';',header=0, names=['StringDate', 'Days', 'From', 'Tweet'])
    tweets = tweets.filter(items=['Days', 'From', 'Tweet'])
    tweets['Days'] = tweets['Days'].progress_apply(days_to_date)
    tweets.columns=['Date', 'From', 'Tweet']
    tweets = tweets[tweets['From'] != '@mashable']
    tweets = tweets.reset_index()
    tweets = tweets.set_index('Date').sort_index()
    end = time.time()
    print (f'Data read in {end-start:.2f} seconds.\n')
    return tweets


def remove_accents(text):
    text = unicodedata.normalize('NFD', str(text)).encode('ascii', 'ignore').decode("utf-8").lower()
    return str(text)

def remove_apostrophes(text):
    text = re.sub(r"\'s", "", text)
    return text

def remove_hashtags(text):
    #hashtags and handles
    text = re.sub(r'\B(\#([0-9]|[a-zA-Z])+|\@([0-9]|[a-zA-Z])+\b)', '', text)
    return text
def remove_urls(text):
    text= re.sub(r'http\S+', '', text)
    return text

def remove_numberwords(text):
    text= re.sub(r'\b[0-9]+\b\s*', '', text)
    return text

def clean_text(df, text_column):
    tqdm.write(f'Cleaning up {text_column} texts...')
    start = time.time()
    tqdm.write('.... removing accents')
    df[text_column] = df[text_column].progress_apply(remove_accents)
    tqdm.write('.... removing URLs')
    df[text_column] = df[text_column].progress_apply(remove_urls)
    tqdm.write('.... removing hashtags')
    df[text_column] = df[text_column].progress_apply(remove_hashtags)
    tqdm.write('.... removing apostrophes')
    df[text_column] = df[text_column].progress_apply(remove_apostrophes) 
    tqdm.write('.... removing numbers')
    df[text_column] = df[text_column].progress_apply(remove_numberwords) 
    # tqdm.write('.... expanding contractions')
    # df[text_column] = df[text_column].progress_apply(expand_text)
    end = time.time()
    tqdm.write (f'Text cleanup finished in {end-start:.2f} seconds.\n')
    return df


def _tokenize(text):
    new_text = []
    for word, tag in pos_tag(tknzr.tokenize(text)):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 's'] else None
        if wntag:  # remove verbs
            lemma = lmtzr.lemmatize(word, wntag)
            new_text.append(lemma)
    return new_text


def tokenize(df, text_column):
    print(f'Tokenizing Dataframe["{text_column}"].')
    start = time.time()
    #df['Unigrams'] = df[text_column].progress_apply(_tokenize)
    df['Unigrams'] = df[text_column].progress_apply(tknzr.tokenize)
    end = time.time()
    print(f'Dataframe["{text_column}"] tokenized in {end-start:.2f} seconds.\n')
    return df


def remove_stopwords(input):
    output = [i for i in input if i not in stop_words]
    return output

def remove_extremewords(input):
    output = [i for i in input if (len(i)<20 and len(i)>1)]   
    return output


def clean_tokens(df):
    tqdm.write('Cleaning up tokens...')
    start = time.time()
    tqdm.write('.... removing stop words')
    df['Unigrams'] = df['Unigrams'].progress_apply(remove_stopwords)
    tqdm.write('.... removing extreme words')
    df['Unigrams'] = df['Unigrams'].progress_apply(remove_extremewords)
    tqdm.write('.... generating bigrams')
    df['Bigrams'] = df['Unigrams'].progress_apply(lambda x: [f'{tuple[0]} {tuple[1]}' for tuple in list(bigrams(x))])
    df['NumTokens']=df['Unigrams'].apply(len)
    df['NumBigrams']=df['Bigrams'].apply(len)
    end = time.time()
    tqdm.write (f'Tokens cleanup finished in {end-start:.2f} seconds.\n')
    return df

#main
setup()
input = './data/twitter.csv'
output = './data/tokenized.data'
text_column = 'Tweet'
tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
lmtzr = WordNetLemmatizer()
cleaned = clean_text(read_data(input), text_column)
# cleaned.to_pickle('./data/cleaned.data')
clean_tokens(tokenize(cleaned, text_column)).to_pickle(output)

