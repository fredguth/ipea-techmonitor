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
from nltk.corpus import stopwords
from configparser import ConfigParser, ExtendedInterpolation

def setup():
    start = time.time()
    print ('Running setup...')
    tqdm.pandas()
    resources = ['taggers/averaged_perceptron_tagger', 'corpora/wordnet', 'corpora/stopwords', 'tokenizers/punkt']
    for path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(path.split('/')[1])
    end = time.time()
    print (f'Setup finished in {end-start:.2f} seconds.\n')

def days_to_date(srl_no):
    args = list(map(int, [number.strip() for number in config['General']['first_date'].split(',')]))
    first = datetime.datetime(args[0], args[1], args[2])
    days = int(srl_no-1)
    new_date = first + datetime.timedelta(days)
    return new_date


def read_data(filename):
    print('Reading data...')
    start = time.time()
    tweets = pd.read_csv(filename, encoding='latin-1', sep=';',header=0, names=['StringDate', 'Days', 'From', 'Tweet'])
    tweets = tweets.filter(items=['Days', 'From', 'Tweet'])
    if config.getboolean('General', 'convert_date'):
        tweets['Days'] = tweets['Days'].progress_apply(days_to_date)
    tweets.columns=['Date', 'From', 'Tweet']
    tweets = tweets[~tweets['From']
        .isin(
            [source.strip().lower()
                for source in 
                    config['General']['exclude_sources'].split(",")])]
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
    actions = [action.strip().lower() for action in config['Text Cleaning']['actions'].split(",")]
    for action in actions:
        tqdm.write('.... ' + action)
        df[text_column] = df[text_column].progress_apply(globals()[action])
    end = time.time()
    tqdm.write(f'Text cleanup finished in {end-start:.2f} seconds.\n')
    return df


def tokenize(df, text_column):
    print(f'Tokenizing Dataframe["{text_column}"].')
    start = time.time()
    df['Unigrams'] = df[text_column].progress_apply(tknzr.tokenize)
    end = time.time()
    print(f'Dataframe["{text_column}"] tokenized in {end-start:.2f} seconds.\n')
    return df


def remove_stopwords(input, stops):
    output = [i for i in input if i not in stops]
    return output

def remove_extremewords(input, min, max):
    output = [i for i in input if (len(i)<=max and len(i)>=min)]
    return output


def clean_tokens(df):
    tqdm.write('Cleaning up tokens...')
    start = time.time()
    tqdm.write('.... removing extreme words')
    min = config['Text Cleaning'].getint('min_word_size') or 2
    max = config['Text Cleaning'].getint('max_word_size') or 20
    df['Unigrams'] = df['Unigrams'].progress_apply(lambda x: remove_extremewords(x, min, max))
    tqdm.write('.... removing stop words')
    ll = [stopwords.words('english') + list(punctuation)] + ["".join(string.split()).split(',') for string in [v for k, v in config.items('Stop Words')]]
    flat = [item for sublist in ll for item in sublist]
    stops = set(flat)
    df['Unigrams'] = df['Unigrams'].progress_apply(lambda x: remove_stopwords(input=x, stops=stops))
    tqdm.write('.... generating bigrams')
    df['Bigrams'] = df['Unigrams'].progress_apply(lambda x: [f'{tuple[0]} {tuple[1]}' for tuple in list(bigrams(x))])
    end = time.time()
    tqdm.write (f'Tokens cleanup finished in {end-start:.2f} seconds.\n')
    return df

#main
start = time.time()
setup()
config = ConfigParser(
    inline_comment_prefixes="#;",
    interpolation=ExtendedInterpolation())
config.read('config.ini')
input = config['General']['input_file']
output = config['Text Cleaning']['tokenized_file']
text_column = config['General']['input_file_text_column']

tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
lmtzr = WordNetLemmatizer()
cleaned = clean_text(read_data(input), text_column)
clean_tokens(tokenize(cleaned, text_column)).to_pickle(output)
end = time.time()
print(f'Data Preparation finished in {end-start:.2f} seconds.\n')

