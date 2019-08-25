# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import re
import unicodedata
import time
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk import bigrams
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import TweetTokenizer
from string import punctuation


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
    contractions = ["ain't", "amn't", "aren't", "S'e", "Ha'ta", "can't", "cain't", "'cause", "could've", "couldn't", "couldn't've", "daren't", "daresn't", "dasn't", "didn't", "doesn't", "don't", "e'er", "everyone's", "finna", "gimme", "giv'n", "gonna", "gon't", "gotta", "hadn't", "hasn't", "haven't", "he'd",
    "he'll", "he's", "he've", "how'd", "howdy", "how'll", "how're", "how's", "I'd", "I'll", "I'm", "I'm'a", "I'm'o", "I've", "isn't", "it'd", "it'll", "it's", "let's", "ma'am", "mayn't", "may've", "mightn't", "might've", "mustn't", "mustn't've", "must've", "needn't", "ne'er", "o'clock", "o'er", "ol'", "oughtn't"]
    too_common = ["awards", "bowl", "election", "elections", "oscars", "olympics", "cup", "christmas", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "wwdc", "ces", "yesterday", "today", "tomorrow", "bloomberg", "gizmodo", "mashable", "forbes", "huffington", "huffpo", "waspo", "post", "techcrunch", "netflix", "hulu", "amazon", "google", "apple", "google", "facebook", "microsoft", "twitter", "snapchat",
    "pokemon", "whatsapp", "groupon", "mozilla", "chrome", "firefox", "softbank", "wikileaks", "wikipedia", "telegram", "tumblr", "samsung", "disney", "hbo", "iphone", "android", "4s", "5s", "6s", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "westworld", "mark", "zuckerberg", "bezos", "gates", "musk", "elon", "bill", "steve", "jobs", "clinton", "obama", "hilary", "trump", "bernie", "warren", "brazil", "canada", "australia", "us", "chicago", "ny", "sf", "francisco", "angeles", "york", "uk", "adele", "aol", "appl", "assange", "avengers", "ballmer", "batman", "beatles", "csco", "donald", "dorsey", "edward", "egypt", "equifax", "godaddy", "gopro", "icloud", "ipod", "iwatch", "kaspersky", "keynote", "kodak", "lexus", "marissa", "meerkat", "nobel", "olympic", "oscar", "ozzie", "polaroid", "quora", "rio", "siemens", "snowden", "sxsw", "tarantino", "thanksgiving", "toshiba", "turkey", "washington", "wii", "xperia", "zappos", "zynga", "ericson", "foxconn", "ipad", "galaxy", "playstation", "htc", "kindle", "sony", "ericsson", "new", "intel" , "ios", "osx", "macbook", "mac", "goog", "youtube", "gmail", "uber", "macos", "thanksgiven", "jan", "feb", "apr", "jun", "jul", "aug", "sept", "sep", "oct", "nov", "asus", "nexus", "aapl", "ev", "skype", "larry", "fb", "spotify", "pichai", "paul", "beatles", "nasa", "tesla", "spacex", "apples", "microsofts", "googles", "facebooks", "billion", "million", "billions", "millions", "raise", "ipo", "funding", "instagram", "pinterest", "france", "germany", "italy", "spain", "russia", "jones", "says", "richard", "lockheed", "boeing", "sandberg", "seattle", "ipads", "macs", "buys", "slack", "nintendo", "saudi", "california", "korea", "elizabeth", "musks", "zuckerbergs", "t-mobile", "huawei", "hp", "windows", "aws", "azure", "tony", "china", "eric", "nokia", "ibm", "spielberg", "ceo", "verizon"]
    stop_words = set(stopwords.words('english') + list(punctuation) + [' ', 'rt', '...', '..', '....', '/:','-->', ']: ', '}: ']+too_common+contractions)
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
    df=df[df['NumTokens']<40]
    df=df[df['NumTokens']>1]
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

