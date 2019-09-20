
# coding: utf-8

# In[1]:


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


# In[2]:


def setup():
    tqdm.pandas()

def flatNestedList(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]


# In[3]:


def readData(filename):
    print('Reading data....')
    start = time.time()
    df = pd.read_pickle(filename)
    end = time.time()
    print(f'Read finished in {end-start:.2f} seconds.\n')
    return df


# In[4]:


setup()
config = ConfigParser(inline_comment_prefixes="#;", interpolation=ExtendedInterpolation())
config.read('config.ini')
inputfile = config['Text Cleaning']['tokenized_file']
output = config['General']['output_file']
writer = pd.ExcelWriter(output, engine='xlsxwriter')
df= readData(inputfile)


# In[5]:


def getSemesterTermFrequencyMatrixFrom(dataframe, column='Unigrams', min_freq=2, max_freq=500, max_features=100000, vocab=None):
    print ('Generating Semester x Term matrix')
    df = pd.DataFrame(dataframe[column])
    df = df.resample('D',closed='left', label='left').apply(flatNestedList)
    cv = CountVectorizer(tokenizer=(lambda x: x), preprocessor=(lambda x: x), vocabulary=vocab, min_df=min_freq, max_df=max_freq, max_features=max_features)
    table = cv.fit_transform(df[column])
    docterm=pd.DataFrame(table.todense())
    docterm.index = df.index
    docterm.columns = cv.get_feature_names()
    semterm = docterm.resample('2QS',closed='left', label='left').sum()
    semterm=semterm.T
    semterm.columns = [ f'{column.year}-{(column.quarter+1)//2}' for column in list(semterm.columns)]
    return semterm, cv.vocabulary_ 


# In[7]:


def applyMask(df, mask):
    mask.loc[list(df.index), list(df.columns)]=df
    return mask


# In[8]:


def getBoostTerm(df, semterm, vocab):
    print ('Generating Semester x Term x Source matrix')
    mask = pd.DataFrame().reindex_like(semterm)
    mask = mask.fillna(0)
    sources = []
    for source in tqdm(df['From'].unique()):
        s, _ = getSemesterTermFrequencyMatrixFrom(df[df['From']==source], min_freq=1, vocab=vocab)
        s = applyMask(s, mask)
        sources.append(s.to_numpy())
    stack =np.stack(sources)
    u_stack = (stack!=0).astype(int)
    count = semterm.to_numpy()
    sources = u_stack.sum(axis=0)
    boost = (5+count-sources)/(4+count)
    bdf = pd.DataFrame(boost)
    bdf.index = semterm.index
    bdf.columns = semterm.columns
    return bdf


# In[10]:


def generateTrends(df, columns, size, threshold):
    print('Creating xls file')
    ll=[]
    for c in df.columns:
        ll.append(np.array(df[df.loc[:,c] < threshold].sort_values(by=[c],ascending=True)[:size].loc[:,c].index))
    trends = pd.DataFrame(ll).T
    trends.columns = columns[1:]
    return trends


# In[11]:


def normalize(df):
    print('Normalizing')
    return df.div(df.sum(axis=0), axis=1)*100000


# In[12]:


def getPoisson(df, transform=None):
    print ('Calculating poisson percentages')
    index = df.index
    columns = df.columns
    p = pd.DataFrame(poisson.cdf(k=getK(df, transform=transform), mu=df.loc[:,2:len(df.columns)]))
    p.columns = columns[1:]
    p.index = index
    return p


# In[13]:


def getK(df, transform=None, past=3):
    if transform=='max':
        table = np.zeros(shape=df.shape)
        for i, (index, row) in tqdm(enumerate(df.iterrows())):
            for j in range(len(df.columns)-1): 
                table[i,j] = max(row[:j+1])
    if transform=='mean':
        table = np.zeros(shape=df.shape)
        for i, (index, row) in tqdm(enumerate(df.iterrows())):
            for j in range(len(df.columns)-1):
                bound = max(0,j-past)
                table[i,j] = row[bound:j+1].mean()
        df = pd.DataFrame(table, index = df.index, columns=df.columns)
        
    return df.loc[:,1:len(df.columns)-1]


# In[14]:


start = time.time()
for column in ['Unigrams', 'Bigrams']:
    print(f'Processing {column}')
    semterm, vocab = getSemesterTermFrequencyMatrixFrom(df, column)
    columns = semterm.columns
    semterm = normalize(semterm)
    boost = getBoostTerm(df, semterm, vocab)
    semterm.columns = np.arange(1,len(semterm.columns)+1).astype(int)
    boost.columns = np.arange(1,len(boost.columns)+1).astype(int)
    p = getPoisson(semterm)
    p = p * boost.loc[:,2:]
    trends = generateTrends(p, columns, 1000, 0.05)
    trends.to_excel(writer, sheet_name=column)
writer.save()
end = time.time()
print(f'Excel file generated in {end-start:.2f} seconds.\n')


# In[16]:




