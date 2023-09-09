
import spacy
import string
import re
from nltk.corpus import stopwords
import nltk
import pandas as pd


def normalize_text(text: str) -> str:
    '''
    Returns the normalized text without any line breaks.

    Args: 
    text: text to normalize

    Example usage:
    normalize_text('Pineapple rind, lemon pith and orange blossom start off the aromas.')
    '''
    tm1 = re.sub('<pre>.*?</pre>', '', text, flags=re.DOTALL)
    tm2 = re.sub('<code>.*?</code>', '', tm1, flags=re.DOTALL)
    tm3 = re.sub('<[^>]+>©', '', tm1, flags=re.DOTALL)
    return tm3.replace("\n", "")


def cleanup_text(text: str, stopwords:list = stopwords.words('english')):
    '''
    Cleanups text by removing personal pronouns, stopwords, and puncuation and returns a series of cleaned text.

    Arg:
    text: the input text documents
    stopwords: a list of stopwords to be removed from the text

    Example usage:
    cleanup_text('Hello, World!')
    '''
    nlp = spacy.load('en_core_web_lg')
    punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~©'
    texts = []
    doc = nlp(text, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    tokens = ' '.join(tokens)
    return tokens


def preprocess_text(text:str):
    '''
    Returns preprocessed text after applying normalization and cleaning

    Arg:
    text: input text to preprocess

    Example usage:
    preprocess_text('Hello, World!')
    '''

    text = normalize_text(text)
    text = cleanup_text(text)

    return text


def preprocess_data(dataset: pd.DataFrame, column: str, preprocess_column: str):
    '''
    preprocesses the text data in the specified column of the dataset and stores the preprocessed text in the preprocess column.

    Args:
    dataset: dataframe to process
    column: the text data to be preprocessed
    preprocess_column: the name of the column in the dataset where the preprocessed text will be stored.

    Return:
    The dataset with the preprocessed text stored in the preprocess column.

    Example usage:
    preprocess_data(df, 'column', 'new_column')
    '''

    dataset[preprocess_column] = dataset[column].apply(lambda x: preprocess_text(x))

    return dataset