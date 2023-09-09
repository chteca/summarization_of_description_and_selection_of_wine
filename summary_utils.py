
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def textrank_summarize(text: str, num_sentences: int = 2):
    '''
    Returns summarized text

    Args:
    text: text to summarize
    num: number of sentences in summarized text, default=2

    Example usage:
    textrank_summarize('Pineapple rind. Lemon pith and orange blossom start off the aromas. The palate is a bit more opulent, semidry finish.', 2)
    '''
    sentences = sent_tokenize(text)
    sentence_matrix = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        sentence_matrix.append(' '.join(words))
    
    vectorizer = CountVectorizer().fit_transform(sentence_matrix)
    sentence_bag_of_words = vectorizer.toarray()
    
    similarity_matrix = cosine_similarity(sentence_bag_of_words)
    
    sentence_graph = nx.from_numpy_array(similarity_matrix)

    sentence_ranks = nx.pagerank(sentence_graph)

    ranked_sentences = sorted(((sentence_ranks[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
    summary_sentences = [sentence for rank, sentence in ranked_sentences[:num_sentences]]
    
    summary = ' '.join(summary_sentences)
    return summary

def summarize_data(dataset: pd.DataFrame, description_col: str, summary_column: str = 'summary'):
    '''
    Returns dataset with summary column

    Args:
    dataset: dataframe with data to process
    description_col: dataframe column based on which summarization will be performed
    summary_column: the name of the new dataframe column in which the summary will be written

    Example usage:
    summarize_data(df, 'description')
    '''

    dataset[summary_column] = dataset[description_column].apply(lambda x: textrank_summarize(x, 2))
    return dataset

