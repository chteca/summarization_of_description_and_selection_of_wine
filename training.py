
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd

def tag_data(dataset: pd.DataFrame, column:str = 'summary'):
    '''
    Tags the data in a specified column of a DataFrame using gensim's TaggedDocument.

    Args:
    dataset: dataframe with data to process
    column: the column name to extract the data from, default is 'summary'

    Returns:
    list of TaggedDocuments

    Example usage:
    tag_data(df, 'summary')
    '''
    data = dataset[column].astype(str).tolist()
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)] 
    return tagged_data


def build_model_and_train(max_epochs:int = 10, vec_size:int = 100, alpha:float = 0.025, min_alpha:float = 0.00025, min_count:int = 1, dm:int = 1, tagged:list):
    '''
    Builds and trains a Doc2Vec model.

    Args:
        max_epochs: the maximum number of training epochs, default is 10
        vec_size: the dimensionality of the feature vectors, default is 100
        alpha: the initial learning rate, default is 0.025
        min_alpha: the minimum learning rate, default is 0.00025
        min_count: ignores all words with total frequency lower than this, default is 1
        dm: training algorithm: distributed memory (1) or distributed bag of words (0), default is 1
        tagged (list): a list of TaggedDocuments to be used for training the model.

    Returns:
        gensim.models.Doc2Vec: The trained Doc2Vec model.
    '''
    model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    return model
