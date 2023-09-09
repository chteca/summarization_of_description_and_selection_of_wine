
import gensim
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import pandas as pd

def get_top_5_wines(text:str, model:gensim.models.doc2vec, dataset: pd.DataFrame = pd.read_csv('./dataset_with_summary.csv')):
    '''

    Args:
    text: wine summary
    model: Doc2Vec model used for comparing the wine summary with the dataset
    dataset: the dataset containing wine information

    The function prints top five most similar wine documents from the dataset (title, point, variety, and price) using the indices from the similarities.. 
    
    Example usage:
    get_top_5_wines('wine summary', d2v.model)

    Output:
    TOP-1
    Title: Rainstorm 2013 Pinot Gris (Willamette Valley)
    Point: 87
    Variety: Pinot Gris
    Price: 14.0 

    TOP-2
    Title: Lucie 2015 Dutton Ranch Widdoes Vineyard Pinot Noir (Russian River Valley)
    Point: 92
    Variety: Pinot Noir
    Price: 60.0 

    TOP-3
    Title: Federico Paternina 2007 Banda Azul Crianza Red  (Rioja)
    Point: 81
    Variety: Tempranillo Blend
    Price: 10.0 

    TOP-4
    Title: Coopers Creek 2013 Pinot Noir (Hawke's Bay)
    Point: 85
    Variety: Pinot Noir
    Price: 16.0 
    ...
    Point: 87
    Variety: Portuguese Red
    Price: 17.0 
    '''
    test_data = word_tokenize(text)
    inferred_vector = model.infer_vector(test_data)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

    print('Test Document : «{}»\n'.format(' '.join(test_data)))
    for label, index in [('TOP-1', 0), ('TOP-2', 1), ('TOP-3', 2), ('TOP-4', 3), ('TOP-5', 4)]:
        print(u'%s\nTitle: %s\nPoint: %s\nVariety: %s\nPrice: %s \n' % (label, dataset.title.iloc[int(sims[index][0])], dataset.points.iloc[int(sims[index][0])], dataset.variety.iloc[int(sims[index][0])], dataset.price.iloc[int(sims[index][0])],))
