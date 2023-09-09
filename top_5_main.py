
if __name__ == '__main__':
    import gensim
    from gensim.models.doc2vec import Doc2Vec
    from preprocessing_utils import preprocess_text
    from summary_utils import textrank_summarize
    from top_5_utils import get_top_5_wines


    description = input('Enter a description of the wine')

    text = preprocess_text(description)
    summary = textrank_summarize(text)

    model = Doc2Vec.load('./d2v2.model')

    get_top_5_wines(summary, model=model)



