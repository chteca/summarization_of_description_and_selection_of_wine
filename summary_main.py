
if __name__ == '__main__':
    from preprocessing_utils import preprocess_text
    from summary_utils import textrank_summarize
    import pandas as pd

    description = input('Enter a description of the wine')

    text = preprocess_text(description)
    summary = textrank_summarize(text)

    print('Summary:', summary)



