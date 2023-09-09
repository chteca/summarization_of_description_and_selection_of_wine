
import pandas as pd

def read_csv_file(file_path: str,
                  use_cols: list,
                   encoding: str = 'latin1' ) -> pd.DataFrame:
    '''
    Reads dataframe and returns the processed dataframe.

    Args:
    file_path: a path to the file
    use_cols: names of the columns required to build the model
    encoding: the encoding used when reading

    Example usage:
    read_csv_file('./wine_reviews.csv', usecols =['points', 'title', 'description', 'variety', 'price'], encoding='latin1')
    
    '''
    reviews = pd.read_csv(file_path, usecols=use_cols, encoding=encoding)
    reviews = reviews.drop_duplicates()
    reviews = reviews.dropna()
    reviews = reviews.reset_index(drop=True)
    reviews['summary'] = ''
    return reviews
