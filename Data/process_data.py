import sys
import numpy as np
import pandas as pd
from time import time
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    function load_data
    loads input data
    """
    message_file="messages.csv"
    messages=pd.read_csv(messages_filepath+"/"+message_file)
    
    categories_file="categories.csv"
    categories=pd.read_csv(categories_filepath+"/"+categories_file)


def clean_data(df):
    """
    function clean_data
    creates flag 1, 0 of input categories
    """
    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id', how='outer')
    
    # clean categories column
    categories_df=df['categories']
    categories = categories_df.str.split(";", expand=True)
    
    # extract row names
    row = categories.iloc[1,:]
    
    # extract string without last two characters
    category_colnames = []
    category_colnames = row.apply(lambda x: x[0:len(x)-2])
    
    # assign column names
    categories.columns = category_colnames

    # instead of string, assign numeric flag - 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[len(x)-1] ).astype(int)
    
    # drop original categories column
    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate dataframe and categories
    frames = [df, categories]
    df=pd.concat(frames, axis=1)

    # drop duplicate rows
    df.drop_duplicates(inplace=True)
    
def save_data(df, database_filename):
    """
    function save_data
    stores data in sql database
    """
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df.to_sql('InsertTableName', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()