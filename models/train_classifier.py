# set base path
#path='/Users/bernd/Documents/Codes/Python/Course_Udacity_Data_Scientist/05_Data_Engineering/Git_develop2/'
#path='/home/bernd/Documents/Python/Udacity_Data_Scientist/05_Emergency_Response/Disaster_Response_Pipelines/'
# path linux
path='/home/bernd/Documents/Python/Disaster_Response_Pipelines/'


# import libraries
import pdb

import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
#import nltk.tokenize.punkt
nltk.download('wordnet')
#from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
import sys
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.base import BaseEstimator, TransformerMixin

import pickle




class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)



def load_data(database_filepath):
    """
    function loads from database and prepares data
    input...filename in database
    output...features base, outcomes, outcome names
    """

    # load
    engine = create_engine('sqlite:///'+path+database_filepath)
    df = pd.read_sql("SELECT * FROM InsertTableName", con=engine)

    # prepare
    # X
    X = df.message.values

    # Y
    # inititalize Y
    Y_df = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    # additionally cleanse values from Y which do not contain at least two class values
    drop_col = list(Y_df.columns[Y_df.nunique() < 2])
    # drop additional columns from data frame
    Y_df2 = Y_df.drop(drop_col, axis=1)
    # Save column names
    Y_names = list(Y_df2.columns)
    # convert to array
    Y = Y_df2.values

    return X, Y, Y_names



def tokenize(text):
    """
    function tokenize
    input...text column of data frame
    output...clean tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
        ('MO', MultiOutputClassifier(estimator=LogisticRegression(C=10)))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # Function to print classification report
    def print_classification_report(y_pred, Y_names):
        for classifier in range(0,y_pred.shape[1]):
            print('Classifier: '+Y_names[classifier])
            target_names = ['class 0', 'class 1','class 2']
            #target_names = ['class 0', 'class 1']
            print(classification_report(Y_test[:,classifier], y_pred[:,classifier],target_names=target_names[0:Y_test[:,classifier].max()+1]))

        accuracy = (y_pred == Y_test).mean()
        print('\n Total Accuracy:', accuracy)

    y_pred = model.predict(X_test)
    #pdb.set_trace()
    print_classification_report(y_pred, category_names)


def save_model(model, model_filepath):
    # open a file, where you ant to store the data
    file = open(model_filepath, 'wb')

    # dump information to that file
    pickle.dump(model, file)

    # close the file
    file.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()