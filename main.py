  ## Import useful libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import html
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import LinearSVC

import random

SEED = 888
np.random.seed(SEED)
random.seed(SEED)


def clean_data(df):
  df = df.fillna('')
  for col in ['article', 'title', 'source']:
    df[col] = df[col].str.casefold().str.strip()
  df.loc[df['article'].str.startswith('read full'), 'article'] = ''
  df['text'] = 2 * df['title'] + ' ' + df['article']
  df = html.escape(df)
  y = pd.Series()
  if 'label' in df.columns:
    mask = df['label'] == 5
    df.loc[mask, 'text'] = 6 * df['title'] + ' ' 3 * df['article']
    df.drop_duplicates(subset = 'article', inplace=True)

    y = df['label']
  X = df[['text', 'source', 'page_rank']]
  return X, y

def load_data(train_path, test_path):
  df_train_val = pd.read_csv(train_path)
  df_eval = pd.read_csv(test_path)
  return df_train_val, df_eval



def get_pipeline(model_params = None):
  if model_params is None:
    model_params = {}

  preprocessor = ColumnTransformer(
    [('tfidf',TfidfVectorizer(stop_words='english'), 'text'),
     ('source', OneHotEncoder(handle_unknown = 'ignore'), ['source']),
    ('page_rank', OrdinalEncoder(handle_unknown = 'error'), ['page_rank'])
  ])

  pipeline = Pipeline(
    [('preprocessor', preprocessor),
     ('model', LinearSVC(random_state=SEED, dual=False))]
  )
  if model_params:
    pipeline.set_params(**model_params)
  return pipeline


def main():
  df_train_val, df_eval = load_data('development.csv', 'evaluation.csv')
  X_train_val, y_train_val = clean_data(df_train_val)
  X_eval, y_eval = clean_data(df_eval)

  params = {'preprocessor__tfidf__sublinear_tf': True, 
            'preprocessor__tfidf__ngram_range': (1, 3), 
            'preprocessor__tfidf__min_df': 10, 
            'preprocessor__tfidf__max_features': 30000, 
            'preprocessor__tfidf__max_df': 0.7, 
            'preprocessor__tfidf__binary': False, 
            'model__max_iter': 3000, 
            'model__class_weight': 'balanced', 
            'model__C': np.float64(0.1)}
  
  model = get_pipeline(params)

  model.fit(X_train_val, y_train_val)

  y_eval = model.predict(X_eval)

  submission = pd.DataFrame({
     'Id' : range(len(y_eval)),
    'Predicted': y_eval
    })
  submission.to_csv('submission.csv', index=False)

  
  return submission


if __name__ == '__main__':
  submission = main()