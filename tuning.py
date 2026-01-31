import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

import main

df_train_val, df_eval = main.load_data('development.csv', 'evaluation.csv')
X_train_val, y_train_val = main.clean_data(df_train_val)
X_eval, y_eval = main.clean_data(df_eval)
pipeline = main.get_pipeline()

param_grid = {
  'preprocessor__tfidf__max_features': [15000, 20000, 30000], 
  'preprocessor__tfidf__ngram_range': [ (1,1), (1, 3)],
  'preprocessor__tfidf__binary': [True, False],
  'preprocessor__tfidf__sublinear_tf': [True, False],
  'preprocessor__tfidf__min_df': [1, 2, 5, 10],
  'preprocessor__tfidf__max_df': [.5, .7, .9],
  'model__C': np.logspace(-1,2,6), 
  'model__class_weight': ['balanced'],
  'model__dual': [False],
  'model__max_iter': [1000,2000,3000]
}

grid_search = RandomizedSearchCV(
  pipeline,
  param_grid,
  n_jobs=4,
  cv = 2,
  scoring='f1_macro',
  verbose = 3,
  random_state=main.SEED
) 

grid_search.fit(X_train_val, y_train_val)
best_params = grid_search.best_params_

print(f'Migliori parameti: {best_params}')