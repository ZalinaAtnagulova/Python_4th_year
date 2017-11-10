import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import model_selection

#load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df = df.rename(columns={'v1': 'target', 'v2': 'text'})
df.head()

# dataset size
df.shape

# Use  df_train for model training
# Use df_test as  hold-out dataset for your final model perfomance estimation.
# You cannot change  this splitting
# All results must be reproducible
SEED = 1337
df_train, df_test = model_selection.train_test_split(df, test_size=0.4, random_state=SEED, shuffle=True, stratify=df.target)
print('train size %d, test size %d' % (df_train.shape[0], df_test.shape[0]))

# MYbaseline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2))
X = vectorizer.fit_transform(df_train.text)
print('feature matrix shape', X.shape)

#encode labels
label_enc = LabelEncoder().fit(df_train.target)
y_train = label_enc.transform(df_train.target)

# Logistic Regression classifier has several hyperparams
# Optimize C (coeff before regularizer) and penalty (type of regularizer) using crossvalidation with grid search
# Basically it means it will look over every combination of hyperparams in the specified region (or lattice)
# and return the best one. 
# Look in docs  for more details
grid = GridSearchCV(LogisticRegression(random_state=SEED), # our model 
                   param_grid={'C': np.logspace(0,5,20), # C in lattice [10^0...10^5]
                               'penalty': ['l1', 'l2']}, 
                    scoring='f1', # our perfomance measure 
                    n_jobs=-1, # multithread 
                    cv=5, # 5-fold stratified cross-validation 
                    verbose=True, return_train_score=True)

grid.fit(X, y_train)
print('best params', grid.best_params_)
print('best estimator', grid.best_score_)
model = grid.best_estimator_

# grid.best_estimator_ is already fitted on whole train dataset
print('train', metrics.f1_score(y_train, model.predict(X)))

# perfomance on test dataset
X_test = vectorizer.transform(df_test.text)
y_pred = model.predict(X_test)
y_test = label_enc.transform(df_test.target)
print('test', metrics.f1_score(y_test, model.predict(X_test)))
