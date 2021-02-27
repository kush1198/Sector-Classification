from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.stats import randint as sp_randint
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import torch
import torch.nn.functional as F
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional
from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical
import torch.nn as nn
import numpy as np


def LSTM_tf(vocabulary_size,embed_size,embedding_matrix,hidden_dim,output_size,seq_length):
  tf.random.set_seed(10)
  model_lstm = Sequential()
  if embedding_matrix==None:
    model_lstm.add(Embedding(vocabulary_size, embed_size, input_length=seq_length,trainable=True))
  else:
    model_lstm.add(Embedding(vocabulary_size, embed_size, input_length=seq_length,trainable=True,embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix)))
  model_lstm.add(Bidirectional(LSTM(hidden_dim)))
  model_lstm.add(Dense(output_size, activation='softmax'))
  model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model_lstm


def CNN_tf(vocabulary_size,embed_size,seq_length,embedding_matrix,pool_size,output_size,filters,kernel_size):
    tf.random.set_seed(0)
    if embedding_matrix==None:
        model_cnn = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabulary_size, embed_size, input_length=seq_length,trainable=True),
        tf.keras.layers.Conv1D(filters,kernel_size),
        tf.keras.layers.MaxPooling1D(pool_size=(pool_size)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_size, activation='softmax')
        ])
    else:
        model_cnn = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabulary_size, embed_size, input_length=seq_length,trainable=True,embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix)),
        tf.keras.layers.Conv1D(filters,kernel_size),
        tf.keras.layers.MaxPooling1D(pool_size=(pool_size)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_size, activation='softmax')
        ])
    model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_cnn

def tuning(Model,train_f,ytrain):
  selection = SelectKBest()
  start,end = (int)(train_f.shape[1]/2),train_f.shape[1]
  model = XGBClassifier(activation='relu',solver='adam',random_state=2020)
  pipeline = Pipeline([("features", selection), (Model, model)])

  if Model == 'SVC':
    model = SVC(kernel="rbf",random_state=2020)
    print(pipeline.get_params)
    param_grid = dict(features__k=np.arange(start,end,3),
                    SVC__C=[0.9, 1, 1.1],
                    SVC__gamma=[0.1,0.9 ,1,1.1,10]
                    )
  elif Model == 'BDT':
    model = BaggingClassifier(random_state=2020)
    param_grid = dict(features__k=np.arange(start,end,1),
                    model__n_estimators = [10, 100, 1000],)    
  elif Model == 'LR':
    model = LogisticRegression(penalty='l2',random_state=2020)
    param_grid = dict(features__k=np.arange(start,end,1),
                    model__solver=['newton-cg', 'lbfgs', 'liblinear'],
                    model__c_values=[100, 10, 1.0, 0.1, 0.01])    
  elif Model == 'RF':
    model = RandomForestClassifier(random_state=2020)
    param_grid = dict(features__k=np.arange(start,end,3),
                    model__n_estimators = [10, 100, 1000],
                    model__max_depth = [5, 8, 15, 25, 30, None]
                   )
  elif Model == 'KNN':
    model = KNeighborsClassifier()
    #print(model.get_params)
    param_grid = dict(features__k=np.arange(start,end,2),
                    model__n_neighbors =range(1, 21, 3),
                   )
  elif Model == 'ANN':
    model = MLPClassifier(activation='relu',solver='adam',random_state=2020)
    param_grid = dict(features__k=np.arange(start,end,1),
                    model__hidden_layer_sizes = [(1,),(30,)],
                    model__alpha = [0.00005,0.0005])
  elif Model == 'XGB':
    model = XGBClassifier(activation='relu',solver='adam',random_state=2020)
    param_grid = dict(features__k=np.arange(start,end,3),
                    model__learning_rate = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
                    model__max_depth = [ 3, 4, 5, 6, 8, 10, 12, 15],
                    model_gamma = [ 0.0, 0.1, 0.2 , 0.3, 0.4 ]
                   )
  
  grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=10,scoring='accuracy')
  grid_search.fit(train_f, ytrain)
  return grid_search.best_estimator_

