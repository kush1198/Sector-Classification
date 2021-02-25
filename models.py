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

class DomainIdLSTM(nn.Module):
  """
  The RNN model that will be used to perform domain identification.
  """

  def __init__(self, vocab_size, output_size, embedding_mat, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
    """
    Initialize the model by setting up the layers.
    """
    super().__init__()

    self.output_size = output_size
    self.n_layers = n_layers
    self.hidden_dim = hidden_dim
    
    # embedding and LSTM layers
    self.embedding = nn.Embedding.from_pretrained(embeddings=embedding_mat,freeze=False)

    # NOTE: batch_first=True 
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
    
    # dropout layer
    self.dropout = nn.Dropout(0.3)
    
    # linear and sigmoid layers
    self.fc = nn.Linear(hidden_dim, output_size)        

  def forward(self, x, hidden):
    """
    Perform a forward pass of our model on some input and hidden state.
    """
    batch_size = x.size(0)

    # embedding: Input: #(batch_size,seq_length)
    # embedding: Output: #(batch_size,seq_length,embedding_dim)
    embeds = self.embedding(x)

    # LSTM: Input: embedding
    # LSTM: Output: 1) #(batch_size,seq_length,hidden_dim)   2) Tuple of size=2 each element of  
    # shape (num_layers*directions,batch_size,hidden_dim)
    lstm_out, hidden = self.lstm(embeds, hidden)

    lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
    
    # dropout and fully-connected layer
    out = self.dropout(lstm_out)
    out = self.fc(out)

    soft_out = F.softmax(out,dim=1)
    
    # reshape to be batch_size first
    soft_out = soft_out.view(batch_size, -1, self.output_size)
    soft_out = soft_out[:, -1, :] # get last batch of labels
    
    # return last sigmoid output and hidden state
    return torch.log(soft_out), soft_out, hidden
  
  
  def init_hidden(self, batch_size):
    ''' Initializes hidden state '''
    # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
    # initialized to zero, for hidden state and cell state of LSTM
    weight = next(self.parameters()).data
    
    if (False):
      hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(), 
      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
    else:
      hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
    
    return hidden

# Model Definition with Conv1D
def CNN_tf(vocabulary_size,embed_size,seq_length,embedding_matrix,pool_size,output_size,filters,kernel_size):
  tf.random.set_seed(0)
  model_cnn = tf.keras.Sequential([
      if embedding_matrix==None:
        tf.keras.layers.Embedding(vocabulary_size, embed_size, input_length=seq_length,trainable=True),
      else:
        tf.keras.layers.Embedding(vocabulary_size, embed_size, input_length=seq_length,trainable=True,embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix)),
      tf.keras.layers.Conv1D(filters,kernel_size),
      tf.keras.layers.MaxPooling1D(pool_size=(pool_size)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(output_size, activation='softmax')
  ])
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
                   # SVC__gamma=[0.1,0.9 ,1,1.1,10]
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
                   # model__n_estimators = [10, 100, 1000],
                   # model__max_depth = [5, 8, 15, 25, 30, None]
                   )
  elif Model == 'KNN':
    model = KNeighborsClassifier()
    #print(model.get_params)
    param_grid = dict(features__k=np.arange(start,end,2),
                    #model__n_neighbors =range(1, 21, 3),
                   )
  elif Model == 'ANN':
    model = MLPClassifier(activation='relu',solver='adam',random_state=2020)
    param_grid = dict(features__k=np.arange(start,end,1),
                    model__hidden_layer_sizes = [(1,),(30,)],
                    model__alpha = [0.00005,0.0005])
  elif Model == 'XGB':
    model = XGBClassifier(activation='relu',solver='adam',random_state=2020)
    param_grid = dict(features__k=np.arange(start,end,3),
                   # model__learning_rate = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
                   # model__max_depth = [ 3, 4, 5, 6, 8, 10, 12, 15],
                   # model_gamma = [ 0.0, 0.1, 0.2 , 0.3, 0.4 ]
                   )
  
  grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=10,scoring='accuracy')
  grid_search.fit(train_f, ytrain)
  return grid_search.best_estimator_

