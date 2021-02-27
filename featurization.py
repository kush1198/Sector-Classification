import numpy as np
import torch
from nltk.corpus import stopwords
#from tqdm import tqdm_notebook
from nltk import word_tokenize
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from numpy import hstack
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from transformers import BertTokenizer, RobertaTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional
from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical
from torch.utils.data import TensorDataset, random_split

tf.compat.v1.disable_eager_execution()

TRANSFORMERS={
    'bert':(BertTokenizer,'bert_base_uncased',)
}

def Glove_Vec(df,glove_model,idf_dict=None):
    vectors = []
    for title in df.Requirements_clean.values:
        vectors.append(np.average(glove_model.query(word_tokenize(title)), axis = 0))
    return np.array(vectors)

def tfidf_glove(df,glove_model,idf_dict=None):
    vectors = []
    for title in df.Requirements_clean.values:
        glove_vectors = glove_model.query(word_tokenize(title))
        weights = [idf_dict.get(word, 1) for word in word_tokenize(title)]
        vectors.append(np.average(glove_vectors, axis = 0, weights = weights))
    return np.array(vectors)


def text_features(df):
    longest_word_length = []
    mean_word_length = []
    length_in_chars = []

    for title in df.Requirements.values:
        length_in_chars.append(len(title))
        longest_word_length.append(len(max(title.split(), key = len)))
        mean_word_length.append(np.mean([len(word) for word in title.split()]))

    longest_word_length = np.array(longest_word_length).reshape(-1,1)
    mean_word_length = np.array(mean_word_length).reshape(-1,1)
    length_in_chars =  np.array(length_in_chars).reshape(-1,1)

    return np.concatenate([longest_word_length, mean_word_length, length_in_chars], axis = 1)


def word_ratio(args,df):
    with open(args.word_dir+'\\easy_words.txt') as f:
        easy_words_list = [line.rstrip(' \n') for line in f]
    
    with open(args.word_dir+'\\stopW.txt') as f:
        terrier_stopword_list = [line.rstrip(' \n') for line in f]

    terrier_stopword_list += stopwords.words('english')

    with open(args.word_dir+'\\common.txt') as f:
        common = [line.rstrip(' \n') for line in f]

    terrier_stopword_list += common

    with open(args.word_dir+'\\contractions.txt') as f:
        contractions_list = [line.rstrip(' \n') for line in f]

    with open(args.word_dir+'\\hyperbolic.txt') as f:
        hyperbolic_list = [line.rstrip(' \n') for line in f]


    easy_words_ratio = []
    stop_words_ratio = []
    contractions_ratio = []
    hyperbolic_ratio = []

    for title in df.Requirements.values:
        easy_words = 0 
        stop_words = 0
        total_words = 0
        contracted_words = 0
        hyperbolic_words = 0

        for word in title.split():
            if word.lower() in easy_words_list:
                easy_words += 1
            if word.lower() in terrier_stopword_list:
                stop_words += 1
            if word.lower() in contractions_list:
                contracted_words += 1
            if word.lower() in hyperbolic_list:
                hyperbolic_words += 1
            total_words += 1
            
        easy_words_ratio.append(easy_words/total_words)
        stop_words_ratio.append(stop_words/total_words)
        contractions_ratio.append(contracted_words/total_words)
        hyperbolic_ratio.append(hyperbolic_words/total_words)

    easy_words_ratio = np.array(easy_words_ratio).reshape(-1,1)
    stop_words_ratio = np.array(stop_words_ratio).reshape(-1,1)
    contractions_ratio = np.array(contractions_ratio).reshape(-1,1)
    hyperbolic_ratio = np.array(hyperbolic_ratio).reshape(-1,1)

    return np.concatenate([easy_words_ratio, stop_words_ratio, contractions_ratio, hyperbolic_ratio], axis = 1)


def featurize(args,train_df,test_df,embd,glove_model,idf_dict=None):
    print('Text Features....')
    train_text_features = text_features(train_df)
    test_text_features = text_features(test_df)
    print('Word ratios....')
    train_word_ratio = word_ratio(args,train_df)
    test_word_ratio = word_ratio(args,test_df)
    
    print('Glove.....') 
    if idf_dict!=None:
        print('IDF weighted')
    train_glove = embd(train_df, glove_model,idf_dict)
    test_glove = embd(test_df, glove_model,idf_dict)

    normalizer_glove = MinMaxScaler()
    train_glove = normalizer_glove.fit_transform(train_glove)
    test_glove = normalizer_glove.transform(test_glove)

    train_embedding_features = train_glove
    test_embedding_features = test_glove
    train_features = hstack((train_text_features,train_word_ratio,))

    normalizer = MinMaxScaler()
    train_features = normalizer.fit_transform(train_features)

    train_features = np.hstack((
                                train_features,
                                train_embedding_features
                                ))

    test_features = np.hstack((
                                test_text_features,
                                test_word_ratio,
                                ))
    test_features = normalizer.transform(test_features)

    test_features = np.hstack((
                                test_features,
                                test_embedding_features
                                ))
    feature_names = [
                        'longest_word_length',
                        'mean_word_length',
                        'length_in_chars',
                        'easy_words_ratio',
                        'stop_words_ratio',
                        'contractions_ratio',
                        'hyperbolic_ratio',
                    ]
    feature_names = feature_names + ['glove_' + str(col) for col in range(100)]            
    return train_features, test_features, feature_names    


def USE_Vec(df):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)
    messages=df.Requirements_clean.values
    with tf.compat.v1.Session() as session:
        session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
        message_embeddings = session.run(embed(messages))
    return np.array(message_embeddings)

def featurize_USE(args,train_df,test_df,embd):
    print('Text Features....')
    train_text_features = text_features(train_df)
    test_text_features = text_features(test_df)
    print('Word ratios....')
    train_word_ratio = word_ratio(args,train_df)
    test_word_ratio = word_ratio(args,test_df)
    
    print('USE.....')
    train_USE = embd(train_df)
    test_USE = embd(test_df)

    normalizer_glove = MinMaxScaler()
    train_USE = normalizer_glove.fit_transform(train_USE)
    test_USE = normalizer_glove.transform(test_USE)

    train_embedding_features = train_USE
    test_embedding_features = test_USE
    train_features = hstack((train_text_features,train_word_ratio,))

    normalizer = MinMaxScaler()
    train_features = normalizer.fit_transform(train_features)

    #train_features = sparse.csr_matrix(train_features)

    train_features = np.hstack((
                                train_features,
                                train_embedding_features
                                ))




    test_features = np.hstack((
                                test_text_features,
                                test_word_ratio,
                                ))
    test_features = normalizer.transform(test_features)
    #test_features = sparse.csr_matrix(test_features)
    test_features = np.hstack((
                                test_features,
                                test_embedding_features
                                ))
    feature_names = [
                        'longest_word_length',
                        'mean_word_length',
                        'length_in_chars',
                        'easy_words_ratio',
                        'stop_words_ratio',
                        'contractions_ratio',
                        'hyperbolic_ratio',
                    ]
    feature_names = feature_names + ['USE' + str(col) for col in range(100)]            
    return train_features, test_features, feature_names    

def preprocessing_for_lstm(xTrain,xTest,yTrain,yTest,glove_model,vocabulary_size,embed_size,max_len):
    ### Create sequence
    tokenizer = Tokenizer(vocabulary_size)
    #print(tokenizer.word_index)
    full_text=np.concatenate((xTrain[:],xTest[:]),axis=0)
    tokenizer.fit_on_texts(full_text)
    vocab=len(tokenizer.word_index)+1
    xTrain_seq = tokenizer.texts_to_sequences(xTrain)
    xTest_seq = tokenizer.texts_to_sequences(xTest)
    xTrain_pad = pad_sequences(xTrain_seq, maxlen=max_len)
    xTest_pad = pad_sequences(xTest_seq, maxlen=max_len)
    encoder = LabelEncoder()
    encoder.fit(yTrain)
    yTrain_enc = to_categorical(encoder.transform(yTrain))
    yTest_enc=to_categorical(encoder.transform(yTest))
    # yTrain_enc and yTest_enc --> one hot encoded
    # xTrain_pad and xTest_pad #(number of samples,max_len) eg. (2521,128)
    embedding_matrix = np.zeros((vocab, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = glove_model.query(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    return xTrain_pad,xTest_pad,yTrain_enc,yTest_enc,embedding_matrix,tokenizer

    