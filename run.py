# NOTE: PLEASE MAKE SURE YOU ARE RUNNING THIS IN A PYTHON3 ENVIRONMENT
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
import nltk
import argparse
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow_hub as hub
from scipy.stats import randint as sp_randint
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm_notebook
from nltk import word_tokenize
from utils import print_model_metrics
import multiprocessing
import os
from tqdm import tqdm_notebook
from transformers import get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from numpy import hstack
from scipy import sparse
import string
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from pymagnitude import *
from nltk.corpus import stopwords 
from contractions import CONTRACTION_MAP
from nltk.stem import WordNetLemmatizer
import pandas as pd
from models import LSTM_tf
from transformers import BertForSequenceClassification, AdamW, BertConfig, RobertaForSequenceClassification
from models import CNN_tf
from featurization import preprocessing_for_lstm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text 
import featurization


def split_dataframe(df,c1,c2,return_vals):
    xtrain, xtest, ytrain, ytest = train_test_split(df.loc[:,[c1,c2]], df.loc[:,'labels'], test_size = 0.15, random_state = 0,stratify=df.loc[:,'labels'])
    traindf=pd.concat([xtrain,ytrain],axis=1)
    testdf=pd.concat([xtest,ytest],axis=1)
    xTrain=xtrain.loc[:,'Requirements_clean'].values
    xTest=xtest.loc[:,'Requirements_clean'].values
    yTrain=ytrain.values
    yTest=ytest.values
    return (traindf,testdf,xTrain,xTest,yTrain,yTest)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        required=True,
        help="Path to dataframe",
    )
    parser.add_argument(
        "--glove_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained glove embeddings",
    )
    parser.add_argument(
        "--word_dir",
        default=None,
        type=str,
        required=True,
        help="Path to directory containing all special hand-crafted features",
    )
    parser.add_argument(
        "--featurization_type",
        default=0,
        type=int,
        required=False,
        help="Type of featurization 0-->Average Glove, 1-->TF-IDF Glove, 2-->USE",
    )
    parser.add_argument(
        "--run_implementation",
        default=0,
        type=int,
        required=True,
        help="run implementation number? (1/2/3)",
    )
    
    args = parser.parse_args()
    # load dataset
    final_df = pd.read_pickle(args.dataset)

    le=LabelEncoder()
    final_df.labels=le.fit_transform(final_df.labels)

    traindf,testdf,xTrain,xTest,yTrain,yTest=split_dataframe(final_df,'Requirements','Requirements_clean','Requirements_clean')

    glove = Magnitude(args.glove_path)
    tfidf = TfidfVectorizer()
    tfidf.fit(xTrain)
    idf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

    if args.run_implementation==1:
        if args.featurization_type==1:
            train_features, test_features, feature_names = featurization.featurize(args,traindf,testdf,featurization.tfidf_glove,glove,idf_dict)
        elif args.featurization_type==2:
            train_features, test_features, feature_names = featurization.featurize_USE(args,traindf,testdf,featurization.USE_Vec)
        else:
            train_features, test_features, feature_names = featurization.featurize(args,traindf,testdf,featurization.Glove_Vec,glove,None)
        
        y_train = np.array(traindf.labels.values)
        y_test = np.array(testdf.labels.values)

        file_names=[('rf.sav','RF'),('lr.sav','LR'),('bdt.sav','BDT'),('svc.sav','SVC'),('xgb.sav','XGB'),('ann.sav','ANN'),('knn.sav','KNN')]
        for _,name in enumerate(file_names):
            est=tuning(name[1],train_features,y_train)
            print(name[1]+": ")
            print_model_metrics(y_testy_test,y_test_pred=np.array(est.predict(test_features)),verbose=False,return_metrics=True)
            print("\n")
            pickle.dump(est,open(name[0],'wb'))

        


    if args.run_implementation==2:
        xTrain_pad,xTest_pad,yTrain_enc,yTest_enc,embed_matrix,tok=preprocessing_for_lstm(xTrain,xTest,yTrain,yTest,glove,20000,100,128)
        y_train = np.array(traindf.labels.values)
        y_test = np.array(testdf.labels.values)

        input_lstm={
            'vocabulary_size':len(tok.word_index)+1,
            'embed_size':100,
            'embedding_matrix':embed_matrix,
            'hidden_dim':16,
            'output_size':5,
            'seq_length':xTrain_pad.shape[1]
        }

        model_lstm=LSTM_tf(**input_dic)

        num_epochs = 5
        history = model_lstm.fit(xTrain_pad, yTrain_enc, epochs=num_epochs, validation_data=(xTest_pad, yTest_enc))
        print_model_metrics(y_test=np.array(yTest),y_test_pred=np.array(model_lstm.predict_classes(xTest_pad)),verbose=False,return_metrics=True)

        input_CNN={
            'vocabulary_size':len(tok.word_index)+1,
            'embed_size':100,
            'embedding_matrix':None,
            'filters':8,
            'output_size':5,
            'pool_size':3,
            'kernel_size':3,
            'seq_length':xTrain_pad.shape[1]
        }
        model_cnn=CNN_tf(**input_CNN)

        history = model_cnn.fit(xTrain_pad, yTrain_enc, epochs=num_epochs)
        print(print_model_metrics(y_test=np.array(yTest),y_test_pred=np.array(model_cnn.predict_classes(xTest_pad)),verbose=False,return_metrics=True))


if __name__ == "__main__":
    main()


