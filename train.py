
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
# We'll use Average Glove here 
from tqdm import tqdm_notebook
from nltk import word_tokenize
from pymagnitude import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
from utils import print_model_metrics
import multiprocessing
import os
from tqdm import tqdm_notebook
from transformers import get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from numpy import hstack
from scipy import sparse
import torch
import random
import string
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from pymagnitude import *
from nltk.corpus import stopwords 
from contractions import CONTRACTION_MAP
from nltk.stem import WordNetLemmatizer
import pandas as pd
from models import LSTM_tf
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, RobertaForSequenceClassification
from models import CNN_tf
from featurization import preprocessing_for_lstm
from sklearn.preprocessing import LabelEncoder
#from keras.utils import np_utils
#from keras.utils.np_utils import to_categorical
#import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
#print(tf.__version__)
from sklearn.feature_extraction import text 
from featurization import preprocessing_for_transformers
# This is needed for the iterator over the data
# But not necessary if you have TF 2.0 installed
#!pip install tensorflow==2.0.0-beta0
import featurization


# split the dataframe into startified train-test 85:15% 
def split_dataframe(df,c1,c2,return_vals):
    xtrain, xtest, ytrain, ytest = train_test_split(df.loc[:,[c1,c2]], df.loc[:,'labels'], test_size = 0.15, random_state = 0,stratify=df.loc[:,'labels'])
    traindf=pd.concat([xtrain,ytrain],axis=1)
    testdf=pd.concat([xtest,ytest],axis=1)
    xTrain=xtrain.loc[:,'Requirements_clean'].values
    xTest=xtest.loc[:,'Requirements_clean'].values
    yTrain=ytrain.values
    yTest=ytest.values
    return (traindf,testdf,xTrain,xTest,yTrain,yTest)


#def Glove_Vec1(df,glove):
#    vectors = []
#    for title in tqdm_notebook(df.Requirements_clean.values):
#        vectors.append(np.average(glove.word_vectors[glove.dictionary[word_tokenize(title)]], axis = 0))
#    return np.array(vectors)

# Now lets create a dict so that for every word in the corpus we have a corresponding IDF value
# Same as Avg Glove except instead of doing a regular average, we'll use the IDF values as weights.

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
        default='avg_glove',
        type=str,
        required=False,
        help="Type of featurization avg_glove, tfidf_glove or use",
    )
    parser.add_argument(
        "--run_implementation",
        default=0,
        type=int,
        required=True,
        help="run implementation number? (1/2/3)",
    )
    parser.add_argument(
        "--transformer_type",
        default='bert',
        type=str,
        required=True,
        help="Type of transformer architecture to use: bert, roberta or distilroberta",
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
    #print(le.classes_)

    traindf,testdf,xTrain,xTest,yTrain,yTest=split_dataframe(final_df,'Requirements','Requirements_clean','Requirements_clean')

    glove = Magnitude(args.glove_path)
    tfidf = TfidfVectorizer()
    tfidf.fit(xTrain)
    idf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

    if args.featurization_type=='avg_glove':
        train_features, test_features, feature_names = featurization.featurize(args,traindf,testdf,featurization.tfidf_glove,glove,idf_dict)
    elif args.featurization_type=='tfidf_glove':
        train_features, test_features, feature_names = featurization.featurize_USE(args,traindf,testdf,featurization.USE_Vec)
    else:
        train_features, test_features, feature_names = featurization.featurize(args,traindf,testdf,featurization.Glove_Vec,glove,None)
    
    
    xTrain_pad,xTest_pad,yTrain_enc,yTest_enc,embed_matrix,tok=preprocessing_for_lstm(xTrain,xTest,yTrain,yTest,glove,20000,100,128)


    if args.run_implementation==2:


        input_dic={
            'vocabulary_size':len(tok.word_index)+1,
            'embed_size':100,
            'embedding_matrix':embed_matrix,
            'hidden_dim':16,
            'output_size':5,
            'seq_length':xTrain_pad.shape[1]
        }

        model_lstm=LSTM_tf(**input_dic)

        num_epochs = 4
        history = model_lstm.fit(xTrain_pad, yTrain_enc, epochs=num_epochs,verbose=2)

        print(print_model_metrics(y_test=np.array(yTest),y_test_pred=np.array(model_lstm.predict_classes(xTest_pad)),verbose=False,return_metrics=True))

        input_CNN={
            'vocabulary_size':len(tok.word_index)+1,
            'embed_size':100,
            'embedding_matrix':embed_matrix,
            'filters':8,
            'output_size':5,
            'pool_size':3,
            'kernel_size':3,
            'seq_length':xTrain_pad.shape[1]
        }
        model_cnn=CNN_tf(**input_CNN)
        # You can train the model on xTrain_pad but I am loading the saved/trained model

        history = model_cnn.fit(xTrain_pad, yTrain_enc, epochs=num_epochs)
        print(print_model_metrics(y_test=np.array(yTest),y_test_pred=np.array(model_cnn.predict_classes(xTest_pad)),verbose=False,return_metrics=True))
       
       
    if args.run_implementation==3:

        if torch.cuda.is_available():    
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        train_dataset,test_datset=preprocessing_for_transformers(final_df,args.transformer_type)
        batch_size=32

        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

        testing_dataloader = DataLoader(
                    train_dataset, # The validation samples.
                    sampler = SequentialSampler(train_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 5, 
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )

        model.cuda()
        optimizer = AdamW(model.parameters(),
                        lr = 2e-5, 
                        eps = 1e-8 
                        )

        epochs = 4
        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_stats = []

        for epoch_i in range(0, epochs):

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            total_train_loss = 0
            model.train()

            for step, batch in enumerate(train_dataloader):

                if step % 40 == 0 and not step == 0:
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()        

                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                loss, logits = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)

                total_train_loss += loss.item()
                loss.backward()
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)            
            

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))                

            # After the completion of each training epoch, measure our performance on
            print("")
            print("Running Testing...")

            model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
            total_acc=0
            total_f=0
            total_p=0
            total_r=0
            # Evaluate data for one epoch
            for batch in testing_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():        

                    (loss, logits) = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                    
                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += flat_accuracy(logits, label_ids)
                #print(logits)
                pred_flat = np.argmax(logits, axis=1).flatten()
                #print(len(pred_flat))
                #print(len(label_ids))
                t_f,t_p,t_r,t_acc=print_model_metrics(y_test=np.array(label_ids),y_test_pred=np.array(pred_flat),verbose=False,return_metrics=True)
                total_f+=t_f
                total_p+=t_p
                total_r+=t_r
                total_acc+=t_acc
                            
            #avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            #print("  Accuracy: {0:.4f}".format(avg_val_accuracy))
            avg_acc=total_acc/len(testing_dataloader)
            avg_f=total_f/len(testing_dataloader)
            avg_p=total_p/len(testing_dataloader)
            avg_r=total_r/len(testing_dataloader)
            
            avg_test_loss = total_eval_loss / len(testing_dataloader)
            print("  Testing Loss: {0:.2f}".format(avg_test_loss))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': "{:.4f}".format(avg_train_loss),
                    'Test. Loss': "{:.4f}".format(avg_val_loss),
                    #'Valid. Accur.': avg_val_accuracy,
                    'Test. acc': "{:.4f}".format(avg_acc),
                    'Test. f.': "{:.4f}".format(avg_f),
                    'Test. p.': "{:.4f}".format(avg_p),
                    'Test. r.': "{:.4f}".format(avg_r),
                }
            )

        print("")
        print("Training complete!")

if __name__ == "__main__":
    main()






