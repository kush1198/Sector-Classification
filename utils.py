from sklearn.metrics import precision_recall_curve,precision_recall_fscore_support, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("muted")

def run_log_reg(train_features, test_features, y_train, y_test,  model, confusion = False, return_f1 = False, verbose = True):
    metrics = np.zeros(4)
    for _ in range(10):
        #log_reg = SGDClassifier(loss = 'log', alpha = alpha, n_jobs = -1, penalty = 'l2', random_state=2020)
        log_reg=model
        log_reg.fit(train_features, y_train)
        y_test_pred = log_reg.predict(test_features)
        metrics += print_model_metrics(y_test, y_test_pred, confusion = confusion, verbose = False, return_metrics = True)
    metrics /=10
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | Accuracy: {:.3f} \n'.format(*metrics))
    if return_f1:
        return metrics[0]
    return log_reg

def print_model_metrics(y_test, y_test_pred, confusion = False, verbose = True, return_metrics = False):

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred,average='weighted')
    acc = accuracy_score(y_test, y_test_pred)

    if confusion:
        # Calculate and Display the confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)

        plt.title('Confusion Matrix')
        sns.set(font_scale=1.0) #for label size
        sns.heatmap(cm, annot = True, fmt = 'd', xticklabels = ['Energy', 'Entertainment', 'Health', 'Other', 'Safety'], yticklabels = ['Energy', 'Entertainment', 'Health', 'Other', 'Safety'], annot_kws={"size": 14}, cmap = 'Blues')# font size

        plt.xlabel('Truth')
        plt.ylabel('Prediction')
        
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(f1, best_precision, best_recall, roc_auc, acc))
    
    if return_metrics:
        return f1, precision, recall, acc