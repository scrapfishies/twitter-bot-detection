'''
Helper functions for evaluating Classification Models
'''

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Model support
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             plot_confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, auc)

from visualize import plot_cm


def multi_model_eval(model_list, X, y, kf): 
    ''' 
    Takes a list of models (at same scale) and performs KFolds cross-validation on each
    Inputs:     * list of models to be evaluated
                * X, y training data
                * KFolds parameters
    Returns:    * Scoring metrics for each CV Round
                    Metrics: Accuracy, Precision, Recall, F1, ROC AUC                
    '''
    for model in model_list:

        # Accuracy scores lists
        acc_scores, prec_scores, recall_scores, f1_scores, roc_auc_scores = [], [], [], [], []

        X_kf, y_kf = np.array(X), np.array(y)

        for train_ind, val_ind in kf.split(X, y):

            X_train, y_train = X_kf[train_ind], y_kf[train_ind]
            X_val, y_val = X_kf[val_ind], y_kf[val_ind]

            # Fit model and make predictions
            model.fit(X_train, y_train)
            pred = model.predict(X_val)

            # Score model and append to list
            acc_scores.append(accuracy_score(y_val, pred))
            prec_scores.append(precision_score(y_val, pred))
            recall_scores.append(recall_score(y_val, pred))
            f1_scores.append(f1_score(y_val, pred))
            roc_auc_scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))

        print(f'Model: {model}')
        print("-"*30)
        print(f'Accuracy:  {np.mean(acc_scores):.5f} +- {np.std(acc_scores):5f}')
        print(f'Precision: {np.mean(prec_scores):.5f} +- {np.std(prec_scores):5f}')
        print(f'Recall:    {np.mean(recall_scores):.5f} +- {np.std(recall_scores):5f}')
        print(f'F1 Score:  {np.mean(f1_scores):.5f} +- {np.std(f1_scores):5f}')
        print(f'ROC AUC:   {np.mean(roc_auc_scores):.5f} +- {np.std(roc_auc_scores):5f}')
        print("")


def roc_curve_cv(model, X, y, kf, model_alias): 
    '''
    Plots ROC Curve with AUC score for each fold in KFold cross-validation
    for a provided model. 
    Inputs: * Classification Model
            * X, y training data
            * KFold parameters
            * Model Alias (for plot)        
    ''' 

    # sets up the figure
    plt.figure(figsize=(8, 8))

    # sets up the X, y for KFolds
    X_kf, y_kf = np.array(X), np.array(y)

    # to return mean and std of CV AUC's
    auc_score_list = []

    # to track the CV rounds
    round = 1

    for train_ind, val_ind in kf.split(X, y):

        # Data split
        X_train, y_train = X_kf[train_ind], y_kf[train_ind]
        X_val, y_val = X_kf[val_ind], y_kf[val_ind]

        # Fit model and make predictions
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:,1]

        # ROC curve calculations and plotting
        fpr, tpr, _ = roc_curve(y_val, proba)
        auc_score = roc_auc_score(y_val, proba)
        auc_score_list.append(auc_score)


        plt.plot(fpr, tpr, lw=2, alpha=0.25, label='Fold %d (AUC = %0.4f)' % (round, auc_score))

        round += 1

    # Final output
    print(f'Average AUC Score: {np.mean(auc_score_list):.4f} +- {np.std(auc_score_list):4f}')

    # Plot formatting
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Chance Line', alpha=.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.title(f'Cross-Validation ROC of {model_alias}',fontsize=16)
    plt.legend(loc="lower right", prop={'size': 14})
    plt.show()


def metrics_report(predicted_values, actual_values): 

    conf_matrix = confusion_matrix(predicted_values, actual_values)

    print("Classification Metrics Report")
    print("-----------------------------")
    print('Accuracy:  {:.4f}'.format(accuracy_score(actual_values, predicted_values)))
    print('Precision: {:.4f}'.format(precision_score(actual_values, predicted_values)))
    print('Recall:    {:.4f}'.format(recall_score(actual_values, predicted_values)))
    print('F1 Score:  {:.4f}'.format(f1_score(actual_values, predicted_values)))
    print("")
    print(classification_report(actual_values, predicted_values))
    print("")
    plot_cm(conf_matrix, normalize=False, target_names=['human', 'bot'], title='Confusion Matrix')

