'''
Helper functions for evaluating Classification Models
'''

# Imports
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
                             plot_confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, auc,
                             precision_recall_curve)


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
    plt.figure(figsize=(6, 6), dpi=100)

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
    plt.xlabel('False Positive Rate',fontsize=10)
    plt.ylabel('True Positive Rate',fontsize=10)
    plt.title(f'Cross-Validation ROC of {model_alias}',fontsize=11)
    plt.legend(loc="lower right", prop={'size': 9}, frameon=False)
    sns.despine()
    plt.show()


def precision_recall_cv(model, X, y, kf, model_alias): 
    '''
    Plots Precision-Recall Curves for each fold in KFold cross-validation
    for a provided model. 
    Inputs: * Classification Model
            * X, y training data
            * KFold parameters
            * Model Alias (for plot)        
    ''' 

    # sets up the figure
    plt.figure(figsize=(6, 6), dpi=100)

    # sets up the X, y for KFolds
    X_kf, y_kf = np.array(X), np.array(y)

    # to return mean and std of CV AUC's
    prec_scores, recall_scores = [], []

    # to track the CV rounds
    round = 1

    for train_ind, val_ind in kf.split(X, y):

        # Data split
        X_train, y_train = X_kf[train_ind], y_kf[train_ind]
        X_val, y_val = X_kf[val_ind], y_kf[val_ind]

        # Fit model and make predictions
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        proba = model.predict_proba(X_val)[:,1]

        # Precicion/Recall curve calculations and plotting
        model_precision, model_recall, _ = precision_recall_curve(y_val, proba)
        
        prec_score = precision_score(y_val, pred)
        rec_score = recall_score(y_val, pred)
        
        prec_scores.append(prec_score)
        recall_scores.append(rec_score)

        plt.plot(model_recall, model_precision, marker=',', alpha=0.2, 
                 label=f'Fold {round}: Precision: {prec_score:.2f} / Recall: {rec_score:.2f}')

        round += 1

    # Final output
    print(f'Average Precision Score: {np.mean(prec_scores):.4f} +- {np.std(prec_scores):4f}')
    print(f'Average Recall Score: {np.mean(recall_scores):.4f} +- {np.std(recall_scores):4f}')

    # Plot formatting
    no_skill = len(y_val[y_val==1]) / len(y_val)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall',fontsize=10)
    plt.ylabel('Precision',fontsize=10)
    plt.title(f'Cross-Validated Precision-Recall Curves: {model_alias}',fontsize=11)
    plt.legend(loc="best", prop={'size': 9}, frameon=False)
    sns.despine()
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


def plot_cm(cm,
            target_names,
            title='Confusion matrix',
            cmap=None,
            normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=10)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=10)
    plt.show();
    

def plot_feature_importance(model, features, model_alias):
    
    importance = model.feature_importances_
    feature_importance = list(zip(features, importance))

    feature_importance.sort(key = lambda x: x[1])

    # split sorted features_importance into x,y
    feat = [f[0] for f in feature_importance]
    imp = [i[1] for i in feature_importance]

    # Plot feature importance
    plt.figure(figsize=(7, 5), dpi=100)
    plt.title(f'Feature Importance: {model_alias}', fontsize=11)
    plt.barh(feat, imp, color='#3298dc')
    plt.xlabel('Feature Score', fontsize=9)
    sns.despine()
    plt.show();



    