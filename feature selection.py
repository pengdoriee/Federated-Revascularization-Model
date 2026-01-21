from sklearn.feature_selection import SelectKBest
from scipy.stats import shapiro
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import kendalltau
import pandas as pd


# =========================================================
# Feature selection
# =========================================================
# This module provides three feature-ranking methods used in the paper:
# 1) ANOVA F-test (with optional normality check)
# 2) Chi-square test (for binary/categorical features)
# 3) Mutual information
#
# Each function returns a DataFrame with columns:
#   - Feature: feature name
#   - Score: importance score (higher = more important)
#
# Notes:
# - Input DataFrame must contain a column named 'Target' (binary label: 0/1)
# - The global variable `continuous_col` must be defined in advance
#   (list of continuous feature names)
#   (e.g., continuous_col = ["Pulse_rate", "Heart_rate", ...])
# =========================================================


# Kendall's tau score for non-normal continuous variables
def kendall_score(X, y):
    scores = []
    for col in X.columns:
        tau, _ = kendalltau(X[col], y)
        scores.append(abs(tau))
    return scores

# ANOVA F-test for continuous features
def ANOVA_F(df, nd_consider = False):
    
    if nd_consider == True:
        X_train, y_train = df.drop('Target', axis=1), df['Target']

        X_train_conti = X_train[continuous_col]

        nd_result = {'feature': [], 'p': []}
        for i in X_train_conti.columns:
            _, p = shapiro(X_train_conti[i])
            nd_result['feature'].append(i)
            nd_result['p'].append(p)

        nd_result = pd.DataFrame(nd_result)

        nd = nd_result[nd_result['p'] > 0.05]['feature']
        non_nd = nd_result[nd_result['p'] <= 0.05]['feature']

        if len(nd) != 0:
            X_train_nd = X_train_conti[nd]

            fs_conti = SelectKBest(score_func=f_classif, k='all')
            fs_conti.fit(X_train_nd, y_train)

            original_feature_names = X_train_nd.columns

            sorted_indices = sorted(range(len(fs_conti.scores_)), key=lambda i: fs_conti.scores_[i], reverse=True)

            sorted_feature_names = [original_feature_names[i] for i in sorted_indices]

            nd_results = pd.DataFrame({
                'Feature': sorted_feature_names,
                'Score': [fs_conti.scores_[i] for i in sorted_indices]
            })

        if len(non_nd) != 0:
            X_train_non_nd = X_train_conti[non_nd]

            # Calculate Kendall scores for non-normal distributed features
            kendall_scores = kendall_score(X_train_non_nd, y_train)

            original_feature_names = X_train_non_nd.columns

            sorted_indices = sorted(range(len(kendall_scores)), key=lambda i: kendall_scores[i], reverse=True)

            sorted_feature_names = [original_feature_names[i] for i in sorted_indices]

            non_nd_results = pd.DataFrame({
                'Feature': sorted_feature_names,
                'Score': [kendall_scores[i] for i in sorted_indices]
            })

        if (len(nd) != 0) & (len(non_nd) != 0):
            conti_results = pd.concat([nd_results, non_nd_results])

        elif (len(nd) != 0) & (len(non_nd) == 0):
            conti_results = nd_results

        elif (len(nd) == 0) & (len(non_nd) != 0):
            conti_results = non_nd_results

    
    else:
        X_train, y_train = df.drop('Target', axis=1), df['Target']
        
        X_train = X_train[continuous_col]

        fs_conti = SelectKBest(score_func=f_classif, k='all')
        fs_conti.fit(X_train, y_train)

        original_feature_names = X_train.columns

        sorted_indices = sorted(range(len(fs_conti.scores_)), key=lambda i: fs_conti.scores_[i], reverse=True)

        sorted_feature_names = [original_feature_names[i] for i in sorted_indices]

        conti_results = pd.DataFrame({
            'Feature': sorted_feature_names,
            'Score': [fs_conti.scores_[i] for i in sorted_indices]
        })

    return conti_results

# Chi-square test for binary / categorical features
def CHI_test(df):
    
    X_train, y_train = df.drop(continuous_col + ['Target'], axis = 1), df['Target']

    fs_binary = SelectKBest(score_func=chi2, k='all')
    fs_binary.fit(X_train, y_train)

    original_binary_feature_names = X_train.columns

    sorted_binary_indices = sorted(range(len(fs_binary.scores_)), key=lambda i: fs_binary.scores_[i], reverse=True)

    sorted_binary_feature_names = [original_binary_feature_names[i] for i in sorted_binary_indices]

    binary_results = pd.DataFrame({
        'Feature': sorted_binary_feature_names,
        'Score': [fs_binary.scores_[i] for i in sorted_binary_indices]
    })
    
    binary_results = binary_results.sort_values(by = 'Score', ascending=False)

    return binary_results

# Mutual information (continuous + categorical)
def Mutual_info(df):
        
    X_train, y_train = df.drop('Target', axis = 1), df['Target']
    
    cat_order = []
    for i, col in enumerate(X_train.columns):
        if col not in continuous_col:
            cat_order.append(i)
       
    mutual_info = mutual_info_classif(X_train, y_train, discrete_features=cat_order)

    mutual_info_df = pd.DataFrame(mutual_info, columns=['Score'], index=X_train.columns)

    mutual_info_df.reset_index(inplace=True)
    
    mutual_info_df.columns = ['Feature', 'Score']
    
    mutual_info_df = mutual_info_df.sort_values(by = 'Score', ascending=False)

    return mutual_info_df