import pandas as pd
import joblib

from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

# =========================================================
# How to use Imputation()
# =========================================================

# The functions follow the same design principle:
# - Fit preprocessing objects (imputers / scalers) on the TRAIN set only
# - Apply the same transformation to validation/test sets
# - Preserve the original column order
# - Automatically detect binary variables (0/1 or single-unique values)

# IMPORTANT:
# - Pass ONLY feature columns (exclude Target/label).
# - Train and additional datasets must share identical column names.
# - Binary variables are kept in their original scale (not standardized).
# =========================================================


# ---------------------------------------------------------
# (1) Imputation: iterative imputation with chained equations
# ---------------------------------------------------------
# Continuous variables:
#   - IterativeImputer with LinearRegression
#   - Median initialization
#
# Binary variables:
#   - IterativeImputer with LogisticRegression
#   - Most frequent initialization


# ---------------------------------------------------------
# (1) Impute ONLY the training set
# ---------------------------------------------------------
# imputed_train = Imputation(X_train)


# ---------------------------------------------------------
# (2) Impute train + test using imputers fitted on train
# ---------------------------------------------------------
# imputed_train, imputed_test = Imputation(X_train, X_test)


# ---------------------------------------------------------
# (3) Impute train + validation + test
# ---------------------------------------------------------
# imputed_train, imputed_val, imputed_test = Imputation(X_train, X_val, X_test)


# ---------------------------------------------------------
# (4) Save the fitted scaler to a .joblib file
# ---------------------------------------------------------
# save expects a tuple/list of two filenames:
#   save = ("continuous_imputer", "binary_imputer")
#
# imputed_train, imputed_test = Imputation(
#     X_train, X_test,
#     save=("cont_imputer", "bin_imputer")
# )
# -> cont_imputer.joblib, bin_imputer.joblib are saved


# ---------------------------------------------------------
# Manually specify binary variables
# ---------------------------------------------------------
# binary_cols = ["Men", "ChestPain", ...]
# imputed_train, imputed_test = Imputation(X_train, X_test, binary=binary_cols)




def Imputation(train, *args, binary = None, save = False):
    
    original_variable = train.columns
    
    if binary == None:
        binary = list()
        for i in train.columns:
            if (len(train[i].unique()) == 1) | (len(train[i].unique()) == 2):
                binary.append(i)
    else:
        binary = binary
        
    train_binary = train[binary].reset_index(drop=True)
    
    max_values = train.max()    
    min_values = train.min()
    max_values[(max_values == 0) & (min_values == 0)] = 1
    
    conti_imputer = IterativeImputer(estimator=LinearRegression(), random_state=42, initial_strategy='median', skip_complete=True, min_value=0, max_value=max_values)
    train_imputed = pd.DataFrame(conti_imputer.fit_transform(train), columns=train.columns)  

    train_imputed[binary] = train_binary
    
    if train_imputed[binary].isnull().values.any():
        binary_imputer = IterativeImputer(estimator=LogisticRegression(), random_state=42, initial_strategy='most_frequent', skip_complete=False, min_value=0, max_value = 1)
    else: 
        binary_imputer = IterativeImputer(estimator=LogisticRegression(), random_state=42, initial_strategy='most_frequent', skip_complete=True, min_value=0, max_value = 1)
        
    train_imputed = pd.DataFrame(binary_imputer.fit_transform(train_imputed), columns=train.columns)[original_variable]    
    
    imputed_args = [] 
    if args:   
        for data in args:
            data_binary = data[binary].reset_index(drop=True)

            data_imputed = pd.DataFrame(conti_imputer.transform(data), columns=train.columns)  

            if data_imputed[binary].isnull().values.any():
                data_imputed[binary] = data_binary

                data_imputed = pd.DataFrame(binary_imputer.transform(data_imputed), columns=train.columns)[original_variable]

            imputed_args.append(data_imputed)
        
    if save != False:
        joblib.dump(conti_imputer, save[0] + '.joblib')          
        joblib.dump(binary_imputer, save[1] + '.joblib')      
            
    if args:
        return [train_imputed, *imputed_args]
    else:
        return train_imputed



def Scaling(train, *args, binary = None, scaler = StandardScaler(), save = False):
    original_variable = train.columns
    
    if binary == None:
        binary = list()
        for i in train.columns:
            if (len(train[i].unique()) == 1) | (len(train[i].unique()) == 2):
                binary.append(i)
    else:
        binary = binary
            
    train_binary = train[binary].reset_index(drop=True)
    train_conti = train.drop(binary, axis=1).reset_index(drop=True)
    
    scaler = scaler
    train_conti = pd.DataFrame(scaler.fit_transform(train_conti), columns=train_conti.columns)
    train_total = pd.concat([train_conti, train_binary], axis=1).astype(float)[original_variable]
            
    scaled_args = list()
    
    if args:
        for data in args:
            data_binary = data[binary].reset_index(drop=True)
            data_conti = data.drop(binary, axis=1).reset_index(drop=True)
        
            data_conti = pd.DataFrame(scaler.transform(data_conti), columns=train_conti.columns)        
            data_total = pd.concat([data_conti, data_binary], axis=1).astype(float)[original_variable]
        
            scaled_args.append(data_total)
        
    if save != False:
        joblib.dump(scaler, save + '.joblib')
    
    if args:
        return [train_total, *scaled_args]
    else:
        return train_total

