import pandas as pd
from sklearn.preprocessing import *
import joblib


# =========================
# How to use Scaling()
# =========================
# Scaling(train, *args, binary=None, scaler=StandardScaler(), save=False)
#
# What it does:
# - Fits StandardScaler on "continuous columns" of train
# - Keeps "binary columns" (0/1 or single-unique) unchanged
# - Applies the same scaler to any additional datasets passed via *args (e.g., val/test)
# - Returns scaled DataFrames in the same column order as train
#
# Notes:
# - Usually, pass only feature columns (exclude Target/label).
# - train and args DataFrames should have the same columns.

# ---------------------------------------------------------
# (1) Scale ONLY the training set
# ---------------------------------------------------------
# X_train = Scaling(X_train)

# ---------------------------------------------------------
# (2) Scale train + test using the scaler fit on train
# ---------------------------------------------------------
# X_train, X_test = Scaling(X_train, X_test)

# ---------------------------------------------------------
# (3) Scale train + validation + test (same scaler fit on train)
# ---------------------------------------------------------
# X_train, X_val, X_test = Scaling(X_train, X_val, X_test)

# ---------------------------------------------------------
# (4) How to use a different scaler/transformer
# ---------------------------------------------------------
# X_train, X_test = Scaling(X_train, X_test, scaler=MinMaxScaler())

# Other commonly used scikit-learn scalers/transformers include:
# - StandardScaler
# - RobustScaler
# - MaxAbsScaler
# - Normalizer
# - QuantileTransformer
# - PowerTransformer

# ---------------------------------------------------------
# (5) Save the fitted scaler to a .joblib file
# ---------------------------------------------------------
# X_train, X_test = Scaling(X_train, X_test, save="standard_scaler")
# -> saves "standard_scaler.joblib"


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
        
    if save:
        joblib.dump(scaler, save + '.joblib')
    
    if args:
        return [train_total, *scaled_args]
    else:
        return train_total


