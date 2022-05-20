import sys; print('Python', sys.version)
import pandas as pd
import datetime
from joblib import dump
from preprop import preprocess_train

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA



def train():
    """
    Function to import and preprocess data to train model
    """
    # Data load
    file = './train_set.csv'
    # Preprocessing
    input_val,output_val = preprocess_train(file)
    # SMOTE & RU
    # Undersampling
    n = int(len(input_val)/5)    # Number of final samples per class 
    sampling_dict = {0:32973, 1:11420, 2:n, 3:39888, 4:n}   # Number of points for each class
    rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_dict)
    X_rus, y_rus = rus.fit_resample(input_val,output_val)
    # SMOTE
    sampling_dict = {0:n, 1:n, 2:n, 3:n, 4:n}    # Balanced classes
    sms = SMOTE(random_state=42, sampling_strategy=sampling_dict)
    X_res, y_res = sms.fit_resample(X_rus,y_rus)
    # Exclusion method
    X_train_sm,X_test_sm,y_train_sm,y_test_sm = train_test_split(X_res,y_res, test_size=0.2,
                                                             stratify=y_res,random_state=42)
    #Model training
    rf = RandomForestClassifier(random_state=42,n_estimators=50, criterion='gini', max_depth=50, max_leaf_nodes=10000,n_jobs=-1)
    scaler = StandardScaler()
    model = Pipeline(steps=[('scaler',scaler),('clf',rf)])
    model.fit(X_train_sm,y_train_sm)
    # Save model
    dump(model, 'RandomF.joblib')

if __name__ == '__main__':
    train()







