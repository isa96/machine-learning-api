#!/usr/bin/python3

import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

def colPreparation():
    labelEncoder = ['Gender','Driving_License','Previously_Insured','Vehicle_Damage']
    oneHotEncoder = ['Vehicle_Age','Region_Code','Policy_Sales_Channel']
    scallingStandar = ['Age','Annual_Premium','Vintage']
    return labelEncoder, oneHotEncoder, scallingStandar

def runModel(data, typed='multi'):
    path = pathPackages = os.getcwd()+"/"+"packages"+"/"
    model = pickle.load(open(path + 'model_InsuranceRecommendation.pkl', 'rb'))
    col_p = pickle.load(open(path + 'columnPreparation.pkl', 'rb'))
    col_m = pickle.load(open(path + 'columnModelling.pkl', 'rb'))

    X = data[col_p]
    colEncoder, colpOneHotEncoder, colStandarScaler = colPreparation()
    for col in X.columns:
        prep = pickle.load(open(path + 'prep' + col + '.pkl', 'rb'))
        if col in colpOneHotEncoder:
            dfTemp = pd.DataFrame(prep.transform(X[[col]]).toarray())
            X = pd.concat([X.drop(col, axis=1), dfTemp], axis=1)
        else:
            dfTemp = pd.DataFrame(prep.transform(X[[col]]))
            X = pd.concat([X.drop(col, axis=1), dfTemp], axis=1)
    X.columns = col_m
    
    if typed == 'multi':
        y = model.predict(X)
        return y
    
    elif typed == 'single':
        y = model.predict(X)[0]
        if y == 0:
            return 0
        else:
            return 1
    else:
        return False