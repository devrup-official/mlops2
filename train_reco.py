#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import sys
# import pickle
import azureml as aml
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.core.run import Run
import argparse
from scipy.stats import anderson
from imblearn.over_sampling import SMOTE
# import json
import time
#import traceback
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# from lightgbm import LGBMClassifier, plot_importance
# from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, precision_score, recall_score

import pandas as pd
import numpy as np
# import re
# import math
# import seaborn as sns
# import matplotlib.pyplot as plt
#from sklearn.externals import joblib
import joblib
# import shap


run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace

datastore = ws.get_default_datastore()
datastore_paths = [(datastore, 'recos/train.csv')]

feature_cols = ['Partition column', 'Customer code', 'Employment index',
       'Country Residence', 'Gender', 'Age', 'Join date', 'New customer Index',
       'Customer seniority', 'Primary cusotmer index',
       'Last date as Primary customer', 'Customer type', 'Customer relation',
       'Residence index', 'Foreigner index', 'Spouse emp index',
       'Channel Index', 'Deceased index', 'Address type', 'Province code',
       'Province name', 'Activity index', 'Gross Income', 'Segmentation']
#         dtype_list = {'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16', 'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16', 'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16', 'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'float16', 'ind_nomina_ult1': 'float16', 'ind_cno_fin_ult1': 'float16', 'ncodpers': 'int64', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16', 'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16', 'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16', 'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16'}
        # target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1'] 
target_cols=['Pensions', 'Credit Card', 'Funds', 'Loans', 'Mortgage']
nominals = ['Employment index',
 'Segmentation',
 'Customer type',
 'Customer relation',
 'Province name',
 'Gender',
 'New customer Index',
 'Primary cusotmer index',
 'Residence index',
 'Foreigner index',
 'Deceased index',
 'Activity index']
ordinal = ['Channel Index']
ordinals = nominals+ordinal

def dataprocessoer(val_Data, ordinals):
    
    # Column names
    column_names = ['Partition column','Customer code', 'Employment index', 'Country Residence', 'Gender', 'Age', 'Join date', 'New customer Index',
'Customer seniority', 'Primary cusotmer index', 'Last date as Primary customer', 'Customer type', 'Customer relation',
'Residence index', 'Foreigner index', 'Spouse emp index', 'Channel Index', 'Deceased index', 'Address type', 'Province code', 
'Province name', 'Activity index','Gross Income', 'Segmentation']
    
    # Independent Columns
    val_Data = val_Data.iloc[:,0:24]
    
    # Setting column names
    val_Data.columns = column_names
    
    val_Data.dropna(subset= ['Customer code'], inplace=True)
    
    # Checking for invalid customer
    if((val_Data.shape[0] == 1) and ((val_Data['Customer code'].iloc[0]=='') or (val_Data['Customer code'].isnull().any()))):
        print("Invalid Customer ID")
    else:
        pass
    
    
    val_Data.drop_duplicates(subset= ['Customer code'], inplace = True)
    
    users_code = val_Data[['Customer code']]
    
    # cus rel age
    val_Data['Join date'] = pd.to_datetime(val_Data['Join date'])
    
    x = []
    for i in val_Data['Join date']:
        if(i is pd.NaT):
            x.append(0)
        else:
            x.append(int(pd.Timedelta(pd.to_datetime('today')-i).days/365))
            
    val_Data['cust_rel_age'] = x
    
    
    
    val_Data.drop(dropbales, axis =1, inplace = True)
    val_Data.drop(['Join date','Country Residence','Partition column','Customer code',
                   'Province code','Address type'], axis =1, inplace = True)
    
    # AGE
    val_Data['Age'].isnull().sum() # Tricky
    val_Data['Age'][val_Data['Age'] == ' NA'] # There it is
    not_null_age = val_Data['Age'][val_Data['Age'] != ' NA'].astype(int)
    
    median_Age = int(not_null_age.median())
    val_Data['Age'].replace(' NA', str(median_Age), inplace= True)
    val_Data['Age'] = val_Data['Age'].astype(int)
    
    
    #CUS AGE
    val_Data['Customer seniority'] = val_Data['Customer seniority'].astype(str)
    val_Data['Customer seniority'] = val_Data['Customer seniority'].str.strip()
    val_Data['Customer seniority'].replace('-999999','0', inplace = True)
    
    customer_seniority_in_months = val_Data['Customer seniority'][(val_Data['Customer seniority']!= 'NA') & 
                                                                  (val_Data['Customer seniority'].notnull())].astype(int)
    median_sen = int(customer_seniority_in_months.median())
    val_Data['Customer seniority'].replace('NA', str(median_sen), inplace= True)
    val_Data['Customer seniority'].fillna(str(median_sen), inplace= True)
    val_Data['Customer seniority'] = val_Data['Customer seniority'].astype(int)
    
    
    # Cust type
    val_Data['Customer type'] = val_Data['Customer type'].astype(str)
    val_Data['Customer type'] = val_Data['Customer type'].str.strip()
    mode_of_indrel_1mes = val_Data['Customer type'].mode()[0]
    val_Data['Customer type'].replace('nan',mode_of_indrel_1mes, inplace = True)
    
    
    # Type casting indrel = says if a customer is primary customer throughout or not
    val_Data['Primary cusotmer index'] = val_Data['Primary cusotmer index'].astype(str)
    # Says new customer or not 1 says yes 
    val_Data['New customer Index'] = val_Data['New customer Index'].astype(str)
    # Says if the customer is active or not
    val_Data['Activity index'] =  val_Data['Activity index'].astype(str)
    
    val_Data['Primary cusotmer index'].replace('nan', '1.0', inplace= True)
    val_Data['New customer Index'].replace('nan','0.0', inplace= True)
    val_Data['Activity index'].replace('nan','0.0', inplace= True)
    
    
    #Missing value imputation
    
    numerical_null_cols = val_Data.select_dtypes(include = [np.number]).columns
    obj_null_cols = val_Data.select_dtypes(include = [np.object]).columns
    
    one_point_mods = {'Employment index': 'N',
                         'Country Residence': 'ES',
                         'Gender': 'V',
                         'New customer Index': '0.0',
                         'Primary cusotmer index': '1.0',
                         'Customer type': '1',
                         'Customer relation': 'I',
                         'Residence index': 'S',
                         'Foreigner index': 'N',
                         'Channel Index': 'KHE',
                         'Deceased index': 'N',
                         'Province name': 'MADRID',
                         'Activity index': '0.0',
                         'Segmentation': '02 - PARTICULARES'}
    
    one_point_meads = {'Customer code': 931405.5,
                         'Age': 39.0,
                         'Customer seniority': 54.0,
                         'Gross Income': 101413.54499999998}

    if(val_Data.shape[0] == 1):
        for i in numerical_null_cols:
            if(val_Data[i].isnull().iloc[0]):
                val_Data[i].fillna(one_point_meads[i], inplace= True)
            else:
                pass     
        for i in obj_null_cols:
            if(val_Data[i].isnull().iloc[0]):
                val_Data[i].fillna(one_point_mods[i], inplace= True)
            else:
                pass    
    else:
        numerical_imputer(numerical_null_cols, val_Data)
        objects_imputer(obj_null_cols, val_Data)

    for i in ordinals:
        val_Data[i] = val_Data[i].astype('category')
        val_Data[i] = val_Data[i].cat.codes
        
        
        
    return users_code,val_Data

traindata = Dataset.Tabular.from_delimited_files(path=datastore_paths)
df = traindata.to_pandas_dataframe()
x=df[feature_cols]
userid,x=dataprocessoer(x, ordinals)
y = df[target_cols]
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}
train_dataset_X = {}
trains_dataset_y = {}
test_Dataset_x ={}
test_Dataset_y = {}
for i in y.columns:
    new_ml_df = pd.concat([x, y[i]], axis=1)
    X = new_ml_df.drop(i, axis =1)
    y = new_ml_df[i]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2)
    train_dataset_X[i] = X_train
    trains_dataset_y[i] = y_train
    test_Dataset_x[i] = X_test
    test_Dataset_y[i] = y_test
    
over_sampler = SMOTE(sampling_strategy= 1)
smoted_X = {}
smoted_y = {}
for i in df_y_products.columns:
    smote_X, smote_y = over_sampler.fit_resample(train_dataset_X[i], trains_dataset_y[i])
    smoted_X[i] = smote_X
    smoted_y[i] = smote_y
    
rf = RandomForestClassifier(n_estimators=20, 
                            max_depth=5, 
                            min_samples_leaf=10, 
                            n_jobs=4)
products = y.columns

def eval_mertics(y_test, preds):
    eval = {}
    eval["Accuracy"]= accuracy_score(y_test, preds)
    eval["f1_Score"]=f1_score(y_test, preds)
    eval["precision"]=precision_score(y_test, preds)
    eval["recall"]=recall_score(y_test, preds)
    eval = pd.DataFrame(eval, index = [0]).T
    return eval

eval_params = {}
# feature_importance_table={}
# X_test = test_Dataset_x['Pensions'].copy()
outputs_folder = './model'
os.makedirs(outputs_folder, exist_ok=True)
for i in y.columns:
    rf.fit(smoted_X[i], smoted_y[i])
    preds = rf.predict(test_Dataset_x[i])
    filename = "rf_model_"+i
    eval_params[i] = eval_mertics(test_Dataset_y[i], preds)
    run.log("metrics for "+filename, eval_params[i])
    
#     model_filename = "sklearn_diabetes_model.pkl"
    model_path = os.path.join(outputs_folder, filename)
    dump(rf, model_path)
    
    print("Uploading the model into run artifacts...")
    run.upload_file(name="./outputs/models/" + filename, path_or_stream=model_path)
    print("Uploaded the model {} to experiment {}".format(filename, run.experiment.name))
    dirpath = os.getcwd()
    print(dirpath)
    print("Following files are uploaded ")
    print(run.get_file_names())

#     probs = np.array(np.round(rf.predict_proba(test_Dataset_x[i])[:,1],5), dtype = float)
#     feature_importance_table[i] = rf.feature_importances_
#     X_test[i] = probs
    pickle.dump(rf, open(filename, 'wb'))
run.complete()

